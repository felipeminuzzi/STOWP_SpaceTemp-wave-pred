# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Import the config file from the current directory
import config

G   = 9.81
EPS = 1e-6

# ===== Config local para IO em chunks do CSV =====
CSV_CHUNKSIZE = 500_000   # linhas por chunk no salvamento
SHOW_SAVE_BAR = True      # mostra barra por chunks na fase de salvamento


def _space_df(dataset, lat, lon):
    """Extrai série em (lat,lon) com seleção 'nearest' (modo laço com barra)."""
    latlon = dataset.sel(latitude=lat, longitude=lon, method='nearest')
    return pd.DataFrame({
        'Time': pd.to_datetime(dataset.time.values),
        'Hs': latlon.swh.values,
        'u10': latlon.u10.values,
        'v10': latlon.v10.values,
        'Peak_period': latlon.pp1d.values,
        'mwd': latlon.mwd.values,
        'mdts': latlon.mdts.values,
        'mdww': latlon.mdww.values,
        'msqs': latlon.msqs.values,
        'mwd1': latlon.mwd1.values,
        'mwd2': latlon.mwd2.values,
        'mwd3': latlon.mwd3.values,        
        'latitude': lat,
        'longitude': lon
    })


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _write_csv_chunked(df: pd.DataFrame, out_path: str, chunksize: int = 500_000, show_bar: bool = True):
    """Escreve DataFrame em CSV por chunks, com barra de progresso e rename atômico."""
    _ensure_dir(out_path)
    tmp_path = out_path + ".tmp"

    n = len(df)
    if n == 0:
        df.head(0).to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)
        return

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    iterator = range(0, n, chunksize)
    if show_bar:
        iterator = tqdm(iterator, desc="Salvando CSV (chunks)", leave=True)

    first = True
    for start in iterator:
        stop = min(start + chunksize, n)
        chunk = df.iloc[start:stop]
        chunk.to_csv(
            tmp_path,
            mode='w' if first else 'a',
            index=False,
            header=first,
            float_format=None,
            encoding='utf-8',
            lineterminator='\n',
        )
        first = False

    os.replace(tmp_path, out_path)


def main():
    print("Carregando arquivo NetCDF bruto...")
    ds = xr.open_dataset(config.raw_df_path)
    
    if hasattr(config, 'use_vectorized') and config.use_vectorized:
        # Increased total from 4 to 5 for the new step
        pbar = tqdm(total=5, desc="Pré-processamento (vetorizado)", leave=True)

        # Fase 1: dataframe vetorizado
        df = ds[['swh', 'u10', 'v10', 'pp1d']].to_dataframe().reset_index()
        df.rename(columns={'swh': 'Hs', 'pp1d': 'Peak_period', 'time': 'Time'}, inplace=True)
        pbar.update(1)

        # Fase 2: cálculos físicos/adimensionais
        df['u10_mod'] = np.sqrt(df['u10']**2 + df['v10']**2)
        df['u10_cosine'] = df['u10'] / (df['u10_mod'] + EPS)
        df['u10_sine']   = df['v10'] / (df['u10_mod'] + EPS)
        df['u10_n'] = (df['u10_mod']**2) / G
        df['Peak_period_n'] = (G * df['Peak_period']) / (df['u10_mod'] + EPS)        # Tp*
        df['Wave_age'] = df['Peak_period_n'] / (2*np.pi)                              # Tp*/(2π)
        df['y'] = df['Hs'] / (df['u10_n'] + EPS)
        pbar.update(1)

        # ======================================================================
        # --- MODIFICATION START: ADDED STEEPNESS CALCULATION ---
        # ======================================================================
        # Fase 3: Engenharia da feature de Steepness
        print("\nEngineering new feature: Instantaneous Wave Steepness...")
        # 1. Calculate Wavelength (L) from Peak Period (T) in deep water
        #    Formula: L = (g * T^2) / (2 * pi)
        df['Wavelength'] = (G * df['Peak_period']**2) / (2 * np.pi)
        
        # 2. Calculate Wave Steepness (Hs / L)
        df['Steepness'] = df['Hs'] / (df['Wavelength'] + EPS)
        pbar.update(1)
        # ======================================================================
        # --- MODIFICATION END ---
        # ======================================================================

        # Fase 4: normalização e codificação cíclica de longitude
        lat_min, lat_max = df['latitude'].min(),  df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        df['lat_norm'] = (df['latitude']  - lat_min) / (lat_max - lat_min + EPS)
        df['lon_norm'] = (df['longitude'] - lon_min) / (lon_max - lon_min + EPS)

        lon_rad = np.deg2rad(df['longitude'])
        df['lon_sin'] = np.sin(lon_rad)
        df['lon_cos'] = np.cos(lon_rad)
        pbar.update(1)

        # Fase 5: seleção e salvamento
        final_cols = [
            'Time', 'latitude', 'longitude',
            'Hs', 'u10', 'v10', 'u10_mod',
            'Peak_period', 'Peak_period_n',   # Tp*
            'u10_n', 'Wave_age',
            'y',
            'Steepness', # <-- ADDED NEW FEATURE
            'lat_norm', 'lon_norm',
            'lon_sin', 'lon_cos',
            'u10_cosine', 'u10_sine',
        ]
        df = df[final_cols]

        est_bytes = df.memory_usage(deep=True).sum()
        print(f"DataFrame pronto: shape={df.shape}, ~{est_bytes/1e6:.1f} MB em memória")
        print(f"Salvando em: {config.processed_df_path} (chunksize={CSV_CHUNKSIZE})")

        _write_csv_chunked(df, config.processed_df_path, chunksize=CSV_CHUNKSIZE, show_bar=SHOW_SAVE_BAR)
        pbar.update(1)
        pbar.close()

    else: # Non-vectorized path (also updated for consistency)
        lats = ds.latitude.values
        lons = ds.longitude.values
        grid_points = [(lo, la) for lo in lons for la in lats]
        print(f"Processando grade: {len(grid_points)} pontos...")

        frames = []
        for lo, la in tqdm(grid_points, desc="Montando DataFrame (laço)", disable=not hasattr(config, 'show_grid_progress') or not config.show_grid_progress):
            frames.append(_space_df(ds, la, lo))
        df = pd.concat(frames, ignore_index=True)

        pbar = tqdm(total=4, desc="Pós-processamento", leave=True)

        # Step 1: Base physics
        df['u10_mod'] = np.sqrt(df['u10']**2 + df['v10']**2)
        df['u10_cosine'] = df['u10'] / (df['u10_mod'] + EPS)
        df['u10_sine']   = df['v10'] / (df['u10_mod'] + EPS)
        df['u10_n'] = (df['u10_mod']**2) / G
        df['Peak_period_n'] = (G * df['Peak_period']) / (df['u10_mod'] + EPS)
        df['Wave_age'] = df['Peak_period_n'] / (2*np.pi)
        df['y'] = df['Hs'] / (df['u10_n'] + EPS)
        pbar.update(1)

        # Step 2: Steepness
        df['Wavelength'] = (G * df['Peak_period']**2) / (2 * np.pi)
        df['Steepness'] = df['Hs'] / (df['Wavelength'] + EPS)
        pbar.update(1)

        # Step 3: Normalization
        lat_min, lat_max = df['latitude'].min(),  df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        df['lat_norm'] = (df['latitude']  - lat_min) / (lat_max - lat_min + EPS)
        df['lon_norm'] = (df['longitude'] - lon_min) / (lon_max - lon_min + EPS)
        lon_rad = np.deg2rad(df['longitude'])
        df['lon_sin'] = np.sin(lon_rad)
        df['lon_cos'] = np.cos(lon_rad)
        pbar.update(1)

        # Step 4: Finalize and save
        final_cols = [
            'Time', 'latitude', 'longitude',
            'Hs', 'u10', 'v10', 'u10_mod',
            'Peak_period', 'Peak_period_n',
            'u10_n', 'Wave_age', 'y', 'Steepness',
            'mdts','mdww','msqs','mwd1','mwd2','mwd3','mwd',
            'lat_norm', 'lon_norm', 'lon_sin', 'lon_cos',
            'u10_cosine', 'u10_sine'
        ]
        df = df[final_cols]
        est_bytes = df.memory_usage(deep=True).sum()
        print(f"DataFrame pronto: shape={df.shape}, ~{est_bytes/1e6:.1f} MB em memória")
        print(f"Salvando em: {config.processed_df_path} (chunksize={CSV_CHUNKSIZE})")

        _write_csv_chunked(df, config.processed_df_path, chunksize=CSV_CHUNKSIZE, show_bar=SHOW_SAVE_BAR)
        pbar.update(1)
        pbar.close()

    print(f"OK: Processed data saved to {config.processed_df_path}")


if __name__ == "__main__":
    main()