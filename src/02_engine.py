import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
import multiprocessing as mp
import config
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.tri as tri 
import shap

# Set up matplotlib configuration
plt.rcParams["figure.figsize"] = (18, 4)

G, EPS = 9.81, 1e-12

# =============================================================================
# SECTION 1: UTILITY FUNCTIONS
# =============================================================================
def ensure_cols_exist(df: pd.DataFrame, cols: list, context: str = ""):
    """
    Check if there is any columns missing in the dataset.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns {missing} {context}. Available: {list(df.columns)}")

def format_and_create_path(path: str) -> str:
    """
    Formats the path string and creates the directory if it doesn't exist.
    Ensures the path ends with a '/'.
    """
    if not path.endswith('/'):
        path = path + '/'

    os.makedirs(path, exist_ok=True)
    return path

# =============================================================================
# SECTION 2: DATA LOADING AND PREPARATION
# =============================================================================
def load_and_split_data(cfg):
    print("1. Loading and splitting data..."); df = pd.read_csv(cfg.processed_df_path)

    df.columns = df.columns.str.strip()
    min_wind_speed = 1.0; initial_rows = len(df)
    df = df[df['u10_mod'] >= min_wind_speed].copy()

    print(f"  Sanitization: Removed {initial_rows - len(df)} rows with wind speed < {min_wind_speed} m/s.")

    df['Time'] = pd.to_datetime(df['Time'], errors='coerce'); df.dropna(subset=['Time'], inplace=True)
    base_cols = ['Time', 'latitude', 'longitude', 'Hs', 'Steepness', 'Wave_age', 'y', 'u10_mod']
    ensure_cols_exist(df, base_cols)
    train_start, test_start = pd.to_datetime(cfg.train_initial_date), pd.to_datetime(cfg.test_initial_date)
    train_mask = (df['Time'] >= train_start) & (df['Time'] < test_start)
    train_set, test_set = df.loc[train_mask].copy(), df.loc[df['Time'] >= test_start].copy()

    print(f"  Train set: {len(train_set)} rows | Test set: {len(test_set)} rows")
    
    return train_set, test_set

def create_climatology_feature(train_df, test_df):
    print("2. Creating spatial climatology features...")
    hs_clim = train_df.groupby(['latitude', 'longitude'])['Hs'].mean().reset_index().rename(columns={'Hs': 'Hs_mean_train'})

    global_hs_mean = train_df['Hs'].mean()
    steep_clim = train_df.groupby(['latitude', 'longitude'])['Steepness'].mean().reset_index().rename(columns={'Steepness': 'Steepness_mean_train'})
    global_steep_mean = train_df['Steepness'].mean()
    train_df = train_df.merge(hs_clim, on=['latitude', 'longitude'], how='left').merge(steep_clim, on=['latitude', 'longitude'], how='left')
    test_df = test_df.merge(hs_clim, on=['latitude', 'longitude'], how='left').merge(steep_clim, on=['latitude', 'longitude'], how='left')
    train_df['Hs_mean_train'] = train_df['Hs_mean_train'].fillna(global_hs_mean)
    test_df['Hs_mean_train'] = test_df['Hs_mean_train'].fillna(global_hs_mean)
    train_df['Steepness_mean_train'] = train_df['Steepness_mean_train'].fillna(global_steep_mean) 
    test_df['Steepness_mean_train'] = test_df['Steepness_mean_train'].fillna(global_steep_mean)

    return train_df, test_df

def stratified_sample(df, cfg):
    print("3. Performing stratified sampling on Wave Age...")
    rs = np.random.RandomState(cfg.random_state)
    wa = df['Wave_age'].to_numpy()

    wa_y_cfg, wa_o_cfg = cfg.piecewise_wa_young, cfg.piecewise_wa_old
    n_young, n_mid, n_old = int(cfg.N_SAMPLES * 0.3), int(cfg.N_SAMPLES * 0.4), int(cfg.N_SAMPLES * 0.3)
    young_idx, old_idx = np.flatnonzero(wa <= wa_y_cfg), np.flatnonzero(wa >= wa_o_cfg)
    
    if len(young_idx) < n_young or len(old_idx) < n_old:
        print("  [Warning] Insufficient samples for fixed thresholds. Falling back to quantiles.")
        wa_y_eff, wa_o_eff = np.nanquantile(wa, [0.35, 0.65])
    else: wa_y_eff, wa_o_eff = wa_y_cfg, wa_o_cfg

    young_mask, old_mask = wa <= wa_y_eff, wa >= wa_o_eff
    mid_mask = ~(young_mask | old_mask)

    def _sample_from_mask(mask, k):
        idx = np.flatnonzero(mask); choice = rs.choice(idx, size=min(k, len(idx)), replace=False); return df.iloc[choice]

    sampled_df = pd.concat([_sample_from_mask(young_mask, n_young), _sample_from_mask(mid_mask, n_mid),
                            _sample_from_mask(old_mask, n_old)], ignore_index=True)
    print(f"  Sampled {len(sampled_df)} data points. Effective thresholds: wa_y={wa_y_eff:.3f}, wa_o={wa_o_eff:.3f}")
    return sampled_df, {'wa_y': wa_y_eff, 'wa_o': wa_o_eff}

# =============================================================================
# SECTION 3: MODEL DEFINITION AND TRAINING
# =============================================================================
def build_cnn_model(input_shape):
    """Builds and compiles a deeper 4-layer 2D CNN model."""
    model = Sequential([
        # Layer 1
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.2),

        # Layer 2
        Conv2D(filters=128, kernel_size=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.2),

        # Layer 3
        Conv2D(filters=256, kernel_size=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.3),
        
        # Layer 4
        Conv2D(filters=512, kernel_size=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.3),

        # Fully Connected Head
        Flatten(),
        Dense(128, activation='relu'), # Increased dense layer size
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    print("\n4. CNN Model Summary (Deep 4-Layer 2D Convolution):")
    model.summary()
    return model

# =============================================================================
# SECTION 4: FEATURE IMPORTANCE
# =============================================================================
def get_integrated_gradients(model, X_test_sample, baseline, n_steps=50):
    """Computes Integrated Gradients for a sample."""
    print("  Calculating Integrated Gradients...")
    X_test_sample_tensor = tf.convert_to_tensor(X_test_sample, dtype=tf.float32)
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
    
    attributions = []
    for i in tqdm(range(X_test_sample.shape[0]), desc="IG Progress"):
        interpolated_path = [baseline + (i/n_steps) * (X_test_sample_tensor[i] - baseline) for i in range(n_steps + 1)]
        path_tensor = tf.stack(interpolated_path)
        
        with tf.GradientTape() as tape:
            tape.watch(path_tensor)
            predictions = model(path_tensor)
        
        grads = tape.gradient(predictions, path_tensor)
        avg_grads = tf.reduce_mean(grads, axis=0)
        integrated_grads = (X_test_sample_tensor[i] - baseline) * avg_grads
        attributions.append(integrated_grads.numpy())
        
    return np.mean(attributions, axis=0)

# =============================================================================
# SECTION 5: VISUALIZATION & METRICS (MATCHING 02_symbwaves.py)
# =============================================================================
def evaluate_performance(y_true, y_pred, test_set, cfg):
    print("\n" + "="*45)
    print("--- Model Performance Metrics ---")
    
    # Handle missing config gracefully
    mape_floor = getattr(cfg, 'mape_floor_y', 1e-6)
    
    mape_geral = 100 * np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, mape_floor)))
    u10n_test = (test_set['u10_mod'].values**2) / G
    hs_true, hs_pred = y_true * u10n_test, y_pred * u10n_test
    mape_hs = 100 * np.mean(np.abs((hs_true - hs_pred) / np.maximum(hs_true, EPS)))
    
    print(f"  MAPE(y)  OVERALL: {mape_geral:.2f}%")
    print(f"  MAPE(Hs) OVERALL: {mape_hs:.2f}%")
    print("="*45 + "\n")
    return {'mape_geral': mape_geral, 'mape_hs': mape_hs}


def plot_single_map(ax, lon, lat, data, title, vmin, vmax, cmap='viridis'):
    m = Basemap(ax=ax, projection='merc', llcrnrlat=lat.min()-1, urcrnrlat=lat.max()+1,
                 llcrnrlon=lon.min()-1, urcrnrlon=lon.max()+1, resolution='h')
    m.drawcoastlines(); m.fillcontinents(color='coral', lake_color='aqua')
    m.drawparallels(np.arange(lat.min(), lat.max()+1, 5), labels=[1,0,0,0])
    m.drawmeridians(np.arange(lon.min(), lon.max()+1, 5), labels=[0,0,0,1])
    lons, lats = np.meshgrid(lon, lat)
    
    # --- Force levels to be identical across subplots ---
    levels = np.linspace(vmin, vmax, 13) # 13 levels gives 12 colored intervals
    cs = m.contourf(lons, lats, data, levels=levels, latlon=True, cmap=cmap, extend='both')
    
    m.colorbar(cs, location='right')
    ax.set_title(title)

def generate_mean_maps(df, title_prefix, output_path, mape_value, vmin=None, vmax=None, mape_vmax=50):
    if df.empty: return
    print(f"  Generating mean map for: {title_prefix}")
    mean_data = df.groupby(['latitude', 'longitude']).agg(y_pred_mean=('y_pred', 'mean'), y_real_mean=('y_real', 'mean'), mape_mean=('error', 'mean')).reset_index()
    lons = np.array(sorted(mean_data['longitude'].unique()))
    lats = np.array(sorted(mean_data['latitude'].unique()))
    grid_pred = mean_data.pivot(index='latitude', columns='longitude', values='y_pred_mean').values
    grid_real = mean_data.pivot(index='latitude', columns='longitude', values='y_real_mean').values
    grid_mape = mean_data.pivot(index='latitude', columns='longitude', values='mape_mean').values
    
    if vmin is None: vmin = np.nanmin([grid_pred, grid_real])
    if vmax is None: vmax = np.nanmax([grid_pred, grid_real])
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)); plt.subplots_adjust(wspace=0.3)
    plot_single_map(axes[0], lons, lats, grid_pred, 'Mean Prediction (ŷ)', vmin, vmax)
    plot_single_map(axes[1], lons, lats, grid_real, 'Mean Ground Truth (y)', vmin, vmax)
    
    # --- Using dynamic mape_vmax ---
    plot_single_map(axes[2], lons, lats, grid_mape, f'MAPE (%) -- Avg: {mape_value:.2f}%', 0, mape_vmax, cmap='Reds')
    
    fig.suptitle(f"{title_prefix} Performance", fontsize=16); plt.savefig(output_path, dpi=150, format='pdf', bbox_inches='tight'); plt.close(fig)


def generate_visualizations(results_df, metrics, cfg, thresholds):
    basin = getattr(cfg, 'basin_name', 'south_atlantic')
    print(f"6. Generating visualizations for {basin.upper()}...")
    
    # Use existing structure or fallback
    if hasattr(cfg, 'PROJECT_ROOT'):
        results_dir = os.path.join(cfg.PROJECT_ROOT, f'results/{basin}/')
    else:
        results_dir = format_and_create_path(cfg.results_path + f'/{cfg.save_name}_viz')
        
    os.makedirs(results_dir, exist_ok=True)
    
    # Use thresholds passed from sampling
    wa_y = thresholds['wa_y']
    wa_o = thresholds['wa_o']
    
    # Timeseries
    daily_stats = results_df.set_index('Time').resample('D').mean(numeric_only=True)
    fig, ax1 = plt.subplots(figsize=(18, 6)); ax1.plot(daily_stats.index, daily_stats['error'], color='tab:red', alpha=0.7)
    ax1.axhline(y=metrics['mape_geral'], color='r', ls='--'); ax1.set_ylabel('MAPE (%)', color='tab:red')
    ax2 = ax1.twinx(); ax2.plot(daily_stats.index, daily_stats['y_real'], color='tab:blue', alpha=0.5)
    plt.title(f"Performance Over Time - {basin.replace('_', ' ').capitalize()}"); plt.savefig(os.path.join(results_dir, 'performance_timeseries.pdf'), format='pdf', bbox_inches='tight'); plt.close(fig)
    
    # MAPE vs Wave Age
    bins = np.linspace(results_df['Wave_age'].min(), results_df['Wave_age'].max(), 31)
    results_df['wa_bin'] = pd.cut(results_df['Wave_age'], bins=bins)
    mape_by_wa = results_df.groupby('wa_bin', observed=True)['error'].mean()
    plt.figure(figsize=(12, 5)); mape_by_wa.plot(marker='o'); plt.title(f"MAPE vs. Wave Age - {basin.replace('_', ' ').capitalize()}"); plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(results_dir, 'mape_vs_wave_age.pdf'), format='pdf', bbox_inches='tight'); plt.close()
    
    # Regime-Specific Errors for Maps
    df_ws = results_df[results_df['Wave_age'] <= wa_y]
    df_sw = results_df[results_df['Wave_age'] >= wa_o]
    mape_ws = df_ws['error'].mean() if not df_ws.empty else 0
    mape_sw = df_sw['error'].mean() if not df_sw.empty else 0

    # Overall: Escalas amplas
    generate_mean_maps(results_df, "Overall", os.path.join(results_dir, 'mean_map_overall.pdf'), metrics['mape_geral'], vmin=0.0, vmax=1.8, mape_vmax=50)
    
    # Wind-Sea: Zoom na física (0.1-0.4) e Zoom no erro (0-20%)
    generate_mean_maps(df_ws, "Wind-Sea", os.path.join(results_dir, 'mean_map_windsea.pdf'), mape_ws, vmin=0.15, vmax=0.4, mape_vmax=20)
    
    # Swell: Escalas amplas
    generate_mean_maps(df_sw, "Swell", os.path.join(results_dir, 'mean_map_swell.pdf'), mape_sw, vmin=0.0, vmax=1.8, mape_vmax=50)
    
    return mape_ws, mape_sw

# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Main function to run the spatio-temporal prediction workflow.
    """
    train_set, test_set = load_and_split_data(config)
    save_path = format_and_create_path(config.results_path + f'/{config.save_name}')
    full_path = format_and_create_path(f'{save_path}{config.add_name_version}')

    if 'Hs_mean_train' in config.feature_var or 'Steepness_mean_train' in config.feature_var:
        train_set, test_set = create_climatology_feature(train_set, test_set)

    if config.use_sampling:
        train_set_sampled, thresholds = stratified_sample(train_set, config)
    else:
        # Define default thresholds if sampling not used, based on typical config
        train_set_sampled = train_set.copy()
        thresholds = {'wa_y': config.piecewise_wa_young, 'wa_o': config.piecewise_wa_old}
        
    train_set_sampled = train_set.copy()
    ensure_cols_exist(train_set_sampled, config.feature_var, "in sampled train set")
    ensure_cols_exist(test_set, config.feature_var, "in test set")
   
    X_train_df, y_train = train_set_sampled[config.feature_var], train_set_sampled[config.target_var]
    X_test_df, y_test = test_set[config.feature_var], test_set[config.target_var]
    
    # 4. Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Reshape data for 2D CNN: (samples, height, width, channels)
    X_train_cnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], 1))
    X_test_cnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], 1))

    # 5. Build and train the CNN model OR load a pre-trained one
    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3])
    model_path = f'{save_path}best_cnn_model.keras'

    if config.load_trained_model:
        print(f"\n5. Loading pre-trained model from: {model_path}")
        try:
            model = tf.keras.models.load_model(model_path)
            print("   Model loaded successfully.")
            model.summary()
        except (IOError, OSError) as e:
            print(f"   FATAL: Error loading model file. {e}")
            return 
    else:
        model = build_cnn_model(input_shape)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        
        print("\n5. Training a new model...")
        history = model.fit(X_train_cnn, y_train, epochs=config.n_epochs, batch_size=64,
                            validation_split = 0.2,
                            callbacks=[early_stopping, model_checkpoint], verbose=1)

    # 6. Make Predictions and Prepare Data for Eval
    print("\n6. Making predictions...")
    y_pred = model.predict(X_test_cnn).flatten()

    # 7. Evaluate and Visualize
    # Create the unified results dataframe required by the new visualization functions
    results_df = pd.DataFrame({
        'Time': test_set['Time'].values, 
        'latitude': test_set['latitude'].values, 
        'longitude': test_set['longitude'].values,
        'y_real': y_test.values, 
        'y_pred': y_pred, 
        'Wave_age': test_set['Wave_age'].values,
        'u10_mod': test_set['u10_mod'].values
    })
    
    # Calculate error column exactly as in symbwaves
    mape_floor = getattr(config, 'mape_floor_y', 1e-6) # Fallback if config is missing it
    results_df['error'] = 100 * np.abs((results_df['y_real'] - results_df['y_pred']) / np.maximum(results_df['y_real'], mape_floor))

    # Save the results to csv
    print(f"  Saving predictions to {full_path}test_predictions.csv...")
    results_df.to_csv(f'{full_path}test_predictions.csv', index=False)

    # Calculate Metrics
    metrics = evaluate_performance(results_df['y_real'].values, results_df['y_pred'].values, test_set, config)
    
    # Generate Plots
    mape_ws, mape_sw = generate_visualizations(results_df, metrics, config, thresholds)

    # 8. Feature Importance Analysis (Integrated Gradients)
    train_sample_indices = np.random.choice(X_train_cnn.shape[0], config.n_explain_samples, replace=False)
    test_sample_indices = np.random.choice(X_test_cnn.shape[0], config.n_explain_samples, replace=False)
    
    X_train_sample_cnn = X_train_cnn[train_sample_indices]
    X_test_sample_cnn = X_test_cnn[test_sample_indices]
    
    baseline = np.zeros(input_shape)
    ig_attributions = get_integrated_gradients(model, X_test_sample_cnn, baseline)
    
    ig_scores = pd.Series(ig_attributions.flatten(), index=config.feature_var).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    ig_scores.plot(kind='bar', color='skyblue')
    plt.title('Integrated Gradients Feature Importance')
    plt.ylabel('Attribution')
    plt.tight_layout()
    plt.savefig(f'{full_path}integrated_gradients_importance.png')
    plt.close()
    print("  Integrated Gradients plot saved to integrated_gradients_importance.png")

    # --- FINAL CONSOLE SUMMARY BLOCK (MATCHING SYMBWAVES) ---
    print("\n" + "#"*55)
    print("--- FINAL SUMMARY OF PERFORMANCE ---")
    print("#"*55)
    print("  Model: CNN (Deep 4-Layer 2D)")
    print(f"  Regime Thresholds: WA <= {thresholds['wa_y']:.2f} (Young), WA >= {thresholds['wa_o']:.2f} (Swell)")
    print("\n--- PERFORMANCE BY REGIME ---")
    print(f"  MAPE Overall:  {metrics['mape_geral']:.2f}%")
    print(f"  MAPE Wind-Sea: {mape_ws:.2f}%")
    print(f"  MAPE Swell:    {mape_sw:.2f}%")
    print("#"*55 + "\n")
    
    print("Workflow completed successfully! ✨")


if __name__ == "__main__":
    main()
