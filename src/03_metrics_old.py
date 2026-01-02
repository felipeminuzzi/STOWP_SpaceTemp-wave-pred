import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================
# Update this to point to the folder containing your cnn_results.csv
# If you are using the paths from config, this usually looks like:
# 'results/cnn_experiment/run_1/'
# Here we search for the most recent file recursively if exact path isn't known
SEARCH_PATTERN = 'results/**/cnn_results.csv' 

def find_results_file():
    """Attempts to find the cnn_results.csv file."""
    files = glob.glob(SEARCH_PATTERN, recursive=True)
    if not files:
        # Fallback for manual testing
        if os.path.exists('cnn_results.csv'):
            return 'cnn_results.csv'
        raise FileNotFoundError(f"Could not find 'cnn_results.csv' in {SEARCH_PATTERN}")
    # Return the most recently modified file
    return max(files, key=os.path.getmtime)

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_spatial_metric(ax, lon, lat, data, title, cmap='viridis', vmin=None, vmax=None):
    """Helper for spatial map plotting."""
    m = Basemap(ax=ax, projection='merc', 
                llcrnrlat=lat.min()-1, urcrnrlat=lat.max()+1,
                llcrnrlon=lon.min()-1, urcrnrlon=lon.max()+1, 
                resolution='h')
    m.drawcoastlines()
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawparallels(np.arange(lat.min(), lat.max()+1, 5), labels=[1,0,0,0], fontsize=8)
    m.drawmeridians(np.arange(lon.min(), lon.max()+1, 5), labels=[0,0,0,1], fontsize=8)
    
    lons, lats = np.meshgrid(lon, lat)
    cs = m.contourf(lons, lats, data, levels=15, latlon=True, cmap=cmap, vmin=vmin, vmax=vmax)
    m.colorbar(cs, location='right', pad='5%')
    ax.set_title(title, fontsize=12)

def run_comprehensive_analysis(file_path):
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    
    save_dir = os.path.dirname(file_path)
    base_name = "CNN"

    # --- PART 1: SPATIAL ANALYSIS (4 METRIC MAPS) ---
    print("1. Generating Spatial Metric Maps...")
    def get_spatial_stats(group):
        y_real = group['y_real']
        y_pred = group['y_pred']
        diff = y_pred - y_real
        
        rmse = np.sqrt((diff**2).mean())
        bias = diff.mean()
        # Scatter Index (SI) = RMSE / mean(observed)
        si = rmse / y_real.mean() if y_real.mean() != 0 else np.nan
        # R2 Score
        r2 = r2_score(y_real, y_pred) if len(group) > 1 else np.nan
        
        return pd.Series({'BIAS': bias, 'RMSE': rmse, 'SI': si, 'R2': r2})

    metrics_df = df.groupby(['latitude', 'longitude']).apply(get_spatial_stats, include_groups=False).reset_index()
    lons = np.sort(metrics_df['longitude'].unique())
    lats = np.sort(metrics_df['latitude'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    
    # 1. BIAS
    grid_bias = metrics_df.pivot(index='latitude', columns='longitude', values='BIAS').values
    plot_spatial_metric(axes[0,0], lons, lats, grid_bias.T, f'BIAS (Pred - Real): {base_name}', cmap='RdBu_r')
    
    # 2. RMSE
    grid_rmse = metrics_df.pivot(index='latitude', columns='longitude', values='RMSE').values
    plot_spatial_metric(axes[0,1], lons, lats, grid_rmse.T, f'RMSE: {base_name}', cmap='YlOrRd')
    
    # 3. Scatter Index (SI)
    grid_si = metrics_df.pivot(index='latitude', columns='longitude', values='SI').values
    plot_spatial_metric(axes[1,0], lons, lats, grid_si.T, f'Scatter Index (SI): {base_name}', cmap='magma_r')
    
    # 4. Correlation Coefficient (R2)
    grid_r2 = metrics_df.pivot(index='latitude', columns='longitude', values='R2').values
    plot_spatial_metric(axes[1,1], lons, lats, grid_r2.T, f'$R^2$ Score: {base_name}', cmap='viridis', vmin=0, vmax=1)

    plt.tight_layout()
    map_path = os.path.join(save_dir, f'cnn_spatial_metrics.png')
    plt.savefig(map_path, dpi=200)
    plt.close()
    print(f"  Saved spatial maps to: {map_path}")

    # --- PART 2: TEMPORAL TIME SERIES ---
    print("2. Generating Temporal Analysis...")
    time_avg = df.groupby('Time')[['y_real', 'y_pred']].mean().reset_index()
    
    plt.figure(figsize=(15, 6))
    plt.plot(time_avg['Time'], time_avg['y_real'], label='Truth ($y$)', color='black', lw=1.5, alpha=0.8)
    plt.plot(time_avg['Time'], time_avg['y_pred'], label='CNN Model ($\hat{y}$)', color='tab:red', linestyle='--', alpha=0.9)
    plt.title(f"Averaged Temporal Trend: {base_name}")
    plt.ylabel("Non-dimensional Wave Height ($y$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ts_path = os.path.join(save_dir, f'cnn_timeseries.png')
    plt.savefig(ts_path, dpi=200)
    plt.close()
    print(f"  Saved time series to: {ts_path}")

    # --- PART 3: SCATTER PLOT & REGRESSION ---
    print("3. Generating Scatter Plot...")
    plt.figure(figsize=(8, 8))
    plt.scatter(time_avg['y_real'], time_avg['y_pred'], alpha=0.5, color='blue', s=20, label='Daily/Hourly Means')
    
    # Reference Line (1:1)
    lims = [min(time_avg['y_real'].min(), time_avg['y_pred'].min()), 
            max(time_avg['y_real'].max(), time_avg['y_pred'].max())]
    plt.plot(lims, lims, 'k--', alpha=0.6, label='1:1 Line')
    
    # Regression Fit
    m, b = np.polyfit(time_avg['y_real'], time_avg['y_pred'], 1)
    correlation = time_avg['y_real'].corr(time_avg['y_pred'])
    
    plt.plot(time_avg['y_real'], m*time_avg['y_real'] + b, color='red', 
             label=f'Fit: $y = {m:.2f}x + {b:.2f}$')

    plt.xlabel('Ground Truth ($y$)')
    plt.ylabel('Model Prediction ($\hat{y}$)')
    plt.title(f"Scatter Analysis: {base_name}\nPearson r: {correlation:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    scatter_path = os.path.join(save_dir, f'cnn_scatter_reg.png')
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"  Saved scatter plot to: {scatter_path}")

def main():
    try:
        file_path = find_results_file()
        run_comprehensive_analysis(file_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Please check that 02_engine.py has been run and generated 'cnn_results.csv'.")

if __name__ == "__main__":
    main()
