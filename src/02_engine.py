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
# import shap

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


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Calculates the Mean Absolute Percentage Error (MAPE), avoiding division by zero.

    When a true value (y_true) is zero, it is replaced by a small positive
    number (epsilon) to prevent division by zero errors.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Create a mask for where y_true is zero
    zero_mask = (y_true == 0)
    
    # Create a copy of y_true to modify
    y_true_safe = np.where(zero_mask, epsilon, y_true)

    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

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
    train_df['Hs_mean_train'].fillna(global_hs_mean, inplace=True); test_df['Hs_mean_train'].fillna(global_hs_mean, inplace=True)
    train_df['Steepness_mean_train'].fillna(global_steep_mean, inplace=True); test_df['Steepness_mean_train'].fillna(global_steep_mean, inplace=True)

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


# def explain_model_with_shap(model, X_train_sample, X_test_sample, feature_names):
#     """Explains the model predictions using SHAP and plots the results."""
#     print("\n7. Explaining model with SHAP...")
#     # Use DeepExplainer for TF/Keras models
#     explainer = shap.DeepExplainer(model, X_train_sample)
#     shap_values = explainer.shap_values(X_test_sample)
    
#     # Generate summary plot
#     plt.figure()
#     shap.summary_plot(shap_values[0].reshape(X_test_sample.shape[0], -1), 
#                       X_test_sample.reshape(X_test_sample.shape[0], -1), 
#                       feature_names=feature_names, show=False)
#     plt.title('SHAP Feature Importance')
#     plt.tight_layout()
#     plt.savefig('shap_feature_importance.png')
#     plt.close()
#     print("  SHAP summary plot saved to shap_feature_importance.png")

# =============================================================================
# SECTION 5: VISUALIZATION
# =============================================================================
def plot_spatial_comparison(df_test, y_true, y_pred, mape):
    """
    Creates and saves a spatial comparison using triangular contour plots.
    """
    print("\n8. Generating spatial comparison plots...")

    # Aggregate data by lat/lon to get mean values at each point
    plot_df = df_test[['latitude', 'longitude']].copy()
    plot_df['y_true'] = y_true
    plot_df['y_pred'] = y_pred
    plot_df['error'] = y_true - y_pred

    spatial_df = plot_df.groupby(['latitude', 'longitude']).mean().reset_index()

    lon = spatial_df['longitude'].values
    lat = spatial_df['latitude'].values

    fig = plt.figure(figsize=(20, 8))
    plt.suptitle(f'Spatio-Temporal Prediction Comparison (Overall MAPE: {mape:.2f}%)', fontsize=16)

    # Plot 1: Real Values
    ax1 = fig.add_subplot(1, 3, 1)
    m1 = Basemap(projection='merc', llcrnrlat=lat.min()-1, urcrnrlat=lat.max()+1,
                 llcrnrlon=lon.min()-1, urcrnrlon=lon.max()+1, resolution='i', ax=ax1)
    m1.drawcoastlines()
    m1.fillcontinents(color='coral', lake_color='aqua')
    m1.drawparallels(np.arange(lat.min(), lat.max()+1, 5), labels=[1,0,0,0])
    m1.drawmeridians(np.arange(lon.min(), lon.max()+1, 5), labels=[0,0,0,1])
    # Use tricontourf for direct plotting from scattered data
    cf1 = m1.tricontourf(lon, lat, spatial_df['y_true'], cmap='viridis', latlon=True, levels=15)
    plt.colorbar(cf1, ax=ax1, label='True "y" Value')
    ax1.set_title('Ground Truth (Mean)')

    # Plot 2: Predicted Values
    ax2 = fig.add_subplot(1, 3, 2)
    m2 = Basemap(projection='merc', llcrnrlat=lat.min()-1, urcrnrlat=lat.max()+1,
                 llcrnrlon=lon.min()-1, urcrnrlon=lon.max()+1, resolution='i', ax=ax2)
    m2.drawcoastlines()
    m2.fillcontinents(color='coral', lake_color='aqua')
    m2.drawparallels(np.arange(lat.min(), lat.max()+1, 5), labels=[1,0,0,0])
    m2.drawmeridians(np.arange(lon.min(), lon.max()+1, 5), labels=[0,0,0,1])
    cf2 = m2.tricontourf(lon, lat, spatial_df['y_pred'], cmap='viridis', latlon=True, levels=15)
    plt.colorbar(cf2, ax=ax2, label='Predicted "y" Value')
    ax2.set_title('CNN Prediction (Mean)')

    # Plot 3: Error (True - Predicted)
    ax3 = fig.add_subplot(1, 3, 3)
    m3 = Basemap(projection='merc', llcrnrlat=lat.min()-1, urcrnrlat=lat.max()+1,
                 llcrnrlon=lon.min()-1, urcrnrlon=lon.max()+1, resolution='i', ax=ax3)
    m3.drawcoastlines()
    m3.fillcontinents(color='coral', lake_color='aqua')
    m3.drawparallels(np.arange(lat.min(), lat.max()+1, 5), labels=[1,0,0,0])
    m3.drawmeridians(np.arange(lon.min(), lon.max()+1, 5), labels=[0,0,0,1])
    cf3 = m3.tricontourf(lon, lat, spatial_df['error'], cmap='coolwarm', latlon=True, levels=15)
    plt.colorbar(cf3, ax=ax3, label='Prediction Error (True - Pred)')
    ax3.set_title('Prediction Error')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("spatio_temporal_comparison.png")
    plt.close()
    print("  Spatial contour plot saved to spatio_temporal_comparison.png")


def main():
    """
    Main function to run the spatio-temporal prediction workflow.
    """
    train_set, test_set = load_and_split_data(config)

    if 'Hs_mean_train' in config.feature_var or 'Steepness_mean_train' in config.feature_var:
        train_set, test_set = create_climatology_feature(train_set, test_set)

    train_set_sampled, thresholds = stratified_sample(train_set, config)
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
    # We treat the feature vector as a grid of size 1 x num_features with 1 channel.
    X_train_cnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], 1))
    X_test_cnn = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], 1))
    
    # 5. Build and train the CNN model
    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3])
    model = build_cnn_model(input_shape)

    # Callbacks for training
    model_path = 'best_cnn_model.keras'
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    print("\n5. Training the model...")
    history = model.fit(X_train_cnn, y_train, epochs=config.n_epochs, batch_size=64,
                        validation_data=(X_test_cnn, y_test),
                        callbacks=[early_stopping, model_checkpoint], verbose=1)

    # 6. Evaluate the model and make predictions
    print("\n6. Evaluating the model...")
    # The best model is already loaded by EarlyStopping's restore_best_weights=True
    y_pred = model.predict(X_test_cnn).flatten()
    mape = calculate_mape(y_test, y_pred)
    print(f"  Final Test Set MAPE: {mape:.2f}%")

    # 7. Feature Importance Analysis
    # Prepare samples for explainers (using a subset for speed)
    n_explain_samples = 500
    train_sample_indices = np.random.choice(X_train_cnn.shape[0], n_explain_samples, replace=False)
    test_sample_indices = np.random.choice(X_test_cnn.shape[0], n_explain_samples, replace=False)
    
    X_train_sample_cnn = X_train_cnn[train_sample_indices]
    X_test_sample_cnn = X_test_cnn[test_sample_indices]
    
    # # SHAP
    # explain_model_with_shap(model, X_train_sample_cnn, X_test_sample_cnn, config.feature_var)
    
    # Integrated Gradients
    baseline = np.zeros(input_shape) # A zero baseline is common
    ig_attributions = get_integrated_gradients(model, X_test_sample_cnn, baseline)
    
    # Plot Integrated Gradients results
    ig_scores = pd.Series(ig_attributions.flatten(), index=config.feature_var).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    ig_scores.plot(kind='bar', color='skyblue')
    plt.title('Integrated Gradients Feature Importance')
    plt.ylabel('Attribution')
    plt.tight_layout()
    plt.savefig('integrated_gradients_importance.png')
    plt.close()
    print("  Integrated Gradients plot saved to integrated_gradients_importance.png")

    # 8. Generate and save visualizations
    plot_spatial_comparison(test_set, y_test.values, y_pred, mape)
    
    print("\nWorkflow completed successfully! ✨")


if __name__ == "__main__":
    main()
