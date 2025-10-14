# -*- coding: utf-8 -*-
"""
config.py

Configuration file for the SymbWaves symbolic regression pipeline.
This version is location-aware and should be kept in the 'scripts' folder.
"""
import os
# ===========================
#  PROJECT ROOT PATH (IMPORTANT!)
# ===========================
# Automatically determine the project's root directory by going up one level
# from the current script's location (the 'scripts' folder).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ===========================
#  PATHS (Now built from PROJECT_ROOT)
# ===========================
# Path to the raw NetCDF file
raw_df_path       = os.path.join(PROJECT_ROOT, 'data', 'raw', 'dados_full2018_2023.nc')
# Path for the processed CSV data
processed_df_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'era5_structured_weighted.csv')
# Path for the results
results_path      = os.path.join(PROJECT_ROOT, 'results')
save_name         = 'first_train_v0'

# ===========================
#  MODEL TRAINING
# ===========================
new_train      = True

# ===========================
#  DATA SPLIT & SAMPLING
# ===========================
train_initial_date = '2018-01-01'
test_initial_date  = '2022-12-31'
n_epochs = 1
random_state = 42
N_SAMPLES = 50_000
# ===========================
#  FEATURES & TARGET
# ===========================
# This is the key change for our new experiment. We are adding the powerful
# 'Steepness_mean_train' feature to give PySR a better physical clue about
# the dominant wave regime (wind-sea vs. swell) at each location.
feature_var = [
    'Hs_mean_train',
    'Steepness_mean_train',
    'Hs', 'u10', 'v10', 'u10_mod',
    'Peak_period', 'Peak_period_n',
    'u10_n', 'Wave_age', 'Steepness',
    'mdts','mdww','msqs','mwd1','mwd2','mwd3','mwd',
    'lat_norm', 'lon_norm', 'lon_sin', 'lon_cos',
    'u10_cosine', 'u10_sine'    
]

# Target variable for the regression.
target_var = 'y'
# ===========================
#  DUAL-MODEL GATING
# ===========================
use_dual_models = True
piecewise_wa_young = 1.3
piecewise_wa_old = 2.0
swell_stability_threshold = 20.0 
gating_type = 'logistic'
logistic_center = 0.5
logistic_width = 0.2
# ===========================
#  VISUALIZATION
# ===========================
region_time = '2019-05-15 18:00:00'
basemap_resolution = 'i'
MIN_COUNT_PER_CELL = 20
SAVE_COUNT_MAPS = True

