# STOWP - SpaceTemporal Ocean Wave Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A CNN framework to predict ocean wave variables in two-dimensional domains.

## Overview

This repository houses the STOWP - SpaceTemporal Ocean Wave Prediction pipeline, a deep learning  project designed to predict non-dimensional wave parameters using ERA5 
reanalysis data. The system utilizes a Convolutional Neural Network (CNN) to 
analyze wind-sea and swell regimes, incorporating features like Wave Age and 
Wave Steepness.

## Prerequisites

1. Python 3.8+
2. System Dependencies: The visualization module uses 'Basemap', which often 
   requires 'libgeos' installed at the system level (e.g., 
   'sudo apt-get install libgeos-dev' on Linux).

## Dependencies

Install the necessary dependencies (TensorFlow, Keras, Xarray, Pandas, 
Matplotlib, Basemap, SHAP, etc.) using:

    pip install -r requirements.txt

## Project Structure

The project is organized modularly to separate configuration, processing, and modeling.

```
├── data/
│   ├── raw/                # Place your input NetCDF files here 
│   └── processed/          # Generated files from 01_create_data.py will be saved here
├── results/                # Model checkpoints, logs, and visualization outputs
├── src/                    # Source code
│   ├── config.py           # Central configuration for paths and parameters
│   ├── 01_create_data.py   # Pre-processing pipeline (NetCDF -> CSV)
│   ├── 02_engine.py        # CNN Model training, evaluation, and visualization
├── requirements.txt        # lib dependencies
├── LICENSE                 # Project License
└── README.md               # Project documentation
```

## Getting Started

Step 0: Clone the repository:
---------------------
    ```bash
    git clone https://github.com/felipeminuzzi/STOWP_SpaceTemp-wave-pred.git
    cd STOWP_SpaceTemp-wave-pred
    ```

Step 1: Configuration
---------------------
Edit 'src/config.py' to adjust paths and hyperparameters.
  - Paths: The script automatically detects the project root relative to the 
    'scripts' folder.
  - Model Control: Set 'load_trained_model = False' to retrain the CNN from 
    scratch.
  - Features: Modify the 'feature_var' list to add or remove inputs. The 
    current setup uses features like 'Hs_mean_train', 'Steepness_mean_train', 
    and circular directional variables.

Step 2: Data Pre-processing
---------------------------
Run the data creation script to transform raw NetCDF data into an engineered 
CSV format:

    python ./src/01_create_data.py

This script performs the following operations:
  - Reads raw NetCDF files via Xarray.
  - Feature Engineering: Calculates physical variables such as Wave Age, 
    Wave Steepness (Hs / L), and non-dimensional peak period.
  - Vectorization: Converts circular directional variables (degrees) into 
    sine/cosine pairs (e.g., u10_sine, mwd_cos).
  - Saves the output in chunks to 'data/processed/'.

Step 3: Model Training & Evaluation
-----------------------------------
Run the main engine to train the model and generate results:

    python src/02_engine.py

This script handles:
  - Climatology: Calculates spatial mean features (Hs_mean_train, 
    Steepness_mean_train) to provide location awareness to the model.
  - Stratified Sampling: Implements optional sampling based on Wave Age 
    thresholds to balance wind-sea and swell representation.
  - CNN Training: Trains a 4-layer 2D Convolutional Neural Network.
  - Explainability: Computes Integrated Gradients to determine feature 
    importance.

================================================================================
OUTPUTS
================================================================================

Results are saved to the 'results/' directory defined in your config file. 
Key artifacts include:

1. Visualizations:
   - spatio_temporal_comparison.png: Spatial maps comparing ERA5 Ground Truth 
     vs. CNN Predictions and relative error.
   - wave_age_comparison.png: Performance breakdown split by "Wind Sea" 
     (Young waves) and "Swell" (Old waves) conditions.
   - integrated_gradients_importance.png: Bar chart showing the attribution 
     of each input feature.

2. Model:
   - best_cnn_model.keras: The saved, trained neural network.