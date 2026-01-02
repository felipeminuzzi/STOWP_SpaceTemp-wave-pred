import numpy as np
import pandas as pd
import sys

# =============================================================================
# CONSTANTS & CONFIG MOCK
# =============================================================================
G = 9.81
EPS = 1e-12

class MockConfig:
    """
    Mock configuration class to replicate the 'cfg' object expected by 
    the evaluate_performance function from 02_symbwaves.py.
    """
    def __init__(self):
        self.mape_floor_y = 1e-6
        self.mape_floor_hs = 0.0
        
        # Default Thresholds used in 02_engine.py default sampling
        # Adjust these if you used different thresholds in your run.
        self.piecewise_wa_young = 1.3   # Default Young/Wind-sea limit
        self.piecewise_wa_old = 2.0     # Default Swell limit
        self.swell_stability_threshold = 20.0 # Upper bound for stable swell

# =============================================================================
# SECTION 1: UTILITY FUNCTIONS (COPIED FROM 02_symbwaves.py)
# =============================================================================
def rmse(true: np.ndarray, pred: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is not None:
        true, pred = true[mask], pred[mask]
    if true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((true - pred) ** 2)))

def mae(true: np.ndarray, pred: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is not None:
        true, pred = true[mask], pred[mask]
    if true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)))

def mape_omega(true: np.ndarray, pred: np.ndarray, floor: float, mask: np.ndarray = None) -> float:
    """
    Sample-based MAPE_Ω (%), with a floor in the denominator:
      100 * mean( |true - pred| / max(true, floor) )
    """
    if mask is not None:
        true, pred = true[mask], pred[mask]
    if true.size == 0:
        return float("nan")
    denom = np.maximum(true, floor)
    return float(100.0 * np.mean(np.abs(true - pred) / denom))

# =============================================================================
# SECTION 2: EVALUATION FUNCTION (EXACT REPLICA FROM 02_symbwaves.py)
# =============================================================================
def evaluate_performance(y_true, y_pred, test_set, cfg):
    """
    Computes global sample-based metrics for y and Hs:
      - MAPE_Ω(y), RMSE(y), MAE(y)
      - MAPE_Ω(Hs), RMSE(Hs), MAE(Hs)
    """
    print("\n" + "=" * 55)
    print("--- Model Performance Metrics (sample-based) ---")

    # floors
    y_floor = float(getattr(cfg, "mape_floor_y", 1e-6))

    # Dimensionless metrics
    mape_y = mape_omega(y_true, y_pred, floor=y_floor)
    rmse_y = rmse(y_true, y_pred)
    mae_y_val = mae(y_true, y_pred)

    # Convert to Hs (meters): Hs = y * U10^2 / g
    # Note: test_set must contain 'u10_mod'
    u10 = test_set["u10_mod"].values
    u10_sq_over_g = (u10 ** 2) / G
    hs_true = y_true * u10_sq_over_g
    hs_pred = y_pred * u10_sq_over_g

    # floor for Hs MAPE: either provided, or derived from y_floor
    hs_floor = float(getattr(cfg, "mape_floor_hs", 0.0))
    if hs_floor <= 0.0:
        # conservative derived per-sample floor based on y_floor
        hs_floor = float(np.nanpercentile(u10_sq_over_g, 5) * y_floor)  
        hs_floor = max(hs_floor, 1e-4)

    mape_hs = mape_omega(hs_true, hs_pred, floor=hs_floor)
    rmse_hs = rmse(hs_true, hs_pred)
    mae_hs_val = mae(hs_true, hs_pred)

    print(f"  MAPE_Ω(y)    OVERALL: {mape_y:6.2f}%")
    print(f"  RMSE(y)      OVERALL: {rmse_y:8.5f}")
    print(f"  MAE(y)       OVERALL: {mae_y_val:8.5f}")
    print("-" * 55)
    print(f"  MAPE_Ω(Hs)   OVERALL: {mape_hs:6.2f}%")
    print(f"  RMSE(Hs)     OVERALL: {rmse_hs:8.3f} m")
    print(f"  MAE(Hs)      OVERALL: {mae_hs_val:8.3f} m")
    print("=" * 55 + "\n")

    return {
        "mape_y": mape_y,
        "rmse_y": rmse_y,
        "mae_y": mae_y_val,
        "mape_hs": mape_hs,
        "rmse_hs": rmse_hs,
        "mae_hs": mae_hs_val,
        "hs_floor_used": hs_floor,
        "y_floor_used": y_floor,
        "hs_true": hs_true, # Returning these to use in regime calculation below
        "hs_pred": hs_pred
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # --- CONFIGURATION: SET YOUR CSV PATH HERE ---
    CSV_PATH = "/home/minuzzi/Documents/STOWP_SpaceTemp-wave-pred/results/south_atlantic/test_predictions.csv"  # Update this path
    # ---------------------------------------------

    print(f"Reading predictions from: {CSV_PATH}")
    
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {CSV_PATH}")
        sys.exit(1)

    # Clean whitespace in columns just in case
    df.columns = df.columns.str.strip()

    # Ensure required columns exist
    required_cols = ['y_real', 'y_pred', 'u10_mod', 'Wave_age']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns in CSV: {missing}")
        sys.exit(1)

    # Initialize Mock Config
    cfg = MockConfig()

    # Extract Arrays
    y_true = df['y_real'].values
    y_pred = df['y_pred'].values

    # 1. Run the Evaluation Function
    # We pass 'df' as 'test_set' because it contains 'u10_mod' required for Hs conversion
    metrics = evaluate_performance(y_true, y_pred, df, cfg)

    # 2. Calculate Regime-Specific Metrics (Wind-Sea vs Swell)
    # This replicates the "Final Summary" logic from 02_symbwaves.py
    
    wa = df['Wave_age'].values
    hs_true = metrics['hs_true']
    hs_pred = metrics['hs_pred']
    
    # Define Regimes
    wa_y_eff = cfg.piecewise_wa_young
    wa_o_eff = cfg.piecewise_wa_old
    
    mask_ws = wa <= wa_y_eff
    mask_sw = (wa >= wa_o_eff) & (wa < cfg.swell_stability_threshold)
    
    # Calculate Regime Metrics
    # -- Overall --
    mape_y_all = metrics['mape_y']
    rmse_y_all = metrics['rmse_y']
    mae_y_all  = metrics['mae_y']
    rmse_hs_all = metrics['rmse_hs']
    mae_hs_all  = metrics['mae_hs']

    # -- Wind-Sea --
    mape_y_ws = mape_omega(y_true, y_pred, floor=metrics['y_floor_used'], mask=mask_ws)
    rmse_y_ws = rmse(y_true, y_pred, mask=mask_ws)
    mae_y_ws  = mae(y_true, y_pred, mask=mask_ws)
    
    rmse_hs_ws = rmse(hs_true, hs_pred, mask=mask_ws)
    mae_hs_ws  = mae(hs_true, hs_pred, mask=mask_ws)

    # -- Swell --
    mape_y_sw = mape_omega(y_true, y_pred, floor=metrics['y_floor_used'], mask=mask_sw)
    rmse_y_sw = rmse(y_true, y_pred, mask=mask_sw)
    mae_y_sw  = mae(y_true, y_pred, mask=mask_sw)

    rmse_hs_sw = rmse(hs_true, hs_pred, mask=mask_sw)
    mae_hs_sw  = mae(hs_true, hs_pred, mask=mask_sw)

    # 3. Print Final Summary (Exactly matching 02_symbwaves.py format)
    print("\n" + "#" * 65)
    print(f"--- FINAL SUMMARY (Calculated from CSV) ---")
    print("#" * 65)
    
    print(f"  Regime Thresholds Used: WA <= {wa_y_eff} (Wind-Sea), WA >= {wa_o_eff} (Swell)")
    print("\n--- PERFORMANCE BY REGIME (sample-based) ---")
    
    # Overall
    print(f"  Overall  -> MAPE_Ω(y): {mape_y_all:6.2f}% | RMSE(y): {rmse_y_all:8.5f} | MAE(y): {mae_y_all:8.5f}"
          f" | RMSE(Hs): {rmse_hs_all:6.3f} m | MAE(Hs): {mae_hs_all:6.3f} m")
    
    # Wind-Sea
    print(f"  Wind-Sea -> MAPE_Ω(y): {mape_y_ws:6.2f}% | RMSE(y): {rmse_y_ws:8.5f} | MAE(y): {mae_y_ws:8.5f}"
          f" | RMSE(Hs): {rmse_hs_ws:6.3f} m | MAE(Hs): {mae_hs_ws:6.3f} m")
    
    # Swell
    print(f"  Swell    -> MAPE_Ω(y): {mape_y_sw:6.2f}% | RMSE(y): {rmse_y_sw:8.5f} | MAE(y): {mae_y_sw:8.5f}"
          f" | RMSE(Hs): {rmse_hs_sw:6.3f} m | MAE(Hs): {mae_hs_sw:6.3f} m")
    
    print("#" * 65 + "\n")
