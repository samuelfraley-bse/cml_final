"""
LOS Data Preparation Module
Handles data loading and preprocessing for Length of Stay regression task
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---- Constants ----
TARGET_COL_REG = "LOS"
TARGET_COL_CLASS = "HOSPITAL_EXPIRE_FLAG"  # Not used in LOS task

ID_COLS = [
    "icustay_id",
    "subject_id", 
    "hadm_id",
]

def get_paths():
    """Get data paths for LOS task"""
    base_dir = Path.cwd()
    
    # Check if we're in scripts/regression folder
    if base_dir.name == "regression":
        base_dir = base_dir.parents[1]  # Go up to cml_final
    
    data_dir = base_dir / "data" / "raw" / "MIMIC III dataset LOS"
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"LOS data directory not found: {data_dir}\n"
            "Expected: data/raw/MIMIC III dataset LOS/"
        )
    
    train_path = data_dir / "mimic_train_LOS.csv"
    test_path = data_dir / "mimic_test_LOS.csv"
    
    return base_dir, train_path, test_path

def load_raw_data():
    """Load raw train and test CSVs for LOS"""
    base_dir, train_path, test_path = get_paths()
    print("Train path:", train_path)
    print("Test path:", test_path)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    
    return train_df, test_df

def split_features_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    leak_cols: list = None,
    id_cols: list = None,
):
    """
    Split into features and target for LOS regression
    
    Returns X_train, y_train (LOS values), X_test
    """
    if id_cols is None:
        id_cols = ID_COLS
    if leak_cols is None:
        leak_cols = []
    
    target_col = TARGET_COL_REG
    
    # Make column names case-insensitive
    id_cols_lower = [c.lower() for c in id_cols]
    leak_cols_lower = [c.lower() for c in leak_cols]
    
    # Drop columns (case-insensitive)
    drop_cols_train = []
    for col in train_df.columns:
        col_lower = col.lower()
        if col_lower in id_cols_lower + leak_cols_lower + [TARGET_COL_CLASS.lower(), TARGET_COL_REG.lower()]:
            drop_cols_train.append(col)
    
    drop_cols_test = []
    for col in test_df.columns:
        col_lower = col.lower()
        if col_lower in id_cols_lower + leak_cols_lower:
            drop_cols_test.append(col)
    
    # Find target column (case-insensitive)
    target_col_actual = None
    for col in train_df.columns:
        if col.lower() == target_col.lower():
            target_col_actual = col
            break
    
    if target_col_actual is None:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    
    # Get target
    y_train = train_df[target_col_actual].copy()
    
    # Get features
    X_train_raw = train_df.drop(columns=drop_cols_train)
    X_test_raw = test_df.drop(columns=drop_cols_test)
    
    print(f"Task: regression (target = {target_col_actual})")
    print("X_train_raw shape:", X_train_raw.shape)
    print("X_test_raw shape:", X_test_raw.shape)
    print("y_train shape:", y_train.shape)
    print(f"LOS statistics: mean={y_train.mean():.2f}, median={y_train.median():.2f}, "
          f"min={y_train.min():.2f}, max={y_train.max():.2f}")
    
    return X_train_raw, y_train, X_test_raw

BP_MIN_LOWER_BOUNDS = {
    "SysBP_Min": 40.0,
    "DiasBP_Min": 10.0,
    "MeanBP_Min": 30.0,
}

def clean_min_bp_outliers(df: pd.DataFrame, lower_bounds: dict = None) -> pd.DataFrame:
    """Clean implausible blood pressure values"""
    if lower_bounds is None:
        lower_bounds = BP_MIN_LOWER_BOUNDS
    
    df = df.copy()
    for col, low in lower_bounds.items():
        if col not in df.columns:
            continue
        s = df[col]
        mask_valid = s.notna()
        n_valid = mask_valid.sum()
        if n_valid == 0:
            continue
        below_mask = (s < low) & mask_valid
        n_below = below_mask.sum()
        if n_below > 0:
            pct_valid = n_below / n_valid * 100.0
            print(
                f"[clean_min_bp_outliers] {col}: "
                f"setting {n_below} values ({pct_valid:.3f}% of valid) "
                f"below {low} mmHg to NaN"
            )
            df.loc[below_mask, col] = np.nan
    return df

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features (same as classification)
    These medical features might predict LOS too!
    """
    df = df.copy()
    eps = 1e-3
    
    # Ranges (instability)
    df["HR_range"] = df["HeartRate_Max"] - df["HeartRate_Min"]
    df["SysBP_range"] = df["SysBP_Max"] - df["SysBP_Min"]
    df["MeanBP_range"] = df["MeanBP_Max"] - df["MeanBP_Min"]
    df["RespRate_range"] = df["RespRate_Max"] - df["RespRate_Min"]
    df["SpO2_range"] = df["SpO2_Max"] - df["SpO2_Min"]
    
    # Critical flags
    df["tachy_flag"] = (df["HeartRate_Max"] >= 120).astype(int)
    df["hypotension_flag"] = ((df["SysBP_Min"] < 90) | (df["MeanBP_Min"] < 65)).astype(int)
    df["hypoxia_flag"] = (df["SpO2_Min"] < 92).astype(int)
    
    # Shock index (HR/SBP) - might indicate severity
    if 'HeartRate_Mean' in df.columns and 'SysBP_Mean' in df.columns:
        df["shock_index_mean"] = df["HeartRate_Mean"] / (df["SysBP_Mean"].clip(lower=eps))
    
    # Pulse pressure
    if 'SysBP_Mean' in df.columns and 'DiasBP_Mean' in df.columns:
        df["pulse_pressure_mean"] = df["SysBP_Mean"] - df["DiasBP_Mean"]
    
    return df

def prepare_data(leak_cols: list = None, apply_fe: bool = False):
    """
    Full data prep pipeline for LOS regression
    
    Parameters
    ----------
    leak_cols : list or None
        List of leakage column names to drop from X
    apply_fe : bool
        If True, applies add_engineered_features()
        If False, returns cleaned raw features only
    
    Returns
    -------
    X_train, y_train (LOS values), X_test
    """
    train_df, test_df = load_raw_data()
    X_train_raw, y_train, X_test_raw = split_features_target(
        train_df=train_df,
        test_df=test_df,
        leak_cols=leak_cols,
    )
    
    X_train_clean = clean_min_bp_outliers(X_train_raw)
    X_test_clean = clean_min_bp_outliers(X_test_raw)
    
    if apply_fe:
        X_train_final = add_engineered_features(X_train_clean)
        X_test_final = add_engineered_features(X_test_clean)
    else:
        X_train_final = X_train_clean
        X_test_final = X_test_clean
    
    print("Final X_train shape:", X_train_final.shape)
    print("Final X_test shape:", X_test_final.shape)
    
    return X_train_final, y_train, X_test_final
