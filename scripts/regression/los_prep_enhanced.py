"""
Enhanced LOS Data Preparation Module
Based on hef_prep.py structure, adapted for Length of Stay regression task

Handles:
- Data loading
- Feature splitting
- Age features
- BP outlier cleaning
- Core engineered features (no flags, no temporal)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_COL_REG = "LOS"
TARGET_COL_CLASS = "HOSPITAL_EXPIRE_FLAG"  # Not used in LOS task

ID_COLS = [
    "icustay_id",
    "subject_id", 
    "hadm_id",
]

BP_MIN_LOWER_BOUNDS = {
    "SysBP_Min": 40.0,
    "DiasBP_Min": 10.0,
    "MeanBP_Min": 30.0,
}

# ============================================================================
# DATA LOADING
# ============================================================================

def get_paths():
    """Get data paths for LOS task"""
    base_dir = Path.cwd()
    
    # Check if we're in scripts folder
    if base_dir.name in ["regression", "scripts"]:
        base_dir = base_dir.parents[1] if base_dir.name != "scripts" else base_dir.parent
    
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
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✓ Train: {train_df.shape}")
    print(f"✓ Test: {test_df.shape}")
    
    return train_df, test_df


# ============================================================================
# FEATURE/TARGET SPLITTING
# ============================================================================

def split_features_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: str = "reg",
    leak_cols: list = None,
    id_cols: list = None,
):
    """
    Split into features and target
    
    Parameters
    ----------
    task : str
        "reg" for LOS regression
    leak_cols : list
        Columns to drop (leakage)
    id_cols : list
        ID columns to drop
    
    Returns
    -------
    X_train, y_train, X_test
    """
    if id_cols is None:
        id_cols = ID_COLS
    if leak_cols is None:
        leak_cols = []
    
    # LOS regression
    target_col = TARGET_COL_REG
    
    # Make case-insensitive
    id_cols_lower = [c.lower() for c in id_cols]
    leak_cols_lower = [c.lower() for c in leak_cols]
    
    # Drop columns
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
    
    # Find target
    target_col_actual = None
    for col in train_df.columns:
        if col.lower() == target_col.lower():
            target_col_actual = col
            break
    
    if target_col_actual is None:
        raise ValueError(f"Target column '{target_col}' not found")
    
    y_train = train_df[target_col_actual].copy()
    X_train_raw = train_df.drop(columns=drop_cols_train)
    X_test_raw = test_df.drop(columns=drop_cols_test)
    
    print(f"\nTask: {task} (target = {target_col_actual})")
    print(f"X_train_raw shape: {X_train_raw.shape}")
    print(f"X_test_raw shape: {X_test_raw.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"LOS stats: mean={y_train.mean():.2f}, median={y_train.median():.2f}, "
          f"min={y_train.min():.2f}, max={y_train.max():.2f}")
    
    return X_train_raw, y_train, X_test_raw


# ============================================================================
# AGE FEATURES
# ============================================================================

def add_age_features(df: pd.DataFrame, dob_col: str = "DOB") -> pd.DataFrame:
    """
    Add age-related features (same as classification)
    """
    df = df.copy()
    
    dob_col_actual = None
    for col in df.columns:
        if col.upper() == dob_col.upper():
            dob_col_actual = col
            break
    
    if dob_col_actual is None:
        print(f"⚠️  Warning: {dob_col} not found, skipping age features")
        return df
    
    dob_series = pd.to_datetime(df[dob_col_actual], errors='coerce')
    reference_date = pd.Timestamp('2100-01-01')
    age_years = ((reference_date - dob_series).dt.days / 365.25).round(1)
    age_years = age_years.clip(upper=90)
    
    df['age_years'] = age_years
    df['is_elderly'] = (age_years >= 75).astype(int)
    df['is_young'] = (age_years < 40).astype(int)
    
    mean_age = age_years.mean()
    age_range = (age_years.min(), age_years.max())
    elderly_count = (age_years >= 75).sum()
    censored_count = (age_years >= 90).sum()
    
    print(f"✓ Age features added. Mean age: {mean_age:.1f} years")
    print(f"  Age range: {age_range[0]:.0f} - {age_range[1]:.0f} years")
    print(f"  Elderly (>=75): {elderly_count} patients ({elderly_count/len(df)*100:.1f}%)")
    print(f"  Censored age (90+): {censored_count} patients ({censored_count/len(df)*100:.1f}%)")
    
    return df


# ============================================================================
# BP OUTLIER CLEANING
# ============================================================================

def clean_min_bp_outliers(df: pd.DataFrame, lower_bounds: dict = None) -> pd.DataFrame:
    """
    Clean implausible blood pressure minimum values
    """
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


# ============================================================================
# CORE ENGINEERED FEATURES (NO FLAGS, NO TEMPORAL)
# ============================================================================

def add_engineered_features_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add core engineered features only - no flags, no temporal.
    
    Includes:
    - Vital ranges (max - min)
    - Composite scores (shock index, pulse pressure, instability)
    - Deviations from normal
    - Proven interaction features
    
    Excludes:
    - Simple threshold flags (XGBoost learns better thresholds)
    - Temporal features (0.000 importance)
    """
    df = df.copy()
    eps = 1e-3
    
    print("\n[add_engineered_features_core]")
    
    # ========================================================================
    # 1. RANGES (Variability/Instability)
    # ========================================================================
    
    if all(c in df.columns for c in ['HeartRate_Max', 'HeartRate_Min']):
        df['HR_range'] = df['HeartRate_Max'] - df['HeartRate_Min']
    
    if all(c in df.columns for c in ['SysBP_Max', 'SysBP_Min']):
        df['SysBP_range'] = df['SysBP_Max'] - df['SysBP_Min']
    
    if all(c in df.columns for c in ['DiasBP_Max', 'DiasBP_Min']):
        df['DiasBP_range'] = df['DiasBP_Max'] - df['DiasBP_Min']
    
    if all(c in df.columns for c in ['MeanBP_Max', 'MeanBP_Min']):
        df['MeanBP_range'] = df['MeanBP_Max'] - df['MeanBP_Min']
    
    if all(c in df.columns for c in ['RespRate_Max', 'RespRate_Min']):
        df['RespRate_range'] = df['RespRate_Max'] - df['RespRate_Min']
    
    if all(c in df.columns for c in ['TempC_Max', 'TempC_Min']):
        df['TempC_range'] = df['TempC_Max'] - df['TempC_Min']
    
    if all(c in df.columns for c in ['SpO2_Max', 'SpO2_Min']):
        df['SpO2_range'] = df['SpO2_Max'] - df['SpO2_Min']
    
    if all(c in df.columns for c in ['Glucose_Max', 'Glucose_Min']):
        df['Glucose_range'] = df['Glucose_Max'] - df['Glucose_Min']
    
    print("  ✓ Ranges calculated (8 features)")
    
    # ========================================================================
    # 2. SHOCK INDEX (HR / SysBP) - Strong predictor
    # ========================================================================
    
    if 'HeartRate_Mean' in df.columns and 'SysBP_Mean' in df.columns:
        df['shock_index_mean'] = df['HeartRate_Mean'] / (df['SysBP_Mean'] + eps)
    
    if 'HeartRate_Min' in df.columns and 'SysBP_Max' in df.columns:
        df['shock_index_min'] = df['HeartRate_Min'] / (df['SysBP_Max'] + eps)
    
    print("  ✓ Shock index calculated (2 features)")
    
    # ========================================================================
    # 3. PULSE PRESSURE (SysBP - DiasBP)
    # ========================================================================
    
    if 'SysBP_Mean' in df.columns and 'DiasBP_Mean' in df.columns:
        df['pulse_pressure_mean'] = df['SysBP_Mean'] - df['DiasBP_Mean']
    
    if 'SysBP_Min' in df.columns and 'DiasBP_Min' in df.columns:
        df['pulse_pressure_min'] = df['SysBP_Min'] - df['DiasBP_Min']
    
    if 'SysBP_Max' in df.columns and 'DiasBP_Max' in df.columns:
        df['pulse_pressure_max'] = df['SysBP_Max'] - df['DiasBP_Max']
    
    print("  ✓ Pulse pressure calculated (3 features)")
    
    # ========================================================================
    # 4. DEVIATIONS FROM NORMAL
    # ========================================================================
    
    # Temperature deviation (normal = 37.0°C)
    if 'TempC_Mean' in df.columns:
        df['temp_dev_mean'] = (df['TempC_Mean'] - 37.0).abs()
    
    # SpO2 deficit (normal = 100%)
    if 'SpO2_Min' in df.columns:
        df['spo2_deficit'] = 100 - df['SpO2_Min']
    
    # Glucose excess (normal = 100 mg/dL)
    if 'Glucose_Max' in df.columns:
        df['glucose_excess'] = (df['Glucose_Max'] - 100).clip(lower=0)
    
    print("  ✓ Deviations from normal calculated (3 features)")
    
    # ========================================================================
    # 5. INSTABILITY COUNT (How many vitals are abnormal)
    # ========================================================================
    
    instability_conditions = []
    
    if 'HeartRate_Mean' in df.columns:
        instability_conditions.append((df['HeartRate_Mean'] > 100) | (df['HeartRate_Mean'] < 60))
    
    if 'SysBP_Min' in df.columns:
        instability_conditions.append(df['SysBP_Min'] < 90)
    
    if 'RespRate_Mean' in df.columns:
        instability_conditions.append((df['RespRate_Mean'] > 20) | (df['RespRate_Mean'] < 12))
    
    if 'TempC_Mean' in df.columns:
        instability_conditions.append((df['TempC_Mean'] > 38.3) | (df['TempC_Mean'] < 36.0))
    
    if 'SpO2_Min' in df.columns:
        instability_conditions.append(df['SpO2_Min'] < 92)
    
    if 'Glucose_Max' in df.columns:
        instability_conditions.append(df['Glucose_Max'] > 180)
    
    # Count abnormalities
    if len(instability_conditions) > 0:
        df['instability_count'] = sum([cond.astype(int) for cond in instability_conditions])
    else:
        df['instability_count'] = 0
    
    print("  ✓ Instability count calculated (1 feature)")
    
    # ========================================================================
    # 6. PROVEN INTERACTION FEATURES
    # ========================================================================
    
    # Hypoxia + Tachypnea (respiratory distress)
    if 'SpO2_Min' in df.columns and 'RespRate_Mean' in df.columns:
        df['hypoxia_and_tachypnea'] = (
            (df['SpO2_Min'] < 92) & (df['RespRate_Mean'] > 20)
        ).astype(int)
        print("  ✓ Interaction feature: hypoxia_and_tachypnea (1 feature)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    n_features_added = 18  # 8+2+3+3+1+1
    
    print(f"\n  ✓ Total core features added: {n_features_added}")
    print(f"  ✓ Excluded: temporal features, simple threshold flags")
    
    return df