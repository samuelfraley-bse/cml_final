# %%
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Constants that almost never change ----

# Targets
TARGET_COL_CLASS = "HOSPITAL_EXPIRE_FLAG"
TARGET_COL_REG   = "LOS"

# ID columns (adjust to match your file)
ID_COLS = [
    "ICUSTAY_ID",
    "SUBJECT_ID",
    "HADM_ID",
]

def get_paths():
    """
    Return BASE_DIR, TRAIN_PATH, TEST_PATH based on the current notebook location.

    Assumes this notebook lives in:
        cml_final/notebooks/HEF/hef_prep.ipynb
    and data lives in:
        cml_final/data/raw/MIMIC III dataset HEF/
    """
    base_dir = Path.cwd().parents[1]  # .../cml_final
    raw_dir = base_dir / "data" / "raw" / "MIMIC III dataset HEF"

    train_path = raw_dir / "mimic_train_HEF.csv"
    test_path  = raw_dir / "mimic_test_HEF.csv"

    return base_dir, train_path, test_path


# %%
def load_raw_data():
    """
    Load raw train and test CSVs and return them as dataframes.
    """
    base_dir, train_path, test_path = get_paths()
    print("Base dir:", base_dir)
    print("Train path:", train_path)
    print("Test path:", test_path)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, test_df


# %%
def split_features_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: str = "class",
    leak_cols: list | None = None,
    id_cols: list | None = None,
):
    """
    From raw train/test:
    - Pick target column based on task ("class" or "reg")
    - Drop ID, leakage, and both target columns from X.
    - Return X_train_raw, y_train, X_test_raw

    Parameters
    ----------
    task : {"class", "reg"}
        "class" -> y = HOSPITAL_EXPIRE_FLAG
        "reg"   -> y = LOS
    leak_cols : list or None
        Extra columns to drop because they leak the target.
        If None, no leak columns are dropped here (you can pass them from a notebook).
    id_cols : list or None
        ID columns to drop. If None, uses the global ID_COLS.
    """
    if id_cols is None:
        id_cols = ID_COLS

    if leak_cols is None:
        leak_cols = []

    if task == "class":
        target_col = TARGET_COL_CLASS
    elif task == "reg":
        target_col = TARGET_COL_REG
    else:
        raise ValueError(f"Unknown task '{task}'. Use 'class' or 'reg'.")

    # drop both targets from X to be safe
    drop_cols_train = id_cols + leak_cols + [TARGET_COL_CLASS, TARGET_COL_REG]
    drop_cols_test  = id_cols + leak_cols

    y_train = train_df[target_col].copy()

    X_train_raw = train_df.drop(columns=[c for c in drop_cols_train if c in train_df.columns])
    X_test_raw  = test_df.drop(columns=[c for c in drop_cols_test  if c in test_df.columns])

    print(f"Task: {task} (target = {target_col})")
    print("X_train_raw shape:", X_train_raw.shape)
    print("X_test_raw shape:", X_test_raw.shape)
    print("y_train shape:", y_train.shape)

    if task == "class":
        print("Positive rate (death):", y_train.mean().round(3))

    return X_train_raw, y_train, X_test_raw


# %%
BP_MIN_LOWER_BOUNDS = {
    "SysBP_Min":  40.0,
    "DiasBP_Min": 10.0,
    "MeanBP_Min": 30.0,
}

def clean_min_bp_outliers(df: pd.DataFrame,
                          lower_bounds: dict = BP_MIN_LOWER_BOUNDS) -> pd.DataFrame:
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
    Add row-level numeric features:
    - Ranges (max - min) for vitals
    - Abnormal vital flags (0/1)
    - instability_count = sum of selected flags
    - Nonlinear/composite features (shock index, pulse pressure, deviations, interactions)
    """
    df = df.copy()
    eps = 1e-3  # avoid divide-by-zero

    # 1. RANGES (instability of vitals)
    df["HR_range"]        = df["HeartRate_Max"]  - df["HeartRate_Min"]
    df["SysBP_range"]     = df["SysBP_Max"]      - df["SysBP_Min"]
    df["DiasBP_range"]    = df["DiasBP_Max"]     - df["DiasBP_Min"]
    df["MeanBP_range"]    = df["MeanBP_Max"]     - df["MeanBP_Min"]
    df["RespRate_range"]  = df["RespRate_Max"]   - df["RespRate_Min"]
    df["TempC_range"]     = df["TempC_Max"]      - df["TempC_Min"]
    df["SpO2_range"]      = df["SpO2_Max"]       - df["SpO2_Min"]
    df["Glucose_range"]   = df["Glucose_Max"]    - df["Glucose_Min"]

    # 2. ABNORMAL VITAL FLAGS (0/1)

    # Cardiovascular
    df["tachy_flag"]       = (df["HeartRate_Max"] >= 120).astype(int)
    df["hypotension_flag"] = (
        (df["SysBP_Min"] < 90) | (df["MeanBP_Min"] < 65)
    ).astype(int)

    # Respiratory
    df["tachypnea_flag"]   = (df["RespRate_Max"] >= 30).astype(int)

    # Temperature
    df["fever_flag"]       = (df["TempC_Max"] >= 38.0).astype(int)

    # Oxygenation
    df["hypoxia_flag"]     = (df["SpO2_Min"] < 92).astype(int)

    # Glucose
    df["hyperglycemia_flag"] = (df["Glucose_Max"] >= 200).astype(int)

    # 3. INSTABILITY COUNT
    instability_flags = [
        "tachy_flag",
        "hypotension_flag",
        "tachypnea_flag",
        "fever_flag",
        "hypoxia_flag",
        "hyperglycemia_flag",
    ]
    df["instability_count"] = df[instability_flags].sum(axis=1)

    # 4. NONLINEAR / COMPOSITE FEATURES

    # 4.1 Shock index (higher is worse): HR / SBP
    df["shock_index_mean"] = df["HeartRate_Mean"] / (df["SysBP_Mean"].clip(lower=eps))
    df["shock_index_min"]  = df["HeartRate_Min"]  / (df["SysBP_Min"].clip(lower=eps))

    # 4.2 Pulse pressure (SBP - DBP)
    df["pulse_pressure_mean"] = df["SysBP_Mean"] - df["DiasBP_Mean"]
    df["pulse_pressure_min"]  = df["SysBP_Min"]  - df["DiasBP_Min"]
    df["pulse_pressure_max"]  = df["SysBP_Max"]  - df["DiasBP_Max"]

    # 4.3 Distance from "normal" temperature (symmetric)
    df["temp_dev_mean"] = (df["TempC_Mean"] - 37.0).abs()

    # 4.4 How far SpO2 fell below 92 (0 if never below)
    df["spo2_deficit"] = (92.0 - df["SpO2_Min"]).clip(lower=0.0)

    # 4.5 How far glucose went above 140 (0 if always <= 140)
    df["glucose_excess"] = (df["Glucose_Max"] - 140.0).clip(lower=0.0)

    # 4.6 Interaction flags: "both bad" situations
    df["tachy_and_hypotension"] = (
        (df["tachy_flag"] == 1) & (df["hypotension_flag"] == 1)
    ).astype(int)

    df["hypoxia_and_tachypnea"] = (
        (df["hypoxia_flag"] == 1) & (df["tachypnea_flag"] == 1)
    ).astype(int)

    return df


# %%
def prepare_data(
    task: str = "class",
    leak_cols: list | None = None,
    apply_fe: bool = True,
):
    """
    Full data prep pipeline (configurable):

    - Load raw train & test
    - Split features/target based on task ("class" or "reg")
    - Clean implausible min BPs
    - Optionally add engineered features

    Parameters
    ----------
    task : {"class", "reg"}
        Which target to use (classification vs regression).
    leak_cols : list or None
        List of leakage column names to drop from X.
        e.g. ["leak1", "leak2", "leak3"]
    apply_fe : bool
        If True, applies add_engineered_features().
        If False, returns cleaned raw features only.

    Returns
    -------
    X_train, y_train, X_test
    """
    train_df, test_df = load_raw_data()
    X_train_raw, y_train, X_test_raw = split_features_target(
        train_df=train_df,
        test_df=test_df,
        task=task,
        leak_cols=leak_cols,
    )

    X_train_clean = clean_min_bp_outliers(X_train_raw)
    X_test_clean  = clean_min_bp_outliers(X_test_raw)

    if apply_fe:
        X_train_final = add_engineered_features(X_train_clean)
        X_test_final  = add_engineered_features(X_test_clean)
    else:
        X_train_final = X_train_clean
        X_test_final  = X_test_clean

    print("Final X_train shape:", X_train_final.shape)
    print("Final X_test shape:", X_test_final.shape)

    return X_train_final, y_train, X_test_final



