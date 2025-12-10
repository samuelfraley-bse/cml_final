"""
Generate LOS Regression Submissions (V2)
Uses improved feature engineering based on exploration analysis.

Creates submissions for:
1. Diagnosis features only (best validation score)
2. Enhanced features (diagnosis + vitals ranges)

Run from project root or scripts/regression/:
    python scripts/regression/generate_los_submissions_v2.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "regression":
    BASE_DIR = SCRIPT_DIR.parents[1]
else:
    BASE_DIR = Path.cwd()

sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(SCRIPT_DIR))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 70)
print("LOS REGRESSION - SUBMISSION GENERATOR V2")
print("=" * 70)

# =============================================================================
# HELPER FUNCTIONS (same as feature_analysis.py)
# =============================================================================

def clean_min_bp_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean implausible blood pressure values"""
    lower_bounds = {
        "SysBP_Min": 40.0,
        "DiasBP_Min": 10.0,
        "MeanBP_Min": 30.0,
    }
    df = df.copy()
    for col, low in lower_bounds.items():
        if col not in df.columns:
            continue
        mask = (df[col] < low) & df[col].notna()
        if mask.sum() > 0:
            df.loc[mask, col] = np.nan
    return df


def add_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features based on DIAGNOSIS and ICD9_diagnosis columns."""
    df = df.copy()

    # --- ICD9 Code Features ---
    if 'ICD9_diagnosis' in df.columns:
        # High-LOS ICD9 codes
        high_los_icd9 = [
            '51884', '5770', '430', '44101', '0380', '51881', '0389', '431',
        ]
        # Low-LOS ICD9 codes
        low_los_icd9 = ['9951', '4373', '29181']
        # Other codes
        other_icd9 = ['41401', '41071', '4241', '486', '5070']

        for code in high_los_icd9 + low_los_icd9 + other_icd9:
            df[f'icd9_{code}'] = (df['ICD9_diagnosis'] == code).astype(int)

        # ICD9 categories
        df['icd9_category'] = df['ICD9_diagnosis'].astype(str).str[:3]
        high_los_cats = ['430', '577', '518', '441', '038']
        for cat in high_los_cats:
            df[f'icd9_cat_{cat}'] = (df['icd9_category'] == cat).astype(int)

    # --- DIAGNOSIS Text Features ---
    if 'DIAGNOSIS' in df.columns:
        diag_upper = df['DIAGNOSIS'].fillna('').str.upper()

        # High-LOS keywords
        df['diag_pancreatitis'] = diag_upper.str.contains('PANCREATITIS').astype(int)
        df['diag_shock'] = diag_upper.str.contains('SHOCK').astype(int)
        df['diag_arrest'] = diag_upper.str.contains('ARREST').astype(int)
        df['diag_transplant'] = diag_upper.str.contains('TRANSPLANT').astype(int)
        df['diag_failure'] = diag_upper.str.contains('FAILURE').astype(int)
        df['diag_sepsis'] = diag_upper.str.contains('SEPSIS').astype(int)
        df['diag_pneumonia'] = diag_upper.str.contains('PNEUMONIA').astype(int)
        df['diag_hemorrhage'] = diag_upper.str.contains('HEMORRHAGE').astype(int)
        df['diag_respiratory'] = diag_upper.str.contains('RESPIRATORY').astype(int)

        # Low-LOS keywords
        df['diag_overdose'] = diag_upper.str.contains('OVERDOSE').astype(int)
        df['diag_diabetic'] = diag_upper.str.contains('DIABETIC|DKA').astype(int)
        df['diag_gi_bleed'] = diag_upper.str.contains('GI BLEED|UPPER GI|LOWER GI').astype(int)

        # Other keywords
        df['diag_cardiac'] = diag_upper.str.contains('HEART|CARDIAC|CORONARY|MI |STEMI|NSTEMI').astype(int)
        df['diag_neurological'] = diag_upper.str.contains('STROKE|SEIZURE|NEUROLOG|MENTAL').astype(int)
        df['diag_trauma'] = diag_upper.str.contains('TRAUMA|FRACTURE|INJURY|MVA|FALL').astype(int)
        df['diag_cancer'] = diag_upper.str.contains('CANCER|TUMOR|MALIGNAN|LYMPHOMA|LEUKEMIA').astype(int)
        df['diag_liver'] = diag_upper.str.contains('LIVER|HEPAT|CIRRHOSIS').astype(int)

        # Complexity proxy
        condition_cols = [c for c in df.columns if c.startswith('diag_')]
        df['diag_count'] = df[condition_cols].sum(axis=1)

    # --- Interaction ---
    if 'FIRST_CAREUNIT' in df.columns and 'ADMISSION_TYPE' in df.columns:
        df['sicu_emergency'] = ((df['FIRST_CAREUNIT'] == 'SICU') &
                                (df['ADMISSION_TYPE'] == 'EMERGENCY')).astype(int)

    return df


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vital sign ranges and clinical indices."""
    df = df.copy()
    eps = 1e-3

    # Vital Sign Ranges
    if 'HeartRate_Max' in df.columns and 'HeartRate_Min' in df.columns:
        df["HR_range"] = df["HeartRate_Max"] - df["HeartRate_Min"]
    if 'SysBP_Max' in df.columns and 'SysBP_Min' in df.columns:
        df["SysBP_range"] = df["SysBP_Max"] - df["SysBP_Min"]
    if 'MeanBP_Max' in df.columns and 'MeanBP_Min' in df.columns:
        df["MeanBP_range"] = df["MeanBP_Max"] - df["MeanBP_Min"]
    if 'RespRate_Max' in df.columns and 'RespRate_Min' in df.columns:
        df["RespRate_range"] = df["RespRate_Max"] - df["RespRate_Min"]
    if 'SpO2_Max' in df.columns and 'SpO2_Min' in df.columns:
        df["SpO2_range"] = df["SpO2_Max"] - df["SpO2_Min"]
    if 'TempC_Max' in df.columns and 'TempC_Min' in df.columns:
        df["Temp_range"] = df["TempC_Max"] - df["TempC_Min"]
    if 'Glucose_Max' in df.columns and 'Glucose_Min' in df.columns:
        df["Glucose_range"] = df["Glucose_Max"] - df["Glucose_Min"]

    # Critical Flags
    if 'HeartRate_Max' in df.columns:
        df["tachy_flag"] = (df["HeartRate_Max"] >= 120).astype(int)
        df["severe_tachy_flag"] = (df["HeartRate_Max"] >= 150).astype(int)
    if 'HeartRate_Min' in df.columns:
        df["brady_flag"] = (df["HeartRate_Min"] <= 50).astype(int)
    if 'SysBP_Min' in df.columns:
        df["hypotension_flag"] = (df["SysBP_Min"] < 90).astype(int)
    if 'MeanBP_Min' in df.columns:
        df["low_map_flag"] = (df["MeanBP_Min"] < 65).astype(int)
    if 'SpO2_Min' in df.columns:
        df["hypoxia_flag"] = (df["SpO2_Min"] < 92).astype(int)
        df["severe_hypoxia_flag"] = (df["SpO2_Min"] < 88).astype(int)
    if 'TempC_Max' in df.columns:
        df["fever_flag"] = (df["TempC_Max"] >= 38.3).astype(int)
    if 'TempC_Min' in df.columns:
        df["hypothermia_flag"] = (df["TempC_Min"] < 36).astype(int)
    if 'RespRate_Max' in df.columns:
        df["tachypnea_flag"] = (df["RespRate_Max"] >= 30).astype(int)
    if 'Glucose_Max' in df.columns:
        df["hyperglycemia_flag"] = (df["Glucose_Max"] >= 180).astype(int)
    if 'Glucose_Min' in df.columns:
        df["hypoglycemia_flag"] = (df["Glucose_Min"] <= 70).astype(int)

    # Clinical Indices
    if 'HeartRate_Mean' in df.columns and 'SysBP_Mean' in df.columns:
        df["shock_index"] = df["HeartRate_Mean"] / (df["SysBP_Mean"].clip(lower=eps))
    if 'SysBP_Mean' in df.columns and 'DiasBP_Mean' in df.columns:
        df["pulse_pressure"] = df["SysBP_Mean"] - df["DiasBP_Mean"]
        df["map_calc"] = df["DiasBP_Mean"] + (df["SysBP_Mean"] - df["DiasBP_Mean"]) / 3

    # Instability Score
    flag_cols = [c for c in df.columns if c.endswith('_flag')]
    if flag_cols:
        df['instability_score'] = df[flag_cols].sum(axis=1)

    return df


# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/5] Loading data...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
train_df = pd.read_csv(DATA_DIR / "mimic_train_LOS.csv")
test_df = pd.read_csv(DATA_DIR / "mimic_test_LOS.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Get test IDs
id_col = [c for c in test_df.columns if c.lower() == 'icustay_id'][0]
test_ids = test_df[id_col]

# Target
y = train_df['LOS'].copy()

# Output directory
OUTPUT_DIR = BASE_DIR / "submissions" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PREPARE FEATURES
# =============================================================================
print("\n[2/5] Preparing features...")

# Columns to drop
drop_cols = [
    'icustay_id', 'subject_id', 'hadm_id',
    'LOS', 'HOSPITAL_EXPIRE_FLAG',
    'DISCHTIME', 'DEATHTIME', 'DOD', 'DOB', 'ADMITTIME',
    'Diff',
]

# --- Version 1: Diagnosis features only ---
X_train_diag = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_test_diag = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

X_train_diag = clean_min_bp_outliers(X_train_diag)
X_test_diag = clean_min_bp_outliers(X_test_diag)

X_train_diag = add_diagnosis_features(X_train_diag)
X_test_diag = add_diagnosis_features(X_test_diag)

# Drop text columns
X_train_diag = X_train_diag.drop(columns=['ICD9_diagnosis', 'DIAGNOSIS', 'icd9_category'], errors='ignore')
X_test_diag = X_test_diag.drop(columns=['ICD9_diagnosis', 'DIAGNOSIS', 'icd9_category'], errors='ignore')

print(f"Diagnosis features: {X_train_diag.shape[1]}")

# --- Version 2: Enhanced features ---
X_train_enh = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_test_enh = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

X_train_enh = clean_min_bp_outliers(X_train_enh)
X_test_enh = clean_min_bp_outliers(X_test_enh)

X_train_enh = add_diagnosis_features(X_train_enh)
X_test_enh = add_diagnosis_features(X_test_enh)

X_train_enh = add_enhanced_features(X_train_enh)
X_test_enh = add_enhanced_features(X_test_enh)

# Drop text columns
X_train_enh = X_train_enh.drop(columns=['ICD9_diagnosis', 'DIAGNOSIS', 'icd9_category'], errors='ignore')
X_test_enh = X_test_enh.drop(columns=['ICD9_diagnosis', 'DIAGNOSIS', 'icd9_category'], errors='ignore')

print(f"Enhanced features: {X_train_enh.shape[1]}")

# =============================================================================
# TRAIN MODELS AND GENERATE SUBMISSIONS
# =============================================================================

def train_and_predict(X_train, y_train, X_test, name):
    """Train GB model and return predictions."""
    # Identify column types
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat_cols),
    ])

    # Model
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
    )

    # Pipeline
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', gb),
    ])

    # Train
    print(f"\nTraining {name}...")
    pipe.fit(X_train, y_train)

    # Predict
    predictions = np.maximum(pipe.predict(X_test), 0)  # Ensure non-negative

    # Validation score (using holdout)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    val_preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat_cols),
    ])

    val_pipe = Pipeline([
        ('preprocess', val_preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            min_samples_split=10, min_samples_leaf=5, subsample=0.8,
            max_features='sqrt', random_state=42,
        )),
    ])

    val_pipe.fit(X_tr, y_tr)
    val_pred = np.maximum(val_pipe.predict(X_val), 0)

    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)

    print(f"  Validation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    return predictions, rmse, mae


# --- Generate submissions ---
print("\n[3/5] Training models and generating predictions...")

results = {}

# Diagnosis features
pred_diag, rmse_diag, mae_diag = train_and_predict(X_train_diag, y, X_test_diag, "GB (Diagnosis)")
results['diagnosis'] = {'pred': pred_diag, 'rmse': rmse_diag, 'mae': mae_diag}

# Enhanced features
pred_enh, rmse_enh, mae_enh = train_and_predict(X_train_enh, y, X_test_enh, "GB (Enhanced)")
results['enhanced'] = {'pred': pred_enh, 'rmse': rmse_enh, 'mae': mae_enh}

# =============================================================================
# SAVE SUBMISSIONS
# =============================================================================
print("\n[4/5] Saving submissions...")

submissions = []

for name, data in results.items():
    submission = pd.DataFrame({
        'icustay_id': test_ids,
        'LOS': data['pred']
    })

    filename = f"submission_los_gb_{name}_v2.csv"
    filepath = OUTPUT_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"\n{name.upper()}:")
    print(f"  File: {filename}")
    print(f"  Validation RMSE: {data['rmse']:.4f}")
    print(f"  Predictions - min: {data['pred'].min():.2f}, max: {data['pred'].max():.2f}, mean: {data['pred'].mean():.2f}")

    submissions.append((name, data['rmse'], filename))

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("[5/5] SUMMARY")
print("=" * 70)

# Sort by RMSE
submissions.sort(key=lambda x: x[1])

print("\nRanked by Validation RMSE:")
for i, (name, rmse, filename) in enumerate(submissions, 1):
    marker = "🏆" if i == 1 else "  "
    print(f"{marker} {i}. {name:15} RMSE: {rmse:.4f}  -> {filename}")

best_name, best_rmse, best_file = submissions[0]
print(f"\n✓ Best model: {best_name}")
print(f"  Upload first: {best_file}")

print("\n" + "=" * 70)
print("SUBMISSIONS COMPLETE")
print("=" * 70)
print(f"\nFiles saved to: {OUTPUT_DIR}")
print("\nUpload to Kaggle and check the score!")
print("=" * 70)
