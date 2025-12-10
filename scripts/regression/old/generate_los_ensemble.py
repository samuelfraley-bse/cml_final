"""
Generate LOS Regression Submissions - Ensemble Version
Combines multiple models for better predictions.

Run: python scripts/regression/generate_los_ensemble.py
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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 70)
print("LOS REGRESSION - ENSEMBLE SUBMISSION")
print("=" * 70)

# =============================================================================
# HELPER FUNCTIONS
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

    if 'ICD9_diagnosis' in df.columns:
        high_los_icd9 = ['51884', '5770', '430', '44101', '0380', '51881', '0389', '431']
        low_los_icd9 = ['9951', '4373', '29181']
        other_icd9 = ['41401', '41071', '4241', '486', '5070']

        for code in high_los_icd9 + low_los_icd9 + other_icd9:
            df[f'icd9_{code}'] = (df['ICD9_diagnosis'] == code).astype(int)

        df['icd9_category'] = df['ICD9_diagnosis'].astype(str).str[:3]
        high_los_cats = ['430', '577', '518', '441', '038']
        for cat in high_los_cats:
            df[f'icd9_cat_{cat}'] = (df['icd9_category'] == cat).astype(int)

    if 'DIAGNOSIS' in df.columns:
        diag_upper = df['DIAGNOSIS'].fillna('').str.upper()

        df['diag_pancreatitis'] = diag_upper.str.contains('PANCREATITIS').astype(int)
        df['diag_shock'] = diag_upper.str.contains('SHOCK').astype(int)
        df['diag_arrest'] = diag_upper.str.contains('ARREST').astype(int)
        df['diag_transplant'] = diag_upper.str.contains('TRANSPLANT').astype(int)
        df['diag_failure'] = diag_upper.str.contains('FAILURE').astype(int)
        df['diag_sepsis'] = diag_upper.str.contains('SEPSIS').astype(int)
        df['diag_pneumonia'] = diag_upper.str.contains('PNEUMONIA').astype(int)
        df['diag_hemorrhage'] = diag_upper.str.contains('HEMORRHAGE').astype(int)
        df['diag_respiratory'] = diag_upper.str.contains('RESPIRATORY').astype(int)
        df['diag_overdose'] = diag_upper.str.contains('OVERDOSE').astype(int)
        df['diag_diabetic'] = diag_upper.str.contains('DIABETIC|DKA').astype(int)
        df['diag_gi_bleed'] = diag_upper.str.contains('GI BLEED|UPPER GI|LOWER GI').astype(int)
        df['diag_cardiac'] = diag_upper.str.contains('HEART|CARDIAC|CORONARY|MI |STEMI|NSTEMI').astype(int)
        df['diag_neurological'] = diag_upper.str.contains('STROKE|SEIZURE|NEUROLOG|MENTAL').astype(int)
        df['diag_trauma'] = diag_upper.str.contains('TRAUMA|FRACTURE|INJURY|MVA|FALL').astype(int)
        df['diag_cancer'] = diag_upper.str.contains('CANCER|TUMOR|MALIGNAN|LYMPHOMA|LEUKEMIA').astype(int)
        df['diag_liver'] = diag_upper.str.contains('LIVER|HEPAT|CIRRHOSIS').astype(int)

        condition_cols = [c for c in df.columns if c.startswith('diag_')]
        df['diag_count'] = df[condition_cols].sum(axis=1)

    if 'FIRST_CAREUNIT' in df.columns and 'ADMISSION_TYPE' in df.columns:
        df['sicu_emergency'] = ((df['FIRST_CAREUNIT'] == 'SICU') &
                                (df['ADMISSION_TYPE'] == 'EMERGENCY')).astype(int)

    return df


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vital sign ranges and clinical indices."""
    df = df.copy()
    eps = 1e-3

    # Ranges
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

    # Flags
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

    # Clinical indices
    if 'HeartRate_Mean' in df.columns and 'SysBP_Mean' in df.columns:
        df["shock_index"] = df["HeartRate_Mean"] / (df["SysBP_Mean"].clip(lower=eps))
    if 'SysBP_Mean' in df.columns and 'DiasBP_Mean' in df.columns:
        df["pulse_pressure"] = df["SysBP_Mean"] - df["DiasBP_Mean"]
        df["map_calc"] = df["DiasBP_Mean"] + (df["SysBP_Mean"] - df["DiasBP_Mean"]) / 3

    # Instability score
    flag_cols = [c for c in df.columns if c.endswith('_flag')]
    if flag_cols:
        df['instability_score'] = df[flag_cols].sum(axis=1)

    return df


# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/6] Loading data...")

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
# PREPARE FEATURES (Enhanced - best on Kaggle)
# =============================================================================
print("\n[2/6] Preparing features...")

drop_cols = [
    'icustay_id', 'subject_id', 'hadm_id',
    'LOS', 'HOSPITAL_EXPIRE_FLAG',
    'DISCHTIME', 'DEATHTIME', 'DOD', 'DOB', 'ADMITTIME',
    'Diff',
]

X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

X_train = clean_min_bp_outliers(X_train)
X_test = clean_min_bp_outliers(X_test)

X_train = add_diagnosis_features(X_train)
X_test = add_diagnosis_features(X_test)

X_train = add_enhanced_features(X_train)
X_test = add_enhanced_features(X_test)

# Drop text columns
X_train = X_train.drop(columns=['ICD9_diagnosis', 'DIAGNOSIS', 'icd9_category'], errors='ignore')
X_test = X_test.drop(columns=['ICD9_diagnosis', 'DIAGNOSIS', 'icd9_category'], errors='ignore')

print(f"Features: {X_train.shape[1]}")

# Identify column types
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}")

# =============================================================================
# CREATE PREPROCESSOR
# =============================================================================
print("\n[3/6] Creating preprocessor...")

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

# =============================================================================
# TRAIN MULTIPLE MODELS
# =============================================================================
print("\n[4/6] Training ensemble models...")

# Define models
models = {
    'GB': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
    ),
    'RF': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
    ),
    'ET': ExtraTreesRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
    ),
}

# Train each model and get predictions
predictions = {}
validation_scores = {}

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Create pipeline
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', model),
    ])

    # Train on full data for test predictions
    pipe.fit(X_train, y)
    test_pred = np.maximum(pipe.predict(X_test), 0)
    predictions[name] = test_pred

    # Validation score
    val_pipe = Pipeline([
        ('preprocess', ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), cat_cols),
        ])),
        ('regressor', model.__class__(**model.get_params())),
    ])

    val_pipe.fit(X_tr, y_tr)
    val_pred = np.maximum(val_pipe.predict(X_val), 0)

    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)

    validation_scores[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"  Validation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# =============================================================================
# CREATE ENSEMBLES
# =============================================================================
print("\n[5/6] Creating ensemble predictions...")

# Simple average
ensemble_avg = np.mean([predictions['GB'], predictions['RF'], predictions['ET']], axis=0)

# Weighted by inverse RMSE (better models get more weight)
weights = {}
total_inv_rmse = sum(1/validation_scores[name]['rmse'] for name in models.keys())
for name in models.keys():
    weights[name] = (1/validation_scores[name]['rmse']) / total_inv_rmse

print(f"\nWeights (by inverse RMSE):")
for name, w in weights.items():
    print(f"  {name}: {w:.3f}")

ensemble_weighted = (
    weights['GB'] * predictions['GB'] +
    weights['RF'] * predictions['RF'] +
    weights['ET'] * predictions['ET']
)

# GB + RF only (often best combo)
ensemble_gb_rf = 0.5 * predictions['GB'] + 0.5 * predictions['RF']

# =============================================================================
# SAVE SUBMISSIONS
# =============================================================================
print("\n[6/6] Saving submissions...")

all_predictions = {
    'gb_single': predictions['GB'],
    'rf_single': predictions['RF'],
    'et_single': predictions['ET'],
    'ensemble_avg': ensemble_avg,
    'ensemble_weighted': ensemble_weighted,
    'ensemble_gb_rf': ensemble_gb_rf,
}

for name, pred in all_predictions.items():
    submission = pd.DataFrame({
        'icustay_id': test_ids,
        'LOS': pred
    })

    filename = f"submission_los_{name}.csv"
    filepath = OUTPUT_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"\n{name}:")
    print(f"  File: {filename}")
    print(f"  Predictions - min: {pred.min():.2f}, max: {pred.max():.2f}, mean: {pred.mean():.2f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nIndividual Model Validation Scores:")
for name, scores in sorted(validation_scores.items(), key=lambda x: x[1]['rmse']):
    print(f"  {name}: RMSE={scores['rmse']:.4f}, MAE={scores['mae']:.4f}, R²={scores['r2']:.4f}")

print("\nSubmissions created:")
print("  1. gb_single - Gradient Boosting alone")
print("  2. rf_single - Random Forest alone")
print("  3. et_single - Extra Trees alone")
print("  4. ensemble_avg - Simple average of all 3")
print("  5. ensemble_weighted - Weighted by validation RMSE")
print("  6. ensemble_gb_rf - Average of GB + RF")

print("\nRecommendation:")
print("  Try 'ensemble_weighted' first - usually best")
print("  Then 'ensemble_gb_rf' if weighted doesn't work")

print("\n" + "=" * 70)
print("Files saved to:", OUTPUT_DIR)
print("=" * 70)
