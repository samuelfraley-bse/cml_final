"""
LOS Regression with Log-Transform Target

The problem: Our models cap predictions at ~19 days but LOS goes to 102 days.
For extreme stays (20-50 days), RMSE is 23.67!

Solution: Train on log(LOS), predict, then exp() to get back to original scale.
This compresses the range and helps the model predict extreme values better.

Run: python scripts/regression/generate_los_logtransform.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 70)
print("LOS REGRESSION - LOG-TRANSFORM APPROACH")
print("=" * 70)

# =============================================================================
# HELPER FUNCTIONS (same as generate_los_ensemble.py)
# =============================================================================

def clean_min_bp_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean outliers in MinBP columns (values > 500 are errors)"""
    df = df.copy()
    for col in ['SysBP_Min', 'DiasBP_Min', 'MeanBP_Min']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.nan if x > 500 else x)
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
    """Enhanced feature engineering"""
    df = df.copy()

    # Vital sign ranges
    vital_pairs = [
        ('HeartRate', 'hr'), ('SysBP', 'sbp'), ('DiasBP', 'dbp'),
        ('MeanBP', 'mbp'), ('RespRate', 'rr'), ('TempC', 'temp'),
        ('SpO2', 'spo2'), ('Glucose', 'gluc')
    ]

    for vital, prefix in vital_pairs:
        max_col = f'{vital}_Max'
        min_col = f'{vital}_Min'
        if max_col in df.columns and min_col in df.columns:
            df[f'{prefix}_range'] = df[max_col] - df[min_col]

    # Binary severity indicators
    if 'SOFA' in df.columns:
        df['high_sofa'] = (df['SOFA'] >= 8).astype(int)
    if 'SAPSII' in df.columns:
        df['high_saps'] = (df['SAPSII'] >= 40).astype(int)
    if 'GCS' in df.columns:
        df['low_gcs'] = (df['GCS'] <= 8).astype(int)

    # Combinations
    if 'high_sofa' in df.columns and 'high_saps' in df.columns:
        df['multi_organ'] = ((df['high_sofa'] == 1) | (df['high_saps'] == 1)).astype(int)

    return df


# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/7] Loading data...")

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
train_df = pd.read_csv(DATA_DIR / "mimic_train_LOS.csv")
test_df = pd.read_csv(DATA_DIR / "mimic_test_LOS.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# =============================================================================
# PREPARE FEATURES
# =============================================================================
print("\n[2/7] Preparing features...")

drop_cols = [
    'icustay_id', 'subject_id', 'hadm_id',
    'LOS', 'HOSPITAL_EXPIRE_FLAG',
    'DISCHTIME', 'DEATHTIME', 'DOD', 'DOB', 'ADMITTIME',
    'Diff', 'DIAGNOSIS', 'ICD9_diagnosis', 'icd9_category'
]

# Process train
X_train_full = train_df.copy()
X_train_full = clean_min_bp_outliers(X_train_full)
X_train_full = add_diagnosis_features(X_train_full)
X_train_full = add_enhanced_features(X_train_full)

# Original target and log-transformed target
y_original = train_df['LOS'].copy()
y_log = np.log1p(y_original)  # log(1 + LOS) to handle LOS near 0

print(f"\nOriginal LOS - min: {y_original.min():.2f}, max: {y_original.max():.2f}, mean: {y_original.mean():.2f}")
print(f"Log LOS      - min: {y_log.min():.2f}, max: {y_log.max():.2f}, mean: {y_log.mean():.2f}")

# Drop columns
X_train_full = X_train_full.drop(columns=[c for c in drop_cols if c in X_train_full.columns])

# Process test
X_test = test_df.copy()
test_ids = X_test['icustay_id'].copy()
X_test = clean_min_bp_outliers(X_test)
X_test = add_diagnosis_features(X_test)
X_test = add_enhanced_features(X_test)
X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

# Align columns
common_cols = list(set(X_train_full.columns) & set(X_test.columns))
X_train_full = X_train_full[common_cols]
X_test = X_test[common_cols]

print(f"Features: {len(common_cols)}")

# Identify column types
numerical_cols = X_train_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train_full.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)}")

# =============================================================================
# CREATE PREPROCESSOR
# =============================================================================
print("\n[3/7] Creating preprocessor...")

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# =============================================================================
# TRAIN/VAL SPLIT
# =============================================================================
print("\n[4/7] Splitting data...")

X_train, X_val, y_train_log, y_val_log, y_train_orig, y_val_orig = train_test_split(
    X_train_full, y_log, y_original, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# =============================================================================
# TRAIN MODELS
# =============================================================================
print("\n[5/7] Training models with log-transformed target...")

models = {
    'GB_log': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    ),
    'RF_log': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
}

results = {}
val_predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Fit on log-transformed target
    pipeline.fit(X_train, y_train_log)

    # Predict (still in log space)
    y_pred_log = pipeline.predict(X_val)

    # Transform back to original scale
    y_pred_orig = np.expm1(y_pred_log)  # exp(x) - 1, inverse of log1p

    # Clip predictions (no negative, reasonable max)
    y_pred_orig = np.clip(y_pred_orig, 0, 100)

    # Calculate metrics on original scale
    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
    mae = mean_absolute_error(y_val_orig, y_pred_orig)
    r2 = r2_score(y_val_orig, y_pred_orig)

    results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'pipeline': pipeline}
    val_predictions[name] = y_pred_orig

    print(f"  Validation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"  Pred range: [{y_pred_orig.min():.2f}, {y_pred_orig.max():.2f}]")

# Also train a model WITHOUT log transform for comparison
print("\nTraining GB_original (no log transform, for comparison)...")
gb_orig = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)
pipeline_orig = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', gb_orig)
])
pipeline_orig.fit(X_train, y_train_orig)
y_pred_no_log = pipeline_orig.predict(X_val)
y_pred_no_log = np.clip(y_pred_no_log, 0, 100)
rmse_orig = np.sqrt(mean_squared_error(y_val_orig, y_pred_no_log))
print(f"  GB_original RMSE: {rmse_orig:.4f}")
print(f"  Pred range: [{y_pred_no_log.min():.2f}, {y_pred_no_log.max():.2f}]")

# =============================================================================
# ANALYZE PERFORMANCE BY LOS BUCKET
# =============================================================================
print("\n[6/7] Analyzing performance by LOS bucket...")

best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
y_pred_best = val_predictions[best_model]

print(f"\n{best_model} RMSE by actual LOS bucket:")
for low, high in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 110)]:
    mask = (y_val_orig >= low) & (y_val_orig < high)
    if mask.sum() > 5:
        bucket_rmse = np.sqrt(np.mean((y_val_orig[mask] - y_pred_best[mask])**2))
        count = mask.sum()
        pct = mask.sum() / len(y_val_orig) * 100
        print(f"  LOS {low:3}-{high:3}: RMSE={bucket_rmse:6.2f} (n={count:4}, {pct:5.1f}%)")

# =============================================================================
# CREATE SUBMISSIONS
# =============================================================================
print("\n[7/7] Creating submissions...")

submission_dir = BASE_DIR / "submissions" / "regression"
submission_dir.mkdir(parents=True, exist_ok=True)

submissions = {}

for name, data in results.items():
    pipeline = data['pipeline']

    # Predict on test (in log space)
    y_test_log = pipeline.predict(X_test)

    # Transform back and clip
    y_test_pred = np.expm1(y_test_log)
    y_test_pred = np.clip(y_test_pred, 0, 100)

    # Create submission
    sub_df = pd.DataFrame({
        'icustay_id': test_ids,
        'LOS': y_test_pred
    })

    fname = f"submission_los_{name.lower()}.csv"
    sub_df.to_csv(submission_dir / fname, index=False)
    submissions[name] = {
        'file': fname,
        'min': y_test_pred.min(),
        'max': y_test_pred.max(),
        'mean': y_test_pred.mean()
    }

    print(f"\n{name}:")
    print(f"  File: {fname}")
    print(f"  Predictions - min: {y_test_pred.min():.2f}, max: {y_test_pred.max():.2f}, mean: {y_test_pred.mean():.2f}")

# Ensemble of log-transformed models
print("\nCreating ensemble of log-transformed models...")

# Weight by inverse RMSE
weights = {}
total_inv_rmse = sum(1/results[k]['rmse'] for k in results)
for name in results:
    weights[name] = (1/results[name]['rmse']) / total_inv_rmse

# Ensemble predictions
ensemble_pred_log = sum(
    weights[name] * results[name]['pipeline'].predict(X_test)
    for name in results
)
ensemble_pred = np.expm1(ensemble_pred_log)
ensemble_pred = np.clip(ensemble_pred, 0, 100)

# Save ensemble
sub_df = pd.DataFrame({
    'icustay_id': test_ids,
    'LOS': ensemble_pred
})
sub_df.to_csv(submission_dir / "submission_los_log_ensemble.csv", index=False)

print(f"\nlog_ensemble:")
print(f"  File: submission_los_log_ensemble.csv")
print(f"  Predictions - min: {ensemble_pred.min():.2f}, max: {ensemble_pred.max():.2f}, mean: {ensemble_pred.mean():.2f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nModel Comparison:")
print(f"  GB_original (no log): RMSE = {rmse_orig:.4f}")
for name, data in results.items():
    print(f"  {name}: RMSE = {data['rmse']:.4f}")

if results[best_model]['rmse'] < rmse_orig:
    improvement = (rmse_orig - results[best_model]['rmse']) / rmse_orig * 100
    print(f"\n✓ Log transform improved validation RMSE by {improvement:.1f}%!")
else:
    print(f"\n⚠ Log transform did not improve validation RMSE")

print("\nKey improvements from log transform:")
print("  - Better handling of extreme LOS values (20+ days)")
print("  - Predictions can go higher (expm1 expands the range)")
print("  - More symmetric error distribution")

print("\nSubmissions to try on Kaggle (in order):")
print("  1. submission_los_log_ensemble.csv - ensemble of log-transformed models")
print("  2. submission_los_gb_log.csv - single best log model")

print("\n" + "=" * 70)
print(f"Files saved to: {submission_dir}")
print("=" * 70)
