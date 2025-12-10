"""
LOS Regression - Two-Stage Model

Problem: Models can't predict long stays (LOS 20-50 days has RMSE 23.67)
Solution:
  Stage 1: Classify if patient will have long stay (>10 days)
  Stage 2: Use different regression models for short vs long stay patients

Run: python scripts/regression/generate_los_twostage.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

print("=" * 70)
print("LOS REGRESSION - TWO-STAGE MODEL")
print("=" * 70)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_min_bp_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['SysBP_Min', 'DiasBP_Min', 'MeanBP_Min']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.nan if x > 500 else x)
    return df


def add_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()

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

    if 'SOFA' in df.columns:
        df['high_sofa'] = (df['SOFA'] >= 8).astype(int)
    if 'SAPSII' in df.columns:
        df['high_saps'] = (df['SAPSII'] >= 40).astype(int)
    if 'GCS' in df.columns:
        df['low_gcs'] = (df['GCS'] <= 8).astype(int)

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

y = train_df['LOS'].copy()

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

numerical_cols = X_train_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train_full.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)}")

# =============================================================================
# DEFINE THRESHOLD FOR LONG STAY
# =============================================================================
# Try different thresholds
THRESHOLD = 10  # days - patients with LOS > 10 are "long stay"

y_long_stay = (y > THRESHOLD).astype(int)
print(f"\nThreshold: {THRESHOLD} days")
print(f"Long stay patients (>{THRESHOLD} days): {y_long_stay.sum()} ({y_long_stay.mean()*100:.1f}%)")

# =============================================================================
# PREPROCESSOR
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
X_train, X_val, y_train, y_val, y_train_long, y_val_long = train_test_split(
    X_train_full, y, y_long_stay, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# =============================================================================
# STAGE 1: CLASSIFIER FOR LONG STAY
# =============================================================================
print("\n[4/7] Stage 1: Training long-stay classifier...")

classifier = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

clf_pipeline.fit(X_train, y_train_long)

# Evaluate classifier
y_pred_long = clf_pipeline.predict(X_val)
y_pred_proba = clf_pipeline.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val_long, y_pred_long)
f1 = f1_score(y_val_long, y_pred_long)

print(f"  Accuracy: {acc:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Predicted long stays: {y_pred_long.sum()} ({y_pred_long.mean()*100:.1f}%)")
print(f"  Actual long stays: {y_val_long.sum()} ({y_val_long.mean()*100:.1f}%)")

# =============================================================================
# STAGE 2: SEPARATE REGRESSORS FOR SHORT AND LONG STAY
# =============================================================================
print("\n[5/7] Stage 2: Training separate regressors...")

# First, fit the preprocessor on ALL training data
preprocessor_fitted = preprocessor.fit(X_train)

# Transform all data once
X_train_transformed = preprocessor_fitted.transform(X_train)
X_val_transformed = preprocessor_fitted.transform(X_val)

# Split training data by long/short stay
short_mask_train = y_train_long == 0
long_mask_train = y_train_long == 1

X_train_short = X_train_transformed[short_mask_train]
y_train_short = y_train[short_mask_train].values
X_train_long = X_train_transformed[long_mask_train]
y_train_long_vals = y_train[long_mask_train].values

print(f"  Short-stay training samples: {len(X_train_short)}")
print(f"  Long-stay training samples: {len(X_train_long)}")

# Regressor for short stays
print("\n  Training short-stay regressor...")
reg_short = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)

reg_short.fit(X_train_short, y_train_short)

# Regressor for long stays - use deeper trees to capture more variance
print("  Training long-stay regressor...")
reg_long = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=7,  # Deeper for complex long-stay patterns
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

reg_long.fit(X_train_long, y_train_long_vals)

# =============================================================================
# EVALUATE TWO-STAGE MODEL
# =============================================================================
print("\n[6/7] Evaluating two-stage model...")

# Method 1: Hard classification (use predicted class to route)
y_pred_twostage_hard = np.zeros(len(X_val))
short_mask = y_pred_long == 0
long_mask = y_pred_long == 1

if short_mask.sum() > 0:
    y_pred_twostage_hard[short_mask] = reg_short.predict(X_val_transformed[short_mask])
if long_mask.sum() > 0:
    y_pred_twostage_hard[long_mask] = reg_long.predict(X_val_transformed[long_mask])

y_pred_twostage_hard = np.clip(y_pred_twostage_hard, 0, 150)

rmse_hard = np.sqrt(mean_squared_error(y_val, y_pred_twostage_hard))
mae_hard = mean_absolute_error(y_val, y_pred_twostage_hard)
print(f"\n  Hard routing (use predicted class):")
print(f"    RMSE: {rmse_hard:.4f}, MAE: {mae_hard:.4f}")
print(f"    Pred range: [{y_pred_twostage_hard.min():.2f}, {y_pred_twostage_hard.max():.2f}]")

# Method 2: Soft routing (weighted average by probability)
pred_short = reg_short.predict(X_val_transformed)
pred_long = reg_long.predict(X_val_transformed)

y_pred_twostage_soft = (1 - y_pred_proba) * pred_short + y_pred_proba * pred_long
y_pred_twostage_soft = np.clip(y_pred_twostage_soft, 0, 150)

rmse_soft = np.sqrt(mean_squared_error(y_val, y_pred_twostage_soft))
mae_soft = mean_absolute_error(y_val, y_pred_twostage_soft)
print(f"\n  Soft routing (weighted by probability):")
print(f"    RMSE: {rmse_soft:.4f}, MAE: {mae_soft:.4f}")
print(f"    Pred range: [{y_pred_twostage_soft.min():.2f}, {y_pred_twostage_soft.max():.2f}]")

# Method 3: Single model baseline for comparison
print("\n  Single model baseline...")
single_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    ))
])
single_pipeline.fit(X_train, y_train)
y_pred_single = single_pipeline.predict(X_val)
y_pred_single = np.clip(y_pred_single, 0, 150)
rmse_single = np.sqrt(mean_squared_error(y_val, y_pred_single))
print(f"    RMSE: {rmse_single:.4f}")
print(f"    Pred range: [{y_pred_single.min():.2f}, {y_pred_single.max():.2f}]")

# Analyze by LOS bucket
print("\n  RMSE by actual LOS bucket:")
best_pred = y_pred_twostage_soft if rmse_soft < rmse_hard else y_pred_twostage_hard
for low, high in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 110)]:
    mask = (y_val >= low) & (y_val < high)
    if mask.sum() > 5:
        bucket_rmse = np.sqrt(np.mean((y_val[mask] - best_pred[mask])**2))
        count = mask.sum()
        pct = mask.sum() / len(y_val) * 100
        print(f"    LOS {low:3}-{high:3}: RMSE={bucket_rmse:6.2f} (n={count:4}, {pct:5.1f}%)")

# =============================================================================
# CREATE SUBMISSIONS
# =============================================================================
print("\n[7/7] Creating submissions...")

submission_dir = BASE_DIR / "submissions" / "regression"
submission_dir.mkdir(parents=True, exist_ok=True)

# Transform test data
X_test_transformed = preprocessor_fitted.transform(X_test)

# Get predictions on test set
y_test_pred_class = clf_pipeline.predict(X_test)
y_test_proba = clf_pipeline.predict_proba(X_test)[:, 1]

# Hard routing
y_test_hard = np.zeros(len(X_test))
short_mask_test = y_test_pred_class == 0
long_mask_test = y_test_pred_class == 1

if short_mask_test.sum() > 0:
    y_test_hard[short_mask_test] = reg_short.predict(X_test_transformed[short_mask_test])
if long_mask_test.sum() > 0:
    y_test_hard[long_mask_test] = reg_long.predict(X_test_transformed[long_mask_test])

y_test_hard = np.clip(y_test_hard, 0, 150)

# Soft routing
pred_short_test = reg_short.predict(X_test_transformed)
pred_long_test = reg_long.predict(X_test_transformed)
y_test_soft = (1 - y_test_proba) * pred_short_test + y_test_proba * pred_long_test
y_test_soft = np.clip(y_test_soft, 0, 150)

# Save submissions
submissions = {
    'twostage_hard': y_test_hard,
    'twostage_soft': y_test_soft,
}

for name, preds in submissions.items():
    sub_df = pd.DataFrame({
        'icustay_id': test_ids,
        'LOS': preds
    })

    fname = f"submission_los_{name}.csv"
    sub_df.to_csv(submission_dir / fname, index=False)

    print(f"\n{name}:")
    print(f"  File: {fname}")
    print(f"  Predictions - min: {preds.min():.2f}, max: {preds.max():.2f}, mean: {preds.mean():.2f}")
    print(f"  Predicted long stays: {(y_test_pred_class == 1).sum()} ({(y_test_pred_class == 1).mean()*100:.1f}%)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nThreshold used: {THRESHOLD} days")
print(f"\nValidation Results:")
print(f"  Single model:     RMSE = {rmse_single:.4f}")
print(f"  Two-stage hard:   RMSE = {rmse_hard:.4f}")
print(f"  Two-stage soft:   RMSE = {rmse_soft:.4f}")

if rmse_soft < rmse_single:
    improvement = (rmse_single - rmse_soft) / rmse_single * 100
    print(f"\n✓ Two-stage soft improved over single model by {improvement:.1f}%!")
    print(f"  Try: submission_los_twostage_soft.csv")
elif rmse_hard < rmse_single:
    improvement = (rmse_single - rmse_hard) / rmse_single * 100
    print(f"\n✓ Two-stage hard improved over single model by {improvement:.1f}%!")
    print(f"  Try: submission_los_twostage_hard.csv")
else:
    print(f"\n⚠ Two-stage model did not improve over single model")
    print(f"  The classifier may not be separating long/short stays well enough")

print("\n" + "=" * 70)
print(f"Files saved to: {submission_dir}")
print("=" * 70)
