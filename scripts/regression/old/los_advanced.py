"""
Advanced LOS Prediction with:
1. Target encoding for ICD9/DIAGNOSIS/CAREUNIT
2. Group-wise models (Surgical vs Medical ICU)
3. Enhanced severity flags
4. Simple ensemble (Linear + XGB/HistGB)

Run: python scripts/regression/los_advanced.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

from los_prep import prepare_data, get_paths, load_raw_data

print("=" * 70)
print("ADVANCED LOS PREDICTION")
print("=" * 70)

# =============================================================================
# 1. TARGET ENCODING FUNCTIONS
# =============================================================================

def target_encode_column(train_col, y_train, test_col, smoothing=20):
    """
    Smoothed target encoding for a categorical column.
    Shrinks rare categories toward global mean to prevent overfitting.
    """
    df = pd.DataFrame({'col': train_col.values, 'y': y_train.values})
    global_mean = df['y'].mean()

    # Stats per category
    stats = df.groupby('col')['y'].agg(['mean', 'count'])

    # Smoothed mean
    smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)

    enc_train = train_col.map(smooth).fillna(global_mean)
    enc_test = test_col.map(smooth).fillna(global_mean)

    return enc_train, enc_test


def target_encode_cv(train_col, y_train, test_col, n_folds=5, smoothing=20):
    """
    K-fold target encoding to prevent leakage on training data.
    """
    global_mean = y_train.mean()
    enc_train = pd.Series(index=train_col.index, dtype=float)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train_col):
        # Compute stats on fold's training portion
        fold_train = train_col.iloc[train_idx]
        fold_y = y_train.iloc[train_idx]

        df = pd.DataFrame({'col': fold_train.values, 'y': fold_y.values})
        stats = df.groupby('col')['y'].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)

        # Apply to validation portion
        enc_train.iloc[val_idx] = train_col.iloc[val_idx].map(smooth).fillna(global_mean)

    # For test, use all training data
    df_full = pd.DataFrame({'col': train_col.values, 'y': y_train.values})
    stats_full = df_full.groupby('col')['y'].agg(['mean', 'count'])
    smooth_full = (stats_full['count'] * stats_full['mean'] + smoothing * global_mean) / (stats_full['count'] + smoothing)
    enc_test = test_col.map(smooth_full).fillna(global_mean)

    return enc_train, enc_test


# =============================================================================
# 2. ENHANCED FEATURE ENGINEERING
# =============================================================================

def add_severity_flags(df):
    """Add clinical severity flags based on vital sign thresholds."""
    df = df.copy()

    # Severe conditions
    if 'MeanBP_Min' in df.columns:
        df['severe_hypotension'] = (df['MeanBP_Min'] < 60).astype(int)
    if 'HeartRate_Max' in df.columns:
        df['severe_tachycardia'] = (df['HeartRate_Max'] > 130).astype(int)
    if 'SpO2_Min' in df.columns:
        df['severe_hypoxemia'] = (df['SpO2_Min'] < 90).astype(int)
    if 'Glucose_Max' in df.columns:
        df['severe_hyperglycemia'] = (df['Glucose_Max'] > 250).astype(int)
    if 'TempC_Max' in df.columns:
        df['high_fever'] = (df['TempC_Max'] > 39).astype(int)
    if 'RespRate_Max' in df.columns:
        df['severe_tachypnea'] = (df['RespRate_Max'] > 35).astype(int)

    # Shock index
    if 'HeartRate_Max' in df.columns and 'SysBP_Min' in df.columns:
        df['shock_index_max'] = df['HeartRate_Max'] / df['SysBP_Min'].clip(lower=1)
        df['shock_index_high'] = (df['shock_index_max'] > 1.0).astype(int)

    # Count severity flags
    severity_cols = ['severe_hypotension', 'severe_tachycardia', 'severe_hypoxemia',
                     'severe_hyperglycemia', 'high_fever', 'severe_tachypnea']
    existing = [c for c in severity_cols if c in df.columns]
    if existing:
        df['severity_count'] = df[existing].sum(axis=1)

    return df


def add_cv_features(df):
    """Add coefficient of variation features."""
    df = df.copy()
    eps = 1e-3

    vitals = [('SysBP', 'SBP'), ('TempC', 'Temp'), ('RespRate', 'RR'),
              ('HeartRate', 'HR'), ('Glucose', 'Gluc')]

    for vital, abbrev in vitals:
        max_col, min_col, mean_col = f'{vital}_Max', f'{vital}_Min', f'{vital}_Mean'
        if all(c in df.columns for c in [max_col, min_col, mean_col]):
            range_val = df[max_col] - df[min_col]
            df[f'{abbrev}_cv'] = range_val / df[mean_col].clip(lower=eps)

    return df


def add_vital_products(df):
    """Add interaction features."""
    df = df.copy()

    if all(c in df.columns for c in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']

    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR_interaction'] = df['HeartRate_Mean'] * df['RespRate_Mean']

    return df


def categorize_diagnosis(diag_series):
    """Map free-text diagnosis to clinical categories."""
    diag_upper = diag_series.fillna('').str.upper()

    categories = pd.Series('OTHER', index=diag_series.index)

    # Order matters - more specific first
    categories[diag_upper.str.contains('SEPSIS|SEPTIC')] = 'SEPSIS'
    categories[diag_upper.str.contains('PNEUMONIA')] = 'PNEUMONIA'
    categories[diag_upper.str.contains('RESPIRATORY|COPD|ASTHMA')] = 'RESPIRATORY'
    categories[diag_upper.str.contains('CARDIAC|HEART|MI |STEMI|NSTEMI|CHF')] = 'CARDIAC'
    categories[diag_upper.str.contains('STROKE|CVA|BLEED|HEMORRHAGE')] = 'NEURO'
    categories[diag_upper.str.contains('TRANSPLANT')] = 'TRANSPLANT'
    categories[diag_upper.str.contains('PANCREATITIS')] = 'GI_SEVERE'
    categories[diag_upper.str.contains('TRAUMA|FRACTURE|INJURY')] = 'TRAUMA'
    categories[diag_upper.str.contains('CABG|VALVE|SURGERY')] = 'CARDIAC_SURG'
    categories[diag_upper.str.contains('OVERDOSE|INTOX')] = 'OVERDOSE'

    return categories


# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# Reset index to align with original data
y_train_full = y_train_full.reset_index(drop=True)

print(f"Base features: {X_train_base.shape[1]}")

# =============================================================================
# APPLY FEATURE ENGINEERING
# =============================================================================
print("\nApplying feature engineering...")

X_train_fe = add_severity_flags(X_train_base)
X_train_fe = add_cv_features(X_train_fe)
X_train_fe = add_vital_products(X_train_fe)

X_test_fe = add_severity_flags(X_test_base)
X_test_fe = add_cv_features(X_test_fe)
X_test_fe = add_vital_products(X_test_fe)

# =============================================================================
# TARGET ENCODING
# =============================================================================
print("\nApplying target encoding...")

# ICD9 target encoding
if 'ICD9_diagnosis' in train_df_orig.columns:
    icd_train = train_df_orig['ICD9_diagnosis'].astype(str).reset_index(drop=True)
    icd_test = test_df_orig['ICD9_diagnosis'].astype(str).reset_index(drop=True)

    icd_te_train, icd_te_test = target_encode_cv(icd_train, y_train_full, icd_test)
    X_train_fe['ICD9_te'] = icd_te_train.values
    X_test_fe['ICD9_te'] = icd_te_test.values
    print(f"  ICD9 target encoded: range [{icd_te_train.min():.2f}, {icd_te_train.max():.2f}]")

# Diagnosis category target encoding
if 'DIAGNOSIS' in train_df_orig.columns:
    diag_cat_train = categorize_diagnosis(train_df_orig['DIAGNOSIS']).reset_index(drop=True)
    diag_cat_test = categorize_diagnosis(test_df_orig['DIAGNOSIS']).reset_index(drop=True)

    diag_te_train, diag_te_test = target_encode_cv(diag_cat_train, y_train_full, diag_cat_test)
    X_train_fe['diag_cat_te'] = diag_te_train.values
    X_test_fe['diag_cat_te'] = diag_te_test.values
    print(f"  Diagnosis category encoded: range [{diag_te_train.min():.2f}, {diag_te_train.max():.2f}]")

# Care unit target encoding
if 'FIRST_CAREUNIT' in train_df_orig.columns:
    cu_train = train_df_orig['FIRST_CAREUNIT'].reset_index(drop=True)
    cu_test = test_df_orig['FIRST_CAREUNIT'].reset_index(drop=True)

    cu_te_train, cu_te_test = target_encode_cv(cu_train, y_train_full, cu_test)
    X_train_fe['careunit_te'] = cu_te_train.values
    X_test_fe['careunit_te'] = cu_te_test.values
    print(f"  Care unit encoded: range [{cu_te_train.min():.2f}, {cu_te_train.max():.2f}]")

# Admission type target encoding
if 'ADMISSION_TYPE' in train_df_orig.columns:
    at_train = train_df_orig['ADMISSION_TYPE'].reset_index(drop=True)
    at_test = test_df_orig['ADMISSION_TYPE'].reset_index(drop=True)

    at_te_train, at_te_test = target_encode_cv(at_train, y_train_full, at_test)
    X_train_fe['admission_te'] = at_te_train.values
    X_test_fe['admission_te'] = at_te_test.values
    print(f"  Admission type encoded: range [{at_te_train.min():.2f}, {at_te_train.max():.2f}]")

print(f"\nFinal features: {X_train_fe.shape[1]}")

# =============================================================================
# PREPARE FOR MODELING
# =============================================================================

# Identify column types
numerical_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)}")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_fe, y_train_full, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# =============================================================================
# MODEL 1: HistGradientBoosting
# =============================================================================
print("\n" + "=" * 70)
print("MODEL 1: HistGradientBoosting")
print("=" * 70)

# Preprocessor for HistGB (just passes through numerics, encodes categoricals)
hist_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

X_train_hist = hist_preprocessor.fit_transform(X_train)
X_val_hist = hist_preprocessor.transform(X_val)

hist_model = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=7,
    min_samples_leaf=10,
    random_state=42
)

hist_model.fit(X_train_hist, y_train)
pred_hist_val = hist_model.predict(X_val_hist)
rmse_hist = np.sqrt(mean_squared_error(y_val, pred_hist_val))

print(f"HistGB RMSE: {rmse_hist:.4f}")
print(f"Pred range: [{pred_hist_val.min():.2f}, {pred_hist_val.max():.2f}]")

# =============================================================================
# MODEL 2: Ridge Regression (Linear)
# =============================================================================
print("\n" + "=" * 70)
print("MODEL 2: Ridge Regression")
print("=" * 70)

# Preprocessor for Ridge (scale numerics, encode categoricals)
ridge_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

X_train_ridge = ridge_preprocessor.fit_transform(X_train)
X_val_ridge = ridge_preprocessor.transform(X_val)

# Impute NaNs for Ridge
imputer = SimpleImputer(strategy='median')
X_train_ridge = imputer.fit_transform(X_train_ridge)
X_val_ridge = imputer.transform(X_val_ridge)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_ridge, y_train)
pred_ridge_val = ridge_model.predict(X_val_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_val, pred_ridge_val))

print(f"Ridge RMSE: {rmse_ridge:.4f}")
print(f"Pred range: [{pred_ridge_val.min():.2f}, {pred_ridge_val.max():.2f}]")

# =============================================================================
# MODEL 3: ENSEMBLE (HistGB + Ridge)
# =============================================================================
print("\n" + "=" * 70)
print("MODEL 3: ENSEMBLE")
print("=" * 70)

# Find optimal alpha
best_alpha = 0.5
best_rmse = float('inf')

for alpha in np.arange(0.0, 1.05, 0.1):
    pred_ens = alpha * pred_hist_val + (1 - alpha) * pred_ridge_val
    rmse_ens = np.sqrt(mean_squared_error(y_val, pred_ens))
    if rmse_ens < best_rmse:
        best_rmse = rmse_ens
        best_alpha = alpha

pred_ens_val = best_alpha * pred_hist_val + (1 - best_alpha) * pred_ridge_val

print(f"Best alpha (HistGB weight): {best_alpha:.1f}")
print(f"Ensemble RMSE: {best_rmse:.4f}")
print(f"Pred range: [{pred_ens_val.min():.2f}, {pred_ens_val.max():.2f}]")

# =============================================================================
# RMSE BY BUCKET
# =============================================================================
print("\n" + "=" * 70)
print("RMSE BY LOS BUCKET (Ensemble)")
print("=" * 70)

buckets = [(0, 3), (3, 7), (7, 14), (14, 30), (30, 200)]
for low, high in buckets:
    mask = (y_val >= low) & (y_val < high)
    if mask.sum() > 0:
        bucket_rmse = np.sqrt(mean_squared_error(y_val[mask], pred_ens_val[mask]))
        pct = mask.sum() / len(y_val) * 100
        print(f"  [{low:2}-{high:3}] days: RMSE={bucket_rmse:6.2f} | n={mask.sum()} ({pct:.1f}%)")

# =============================================================================
# GENERATE SUBMISSIONS
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSIONS")
print("=" * 70)

# Transform full data
X_full_hist = hist_preprocessor.fit_transform(X_train_fe)
X_test_hist = hist_preprocessor.transform(X_test_fe)

X_full_ridge = ridge_preprocessor.fit_transform(X_train_fe)
X_test_ridge = ridge_preprocessor.transform(X_test_fe)
X_full_ridge = imputer.fit_transform(X_full_ridge)
X_test_ridge = imputer.transform(X_test_ridge)

# Retrain on full data
hist_model_full = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_depth=7,
    min_samples_leaf=10, random_state=42
)
hist_model_full.fit(X_full_hist, y_train_full)
pred_hist_test = hist_model_full.predict(X_test_hist)

ridge_model_full = Ridge(alpha=1.0)
ridge_model_full.fit(X_full_ridge, y_train_full)
pred_ridge_test = ridge_model_full.predict(X_test_ridge)

# Ensemble
pred_ens_test = best_alpha * pred_hist_test + (1 - best_alpha) * pred_ridge_test

print(f"\nHistGB test range: [{pred_hist_test.min():.2f}, {pred_hist_test.max():.2f}]")
print(f"Ridge test range: [{pred_ridge_test.min():.2f}, {pred_ridge_test.max():.2f}]")
print(f"Ensemble test range: [{pred_ens_test.min():.2f}, {pred_ens_test.max():.2f}]")

# Save submissions
test_ids = test_df_orig['icustay_id'].values
base_dir, _, _ = get_paths()
output_dir = base_dir / "submissions" / "regression"
output_dir.mkdir(parents=True, exist_ok=True)

# HistGB submission
sub_hist = pd.DataFrame({'icustay_id': test_ids, 'LOS': pred_hist_test})
file_hist = output_dir / "submission_los_advanced_histgb.csv"
sub_hist.to_csv(file_hist, index=False)
print(f"\nSaved: {file_hist.name}")

# Ensemble submission
sub_ens = pd.DataFrame({'icustay_id': test_ids, 'LOS': pred_ens_test})
file_ens = output_dir / "submission_los_advanced_ensemble.csv"
sub_ens.to_csv(file_ens, index=False)
print(f"Saved: {file_ens.name}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
ADVANCED TECHNIQUES USED:
  1. Target encoding for ICD9, Diagnosis category, Care unit, Admission type
  2. Enhanced severity flags (shock index, severe conditions)
  3. CV features and vital products
  4. Simple ensemble (HistGB + Ridge)

VALIDATION RESULTS:
  HistGB:   {rmse_hist:.4f}
  Ridge:    {rmse_ridge:.4f}
  Ensemble: {best_rmse:.4f} (alpha={best_alpha:.1f})

FILES GENERATED:
  1. {file_hist.name} - HistGB with target encoding
  2. {file_ens.name} - Ensemble (HistGB + Ridge)

PREVIOUS: 20.0 on Kaggle
TARGET:   15.7

Submit both and compare!
""")

print("=" * 70)
