"""
Feature Ablation Test - Which features actually help?

Test different feature sets to see what's helping and what might be hurting.

Run: python scripts/regression/feature_ablation.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from los_prep import prepare_data, get_paths, load_raw_data

print("=" * 70)
print("FEATURE ABLATION TEST")
print("=" * 70)

# =============================================================================
# Load base data
# =============================================================================
print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# Split once - use same split for all tests
X_train, X_val, y_train, y_val = train_test_split(
    X_train_base, y_train_full, test_size=0.2, random_state=42
)

# Also split original data for diagnosis features
train_orig_split, val_orig_split, _, _ = train_test_split(
    train_df_orig, y_train_full, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# =============================================================================
# Feature engineering functions
# =============================================================================

def add_cv_features(df):
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

def add_vital_product(df):
    df = df.copy()
    if all(c in df.columns for c in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']
    return df

def add_interactions(df):
    df = df.copy()
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR_interaction'] = df['HeartRate_Mean'] * df['RespRate_Mean']
    if 'RespRate_Mean' in df.columns and 'TempC_Mean' in df.columns:
        df['RR_Temp_interaction'] = df['RespRate_Mean'] * df['TempC_Mean']
    return df

def add_squared_terms(df):
    df = df.copy()
    for col in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']:
        if col in df.columns:
            df[f'{col}_sq'] = df[col] ** 2
    return df

def add_diagnosis_flags(df, orig_df):
    df = df.copy()
    if 'DIAGNOSIS' in orig_df.columns and len(orig_df) == len(df):
        diag_upper = orig_df['DIAGNOSIS'].fillna('').str.upper()
        for kw in ['PANCREATITIS', 'SHOCK', 'SEPSIS', 'PNEUMONIA', 'TRANSPLANT']:
            df[f'diag_{kw.lower()}'] = diag_upper.str.contains(kw, na=False).astype(int)
    return df

def add_careunit_flags(df, orig_df):
    df = df.copy()
    if 'FIRST_CAREUNIT' in orig_df.columns and len(orig_df) == len(df):
        careunit = orig_df['FIRST_CAREUNIT']
        df['is_sicu'] = (careunit == 'SICU').astype(int)
        df['is_micu'] = (careunit == 'MICU').astype(int)
    return df

# =============================================================================
# Define feature sets to test
# =============================================================================

def train_and_eval(X_tr, X_va, y_tr, y_va, name):
    """Train model and return RMSE."""
    numerical_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_tr.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    X_tr_t = preprocessor.fit_transform(X_tr)
    X_va_t = preprocessor.transform(X_va)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=20, random_state=42
    )
    model.fit(X_tr_t, y_tr)
    pred = model.predict(X_va_t)
    rmse = np.sqrt(mean_squared_error(y_va, pred))

    return rmse, pred

# =============================================================================
# Test different feature combinations
# =============================================================================
print("\n" + "=" * 70)
print("TESTING FEATURE COMBINATIONS")
print("=" * 70)

results = []

# 1. Baseline (los_prep features only)
print("\n1. BASELINE (los_prep only)...")
rmse, pred = train_and_eval(X_train.copy(), X_val.copy(), y_train, y_val, "baseline")
print(f"   RMSE: {rmse:.4f}, Features: {X_train.shape[1]}")
results.append(('baseline', rmse, X_train.shape[1]))

# 2. + CV features
print("\n2. + CV features...")
X_tr = add_cv_features(X_train.copy())
X_va = add_cv_features(X_val.copy())
rmse, pred = train_and_eval(X_tr, X_va, y_train, y_val, "+cv")
print(f"   RMSE: {rmse:.4f}, Features: {X_tr.shape[1]}")
results.append(('+cv_features', rmse, X_tr.shape[1]))

# 3. + Vital product
print("\n3. + Vital product...")
X_tr = add_vital_product(X_train.copy())
X_va = add_vital_product(X_val.copy())
rmse, pred = train_and_eval(X_tr, X_va, y_train, y_val, "+vital_product")
print(f"   RMSE: {rmse:.4f}, Features: {X_tr.shape[1]}")
results.append(('+vital_product', rmse, X_tr.shape[1]))

# 4. + Interactions
print("\n4. + Interactions...")
X_tr = add_interactions(X_train.copy())
X_va = add_interactions(X_val.copy())
rmse, pred = train_and_eval(X_tr, X_va, y_train, y_val, "+interactions")
print(f"   RMSE: {rmse:.4f}, Features: {X_tr.shape[1]}")
results.append(('+interactions', rmse, X_tr.shape[1]))

# 5. + Squared terms
print("\n5. + Squared terms...")
X_tr = add_squared_terms(X_train.copy())
X_va = add_squared_terms(X_val.copy())
rmse, pred = train_and_eval(X_tr, X_va, y_train, y_val, "+squared")
print(f"   RMSE: {rmse:.4f}, Features: {X_tr.shape[1]}")
results.append(('+squared_terms', rmse, X_tr.shape[1]))

# 6. + Diagnosis flags
print("\n6. + Diagnosis flags...")
X_tr = add_diagnosis_flags(X_train.copy(), train_orig_split)
X_va = add_diagnosis_flags(X_val.copy(), val_orig_split)
rmse, pred = train_and_eval(X_tr, X_va, y_train, y_val, "+diagnosis")
print(f"   RMSE: {rmse:.4f}, Features: {X_tr.shape[1]}")
results.append(('+diagnosis_flags', rmse, X_tr.shape[1]))

# 7. + Careunit flags
print("\n7. + Careunit flags...")
X_tr = add_careunit_flags(X_train.copy(), train_orig_split)
X_va = add_careunit_flags(X_val.copy(), val_orig_split)
rmse, pred = train_and_eval(X_tr, X_va, y_train, y_val, "+careunit")
print(f"   RMSE: {rmse:.4f}, Features: {X_tr.shape[1]}")
results.append(('+careunit_flags', rmse, X_tr.shape[1]))

# 8. CV + Vital product (best combo?)
print("\n8. CV + Vital product...")
X_tr = add_cv_features(X_train.copy())
X_tr = add_vital_product(X_tr)
X_va = add_cv_features(X_val.copy())
X_va = add_vital_product(X_va)
rmse, pred = train_and_eval(X_tr, X_va, y_train, y_val, "cv+vital")
print(f"   RMSE: {rmse:.4f}, Features: {X_tr.shape[1]}")
results.append(('cv+vital_product', rmse, X_tr.shape[1]))

# 9. All engineered features
print("\n9. ALL engineered features...")
X_tr = add_cv_features(X_train.copy())
X_tr = add_vital_product(X_tr)
X_tr = add_interactions(X_tr)
X_tr = add_squared_terms(X_tr)
X_tr = add_diagnosis_flags(X_tr, train_orig_split)
X_tr = add_careunit_flags(X_tr, train_orig_split)

X_va = add_cv_features(X_val.copy())
X_va = add_vital_product(X_va)
X_va = add_interactions(X_va)
X_va = add_squared_terms(X_va)
X_va = add_diagnosis_flags(X_va, val_orig_split)
X_va = add_careunit_flags(X_va, val_orig_split)

rmse, pred = train_and_eval(X_tr, X_va, y_train, y_val, "all")
print(f"   RMSE: {rmse:.4f}, Features: {X_tr.shape[1]}")
results.append(('all_features', rmse, X_tr.shape[1]))

# 10. Minimal: only top 3 features
print("\n10. MINIMAL (only top correlated features)...")
X_tr = add_cv_features(X_train.copy())
X_tr = add_vital_product(X_tr)
# Keep only vital_product and top CV features
minimal_cols = [c for c in X_tr.columns if c in X_train.columns or
                c in ['vital_product', 'SBP_cv', 'Temp_cv', 'RR_cv', 'HR_cv']]
X_tr_min = X_tr[minimal_cols]

X_va = add_cv_features(X_val.copy())
X_va = add_vital_product(X_va)
X_va_min = X_va[minimal_cols]

rmse, pred = train_and_eval(X_tr_min, X_va_min, y_train, y_val, "minimal")
print(f"   RMSE: {rmse:.4f}, Features: {X_tr_min.shape[1]}")
results.append(('minimal_top_features', rmse, X_tr_min.shape[1]))

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY - Ranked by RMSE")
print("=" * 70)

results_sorted = sorted(results, key=lambda x: x[1])

print("\nFeature Set                    | RMSE   | # Features")
print("-" * 55)
for name, rmse, n_feat in results_sorted:
    print(f"{name:30} | {rmse:.4f} | {n_feat}")

best_name, best_rmse, _ = results_sorted[0]
baseline_rmse = [r[1] for r in results if r[0] == 'baseline'][0]

print(f"\nBest: {best_name} (RMSE={best_rmse:.4f})")
print(f"Baseline: {baseline_rmse:.4f}")
print(f"Improvement: {baseline_rmse - best_rmse:.4f}")

# Check what hurts
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

baseline_rmse = [r[1] for r in results if r[0] == 'baseline'][0]
for name, rmse, _ in results:
    diff = rmse - baseline_rmse
    if diff > 0.01:
        print(f"  {name}: HURTS (+{diff:.4f})")
    elif diff < -0.01:
        print(f"  {name}: HELPS ({diff:.4f})")
    else:
        print(f"  {name}: NEUTRAL ({diff:+.4f})")

print("\n" + "=" * 70)
