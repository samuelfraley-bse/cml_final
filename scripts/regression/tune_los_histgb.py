"""
Quick Hyperparameter Tuning for LOS with HistGradientBoosting

Current best: 20.0 on Kaggle (with improved features, no weighting)
Target: 15.7

Run: python scripts/regression/tune_los_histgb.py
"""

import sys
from pathlib import Path

# Add scripts/regression to path for los_prep import
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from itertools import product

from los_prep import prepare_data, get_paths, load_raw_data

print("=" * 70)
print("HYPERPARAMETER TUNING FOR LOS")
print("=" * 70)

# =============================================================================
# Feature Engineering (same as generate_los_improved_features.py)
# =============================================================================

def add_cv_features(df):
    df = df.copy()
    eps = 1e-3
    vital_configs = [
        ('SysBP', 'SBP'), ('TempC', 'Temp'), ('RespRate', 'RR'),
        ('HeartRate', 'HR'), ('Glucose', 'Gluc'), ('DiasBP', 'DBP'),
        ('MeanBP', 'MBP'), ('SpO2', 'SpO2'),
    ]
    for vital, abbrev in vital_configs:
        max_col, min_col, mean_col = f'{vital}_Max', f'{vital}_Min', f'{vital}_Mean'
        if max_col in df.columns and min_col in df.columns:
            range_col = f'{abbrev}_range'
            if range_col not in df.columns:
                df[range_col] = df[max_col] - df[min_col]
            if mean_col in df.columns:
                df[f'{abbrev}_cv'] = df[range_col] / (df[mean_col].clip(lower=eps))
    return df

def add_vital_products(df):
    df = df.copy()
    if all(col in df.columns for col in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']
        df['vital_product_mean'] = df['RespRate_Mean'] * df['TempC_Mean'] * df['HeartRate_Mean']
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR_interaction'] = df['HeartRate_Mean'] * df['RespRate_Mean']
    if 'RespRate_Mean' in df.columns and 'TempC_Mean' in df.columns:
        df['RR_Temp_interaction'] = df['RespRate_Mean'] * df['TempC_Mean']
    if 'HeartRate_Max' in df.columns and 'RespRate_Max' in df.columns:
        df['HRmax_RRmax'] = df['HeartRate_Max'] * df['RespRate_Max']
    return df

def add_severity_indicators(df):
    df = df.copy()
    critical_flags = ['tachy_flag', 'hypotension_flag', 'hypoxia_flag']
    existing_flags = [f for f in critical_flags if f in df.columns]
    if existing_flags:
        df['critical_flag_count'] = df[existing_flags].sum(axis=1)
    for col in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max', 'vital_product']:
        if col in df.columns:
            df[f'{col}_sq'] = df[col] ** 2
    for col in ['Glucose_Max', 'Glucose_Mean']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
    if 'RespRate_Max' in df.columns:
        df['high_resp_rate'] = (df['RespRate_Max'] > 30).astype(int)
    if 'TempC_Max' in df.columns:
        df['high_temp'] = (df['TempC_Max'] > 38.5).astype(int)
    if 'Glucose_Max' in df.columns:
        df['high_glucose'] = (df['Glucose_Max'] > 200).astype(int)
    return df

def add_diagnosis_features(df, orig_df):
    df = df.copy()
    if 'DIAGNOSIS' in orig_df.columns and len(orig_df) == len(df):
        diag_upper = orig_df['DIAGNOSIS'].fillna('').str.upper()
        for kw in ['PANCREATITIS', 'SHOCK', 'TRANSPLANT', 'ARREST', 'RESPIRATORY FAILURE', 'SEPSIS', 'PNEUMONIA']:
            df[f'diag_{kw.lower().replace(" ", "_")}'] = diag_upper.str.contains(kw, na=False).astype(int)
        for kw in ['OVERDOSE', 'DIABETIC', 'DKA', 'CABG']:
            df[f'diag_{kw.lower()}'] = diag_upper.str.contains(kw, na=False).astype(int)
    if 'ICD9_diagnosis' in orig_df.columns and len(orig_df) == len(df):
        icd9 = orig_df['ICD9_diagnosis'].astype(str)
        df['high_los_icd9'] = icd9.isin(['51884', '5770', '430', '44101', '0380', '51881', '0389', '431']).astype(int)
    return df

def add_careunit_features(df, orig_df):
    df = df.copy()
    if 'FIRST_CAREUNIT' in orig_df.columns and len(orig_df) == len(df):
        careunit = orig_df['FIRST_CAREUNIT']
        df['is_sicu'] = (careunit == 'SICU').astype(int)
        df['is_ccu'] = (careunit == 'CCU').astype(int)
        df['is_micu'] = (careunit == 'MICU').astype(int)
        df['is_csru'] = (careunit == 'CSRU').astype(int)
    return df

# =============================================================================
# Load and Prepare Data
# =============================================================================
print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# Apply feature engineering
for func in [add_cv_features, add_vital_products, add_severity_indicators]:
    X_train_base = func(X_train_base)
    X_test_base = func(X_test_base)

X_train_fe = add_diagnosis_features(X_train_base, train_df_orig)
X_train_fe = add_careunit_features(X_train_fe, train_df_orig)
X_test_fe = add_diagnosis_features(X_test_base, test_df_orig)
X_test_fe = add_careunit_features(X_test_fe, test_df_orig)

print(f"Features: {X_train_fe.shape[1]}")

# Prepare preprocessor
numerical_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_fe, y_train_full, test_size=0.2, random_state=42
)

# Transform
X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# =============================================================================
# Grid Search
# =============================================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER GRID SEARCH")
print("=" * 70)

# Parameter grid - reduced for speed (16 combinations instead of 81)
param_grid = {
    'max_iter': [300, 500],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5, 7],
    'min_samples_leaf': [10, 20],
}

# Generate all combinations
keys = list(param_grid.keys())
combinations = list(product(*param_grid.values()))

print(f"\nTesting {len(combinations)} parameter combinations...")
print(f"Parameters: {keys}\n")

results = []

for i, combo in enumerate(combinations):
    params = dict(zip(keys, combo))

    model = HistGradientBoostingRegressor(
        max_iter=params['max_iter'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )

    model.fit(X_train_t, y_train)
    pred = model.predict(X_val_t)
    rmse = np.sqrt(mean_squared_error(y_val, pred))

    results.append({
        'params': params,
        'rmse': rmse,
        'pred_min': pred.min(),
        'pred_max': pred.max()
    })

    # Print progress every 10
    if (i + 1) % 10 == 0 or (i + 1) == len(combinations):
        print(f"  [{i+1}/{len(combinations)}] Best so far: {min(r['rmse'] for r in results):.4f}")

# Sort by RMSE
results_sorted = sorted(results, key=lambda x: x['rmse'])

# =============================================================================
# Results
# =============================================================================
print("\n" + "=" * 70)
print("TOP 10 CONFIGURATIONS")
print("=" * 70)

for i, r in enumerate(results_sorted[:10]):
    p = r['params']
    print(f"\n{i+1}. RMSE: {r['rmse']:.4f}")
    print(f"   iter={p['max_iter']}, lr={p['learning_rate']}, "
          f"depth={p['max_depth']}, min_leaf={p['min_samples_leaf']}")
    print(f"   Pred range: [{r['pred_min']:.2f}, {r['pred_max']:.2f}]")

# =============================================================================
# Generate Submission with Best Params
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSION WITH BEST PARAMS")
print("=" * 70)

best = results_sorted[0]
best_params = best['params']

print(f"\nBest params: {best_params}")
print(f"Validation RMSE: {best['rmse']:.4f}")

# Train on full data
X_full_t = preprocessor.fit_transform(X_train_fe)
X_test_t = preprocessor.transform(X_test_fe)

final_model = HistGradientBoostingRegressor(
    max_iter=best_params['max_iter'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)

final_model.fit(X_full_t, y_train_full)
test_pred = final_model.predict(X_test_t)

print(f"Test pred range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")

# Save
test_ids = test_df_orig['icustay_id'].values
base_dir, _, _ = get_paths()
output_dir = base_dir / "submissions" / "regression"
output_dir.mkdir(parents=True, exist_ok=True)

submission = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred})
output_file = output_dir / "submission_los_tuned.csv"
submission.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
BEST CONFIGURATION:
  max_iter:        {best_params['max_iter']}
  learning_rate:   {best_params['learning_rate']}
  max_depth:       {best_params['max_depth']}
  min_samples_leaf: {best_params['min_samples_leaf']}

VALIDATION RMSE: {best['rmse']:.4f}

PREVIOUS:
  Default params: 4.6877 validation RMSE → 20.0 Kaggle

Submit {output_file.name} and compare to previous 20.0 score!
""")

print("=" * 70)
