"""
Test Additional Categorical Features - Quick test

Add INSURANCE, MARITAL_STATUS, ETHNICITY to see if they help.
Also test ICD9 grouping.

Run: python scripts/regression/los_extra_cats.py
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
print("EXTRA CATEGORICAL FEATURES TEST")
print("=" * 70)

# =============================================================================
# Load data
# =============================================================================
print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# =============================================================================
# Add extra categorical features
# =============================================================================

def add_extra_cats(X_df, orig_df):
    """Add additional categorical columns from original data."""
    df = X_df.copy()

    # Add columns if they exist and match length
    extra_cats = ['INSURANCE', 'MARITAL_STATUS', 'ETHNICITY', 'RELIGION']

    for col in extra_cats:
        if col in orig_df.columns and len(orig_df) == len(df):
            df[col] = orig_df[col].values
            # Fill NaN with 'Unknown'
            df[col] = df[col].fillna('Unknown')

    return df

def add_icd9_group(X_df, orig_df):
    """Add ICD9 category (first 3 digits) instead of full code."""
    df = X_df.copy()

    if 'ICD9_diagnosis' in orig_df.columns and len(orig_df) == len(df):
        # Extract first 3 characters as category
        icd9 = orig_df['ICD9_diagnosis'].astype(str).fillna('000')
        df['ICD9_category'] = icd9.str[:3]

        # Also create broader groups
        # V codes = supplementary, E codes = external causes
        # 001-139 = infectious, 140-239 = neoplasms, 240-279 = endocrine
        # 280-289 = blood, 290-319 = mental, 320-389 = nervous
        # 390-459 = circulatory, 460-519 = respiratory, 520-579 = digestive
        # 580-629 = genitourinary, 680-709 = skin, 710-739 = musculoskeletal
        # 780-799 = symptoms, 800-999 = injury

        def icd9_to_system(code):
            try:
                if code.startswith('V'):
                    return 'supplementary'
                elif code.startswith('E'):
                    return 'external'
                num = int(code[:3])
                if num < 140:
                    return 'infectious'
                elif num < 240:
                    return 'neoplasm'
                elif num < 280:
                    return 'endocrine'
                elif num < 290:
                    return 'blood'
                elif num < 320:
                    return 'mental'
                elif num < 390:
                    return 'nervous'
                elif num < 460:
                    return 'circulatory'
                elif num < 520:
                    return 'respiratory'
                elif num < 580:
                    return 'digestive'
                elif num < 630:
                    return 'genitourinary'
                elif num < 710:
                    return 'skin'
                elif num < 740:
                    return 'musculoskeletal'
                elif num < 800:
                    return 'symptoms'
                else:
                    return 'injury'
            except:
                return 'other'

        df['ICD9_system'] = icd9.apply(icd9_to_system)

    return df

def add_best_features(df):
    """Add our best engineered features."""
    df = df.copy()

    # vital_product
    if all(c in df.columns for c in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']

    # HR_RR interaction
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR'] = df['HeartRate_Mean'] * df['RespRate_Mean']

    return df

# =============================================================================
# Test different feature sets
# =============================================================================
print("\n" + "=" * 70)
print("TESTING FEATURE COMBINATIONS")
print("=" * 70)

results = []

# Split once
X_train, X_val, y_train, y_val = train_test_split(
    X_train_base, y_train_full, test_size=0.2, random_state=42
)
train_orig_split, val_orig_split, _, _ = train_test_split(
    train_df_orig, y_train_full, test_size=0.2, random_state=42
)

def train_and_eval(X_tr, X_va, name):
    """Train Poisson model and return score."""
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

    # Use Poisson since it scored best
    model = HistGradientBoostingRegressor(
        loss='poisson',
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_tr_t, y_train)
    pred = model.predict(X_va_t)

    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return rmse, pred

# 1. Baseline (what we have)
print("\n1. BASELINE (current features)...")
X_tr = add_best_features(X_train.copy())
X_va = add_best_features(X_val.copy())
rmse, _ = train_and_eval(X_tr, X_va, "baseline")
print(f"   RMSE: {rmse:.4f}")
results.append(('baseline', rmse))

# 2. + Extra categoricals
print("\n2. + INSURANCE, MARITAL_STATUS, ETHNICITY, RELIGION...")
X_tr = add_best_features(X_train.copy())
X_tr = add_extra_cats(X_tr, train_orig_split)
X_va = add_best_features(X_val.copy())
X_va = add_extra_cats(X_va, val_orig_split)
rmse, _ = train_and_eval(X_tr, X_va, "+extra_cats")
print(f"   RMSE: {rmse:.4f}")
results.append(('+extra_cats', rmse))

# 3. + ICD9 system groups
print("\n3. + ICD9 system groups...")
X_tr = add_best_features(X_train.copy())
X_tr = add_icd9_group(X_tr, train_orig_split)
X_va = add_best_features(X_val.copy())
X_va = add_icd9_group(X_va, val_orig_split)
rmse, _ = train_and_eval(X_tr, X_va, "+icd9_system")
print(f"   RMSE: {rmse:.4f}")
results.append(('+icd9_system', rmse))

# 4. + Both extra cats and ICD9
print("\n4. + Extra cats AND ICD9 system...")
X_tr = add_best_features(X_train.copy())
X_tr = add_extra_cats(X_tr, train_orig_split)
X_tr = add_icd9_group(X_tr, train_orig_split)
X_va = add_best_features(X_val.copy())
X_va = add_extra_cats(X_va, val_orig_split)
X_va = add_icd9_group(X_va, val_orig_split)
rmse, _ = train_and_eval(X_tr, X_va, "+all")
print(f"   RMSE: {rmse:.4f}")
results.append(('+all_extra', rmse))

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results_sorted = sorted(results, key=lambda x: x[1])
print("\nRanked by RMSE:")
for name, rmse in results_sorted:
    print(f"  {name:20} RMSE={rmse:.4f}")

best_name, best_rmse = results_sorted[0]
baseline_rmse = [r[1] for r in results if r[0] == 'baseline'][0]

print(f"\nBest: {best_name} (RMSE={best_rmse:.4f})")
print(f"Improvement over baseline: {baseline_rmse - best_rmse:.4f}")

# =============================================================================
# Generate submission if improvement found
# =============================================================================
if best_rmse < baseline_rmse - 0.01:
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSION")
    print("=" * 70)

    # Prepare full data with best features
    X_train_fe = add_best_features(X_train_base)
    X_test_fe = add_best_features(X_test_base)

    if 'extra' in best_name or 'all' in best_name:
        X_train_fe = add_extra_cats(X_train_fe, train_df_orig)
        X_test_fe = add_extra_cats(X_test_fe, test_df_orig)

    if 'icd9' in best_name or 'all' in best_name:
        X_train_fe = add_icd9_group(X_train_fe, train_df_orig)
        X_test_fe = add_icd9_group(X_test_fe, test_df_orig)

    numerical_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    X_full_t = preprocessor.fit_transform(X_train_fe)
    X_test_t = preprocessor.transform(X_test_fe)

    model = HistGradientBoostingRegressor(
        loss='poisson',
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_full_t, y_train_full)
    test_pred = model.predict(X_test_t)

    test_ids = test_df_orig['icustay_id'].values
    base_dir, _, _ = get_paths()
    output_dir = base_dir / "submissions" / "regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    sub = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred})
    filename = output_dir / f"submission_los_poisson_extra_cats.csv"
    sub.to_csv(filename, index=False)
    print(f"\nSaved: {filename.name}")
    print(f"Range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
else:
    print("\nNo significant improvement - not generating submission.")

print("\n" + "=" * 70)
