"""
LOS Fresh Start - Clean, minimal approach

Best so far: 20.0 (improved features, HistGB, no tricks)
Target: 15.7

New ideas to try:
1. Quantile regression (predict median, not mean - robust to outliers)
2. Different loss functions (Huber, MAE)
3. Feature selection (maybe we have too many features?)

Run: python scripts/regression/los_fresh_start.py
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

from los_prep import prepare_data, get_paths, load_raw_data

print("=" * 70)
print("LOS FRESH START")
print("=" * 70)

# =============================================================================
# MINIMAL FEATURE ENGINEERING - Only what we know helps
# =============================================================================

def add_core_features(df):
    """Only the features with highest correlation."""
    df = df.copy()
    eps = 1e-3

    # 1. Vital product (corr 0.2010) - BEST feature
    if all(c in df.columns for c in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']

    # 2. CV features (corr 0.17+)
    for vital, abbrev in [('SysBP', 'SBP'), ('TempC', 'Temp'), ('RespRate', 'RR'), ('HeartRate', 'HR')]:
        max_col, min_col, mean_col = f'{vital}_Max', f'{vital}_Min', f'{vital}_Mean'
        if all(c in df.columns for c in [max_col, min_col, mean_col]):
            range_val = df[max_col] - df[min_col]
            df[f'{abbrev}_cv'] = range_val / df[mean_col].clip(lower=eps)

    # 3. Key interaction (corr 0.16)
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR'] = df['HeartRate_Mean'] * df['RespRate_Mean']

    return df

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# Add core features
X_train_fe = add_core_features(X_train_base)
X_test_fe = add_core_features(X_test_base)

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

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_fe, y_train_full, test_size=0.2, random_state=42
)

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# =============================================================================
# TEST DIFFERENT LOSS FUNCTIONS
# =============================================================================
print("\n" + "=" * 70)
print("TESTING DIFFERENT APPROACHES")
print("=" * 70)

results = []

# 1. Squared Error (default) - what we've been using
print("\n1. SQUARED ERROR (default)...")
model_se = HistGradientBoostingRegressor(
    loss='squared_error',
    max_iter=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
model_se.fit(X_train_t, y_train)
pred_se = model_se.predict(X_val_t)
rmse_se = np.sqrt(mean_squared_error(y_val, pred_se))
mae_se = mean_absolute_error(y_val, pred_se)
print(f"   RMSE: {rmse_se:.4f}, MAE: {mae_se:.4f}")
print(f"   Range: [{pred_se.min():.2f}, {pred_se.max():.2f}]")
results.append(('squared_error', rmse_se, pred_se, model_se))

# 2. Absolute Error (MAE) - more robust to outliers
print("\n2. ABSOLUTE ERROR (MAE loss)...")
model_ae = HistGradientBoostingRegressor(
    loss='absolute_error',
    max_iter=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
model_ae.fit(X_train_t, y_train)
pred_ae = model_ae.predict(X_val_t)
rmse_ae = np.sqrt(mean_squared_error(y_val, pred_ae))
mae_ae = mean_absolute_error(y_val, pred_ae)
print(f"   RMSE: {rmse_ae:.4f}, MAE: {mae_ae:.4f}")
print(f"   Range: [{pred_ae.min():.2f}, {pred_ae.max():.2f}]")
results.append(('absolute_error', rmse_ae, pred_ae, model_ae))

# 3. Quantile Regression (median) - predicts 50th percentile
print("\n3. QUANTILE REGRESSION (median)...")
model_q50 = HistGradientBoostingRegressor(
    loss='quantile',
    quantile=0.5,
    max_iter=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
model_q50.fit(X_train_t, y_train)
pred_q50 = model_q50.predict(X_val_t)
rmse_q50 = np.sqrt(mean_squared_error(y_val, pred_q50))
mae_q50 = mean_absolute_error(y_val, pred_q50)
print(f"   RMSE: {rmse_q50:.4f}, MAE: {mae_q50:.4f}")
print(f"   Range: [{pred_q50.min():.2f}, {pred_q50.max():.2f}]")
results.append(('quantile_50', rmse_q50, pred_q50, model_q50))

# 4. Quantile Regression (60th percentile) - slightly higher predictions
print("\n4. QUANTILE REGRESSION (60th percentile)...")
model_q60 = HistGradientBoostingRegressor(
    loss='quantile',
    quantile=0.6,
    max_iter=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
model_q60.fit(X_train_t, y_train)
pred_q60 = model_q60.predict(X_val_t)
rmse_q60 = np.sqrt(mean_squared_error(y_val, pred_q60))
mae_q60 = mean_absolute_error(y_val, pred_q60)
print(f"   RMSE: {rmse_q60:.4f}, MAE: {mae_q60:.4f}")
print(f"   Range: [{pred_q60.min():.2f}, {pred_q60.max():.2f}]")
results.append(('quantile_60', rmse_q60, pred_q60, model_q60))

# 5. Poisson (for count-like data)
print("\n5. POISSON LOSS...")
model_pois = HistGradientBoostingRegressor(
    loss='poisson',
    max_iter=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
model_pois.fit(X_train_t, y_train)
pred_pois = model_pois.predict(X_val_t)
rmse_pois = np.sqrt(mean_squared_error(y_val, pred_pois))
mae_pois = mean_absolute_error(y_val, pred_pois)
print(f"   RMSE: {rmse_pois:.4f}, MAE: {mae_pois:.4f}")
print(f"   Range: [{pred_pois.min():.2f}, {pred_pois.max():.2f}]")
results.append(('poisson', rmse_pois, pred_pois, model_pois))

# =============================================================================
# COMPARE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

results_sorted = sorted(results, key=lambda x: x[1])
print("\nRanked by validation RMSE:")
for i, (name, rmse, pred, model) in enumerate(results_sorted):
    print(f"  {i+1}. {name:20} RMSE={rmse:.4f}  range=[{pred.min():.2f}, {pred.max():.2f}]")

best_name, best_rmse, best_pred, best_model = results_sorted[0]
print(f"\n>>> Best: {best_name} (RMSE={best_rmse:.4f})")

# =============================================================================
# BUCKET ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print(f"RMSE BY BUCKET ({best_name})")
print("=" * 70)

buckets = [(0, 3), (3, 7), (7, 14), (14, 30), (30, 200)]
for low, high in buckets:
    mask = (y_val >= low) & (y_val < high)
    if mask.sum() > 0:
        bucket_rmse = np.sqrt(mean_squared_error(y_val[mask], best_pred[mask]))
        pct = mask.sum() / len(y_val) * 100
        avg_pred = best_pred[mask].mean()
        avg_actual = y_val[mask].mean()
        print(f"  [{low:2}-{high:3}] days: RMSE={bucket_rmse:6.2f} | "
              f"pred={avg_pred:5.1f} vs actual={avg_actual:5.1f} | "
              f"n={mask.sum()} ({pct:.1f}%)")

# =============================================================================
# GENERATE SUBMISSIONS
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSIONS")
print("=" * 70)

# Transform full data
X_full_t = preprocessor.fit_transform(X_train_fe)
X_test_t = preprocessor.transform(X_test_fe)

test_ids = test_df_orig['icustay_id'].values
base_dir, _, _ = get_paths()
output_dir = base_dir / "submissions" / "regression"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Best model submission
print(f"\n1. Best model ({best_name})...")
if best_name == 'squared_error':
    final_model = HistGradientBoostingRegressor(
        loss='squared_error', max_iter=300, learning_rate=0.05,
        max_depth=5, min_samples_leaf=20, random_state=42
    )
elif best_name == 'absolute_error':
    final_model = HistGradientBoostingRegressor(
        loss='absolute_error', max_iter=300, learning_rate=0.05,
        max_depth=5, min_samples_leaf=20, random_state=42
    )
elif best_name == 'quantile_50':
    final_model = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.5, max_iter=300, learning_rate=0.05,
        max_depth=5, min_samples_leaf=20, random_state=42
    )
elif best_name == 'quantile_60':
    final_model = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.6, max_iter=300, learning_rate=0.05,
        max_depth=5, min_samples_leaf=20, random_state=42
    )
else:  # poisson
    final_model = HistGradientBoostingRegressor(
        loss='poisson', max_iter=300, learning_rate=0.05,
        max_depth=5, min_samples_leaf=20, random_state=42
    )

final_model.fit(X_full_t, y_train_full)
test_pred = final_model.predict(X_test_t)
print(f"   Range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")

sub = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred})
file_best = output_dir / f"submission_los_{best_name}.csv"
sub.to_csv(file_best, index=False)
print(f"   Saved: {file_best.name}")

# 2. Also save squared error (our baseline)
if best_name != 'squared_error':
    print("\n2. Squared error (baseline)...")
    model_se_full = HistGradientBoostingRegressor(
        loss='squared_error', max_iter=300, learning_rate=0.05,
        max_depth=5, min_samples_leaf=20, random_state=42
    )
    model_se_full.fit(X_full_t, y_train_full)
    test_pred_se = model_se_full.predict(X_test_t)
    print(f"   Range: [{test_pred_se.min():.2f}, {test_pred_se.max():.2f}]")

    sub_se = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred_se})
    file_se = output_dir / "submission_los_squared_error.csv"
    sub_se.to_csv(file_se, index=False)
    print(f"   Saved: {file_se.name}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
APPROACH:
  - Minimal features (only high-correlation ones)
  - Test different loss functions

RESULTS:
  Best: {best_name} (RMSE={best_rmse:.4f})

CURRENT BEST ON KAGGLE: 20.0
TARGET: 15.7

The 4.3 point gap likely requires:
  - Different features we don't have access to
  - A fundamentally different modeling approach
  - Or the scoring metric is not pure MSE/RMSE
""")

print("=" * 70)
