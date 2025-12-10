"""
Quick Metric Test - Generate submissions optimized for different metrics

Since we don't know the Kaggle metric, try different loss functions
and submit to see which improves.

Run: python scripts/regression/los_metric_test.py
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
print("METRIC TEST - Different Loss Functions")
print("=" * 70)

# =============================================================================
# Load data (minimal features - just what works)
# =============================================================================
print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# Add just the best features
def add_best_features(df):
    df = df.copy()
    eps = 1e-3

    # vital_product
    if all(c in df.columns for c in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']

    # HR_RR interaction
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR'] = df['HeartRate_Mean'] * df['RespRate_Mean']

    return df

X_train_fe = add_best_features(X_train_base)
X_test_fe = add_best_features(X_test_base)

# Preprocessor
numerical_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Transform
X_full_t = preprocessor.fit_transform(X_train_fe)
X_test_t = preprocessor.transform(X_test_fe)

# Quick validation split for metrics
X_train, X_val, y_train, y_val = train_test_split(
    X_train_fe, y_train_full, test_size=0.2, random_state=42
)
X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)

test_ids = test_df_orig['icustay_id'].values
base_dir, _, _ = get_paths()
output_dir = base_dir / "submissions" / "regression"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Features: {X_train_fe.shape[1]}")

# =============================================================================
# Test different approaches
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSIONS")
print("=" * 70)

submissions = []

# 1. MSE (baseline)
print("\n1. MSE (squared_error)...")
model = HistGradientBoostingRegressor(
    loss='squared_error', max_iter=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=20, random_state=42
)
model.fit(X_train_t, y_train)
val_pred = model.predict(X_val_t)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

model.fit(X_full_t, y_train_full)
test_pred = model.predict(X_test_t)
print(f"   Val RMSE: {val_rmse:.4f}, Range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
submissions.append(('mse', test_pred, val_rmse))

# 2. MAE (absolute_error)
print("\n2. MAE (absolute_error)...")
model = HistGradientBoostingRegressor(
    loss='absolute_error', max_iter=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=20, random_state=42
)
model.fit(X_train_t, y_train)
val_pred = model.predict(X_val_t)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

model.fit(X_full_t, y_train_full)
test_pred = model.predict(X_test_t)
print(f"   Val RMSE: {val_rmse:.4f}, Range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
submissions.append(('mae', test_pred, val_rmse))

# 3. Poisson (for count-like data)
print("\n3. Poisson...")
model = HistGradientBoostingRegressor(
    loss='poisson', max_iter=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=20, random_state=42
)
model.fit(X_train_t, y_train)
val_pred = model.predict(X_val_t)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

model.fit(X_full_t, y_train_full)
test_pred = model.predict(X_test_t)
print(f"   Val RMSE: {val_rmse:.4f}, Range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
submissions.append(('poisson', test_pred, val_rmse))

# 4. Log-transformed target (for RMSLE metric)
print("\n4. Log-transformed target...")
y_train_log = np.log1p(y_train)
y_full_log = np.log1p(y_train_full)

model = HistGradientBoostingRegressor(
    loss='squared_error', max_iter=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=20, random_state=42
)
model.fit(X_train_t, y_train_log)
val_pred_log = model.predict(X_val_t)
val_pred = np.expm1(val_pred_log)  # Transform back
val_pred = np.clip(val_pred, 0, None)  # No negatives
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

model.fit(X_full_t, y_full_log)
test_pred_log = model.predict(X_test_t)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)
print(f"   Val RMSE: {val_rmse:.4f}, Range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
submissions.append(('log_target', test_pred, val_rmse))

# 5. Quantile 50 (median)
print("\n5. Quantile 50 (median)...")
model = HistGradientBoostingRegressor(
    loss='quantile', quantile=0.5, max_iter=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=20, random_state=42
)
model.fit(X_train_t, y_train)
val_pred = model.predict(X_val_t)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

model.fit(X_full_t, y_train_full)
test_pred = model.predict(X_test_t)
print(f"   Val RMSE: {val_rmse:.4f}, Range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
submissions.append(('quantile50', test_pred, val_rmse))

# =============================================================================
# Save all submissions
# =============================================================================
print("\n" + "=" * 70)
print("SAVING FILES")
print("=" * 70)

for name, pred, rmse in submissions:
    sub = pd.DataFrame({'icustay_id': test_ids, 'LOS': pred})
    filename = output_dir / f"submission_los_{name}.csv"
    sub.to_csv(filename, index=False)
    print(f"  {filename.name} (val RMSE: {rmse:.4f})")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
APPROACH:
  Generate 5 submissions with different loss functions to identify
  which metric Kaggle might be using.

FILES TO SUBMIT (in order of priority):
  1. submission_los_mse.csv - baseline MSE
  2. submission_los_log_target.csv - if metric is RMSLE
  3. submission_los_poisson.csv - alternative for right-skewed
  4. submission_los_mae.csv - if metric is MAE-based
  5. submission_los_quantile50.csv - if metric is median-based

INTERPRETATION:
  - If log_target improves: metric is likely RMSLE
  - If mae improves: metric is MAE-based
  - If poisson improves: metric penalizes under-prediction
  - If quantile50 improves: metric ignores outliers
  - If mse is still best: we need better features, not different metric
""")

print("=" * 70)
