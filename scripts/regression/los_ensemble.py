"""
Ensemble Models - Combine diverse models for better predictions

Averages predictions from multiple models with different loss functions
and algorithms to reduce variance.

Run: python scripts/regression/los_ensemble.py
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error

from los_prep import prepare_data, get_paths, load_raw_data

print("=" * 70)
print("ENSEMBLE MODELS")
print("=" * 70)

# =============================================================================
# Load and prepare data
# =============================================================================
print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# =============================================================================
# Feature engineering (best features + extra cats)
# =============================================================================

def add_extra_cats(X_df, orig_df):
    """Add additional categorical columns."""
    df = X_df.copy()
    extra_cats = ['INSURANCE', 'MARITAL_STATUS', 'ETHNICITY', 'RELIGION']
    for col in extra_cats:
        if col in orig_df.columns and len(orig_df) == len(df):
            df[col] = orig_df[col].values
            df[col] = df[col].fillna('Unknown')
    return df

def add_best_features(df):
    """Add our best engineered features."""
    df = df.copy()
    if all(c in df.columns for c in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR'] = df['HeartRate_Mean'] * df['RespRate_Mean']
    return df

# Prepare features
X_train_fe = add_best_features(X_train_base)
X_train_fe = add_extra_cats(X_train_fe, train_df_orig)
X_test_fe = add_best_features(X_test_base)
X_test_fe = add_extra_cats(X_test_fe, test_df_orig)

print(f"Features: {X_train_fe.shape[1]}")

# Preprocessor
numerical_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_fe, y_train_full, test_size=0.2, random_state=42
)

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# =============================================================================
# Train individual models
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING INDIVIDUAL MODELS")
print("=" * 70)

models = {}
predictions = {}

# 1. Poisson HistGB (our best)
print("\n1. Poisson HistGB...")
model_poisson = HistGradientBoostingRegressor(
    loss='poisson',
    max_iter=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
model_poisson.fit(X_train_t, y_train)
pred_poisson = model_poisson.predict(X_val_t)
rmse_poisson = np.sqrt(mean_squared_error(y_val, pred_poisson))
print(f"   RMSE: {rmse_poisson:.4f}")
models['poisson'] = model_poisson
predictions['poisson'] = pred_poisson

# 2. MSE HistGB
print("\n2. MSE HistGB...")
model_mse = HistGradientBoostingRegressor(
    loss='squared_error',
    max_iter=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
model_mse.fit(X_train_t, y_train)
pred_mse = model_mse.predict(X_val_t)
rmse_mse = np.sqrt(mean_squared_error(y_val, pred_mse))
print(f"   RMSE: {rmse_mse:.4f}")
models['mse'] = model_mse
predictions['mse'] = pred_mse

# 3. HistGB with different hyperparameters
print("\n3. HistGB (deeper, slower lr)...")
model_deep = HistGradientBoostingRegressor(
    loss='poisson',
    max_iter=500,
    learning_rate=0.03,
    max_depth=7,
    min_samples_leaf=15,
    random_state=42
)
model_deep.fit(X_train_t, y_train)
pred_deep = model_deep.predict(X_val_t)
rmse_deep = np.sqrt(mean_squared_error(y_val, pred_deep))
print(f"   RMSE: {rmse_deep:.4f}")
models['deep'] = model_deep
predictions['deep'] = pred_deep

# 4. Random Forest (different algorithm for diversity)
print("\n4. Random Forest...")
model_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=42
)
model_rf.fit(X_train_t, y_train)
pred_rf = model_rf.predict(X_val_t)
rmse_rf = np.sqrt(mean_squared_error(y_val, pred_rf))
print(f"   RMSE: {rmse_rf:.4f}")
models['rf'] = model_rf
predictions['rf'] = pred_rf

# =============================================================================
# Test ensemble combinations
# =============================================================================
print("\n" + "=" * 70)
print("TESTING ENSEMBLE COMBINATIONS")
print("=" * 70)

ensemble_results = []

# Simple average of all
print("\n1. Simple average (all 4)...")
pred_avg_all = (pred_poisson + pred_mse + pred_deep + pred_rf) / 4
rmse_avg_all = np.sqrt(mean_squared_error(y_val, pred_avg_all))
print(f"   RMSE: {rmse_avg_all:.4f}")
ensemble_results.append(('avg_all_4', rmse_avg_all, [0.25, 0.25, 0.25, 0.25]))

# Average of best 3 (no RF if it's bad)
print("\n2. Average (Poisson + MSE + Deep)...")
pred_avg_3 = (pred_poisson + pred_mse + pred_deep) / 3
rmse_avg_3 = np.sqrt(mean_squared_error(y_val, pred_avg_3))
print(f"   RMSE: {rmse_avg_3:.4f}")
ensemble_results.append(('avg_histgb_3', rmse_avg_3, [0.33, 0.33, 0.33, 0]))

# Average of 2 best HistGB
print("\n3. Average (Poisson + MSE)...")
pred_avg_2 = (pred_poisson + pred_mse) / 2
rmse_avg_2 = np.sqrt(mean_squared_error(y_val, pred_avg_2))
print(f"   RMSE: {rmse_avg_2:.4f}")
ensemble_results.append(('avg_poisson_mse', rmse_avg_2, [0.5, 0.5, 0, 0]))

# Weighted average (favor Poisson)
print("\n4. Weighted (0.5 Poisson + 0.3 MSE + 0.2 Deep)...")
pred_weighted = 0.5 * pred_poisson + 0.3 * pred_mse + 0.2 * pred_deep
rmse_weighted = np.sqrt(mean_squared_error(y_val, pred_weighted))
print(f"   RMSE: {rmse_weighted:.4f}")
ensemble_results.append(('weighted_favor_poisson', rmse_weighted, [0.5, 0.3, 0.2, 0]))

# Weighted with RF
print("\n5. Weighted (0.4 Poisson + 0.3 MSE + 0.2 Deep + 0.1 RF)...")
pred_weighted_rf = 0.4 * pred_poisson + 0.3 * pred_mse + 0.2 * pred_deep + 0.1 * pred_rf
rmse_weighted_rf = np.sqrt(mean_squared_error(y_val, pred_weighted_rf))
print(f"   RMSE: {rmse_weighted_rf:.4f}")
ensemble_results.append(('weighted_with_rf', rmse_weighted_rf, [0.4, 0.3, 0.2, 0.1]))

# =============================================================================
# Find best ensemble
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Add individual models for comparison
all_results = [
    ('poisson_single', rmse_poisson),
    ('mse_single', rmse_mse),
    ('deep_single', rmse_deep),
    ('rf_single', rmse_rf),
]
all_results.extend([(name, rmse) for name, rmse, _ in ensemble_results])

all_results_sorted = sorted(all_results, key=lambda x: x[1])

print("\nRanked by RMSE:")
for name, rmse in all_results_sorted:
    marker = " <-- BEST" if name == all_results_sorted[0][0] else ""
    print(f"  {name:25} RMSE={rmse:.4f}{marker}")

best_name = all_results_sorted[0][0]
best_rmse = all_results_sorted[0][1]

# =============================================================================
# Generate submission
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSION")
print("=" * 70)

# Retrain on full data
X_full_t = preprocessor.fit_transform(X_train_fe)
X_test_t = preprocessor.transform(X_test_fe)

# Train all models on full data
print("\nTraining models on full data...")

model_poisson_full = HistGradientBoostingRegressor(
    loss='poisson', max_iter=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=20, random_state=42
)
model_poisson_full.fit(X_full_t, y_train_full)
test_pred_poisson = model_poisson_full.predict(X_test_t)

model_mse_full = HistGradientBoostingRegressor(
    loss='squared_error', max_iter=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=20, random_state=42
)
model_mse_full.fit(X_full_t, y_train_full)
test_pred_mse = model_mse_full.predict(X_test_t)

model_deep_full = HistGradientBoostingRegressor(
    loss='poisson', max_iter=500, learning_rate=0.03,
    max_depth=7, min_samples_leaf=15, random_state=42
)
model_deep_full.fit(X_full_t, y_train_full)
test_pred_deep = model_deep_full.predict(X_test_t)

model_rf_full = RandomForestRegressor(
    n_estimators=200, max_depth=10, min_samples_leaf=20,
    n_jobs=-1, random_state=42
)
model_rf_full.fit(X_full_t, y_train_full)
test_pred_rf = model_rf_full.predict(X_test_t)

# Get best weights
best_weights = None
for name, rmse, weights in ensemble_results:
    if name == best_name:
        best_weights = weights
        break

if best_weights is None:
    # It's a single model
    if 'poisson' in best_name:
        test_pred = test_pred_poisson
    elif 'mse' in best_name:
        test_pred = test_pred_mse
    elif 'deep' in best_name:
        test_pred = test_pred_deep
    else:
        test_pred = test_pred_rf
else:
    # Ensemble
    test_pred = (best_weights[0] * test_pred_poisson +
                 best_weights[1] * test_pred_mse +
                 best_weights[2] * test_pred_deep +
                 best_weights[3] * test_pred_rf)

print(f"\nBest ensemble: {best_name}")
print(f"Validation RMSE: {best_rmse:.4f}")
print(f"Test pred range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")

# Save submission
test_ids = test_df_orig['icustay_id'].values
base_dir, _, _ = get_paths()
output_dir = base_dir / "submissions" / "regression"
output_dir.mkdir(parents=True, exist_ok=True)

sub = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred})
filename = output_dir / f"submission_los_ensemble.csv"
sub.to_csv(filename, index=False)
print(f"\nSaved: {filename.name}")

# Also save best single if ensemble didn't win
if 'single' not in best_name:
    # Save the simple poisson + mse average as backup
    test_pred_simple = (test_pred_poisson + test_pred_mse) / 2
    sub_simple = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred_simple})
    filename_simple = output_dir / "submission_los_ensemble_simple.csv"
    sub_simple.to_csv(filename_simple, index=False)
    print(f"Also saved: {filename_simple.name} (Poisson + MSE average)")

print("\n" + "=" * 70)
