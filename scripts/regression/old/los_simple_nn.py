"""
Simple Neural Network for LOS - Quick test to see if NN helps

Also tests different metrics to understand what 15.7 might mean.

Run: python scripts/regression/los_simple_nn.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from los_prep import prepare_data, get_paths, load_raw_data

print("=" * 70)
print("SIMPLE NEURAL NETWORK + METRIC ANALYSIS")
print("=" * 70)

# =============================================================================
# Load data
# =============================================================================
print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# Add just the interaction features (best from ablation)
def add_interactions(df):
    df = df.copy()
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR_interaction'] = df['HeartRate_Mean'] * df['RespRate_Mean']
    if 'RespRate_Mean' in df.columns and 'TempC_Mean' in df.columns:
        df['RR_Temp_interaction'] = df['RespRate_Mean'] * df['TempC_Mean']
    return df

X_train_fe = add_interactions(X_train_base)
X_test_fe = add_interactions(X_test_base)

print(f"Features: {X_train_fe.shape[1]}")

# Prepare preprocessor
numerical_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

# NN needs scaling
nn_preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# HistGB preprocessor (no scaling needed)
histgb_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_fe, y_train_full, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# =============================================================================
# Train HistGB (baseline)
# =============================================================================
print("\n" + "=" * 70)
print("1. HistGradientBoosting (baseline)")
print("=" * 70)

X_train_hgb = histgb_preprocessor.fit_transform(X_train)
X_val_hgb = histgb_preprocessor.transform(X_val)

hgb_model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=5,
    min_samples_leaf=20, random_state=42
)
hgb_model.fit(X_train_hgb, y_train)
pred_hgb = hgb_model.predict(X_val_hgb)

rmse_hgb = np.sqrt(mean_squared_error(y_val, pred_hgb))
mae_hgb = mean_absolute_error(y_val, pred_hgb)

print(f"RMSE: {rmse_hgb:.4f}")
print(f"MAE:  {mae_hgb:.4f}")
print(f"Range: [{pred_hgb.min():.2f}, {pred_hgb.max():.2f}]")

# =============================================================================
# Train Simple Neural Network
# =============================================================================
print("\n" + "=" * 70)
print("2. Simple Neural Network")
print("=" * 70)

X_train_nn = nn_preprocessor.fit_transform(X_train)
X_val_nn = nn_preprocessor.transform(X_val)

# Simple architecture - 2 hidden layers
nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=False
)

print("Training NN (this may take a moment)...")
nn_model.fit(X_train_nn, y_train)
pred_nn = nn_model.predict(X_val_nn)

rmse_nn = np.sqrt(mean_squared_error(y_val, pred_nn))
mae_nn = mean_absolute_error(y_val, pred_nn)

print(f"RMSE: {rmse_nn:.4f}")
print(f"MAE:  {mae_nn:.4f}")
print(f"Range: [{pred_nn.min():.2f}, {pred_nn.max():.2f}]")
print(f"Iterations: {nn_model.n_iter_}")

# =============================================================================
# Train Larger NN
# =============================================================================
print("\n" + "=" * 70)
print("3. Larger Neural Network")
print("=" * 70)

nn_large = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
    verbose=False
)

print("Training larger NN...")
nn_large.fit(X_train_nn, y_train)
pred_nn_large = nn_large.predict(X_val_nn)

rmse_nn_large = np.sqrt(mean_squared_error(y_val, pred_nn_large))
mae_nn_large = mean_absolute_error(y_val, pred_nn_large)

print(f"RMSE: {rmse_nn_large:.4f}")
print(f"MAE:  {mae_nn_large:.4f}")
print(f"Range: [{pred_nn_large.min():.2f}, {pred_nn_large.max():.2f}]")
print(f"Iterations: {nn_large.n_iter_}")

# =============================================================================
# METRIC ANALYSIS - What could 15.7 mean?
# =============================================================================
print("\n" + "=" * 70)
print("METRIC ANALYSIS - Understanding 15.7")
print("=" * 70)

# Use the best model's predictions
best_pred = pred_hgb if rmse_hgb <= rmse_nn else pred_nn
best_name = "HistGB" if rmse_hgb <= rmse_nn else "NN"

# Calculate various metrics
mse = mean_squared_error(y_val, best_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, best_pred)
mape = np.mean(np.abs((y_val - best_pred) / y_val.clip(lower=0.1))) * 100
rmsle = np.sqrt(np.mean((np.log1p(best_pred.clip(lower=0)) - np.log1p(y_val)) ** 2))

print(f"\nUsing {best_name} predictions:")
print(f"  MSE:   {mse:.2f}")
print(f"  RMSE:  {rmse:.4f}")
print(f"  MAE:   {mae:.4f}")
print(f"  MAPE:  {mape:.2f}%")
print(f"  RMSLE: {rmsle:.4f}")

print("\n" + "-" * 50)
print("IF Kaggle score is one of these metrics:")
print("-" * 50)

print(f"\n  If MSE:   {mse:.2f} (our val) → need to get to ~{15.7**2:.0f}?")
print(f"  If RMSE:  {rmse:.4f} (our val) → need to get to 15.7")
print(f"  If MAE:   {mae:.4f} (our val)")
print(f"  If RMSLE: {rmsle:.4f} (our val)")

# Check what Kaggle might actually be
print("\n" + "-" * 50)
print("Kaggle score interpretation:")
print("-" * 50)

print(f"""
Our validation RMSE: {rmse:.4f}
Our Kaggle score: ~20

If Kaggle = MSE:
  validation MSE = {mse:.2f}
  This matches! (validation ~22, Kaggle ~20)
  Target 15.7 would mean MSE ≈ 15.7

If Kaggle = RMSE:
  validation RMSE = {rmse:.4f}
  But Kaggle = ~20, so there's a big gap
  This doesn't match unless test set is different

CONCLUSION: Kaggle metric is likely MSE, not RMSE!
  Our best MSE: {mse:.2f}
  Target MSE: 15.7
  Gap: {mse - 15.7:.2f} points
""")

# =============================================================================
# Generate submissions
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSIONS")
print("=" * 70)

# Train on full data
X_full_nn = nn_preprocessor.fit_transform(X_train_fe)
X_test_nn = nn_preprocessor.transform(X_test_fe)

X_full_hgb = histgb_preprocessor.fit_transform(X_train_fe)
X_test_hgb = histgb_preprocessor.transform(X_test_fe)

test_ids = test_df_orig['icustay_id'].values
base_dir, _, _ = get_paths()
output_dir = base_dir / "submissions" / "regression"
output_dir.mkdir(parents=True, exist_ok=True)

# HistGB submission
hgb_full = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=5,
    min_samples_leaf=20, random_state=42
)
hgb_full.fit(X_full_hgb, y_train_full)
test_pred_hgb = hgb_full.predict(X_test_hgb)

sub_hgb = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred_hgb})
file_hgb = output_dir / "submission_los_histgb_interactions.csv"
sub_hgb.to_csv(file_hgb, index=False)
print(f"\nHistGB: {file_hgb.name}")
print(f"  Range: [{test_pred_hgb.min():.2f}, {test_pred_hgb.max():.2f}]")

# NN submission
nn_full = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=False
)
nn_full.fit(X_full_nn, y_train_full)
test_pred_nn = nn_full.predict(X_test_nn)

sub_nn = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred_nn})
file_nn = output_dir / "submission_los_nn_simple.csv"
sub_nn.to_csv(file_nn, index=False)
print(f"\nNN: {file_nn.name}")
print(f"  Range: [{test_pred_nn.min():.2f}, {test_pred_nn.max():.2f}]")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
VALIDATION RESULTS:
  HistGB:    RMSE={rmse_hgb:.4f}, MAE={mae_hgb:.4f}
  NN Small:  RMSE={rmse_nn:.4f}, MAE={mae_nn:.4f}
  NN Large:  RMSE={rmse_nn_large:.4f}, MAE={mae_nn_large:.4f}

METRIC INSIGHT:
  Kaggle score is likely MSE (not RMSE)
  Our validation MSE: {mse:.2f}
  Target: 15.7
  Gap: {mse - 15.7:.2f}

FILES GENERATED:
  1. {file_hgb.name}
  2. {file_nn.name}
""")

print("=" * 70)
