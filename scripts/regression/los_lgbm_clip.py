"""
LightGBM + Clipping - Try to break 20

Test LightGBM (often better than sklearn) and clip extreme predictions.

Run: python scripts/regression/los_lgbm_clip.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed. Run: pip install lightgbm")

from los_prep import prepare_data, get_paths, load_raw_data

print("=" * 70)
print("LIGHTGBM + CLIPPING TEST")
print("=" * 70)

# =============================================================================
# Load data
# =============================================================================
print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# =============================================================================
# Feature engineering (just the best features, no extra cats)
# =============================================================================

def add_best_features(df):
    """Add our best engineered features."""
    df = df.copy()
    if all(c in df.columns for c in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR'] = df['HeartRate_Mean'] * df['RespRate_Mean']
    return df

X_train_fe = add_best_features(X_train_base)
X_test_fe = add_best_features(X_test_base)

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
# Test models
# =============================================================================
print("\n" + "=" * 70)
print("TESTING MODELS")
print("=" * 70)

results = []

# 1. Baseline Poisson HistGB (no clipping)
print("\n1. Poisson HistGB (baseline)...")
model_base = HistGradientBoostingRegressor(
    loss='poisson', max_iter=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=20, random_state=42
)
model_base.fit(X_train_t, y_train)
pred_base = model_base.predict(X_val_t)
rmse_base = np.sqrt(mean_squared_error(y_val, pred_base))
print(f"   RMSE: {rmse_base:.4f}, Range: [{pred_base.min():.2f}, {pred_base.max():.2f}]")
results.append(('poisson_baseline', rmse_base, pred_base, model_base))

# 2. Poisson HistGB with clipping
print("\n2. Poisson HistGB (clipped 0.5-30)...")
pred_clipped = np.clip(pred_base, 0.5, 30)
rmse_clipped = np.sqrt(mean_squared_error(y_val, pred_clipped))
print(f"   RMSE: {rmse_clipped:.4f}, Range: [{pred_clipped.min():.2f}, {pred_clipped.max():.2f}]")
results.append(('poisson_clip_30', rmse_clipped, pred_clipped, model_base))

# 3. Different clip ranges
for clip_max in [25, 35, 40]:
    pred_clip = np.clip(pred_base, 0.5, clip_max)
    rmse_clip = np.sqrt(mean_squared_error(y_val, pred_clip))
    print(f"   Clip 0.5-{clip_max}: RMSE={rmse_clip:.4f}")
    results.append((f'poisson_clip_{clip_max}', rmse_clip, pred_clip, model_base))

# 4. LightGBM
if HAS_LGBM:
    print("\n3. LightGBM (Poisson)...")

    # LightGBM can handle categoricals directly, but we'll use encoded for consistency
    lgb_train = lgb.Dataset(X_train_t, label=y_train)
    lgb_val = lgb.Dataset(X_val_t, label=y_val, reference=lgb_train)

    params = {
        'objective': 'poisson',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    model_lgb = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    pred_lgb = model_lgb.predict(X_val_t)
    rmse_lgb = np.sqrt(mean_squared_error(y_val, pred_lgb))
    print(f"   RMSE: {rmse_lgb:.4f}, Range: [{pred_lgb.min():.2f}, {pred_lgb.max():.2f}]")
    results.append(('lgbm_poisson', rmse_lgb, pred_lgb, model_lgb))

    # LightGBM with clipping
    pred_lgb_clip = np.clip(pred_lgb, 0.5, 30)
    rmse_lgb_clip = np.sqrt(mean_squared_error(y_val, pred_lgb_clip))
    print(f"   LGBM clipped: RMSE={rmse_lgb_clip:.4f}")
    results.append(('lgbm_clip_30', rmse_lgb_clip, pred_lgb_clip, model_lgb))

    # LightGBM MSE objective
    print("\n4. LightGBM (MSE)...")
    params_mse = params.copy()
    params_mse['objective'] = 'regression'

    model_lgb_mse = lgb.train(
        params_mse,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    pred_lgb_mse = model_lgb_mse.predict(X_val_t)
    rmse_lgb_mse = np.sqrt(mean_squared_error(y_val, pred_lgb_mse))
    print(f"   RMSE: {rmse_lgb_mse:.4f}, Range: [{pred_lgb_mse.min():.2f}, {pred_lgb_mse.max():.2f}]")
    results.append(('lgbm_mse', rmse_lgb_mse, pred_lgb_mse, model_lgb_mse))

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results_sorted = sorted(results, key=lambda x: x[1])

print("\nRanked by RMSE:")
for name, rmse, _, _ in results_sorted:
    print(f"  {name:25} RMSE={rmse:.4f}")

best_name, best_rmse, best_pred, best_model = results_sorted[0]
print(f"\nBest: {best_name} (RMSE={best_rmse:.4f})")

# =============================================================================
# Generate submissions
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

# 1. Best HistGB Poisson (with optimal clip if that won)
print("\n1. HistGB Poisson...")
model_full = HistGradientBoostingRegressor(
    loss='poisson', max_iter=300, learning_rate=0.05,
    max_depth=5, min_samples_leaf=20, random_state=42
)
model_full.fit(X_full_t, y_train_full)
test_pred_histgb = model_full.predict(X_test_t)

# Find best clip value
best_clip = None
for name, rmse, _, _ in results_sorted:
    if 'clip' in name and 'poisson' in name and 'lgbm' not in name:
        best_clip = int(name.split('_')[-1])
        break

if best_clip and 'clip' in best_name and 'lgbm' not in best_name:
    test_pred_histgb = np.clip(test_pred_histgb, 0.5, best_clip)
    print(f"   Clipped to [0.5, {best_clip}]")

print(f"   Range: [{test_pred_histgb.min():.2f}, {test_pred_histgb.max():.2f}]")

sub = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred_histgb})
filename = output_dir / "submission_los_poisson_tuned.csv"
sub.to_csv(filename, index=False)
print(f"   Saved: {filename.name}")

# 2. LightGBM if available and good
if HAS_LGBM:
    print("\n2. LightGBM Poisson...")

    lgb_full = lgb.Dataset(X_full_t, label=y_train_full)
    model_lgb_full = lgb.train(
        params,
        lgb_full,
        num_boost_round=300
    )
    test_pred_lgb = model_lgb_full.predict(X_test_t)

    # Clip if it helped
    if 'lgbm_clip' in best_name:
        test_pred_lgb = np.clip(test_pred_lgb, 0.5, 30)
        print("   Clipped to [0.5, 30]")

    print(f"   Range: [{test_pred_lgb.min():.2f}, {test_pred_lgb.max():.2f}]")

    sub_lgb = pd.DataFrame({'icustay_id': test_ids, 'LOS': test_pred_lgb})
    filename_lgb = output_dir / "submission_los_lgbm.csv"
    sub_lgb.to_csv(filename_lgb, index=False)
    print(f"   Saved: {filename_lgb.name}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print(f"""
Best validation: {best_name} (RMSE={best_rmse:.4f})

Submit in this order:
1. submission_los_lgbm.csv (if LGBM was best)
2. submission_los_poisson_tuned.csv

Current best on Kaggle: 20.28
Target: < 20.0
""")
print("=" * 70)
