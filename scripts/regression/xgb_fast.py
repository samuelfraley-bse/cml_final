"""
FAST XGBoost for LOS
Uses good default parameters with minimal tuning
Should run in 3-5 minutes
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "regression":
    BASE_DIR = SCRIPT_DIR.parents[1]
    sys.path.append(str(BASE_DIR / "src"))
    sys.path.append(str(SCRIPT_DIR))
else:
    BASE_DIR = Path.cwd()
    sys.path.append(str(BASE_DIR / "src"))

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: XGBoost not installed!")
    print("Install with: pip install xgboost")
    exit(1)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("FAST XGBOOST FOR LOS")
print("Target: Beat 15.7 benchmark | Runtime: 3-5 minutes")
print("="*70)

from los_prep import prepare_data, load_raw_data

OUTPUT_DIR = BASE_DIR / "submissions" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/4] Loading data with feature engineering...")
X, y, X_test = prepare_data(
    leak_cols=[
        "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS",
        "DOB", "DEATHTIME", "DISCHTIME", "DOD",
        "LOS", "HOSPITAL_EXPIRE_FLAG"
    ],
    apply_fe=True,
)

# Split
print("\n[2/4] Creating split...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
print("\n[3/4] Preprocessing...")
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols),
])

X_train_proc = preprocessor.fit_transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)
X_test_proc = preprocessor.transform(X_test)

# Train XGBoost with good defaults
print("\n[4/4] Training XGBoost...")
print("  Using optimized parameters (no search)")

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.01,
    reg_lambda=1.5,
    tree_method='hist',
    eval_metric='rmse',
    early_stopping_rounds=50,
    random_state=42,
)

# Train with validation for early stopping
xgb_model.fit(
    X_train_proc, 
    y_train,
    eval_set=[(X_valid_proc, y_valid)],
    verbose=False
)

print(f"  Training completed at iteration: {xgb_model.best_iteration}")

# Validate
y_valid_pred = np.maximum(xgb_model.predict(X_valid_proc), 0)
rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
mae = mean_absolute_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)

print(f"\nValidation metrics:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")

print(f"\n  vs Your GB:      21.3")
print(f"  vs NN Benchmark: 15.7")

# Feature importance
feature_importance = xgb_model.feature_importances_
print(f"\nTop 5 most important features:")
top_indices = np.argsort(feature_importance)[-5:][::-1]
for idx in top_indices:
    print(f"  Feature {idx}: importance = {feature_importance[idx]:.4f}")

# Predict
test_pred = np.maximum(xgb_model.predict(X_test_proc), 0)

print(f"\nTest predictions:")
print(f"  Min:    {test_pred.min():.2f} days")
print(f"  Max:    {test_pred.max():.2f} days")
print(f"  Mean:   {test_pred.mean():.2f} days")
print(f"  Median: {np.median(test_pred):.2f} days")

# Get IDs
try:
    _, test_df = load_raw_data()
    id_col = [c for c in test_df.columns if c.lower() == 'icustay_id'][0]
    icustay_ids = test_df[id_col]
except:
    icustay_ids = range(len(test_pred))

submission = pd.DataFrame({
    'icustay_id': icustay_ids,
    'LOS': test_pred
})

output_file = OUTPUT_DIR / "submission_los_xgb_fast.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ FAST XGBOOST SUBMISSION CREATED!")
print("="*70)
print(f"\nFile: {output_file}")
print(f"Validation RMSE: {rmse:.4f}")

if rmse < 21.3:
    improvement = 21.3 - rmse
    print(f"\n🎉 IMPROVED over GB! -{improvement:.2f} points")
    
    if rmse < 15.7:
        print(f"🏆 BEAT NN BENCHMARK! ({rmse:.4f} < 15.7)")
        print("   Upload this immediately!")
    elif rmse < 18:
        print(f"📈 Close to benchmark! Gap: {rmse - 15.7:.2f} points")
        print("   Very promising - definitely submit")
    else:
        print(f"📊 Better than GB, gap: {rmse - 15.7:.2f} points")
        print("   Worth trying")
else:
    print(f"\n⚠️  Similar to GB ({rmse:.4f} vs 21.3)")
    print("   XGBoost often performs better on test though!")

print("\n💡 XGBoost is often the best for tabular regression")
print("   Even if validation is similar, try on leaderboard!")
print("="*70)