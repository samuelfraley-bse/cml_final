"""
Fine-tune GB for LOS - Beat 15.7 Benchmark
Current: 21.3 → Target: < 15.7

Strategy:
1. Grid search hyperparameters
2. Try with feature engineering
3. More trees, deeper trees
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("GB TUNING FOR LOS - TARGET: BEAT 15.7")
print("Current best: 21.3 | Gap: -5.6 points")
print("="*70)

from los_prep import prepare_data, load_raw_data

OUTPUT_DIR = BASE_DIR / "submissions" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load WITH feature engineering this time
print("\n[1/5] Loading data WITH feature engineering...")
X, y, X_test = prepare_data(
    leak_cols=[
        "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS",
        "DOB", "DEATHTIME", "DISCHTIME", "DOD",
        "LOS", "HOSPITAL_EXPIRE_FLAG"
    ],
    apply_fe=True,  # Try FE for LOS
)

# Split
print("\n[2/5] Creating split...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
print("\n[3/5] Preprocessing...")
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

# Grid search with MORE aggressive parameters
print("\n[4/5] Grid search (this will take 10-15 minutes)...")
print("  Testing more trees, deeper trees, different learning rates")

gb = GradientBoostingRegressor(random_state=42)

param_grid = {
    'regressor__n_estimators': [300, 400, 500],
    'regressor__learning_rate': [0.03, 0.05, 0.07, 0.1],
    'regressor__max_depth': [5, 6, 7, 8],
    'regressor__min_samples_split': [8, 10, 12],
    'regressor__min_samples_leaf': [3, 5, 7],
    'regressor__subsample': [0.8, 0.9],
    'regressor__max_features': ['sqrt', 0.7, 0.8],
}

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', gb),
])

# Use negative MSE (sklearn convention)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='neg_mean_squared_error',  # Will be negative
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Best params
print(f"\n  Best CV score: {-grid_search.best_score_:.4f} (RMSE)")
print(f"  Best params:")
for param, value in grid_search.best_params_.items():
    print(f"    {param}: {value}")

# Validate
print("\n[5/5] Validating on holdout...")
best_model = grid_search.best_estimator_
y_valid_pred = np.maximum(best_model.predict(X_valid), 0)

rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
mae = mean_absolute_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)

print(f"  Validation RMSE: {rmse:.4f}")
print(f"  Validation MAE:  {mae:.4f}")
print(f"  Validation R²:   {r2:.4f}")
print(f"\n  vs Current best: 21.3")
print(f"  vs Benchmark:    15.7")

# Predict
test_pred = np.maximum(best_model.predict(X_test), 0)

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

output_file = OUTPUT_DIR / "submission_los_gb_tuned.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ SUCCESS!")
print("="*70)
print(f"\nSubmission: {output_file}")
print(f"Validation RMSE: {rmse:.4f}")

if rmse < 21.3:
    improvement = 21.3 - rmse
    print(f"\n🎉 IMPROVED! -{improvement:.2f} points")
    if rmse < 15.7:
        print(f"🏆 BEAT BENCHMARK! ({rmse:.4f} < 15.7)")
    else:
        gap = rmse - 15.7
        print(f"   Gap to benchmark: {gap:.2f} points")
else:
    print(f"\n⚠️  No improvement ({rmse:.4f} vs 21.3)")

print("="*70)