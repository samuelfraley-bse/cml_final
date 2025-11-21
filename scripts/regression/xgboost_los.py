"""
XGBoost for LOS Regression
Often the best for tabular regression tasks
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

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("XGBOOST FOR LOS - BEAT 15.7 BENCHMARK")
print("="*70)

from los_prep import prepare_data, load_raw_data

OUTPUT_DIR = BASE_DIR / "submissions" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/5] Loading data with feature engineering...")
X, y, X_test = prepare_data(
    leak_cols=[
        "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS",
        "DOB", "DEATHTIME", "DISCHTIME", "DOD",
        "LOS", "HOSPITAL_EXPIRE_FLAG"
    ],
    apply_fe=True,
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

# Preprocess data
X_train_proc = preprocessor.fit_transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)
X_test_proc = preprocessor.transform(X_test)

# XGBoost with tuning
print("\n[4/5] Training XGBoost with hyperparameter search...")
print("  This will take 10-15 minutes...")

xgb_model = xgb.XGBRegressor(
    tree_method='hist',
    random_state=42,
    eval_metric='rmse',
)

param_distributions = {
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [5, 6, 7, 8, 9],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],  # L1 regularization
    'reg_lambda': [1, 1.5, 2],     # L2 regularization
}

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions,
    n_iter=40,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

random_search.fit(X_train_proc, y_train)

print(f"\n  Best CV score: {-random_search.best_score_:.4f} (RMSE)")
print(f"  Best params:")
for param, value in random_search.best_params_.items():
    print(f"    {param}: {value}")

# Validate
print("\n[5/5] Validating...")
best_model = random_search.best_estimator_
y_valid_pred = np.maximum(best_model.predict(X_valid_proc), 0)

rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
mae = mean_absolute_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)

print(f"  Validation RMSE: {rmse:.4f}")
print(f"  Validation MAE:  {mae:.4f}")
print(f"  Validation R²:   {r2:.4f}")

print(f"\n  vs Your GB:      21.3")
print(f"  vs NN Benchmark: 15.7")

# Predict
test_pred = np.maximum(best_model.predict(X_test_proc), 0)

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

output_file = OUTPUT_DIR / "submission_los_xgboost.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ XGBOOST SUBMISSION CREATED!")
print("="*70)
print(f"\nFile: {output_file}")
print(f"Validation RMSE: {rmse:.4f}")

if rmse < 21.3:
    improvement = 21.3 - rmse
    print(f"\n🎉 IMPROVED over GB! -{improvement:.2f} points")
    
    if rmse < 15.7:
        print(f"🏆 BEAT NN BENCHMARK! ({rmse:.4f} < 15.7)")
    elif rmse < 18:
        print(f"📈 Close to benchmark! Gap: {rmse - 15.7:.2f} points")
    else:
        print(f"📊 Better than GB but gap remains: {rmse - 15.7:.2f} points")
else:
    print(f"\n⚠️  Similar to GB ({rmse:.4f} vs 21.3)")

print("\nXGBoost often outperforms on leaderboard - worth trying!")
print("="*70)