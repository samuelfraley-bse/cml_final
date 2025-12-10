"""
Generate Multiple LOS Regression Submissions
Creates submissions for RF, GB, ET, and Ensemble models

Run from scripts/regression/:
    python generate_los_submissions.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "regression":
    BASE_DIR = SCRIPT_DIR.parents[1]
    sys.path.append(str(BASE_DIR / "src"))
    sys.path.append(str(SCRIPT_DIR))
else:
    BASE_DIR = Path.cwd()
    sys.path.append(str(BASE_DIR / "src"))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("LOS REGRESSION - MULTIPLE MODEL GENERATOR")
print("="*70)

# Import data prep
try:
    from los_prep import prepare_data, load_raw_data
except ImportError:
    print("\nERROR: Cannot import los_prep!")
    print("Make sure los_prep.py is in scripts/regression/ or src/")
    exit(1)

# Paths
OUTPUT_DIR = BASE_DIR / "submissions" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/6] Loading data...")
X, y, X_test = prepare_data(
    leak_cols=[
        "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS",
        "DOB", "DEATHTIME", "DISCHTIME", "DOD",
        "LOS", "HOSPITAL_EXPIRE_FLAG"
    ],
    apply_fe=False,
)

# Split
print("\n[2/6] Creating split...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
print("\n[3/6] Setting up preprocessing...")
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

# Store models and predictions
models = {}
predictions = {}
validation_scores = {}

# Model 1: Random Forest
print("\n[4/6] Training Random Forest Regressor...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=7,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
)

rf_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', rf),
])

rf_pipe.fit(X_train, y_train)

y_valid_pred_rf = np.maximum(rf_pipe.predict(X_valid), 0)
rf_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred_rf))
rf_mae = mean_absolute_error(y_valid, y_valid_pred_rf)
rf_r2 = r2_score(y_valid, y_valid_pred_rf)

print(f"  RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")

models['rf'] = rf_pipe
predictions['rf'] = np.maximum(rf_pipe.predict(X_test), 0)
validation_scores['rf'] = {'rmse': rf_rmse, 'mae': rf_mae, 'r2': rf_r2}

# Model 2: Gradient Boosting
print("\n[5/6] Training Gradient Boosting Regressor...")
gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42,
)

gb_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', gb),
])

gb_pipe.fit(X_train, y_train)

y_valid_pred_gb = np.maximum(gb_pipe.predict(X_valid), 0)
gb_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred_gb))
gb_mae = mean_absolute_error(y_valid, y_valid_pred_gb)
gb_r2 = r2_score(y_valid, y_valid_pred_gb)

print(f"  RMSE: {gb_rmse:.4f}, MAE: {gb_mae:.4f}, R²: {gb_r2:.4f}")

models['gb'] = gb_pipe
predictions['gb'] = np.maximum(gb_pipe.predict(X_test), 0)
validation_scores['gb'] = {'rmse': gb_rmse, 'mae': gb_mae, 'r2': gb_r2}

# Model 3: Extra Trees
print("\n[6/6] Training Extra Trees Regressor...")
et = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=7,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
)

et_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', et),
])

et_pipe.fit(X_train, y_train)

y_valid_pred_et = np.maximum(et_pipe.predict(X_valid), 0)
et_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred_et))
et_mae = mean_absolute_error(y_valid, y_valid_pred_et)
et_r2 = r2_score(y_valid, y_valid_pred_et)

print(f"  RMSE: {et_rmse:.4f}, MAE: {et_mae:.4f}, R²: {et_r2:.4f}")

models['et'] = et_pipe
predictions['et'] = np.maximum(et_pipe.predict(X_test), 0)
validation_scores['et'] = {'rmse': et_rmse, 'mae': et_mae, 'r2': et_r2}

# Ensemble (weighted by R²)
print("\nCreating weighted ensemble...")
total_r2 = rf_r2 + gb_r2 + et_r2
if total_r2 > 0:
    weights = {
        'rf': rf_r2 / total_r2,
        'gb': gb_r2 / total_r2,
        'et': et_r2 / total_r2,
    }
    
    print(f"  Weights: RF={weights['rf']:.3f}, GB={weights['gb']:.3f}, ET={weights['et']:.3f}")
    
    ensemble_pred = (
        weights['rf'] * predictions['rf'] +
        weights['gb'] * predictions['gb'] +
        weights['et'] * predictions['et']
    )
    
    # Validate ensemble
    ensemble_valid_pred = (
        weights['rf'] * y_valid_pred_rf +
        weights['gb'] * y_valid_pred_gb +
        weights['et'] * y_valid_pred_et
    )
    
    ens_rmse = np.sqrt(mean_squared_error(y_valid, ensemble_valid_pred))
    ens_mae = mean_absolute_error(y_valid, ensemble_valid_pred)
    ens_r2 = r2_score(y_valid, ensemble_valid_pred)
    
    print(f"  Ensemble - RMSE: {ens_rmse:.4f}, MAE: {ens_mae:.4f}, R²: {ens_r2:.4f}")
    
    predictions['ensemble'] = ensemble_pred
    validation_scores['ensemble'] = {'rmse': ens_rmse, 'mae': ens_mae, 'r2': ens_r2}

# Get test IDs
try:
    _, test_df = load_raw_data()
    id_col = [c for c in test_df.columns if c.lower() == 'icustay_id'][0]
    icustay_ids = test_df[id_col]
except Exception as e:
    print(f"\n⚠️  WARNING: Could not load IDs: {e}")
    icustay_ids = range(len(X_test))

# Save all submissions
print("\n" + "="*70)
print("SAVING SUBMISSIONS")
print("="*70)

saved_files = []
for name, pred in predictions.items():
    submission = pd.DataFrame({
        'icustay_id': icustay_ids,
        'LOS': pred
    })
    
    filepath = OUTPUT_DIR / f"submission_los_{name}.csv"
    submission.to_csv(filepath, index=False)
    
    scores = validation_scores[name]
    print(f"\n{name.upper()}:")
    print(f"  File: {filepath.name}")
    print(f"  Validation - RMSE: {scores['rmse']:.4f}, MAE: {scores['mae']:.4f}, R²: {scores['r2']:.4f}")
    print(f"  Predictions - min: {pred.min():.2f}, max: {pred.max():.2f}, mean: {pred.mean():.2f}")
    
    saved_files.append(filepath)

# Summary
print("\n" + "="*70)
print("SUMMARY - VALIDATION SCORES")
print("="*70)

# Sort by RMSE (lower is better)
sorted_models = sorted(validation_scores.items(), key=lambda x: x[1]['rmse'])

print("\nRanked by RMSE (lower = better):")
for i, (name, scores) in enumerate(sorted_models, 1):
    print(f"{i}. {name.upper()}: RMSE={scores['rmse']:.4f}, MAE={scores['mae']:.4f}, R²={scores['r2']:.4f}")

best_model = sorted_models[0][0]
print(f"\n🏆 Best model: {best_model.upper()}")
print(f"   Submit: submission_los_{best_model}.csv first")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print(f"\n1. Upload: submission_los_{best_model}.csv")
print("2. If good, try: submission_los_ensemble.csv")
print("3. Experiment with others based on leaderboard feedback")
print("\n" + "="*70)
