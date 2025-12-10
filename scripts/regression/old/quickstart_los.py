"""
QUICKSTART: LOS (Length of Stay) Regression Submission
Predicts how many days patients will stay in ICU

Run from scripts/regression/:
    python quickstart_los.py

Or from project root:
    python scripts/regression/quickstart_los.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "regression":
    BASE_DIR = SCRIPT_DIR.parents[1]  # cml_final
    sys.path.append(str(BASE_DIR / "src"))
    sys.path.append(str(SCRIPT_DIR))
else:
    BASE_DIR = Path.cwd()
    sys.path.append(str(BASE_DIR / "src"))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*60)
print("QUICKSTART LOS REGRESSION SUBMISSION")
print("="*60)

# Import data prep
try:
    from los_prep import prepare_data
except ImportError:
    print("\nERROR: Cannot import los_prep module!")
    print("Make sure los_prep.py is in:")
    print("  - scripts/regression/ folder, OR")
    print("  - src/ folder")
    exit(1)

# Paths
OUTPUT_DIR = BASE_DIR / "submissions" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/5] Loading LOS data...")
try:
    X, y, X_test = prepare_data(
        leak_cols=[
            "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS", 
            "DOB", "DEATHTIME", "DISCHTIME", "DOD", 
            "LOS", "HOSPITAL_EXPIRE_FLAG"
        ],
        apply_fe=False,  # Start without FE
    )
except Exception as e:
    print(f"\nERROR loading data: {e}")
    print("\nMake sure you're running from the correct directory!")
    exit(1)

print(f"\n  Features: {X.shape[1]}")
print(f"  LOS range: [{y.min():.2f}, {y.max():.2f}] days")
print(f"  LOS mean: {y.mean():.2f} days")

# Train/valid split
print("\n[2/5] Creating train/validation split...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
print("\n[3/5] Setting up preprocessing...")
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

# Train Gradient Boosting Regressor
print("\n[4/5] Training Gradient Boosting Regressor...")
print("  (Using parameters similar to classification task)")

gb_reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42,
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', gb_reg),
])

pipeline.fit(X_train, y_train)

# Validate
print("\n[5/5] Validating...")
y_valid_pred = pipeline.predict(X_valid)

# Clip negative predictions (LOS can't be negative)
y_valid_pred_clipped = np.maximum(y_valid_pred, 0)

rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred_clipped))
mae = mean_absolute_error(y_valid, y_valid_pred_clipped)
r2 = r2_score(y_valid, y_valid_pred_clipped)

print(f"  Validation RMSE: {rmse:.4f} days")
print(f"  Validation MAE:  {mae:.4f} days")
print(f"  Validation R²:   {r2:.4f}")

# Predict on test
print("\nGenerating predictions for test set...")
test_pred = pipeline.predict(X_test)

# Clip negative predictions
test_pred_clipped = np.maximum(test_pred, 0)

# Check if predictions were scaled (they shouldn't be with StandardScaler on features only)
print(f"\nPrediction statistics:")
print(f"  Min:    {test_pred_clipped.min():.2f} days")
print(f"  Max:    {test_pred_clipped.max():.2f} days")
print(f"  Mean:   {test_pred_clipped.mean():.2f} days")
print(f"  Median: {np.median(test_pred_clipped):.2f} days")

# Compare with training distribution
print(f"\nTraining LOS distribution:")
print(f"  Min:    {y.min():.2f} days")
print(f"  Max:    {y.max():.2f} days")
print(f"  Mean:   {y.mean():.2f} days")
print(f"  Median: {y.median():.2f} days")

if test_pred_clipped.mean() < 0.1 or test_pred_clipped.mean() > 100:
    print("\n⚠️  WARNING: Predictions seem unusual!")
    print("   Check if target needs de-standardization")

# Get test IDs
try:
    from los_prep import load_raw_data
    _, test_df = load_raw_data()
    
    # Find ID column
    id_col = None
    for col in test_df.columns:
        if col.lower() == 'icustay_id':
            id_col = col
            break
    
    if id_col is None:
        print("\n⚠️  WARNING: icustay_id not found in test data!")
        print("   Creating submission without IDs...")
        icustay_ids = range(len(test_pred_clipped))
    else:
        icustay_ids = test_df[id_col]
        
except Exception as e:
    print(f"\n⚠️  WARNING: Could not load test IDs: {e}")
    icustay_ids = range(len(test_pred_clipped))

# Create submission
submission = pd.DataFrame({
    'icustay_id': icustay_ids,
    'LOS': test_pred_clipped
})

output_file = OUTPUT_DIR / "submission_los_gb.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*60)
print("✓ SUCCESS!")
print("="*60)
print(f"\nSubmission saved: {output_file}")
print(f"\nValidation metrics:")
print(f"  RMSE: {rmse:.4f} days")
print(f"  MAE:  {mae:.4f} days")
print(f"  R²:   {r2:.4f}")
print("\n📤 Upload to Kaggle and see how it scores!")
print("="*60)
