"""
QUICKSTART: Generate Best Submission
This is the fastest path to your best Kaggle submission

Run from your project root:
    python quickstart_submission.py

This will create ONE submission file using your best model (RF without FE)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

print("="*60)
print("QUICKSTART KAGGLE SUBMISSION")
print("="*60)

# Paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
OUTPUT_DIR = BASE_DIR / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/5] Loading data...")
train_df = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_df = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")
print(f"  Train: {train_df.shape}")
print(f"  Test: {test_df.shape}")

# Find target and prepare data
print("\n[2/5] Preparing features...")
target_col = [c for c in train_df.columns if c.lower() == "hospital_expire_flag"][0]
y = train_df[target_col].copy()

# Drop IDs and leakage columns
drop_cols = ["icustay_id", "subject_id", "hadm_id", "ADMITTIME", 
              "DOB", "DEATHTIME", 
             "DISCHTIME", "DOD", "LOS", "HOSPITAL_EXPIRE_FLAG"]

# Case-insensitive drop
cols_to_drop_train = [c for c in train_df.columns if c.lower() in [x.lower() for x in drop_cols]]
cols_to_drop_test = [c for c in test_df.columns if c.lower() in [x.lower() for x in drop_cols] and c.lower() != "hospital_expire_flag"]

X = train_df.drop(columns=cols_to_drop_train)
X_test = test_df.drop(columns=cols_to_drop_test)

# Clean BP outliers
for col, threshold in [("SysBP_Min", 40), ("DiasBP_Min", 10), ("MeanBP_Min", 30)]:
    if col in X.columns:
        X.loc[X[col] < threshold, col] = np.nan
    if col in X_test.columns:
        X_test.loc[X_test[col] < threshold, col] = np.nan

print(f"  Features: {X.shape[1]}")
print(f"  Positive rate: {y.mean():.3f}")

# Train/valid split
print("\n[3/5] Creating split...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing
print("\n[4/5] Training model...")
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

# Train Random Forest (best model from your tests)
rf = RandomForestClassifier(
    n_estimators=203,
    max_depth=None,
    min_samples_split=7,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', rf),
])

pipeline.fit(X_train, y_train)

# Validate
y_valid_proba = pipeline.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, y_valid_proba)
print(f"  Validation AUC: {valid_auc:.4f}")

# Predict on test
print("\n[5/5] Generating submission...")
test_proba = pipeline.predict_proba(X_test)[:, 1]

# Get icustay_id from test data
id_col = [c for c in test_df.columns if c.lower() == 'icustay_id'][0]
icustay_ids = test_df[id_col]

submission = pd.DataFrame({
    'icustay_id': icustay_ids,
    'HOSPITAL_EXPIRE_FLAG': test_proba
})

output_file = OUTPUT_DIR / "submission_best_v2.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*60)
print("✓ SUCCESS!")
print("="*60)
print(f"\nSubmission saved: {output_file}")
print(f"\nPrediction stats:")
print(f"  Min:    {test_proba.min():.4f}")
print(f"  Max:    {test_proba.max():.4f}")
print(f"  Mean:   {test_proba.mean():.4f}")
print(f"  Median: {np.median(test_proba):.4f}")
print(f"\nValidation AUC: {valid_auc:.4f}")
print("\nUpload to Kaggle and see how it scores!")
print("="*60)