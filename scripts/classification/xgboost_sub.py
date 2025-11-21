"""
XGBoost Submission Generator
XGBoost often outperforms RF/GB on medical/tabular data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: XGBoost not installed!")
    print("Install with: pip install xgboost")
    exit(1)

print("="*60)
print("XGBOOST SUBMISSION GENERATOR")
print("="*60)

# Paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
OUTPUT_DIR = BASE_DIR / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/6] Loading data...")
train_df = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_df = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

# Get target
target_col = [c for c in train_df.columns if c.lower() == "hospital_expire_flag"][0]
y = train_df[target_col].copy()

# Drop columns
drop_cols = ["icustay_id", "subject_id", "hadm_id", "ADMITTIME", 
             "ICD9_diagnosis", "DIAGNOSIS", "DOB", "DEATHTIME", 
             "DISCHTIME", "DOD", "LOS", "HOSPITAL_EXPIRE_FLAG"]

cols_to_drop_train = [c for c in train_df.columns if c.lower() in [x.lower() for x in drop_cols]]
cols_to_drop_test = [c for c in test_df.columns if c.lower() in [x.lower() for x in drop_cols] and c.lower() != "hospital_expire_flag"]

X = train_df.drop(columns=cols_to_drop_train)
X_test = test_df.drop(columns=cols_to_drop_test)

# Clean BP
for col, threshold in [("SysBP_Min", 40), ("DiasBP_Min", 10), ("MeanBP_Min", 30)]:
    if col in X.columns:
        X.loc[X[col] < threshold, col] = np.nan
    if col in X_test.columns:
        X_test.loc[X_test[col] < threshold, col] = np.nan

print(f"  Features: {X.shape[1]}, Positive rate: {y.mean():.3f}")

# Split
print("\n[2/6] Creating split...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing
print("\n[3/6] Preprocessing...")
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

print(f"  Processed shape: {X_train_proc.shape}")

# Train XGBoost with hyperparameter tuning
print("\n[4/6] Training XGBoost with RandomizedSearch...")
print("  This may take 5-10 minutes...")

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    tree_method='hist',
    random_state=42,
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
)

param_distributions = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
}

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring='roc_auc',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train_proc, y_train)

print(f"\n  Best CV AUC: {random_search.best_score_:.4f}")
print(f"  Best params: {random_search.best_params_}")

# Validate
print("\n[5/6] Validating...")
best_model = random_search.best_estimator_
y_valid_proba = best_model.predict_proba(X_valid_proc)[:, 1]
valid_auc = roc_auc_score(y_valid, y_valid_proba)
print(f"  Validation AUC: {valid_auc:.4f}")

# Predict on test
print("\n[6/6] Generating submission...")
test_proba = best_model.predict_proba(X_test_proc)[:, 1]

# Get IDs
id_col = [c for c in test_df.columns if c.lower() == 'icustay_id'][0]
icustay_ids = test_df[id_col]

submission = pd.DataFrame({
    'icustay_id': icustay_ids,
    'HOSPITAL_EXPIRE_FLAG': test_proba
})

output_file = OUTPUT_DIR / "PROB_submission_xgboost.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*60)
print("✓ SUCCESS!")
print("="*60)
print(f"\nSubmission: {output_file}")
print(f"Validation AUC: {valid_auc:.4f}")
print(f"\nPrediction stats:")
print(f"  Min:    {test_proba.min():.4f}")
print(f"  Max:    {test_proba.max():.4f}")
print(f"  Mean:   {test_proba.mean():.4f}")

if valid_auc > 0.727:
    print(f"\n🎉 This beats your GB score! ({valid_auc:.4f} > 0.727)")
else:
    print(f"\n⚠️  Similar to GB ({valid_auc:.4f} vs 0.727)")

print("="*60)