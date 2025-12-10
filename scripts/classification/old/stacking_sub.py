"""
Stacking Ensemble - Meta-learner on top of base models
Often gets 1-2% boost over single models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               ExtraTreesClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

print("="*60)
print("STACKING ENSEMBLE SUBMISSION")
print("="*60)

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
OUTPUT_DIR = BASE_DIR / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/5] Loading data...")
train_df = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_df = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

target_col = [c for c in train_df.columns if c.lower() == "hospital_expire_flag"][0]
y = train_df[target_col].copy()

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

print(f"  Features: {X.shape[1]}")

# Split
print("\n[2/5] Creating split...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
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

# Base models
print("\n[4/5] Training stacking ensemble...")
print("  This will take 5-10 minutes (training multiple models)...")

base_estimators = [
    ('rf', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=7,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=7,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )),
]

# Meta-learner (logistic regression)
meta_learner = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
)

# Stacking classifier
stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,  # 5-fold CV for meta-features
    n_jobs=-1,
)

# Full pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('stacking', stacking),
])

pipeline.fit(X_train, y_train)
print("  ✓ Stacking ensemble trained")

# Validate
print("\n[5/5] Validating...")
y_valid_proba = pipeline.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, y_valid_proba)
print(f"  Validation AUC: {valid_auc:.4f}")
print(f"  vs GB baseline: 0.727")

# Predict
test_proba = pipeline.predict_proba(X_test)[:, 1]

id_col = [c for c in test_df.columns if c.lower() == 'icustay_id'][0]
submission = pd.DataFrame({
    'icustay_id': test_df[id_col],
    'HOSPITAL_EXPIRE_FLAG': test_proba
})

output_file = OUTPUT_DIR / "PROB_submission_stacking.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*60)
print("✓ SUCCESS!")
print("="*60)
print(f"\nSubmission: {output_file}")
print(f"Validation AUC: {valid_auc:.4f}")

if valid_auc > 0.727:
    print(f"\n🎉 Stacking wins! ({valid_auc:.4f} > 0.727)")
else:
    print(f"\n⚠️  Similar to GB ({valid_auc:.4f} vs 0.727)")

print("\nStacking often improves test performance even with similar validation!")
print("="*60)