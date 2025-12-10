"""
Fine-tune Gradient Boosting
Your GB scored 0.727 - let's optimize around those parameters
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

print("="*60)
print("GB FINE-TUNING (NARROW GRID SEARCH)")
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

# Grid search around your best parameters
print("\n[4/5] Grid search (narrow range around current best)...")
print("  Current best: n_estimators=200, lr=0.05, max_depth=5")
print("  This will take 5-10 minutes...")

gb = GradientBoostingClassifier(
    random_state=42,
)

# Narrow grid around current best
param_grid = {
    'clf__n_estimators': [180, 200, 220, 250],
    'clf__learning_rate': [0.03, 0.05, 0.07],
    'clf__max_depth': [4, 5, 6],
    'clf__min_samples_split': [8, 10, 12],
    'clf__min_samples_leaf': [4, 5, 6],
    'clf__subsample': [0.75, 0.8, 0.85],
    'clf__max_features': ['sqrt', 0.8],
}

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', gb),
])

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"\n  Best CV AUC: {grid_search.best_score_:.6f}")
print(f"  Best params:")
for param, value in grid_search.best_params_.items():
    print(f"    {param}: {value}")

# Validate on holdout
print("\n[5/5] Validating on holdout set...")
best_model = grid_search.best_estimator_
y_valid_proba = best_model.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, y_valid_proba)
print(f"  Holdout validation AUC: {valid_auc:.6f}")
print(f"  vs Current best: 0.727226")

# Predict on test
test_proba = best_model.predict_proba(X_test)[:, 1]

id_col = [c for c in test_df.columns if c.lower() == 'icustay_id'][0]
submission = pd.DataFrame({
    'icustay_id': test_df[id_col],
    'HOSPITAL_EXPIRE_FLAG': test_proba
})

output_file = OUTPUT_DIR / "PROB_submission_gb_tuned.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*60)
print("✓ SUCCESS!")
print("="*60)
print(f"\nSubmission: {output_file}")
print(f"Validation AUC: {valid_auc:.6f}")

if valid_auc > 0.7273:
    print(f"\n🎉 Improvement found! ({valid_auc:.6f} > 0.7273)")
    print("   Submit this one!")
elif valid_auc > 0.7270:
    print(f"\n⚠️  Marginal change ({valid_auc:.6f})")
    print("   Might be worth trying on leaderboard")
else:
    print(f"\n⚠️  No improvement in validation")
    print("   Current model is already well-tuned")

print("="*60)