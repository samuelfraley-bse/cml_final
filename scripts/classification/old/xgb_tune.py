"""
XGBoost Hyperparameter Tuning Script - FIXED VERSION

Uses RandomizedSearchCV for efficient exploration of hyperparameter space.
Based on the 85-feature version that scored 74.6.

Run from project root:
    python scripts/classification/xgb_tune_v2.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add notebooks/HEF to path
BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

from hef_prep import (
    split_features_target,
    add_age_features,
    clean_min_bp_outliers, 
    add_engineered_features,
    add_age_interactions,
    TARGET_COL_CLASS,
    ID_COLS
)

print("="*80)
print("XGBOOST HYPERPARAMETER TUNING")
print("="*80)

# ============================================================================
# DATA PREPARATION (same as 85-feature version that scored 74.6)
# ============================================================================
print("\n[1/4] Preparing data (85-feature version)...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

# Check for ICD9
icd9_col = None
for col in train_raw.columns:
    if 'ICD9' in col.upper() and 'DIAG' in col.upper():
        icd9_col = col
        break

# Split features - drop raw ICD9 here, add encoded version later
leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

# Remove remaining IDs
id_cols_to_check = ['icustay_id', 'subject_id', 'hadm_id']
for col in [c for c in X_train_raw.columns if c.lower() in id_cols_to_check]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in id_cols_to_check]:
    X_test_raw = X_test_raw.drop(columns=[col])

# Feature engineering pipeline
X_train_fe = add_age_features(X_train_raw)
X_test_fe = add_age_features(X_test_raw)

X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)

X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)

X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

# Add ICD9 back as encoded features (NOT leakage - it's admission diagnosis)
if icd9_col:
    train_icd9 = train_raw[icd9_col].copy().fillna('MISSING')
    test_icd9 = test_raw[icd9_col].copy().fillna('MISSING')
    
    le = LabelEncoder()
    le.fit(pd.concat([train_icd9, test_icd9]))
    X_train_fe['ICD9_encoded'] = le.transform(train_icd9)
    X_test_fe['ICD9_encoded'] = le.transform(test_icd9)
    
    train_cat = train_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    test_cat = test_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    le_cat = LabelEncoder()
    le_cat.fit(pd.concat([train_cat, test_cat]))
    X_train_fe['ICD9_category'] = le_cat.transform(train_cat)
    X_test_fe['ICD9_category'] = le_cat.transform(test_cat)

# Encode categoricals
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le_c.fit(combined)
    X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))

X_train_final = X_train_fe
X_test_final = X_test_fe

print(f"✓ Data prepared: {X_train_final.shape[1]} features")

# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================
print("\n[2/4] Defining search space...")

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

# Parameter distributions for RandomizedSearchCV
param_distributions = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
    'min_child_weight': [3, 5, 7, 10],
    'subsample': [0.6, 0.7, 0.8, 0.85, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.85, 0.9],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.1, 0.2, 0.5],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0],
}

print("  Search space defined")

# ============================================================================
# RANDOMIZED SEARCH
# ============================================================================
print("\n[3/4] Running RandomizedSearchCV...")
print("  Testing 50 configurations with 5-fold CV...")
print("  This may take 10-15 minutes...\n")

# Base model
base_model = XGBClassifier(
    n_estimators=300,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    n_jobs=-1,
    verbosity=0
)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized search - use 'roc_auc' string (sklearn handles predict_proba)
n_iter = 50
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=n_iter,
    scoring='roc_auc',
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

start_time = time.time()
search.fit(X_train_final, y_train)
elapsed = time.time() - start_time

print(f"\n✓ Search completed in {elapsed/60:.1f} minutes")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*80)
print("TUNING RESULTS")
print("="*80)

print(f"\n🏆 Best CV AUC: {search.best_score_:.4f}")
print(f"\nBest Parameters:")
for param, value in search.best_params_.items():
    if isinstance(value, float):
        print(f"  {param}: {value:.4f}")
    else:
        print(f"  {param}: {value}")

# Show top 10 configurations
print("\n\nTop 10 Configurations:")
results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values('rank_test_score')
cols_to_show = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_train_score']
print(results_df[cols_to_show].head(10).to_string())

# ============================================================================
# TRAIN BEST MODEL WITH EARLY STOPPING
# ============================================================================
print("\n" + "="*80)
print("TRAINING BEST MODEL WITH EARLY STOPPING")
print("="*80)

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# Get best params and build final model
best_params = search.best_params_.copy()

best_model = XGBClassifier(
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    min_child_weight=best_params['min_child_weight'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    n_estimators=1500,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    early_stopping_rounds=50,
    n_jobs=-1
)

print("\nTraining with early stopping...")
best_model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=50
)

print(f"\n✓ Best iteration: {best_model.best_iteration}")

# Evaluate
y_val_pred = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"\n✓ Validation AUC: {val_auc:.4f}")

y_val_binary = (y_val_pred >= 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_val, y_val_binary, target_names=['Survived', 'Died']))

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

y_test_pred = best_model.predict_proba(X_test_final)[:, 1]

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
test_ids = test_raw[id_col].values

submission = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': y_test_pred
})

output_dir = BASE_DIR / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "xgboost_tuned.csv"

submission.to_csv(output_file, index=False)

print(f"\n✓ Submission saved: {output_file}")
print(f"\nPrediction stats: Min={y_test_pred.min():.4f}, Max={y_test_pred.max():.4f}, Mean={y_test_pred.mean():.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
📊 Tuning Results:
  - Configurations tested: {n_iter}
  - Best CV AUC: {search.best_score_:.4f}
  - Validation AUC: {val_auc:.4f}
  - Best n_estimators (with early stopping): {best_model.best_iteration}

🎛️ Best Hyperparameters (copy this for future use):
""")

print("best_params = {")
for param, value in best_params.items():
    if isinstance(value, float):
        print(f"    '{param}': {value:.4f},")
    else:
        print(f"    '{param}': {value},")
print("}")

print(f"""
📁 Submission: {output_file}

🔄 Compare:
  - Previous best (default params): 74.6
  - Tuned model: Submit to see!
""")

print("="*80)
print("✓ TUNING COMPLETE!")
print("="*80)