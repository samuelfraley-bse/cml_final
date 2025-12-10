"""
Ensemble Model: XGBoost + LightGBM

Combines predictions from both models. Since they use different
algorithms (level-wise vs leaf-wise), they make different errors.
Averaging often improves generalization.

Run from project root:
    python scripts/classification/ensemble_xgb_lgbm.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

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
print("ENSEMBLE MODEL: XGBOOST + LIGHTGBM")
print("="*80)

# ============================================================================
# DATA PREPARATION (same as best 85-feature version)
# ============================================================================
print("\n[1/6] Preparing data...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")
print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

# Check for ICD9
icd9_col = None
for col in train_raw.columns:
    if 'ICD9' in col.upper() and 'DIAG' in col.upper():
        icd9_col = col
        break

# Split features
leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

# Remove IDs
id_cols_to_check = ['icustay_id', 'subject_id', 'hadm_id']
for col in [c for c in X_train_raw.columns if c.lower() in id_cols_to_check]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in id_cols_to_check]:
    X_test_raw = X_test_raw.drop(columns=[col])

# Feature engineering
print("\n[2/6] Feature engineering...")
X_train_fe = add_age_features(X_train_raw)
X_test_fe = add_age_features(X_test_raw)

X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)

X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)

X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

# Add ICD9
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

# Track categorical columns for LightGBM
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

# Encode categoricals
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le_c.fit(combined)
    X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))

X_train_final = X_train_fe
X_test_final = X_test_fe

print(f"✓ Features prepared: {X_train_final.shape[1]} features")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
print("\n[3/6] Creating train/validation split...")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print(f"  Train: {X_tr.shape[0]} samples, Valid: {X_val.shape[0]} samples")

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

# ============================================================================
# TRAIN XGBOOST
# ============================================================================
print("\n[4/6] Training XGBoost...")

xgb_model = XGBClassifier(
    max_depth=5,
    learning_rate=0.03,
    n_estimators=1500,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    early_stopping_rounds=50,
    n_jobs=-1
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=100
)

xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_pred = xgb_model.predict_proba(X_test_final)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_val_pred)
print(f"  ✓ XGBoost Val AUC: {xgb_auc:.4f}")

# ============================================================================
# TRAIN LIGHTGBM
# ============================================================================
print("\n[5/6] Training LightGBM...")

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.03,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

train_data = lgb.Dataset(X_tr, label=y_tr)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

lgb_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1500,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_test_pred = lgb_model.predict(X_test_final, num_iteration=lgb_model.best_iteration)
lgb_auc = roc_auc_score(y_val, lgb_val_pred)
print(f"  ✓ LightGBM Val AUC: {lgb_auc:.4f}")

# ============================================================================
# ENSEMBLE PREDICTIONS
# ============================================================================
print("\n[6/6] Creating ensemble predictions...")

# Try different weights
weights_to_try = [
    (0.5, 0.5),   # Equal
    (0.6, 0.4),   # Favor XGBoost
    (0.7, 0.3),   # More XGBoost
    (0.4, 0.6),   # Favor LightGBM
]

print("\n  Testing different ensemble weights:")
print("  " + "-"*50)

best_weight = None
best_auc = 0

for xgb_w, lgb_w in weights_to_try:
    ensemble_val = xgb_w * xgb_val_pred + lgb_w * lgb_val_pred
    ensemble_auc = roc_auc_score(y_val, ensemble_val)
    print(f"  XGB:{xgb_w:.1f} + LGB:{lgb_w:.1f} → Val AUC: {ensemble_auc:.4f}")
    
    if ensemble_auc > best_auc:
        best_auc = ensemble_auc
        best_weight = (xgb_w, lgb_w)

print(f"\n  🏆 Best weights: XGB:{best_weight[0]:.1f} + LGB:{best_weight[1]:.1f}")

# Generate final predictions with best weights
ensemble_val_pred = best_weight[0] * xgb_val_pred + best_weight[1] * lgb_val_pred
ensemble_test_pred = best_weight[0] * xgb_test_pred + best_weight[1] * lgb_test_pred

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)

print(f"""
Model Performance (Validation AUC):
  XGBoost:           {xgb_auc:.4f}
  LightGBM:          {lgb_auc:.4f}
  Ensemble:          {best_auc:.4f}
  
  Improvement:       {(best_auc - max(xgb_auc, lgb_auc)):.4f}
""")

# Classification report for ensemble
y_val_binary = (ensemble_val_pred >= 0.5).astype(int)
print("Ensemble Classification Report:")
print(classification_report(y_val, y_val_binary, target_names=['Survived', 'Died']))

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
test_ids = test_raw[id_col].values

output_dir = BASE_DIR / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)

# Save ensemble
submission_ensemble = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': ensemble_test_pred
})
ensemble_file = output_dir / "ensemble_xgb_lgbm.csv"
submission_ensemble.to_csv(ensemble_file, index=False)
print(f"✓ Ensemble saved: {ensemble_file}")

# Also save individual models for comparison
submission_xgb = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': xgb_test_pred
})
xgb_file = output_dir / "xgb_for_ensemble.csv"
submission_xgb.to_csv(xgb_file, index=False)
print(f"✓ XGBoost saved: {xgb_file}")

submission_lgb = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': lgb_test_pred
})
lgb_file = output_dir / "lgb_for_ensemble.csv"
submission_lgb.to_csv(lgb_file, index=False)
print(f"✓ LightGBM saved: {lgb_file}")

print(f"""
📊 Final Summary:
  XGBoost Val AUC:   {xgb_auc:.4f}
  LightGBM Val AUC:  {lgb_auc:.4f}
  Ensemble Val AUC:  {best_auc:.4f}
  
  Best weights: {best_weight[0]:.1f} XGB + {best_weight[1]:.1f} LGB
  
  Submit ensemble_xgb_lgbm.csv to Kaggle!
""")

print("="*80)
print("✓ COMPLETE!")
print("="*80)