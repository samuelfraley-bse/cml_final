"""
LightGBM Classification Script

LightGBM often outperforms XGBoost due to:
- Faster training (histogram-based)
- Better handling of categorical features
- Leaf-wise growth (vs level-wise)

Uses the same 85-feature setup that scored 74.6 with XGBoost.

Run from project root:
    python scripts/classification/lgbm_model.py

Requirements:
    pip install lightgbm
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM
try:
    import scripts.classification.lgb_hef as lgb
    print("✓ LightGBM imported successfully")
except ImportError:
    print("❌ LightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

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
print("LIGHTGBM CLASSIFICATION MODEL")
print("="*80)

# ============================================================================
# DATA PREPARATION (same as 85-feature version)
# ============================================================================
print("\n[1/5] Preparing data...")

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

# Remove remaining IDs
id_cols_to_check = ['icustay_id', 'subject_id', 'hadm_id']
for col in [c for c in X_train_raw.columns if c.lower() in id_cols_to_check]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in id_cols_to_check]:
    X_test_raw = X_test_raw.drop(columns=[col])

# Feature engineering pipeline
print("\n[2/5] Feature engineering...")
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

# Track categorical columns BEFORE encoding (LightGBM can use them natively)
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
print(f"  Categorical columns: {cat_cols}")

# Encode categoricals (LightGBM prefers integer encoding over one-hot)
cat_col_indices = []
for i, col in enumerate(X_train_fe.columns):
    if col in cat_cols:
        le_c = LabelEncoder()
        combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
        le_c.fit(combined)
        X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
        X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))
        cat_col_indices.append(i)

X_train_final = X_train_fe
X_test_final = X_test_fe

print(f"✓ Features prepared: {X_train_final.shape[1]} features")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
print("\n[3/5] Creating train/validation split...")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print(f"  Train: {X_tr.shape[0]} samples")
print(f"  Valid: {X_val.shape[0]} samples")

# Calculate class weight
scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

# ============================================================================
# LIGHTGBM MODEL
# ============================================================================
print("\n[4/5] Training LightGBM model...")

# LightGBM parameters (tuned for medical data)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,              # Max leaves per tree
    'max_depth': 6,                # Limit depth to prevent overfitting
    'learning_rate': 0.03,
    'feature_fraction': 0.85,      # Like colsample_bytree
    'bagging_fraction': 0.85,      # Like subsample
    'bagging_freq': 5,             # Bagging every 5 iterations
    'min_child_samples': 20,       # Min samples per leaf
    'min_child_weight': 0.001,
    'reg_alpha': 0.1,              # L1 regularization
    'reg_lambda': 1.0,             # L2 regularization
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

# Create datasets
train_data = lgb.Dataset(
    X_tr, 
    label=y_tr,
    categorical_feature=cat_cols if cat_cols else 'auto'
)
val_data = lgb.Dataset(
    X_val, 
    label=y_val,
    reference=train_data,
    categorical_feature=cat_cols if cat_cols else 'auto'
)

# Train with early stopping
print("\n  Training...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

print(f"\n✓ Training complete!")
print(f"  Best iteration: {model.best_iteration}")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "="*80)
print("EVALUATION")
print("="*80)

y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
y_test_pred = model.predict(X_test_final, num_iteration=model.best_iteration)

val_auc = roc_auc_score(y_val, y_val_pred)
print(f"\n✓ Validation AUC: {val_auc:.4f}")

y_val_binary = (y_val_pred >= 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_val, y_val_binary, target_names=['Survived', 'Died']))

# Feature importance
print("\nTop 25 Most Important Features:")
importance_df = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(importance_df.head(25).to_string(index=False))

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
test_ids = test_raw[id_col].values

submission = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': y_test_pred
})

output_dir = BASE_DIR / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "lightgbm_model.csv"

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
📊 LightGBM Results:
  - Validation AUC: {val_auc:.4f}
  - Best iteration: {model.best_iteration}
  - Features used: {X_train_final.shape[1]}

🎛️ Parameters Used:
  - num_leaves: {params['num_leaves']}
  - max_depth: {params['max_depth']}
  - learning_rate: {params['learning_rate']}
  - feature_fraction: {params['feature_fraction']}
  - bagging_fraction: {params['bagging_fraction']}
  - min_child_samples: {params['min_child_samples']}

📁 Submission: {output_file}

🔄 Compare with XGBoost:
  - XGBoost best: 74.6 (Val AUC ~0.856)
  - LightGBM: {val_auc:.4f} (submit to see Kaggle score)
""")

print("="*80)
print("✓ COMPLETE!")
print("="*80)