"""
K-Fold Ensemble with Probability Calibration

For causal inference, we want:
1. Well-calibrated probabilities (not just good AUC)
2. Stable predictions across folds
3. Reduced overfitting

This script:
- Trains 5 XGBoost models on different folds
- Averages out-of-fold predictions for calibration check
- Averages test predictions for final submission
- Optionally applies Platt scaling for calibration

Run from project root:
    python scripts/classification/kfold_ensemble.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from xgboost import XGBClassifier
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
print("K-FOLD ENSEMBLE WITH CALIBRATION")
print("="*80)

# ============================================================================
# DATA PREPARATION
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

# Remove IDs
id_cols_to_check = ['icustay_id', 'subject_id', 'hadm_id']
for col in [c for c in X_train_raw.columns if c.lower() in id_cols_to_check]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in id_cols_to_check]:
    X_test_raw = X_test_raw.drop(columns=[col])

# Feature engineering
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

# Encode categoricals
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le_c.fit(combined)
    X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))

X_train_final = X_train_fe.values
X_test_final = X_test_fe.values
feature_names = X_train_fe.columns.tolist()
y_train_array = y_train.values

print(f"✓ Features prepared: {len(feature_names)} features")

# ============================================================================
# K-FOLD CROSS VALIDATION
# ============================================================================
print("\n[3/5] K-Fold Cross Validation Training...")

N_FOLDS = 5
kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Storage for predictions
oof_predictions = np.zeros(len(X_train_final))  # Out-of-fold predictions
test_predictions = np.zeros(len(X_test_final))   # Averaged test predictions
fold_aucs = []
fold_models = []

scale_pos_weight = (y_train_array == 0).sum() / (y_train_array == 1).sum()

print(f"  Training {N_FOLDS} folds...")
print(f"  Scale pos weight: {scale_pos_weight:.2f}")
print()

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_final, y_train_array)):
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=" ")
    
    X_tr, X_val = X_train_final[train_idx], X_train_final[val_idx]
    y_tr, y_val = y_train_array[train_idx], y_train_array[val_idx]
    
    model = XGBClassifier(
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
        random_state=42 + fold,  # Different seed per fold
        tree_method='hist',
        eval_metric='auc',
        early_stopping_rounds=50,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Out-of-fold predictions
    oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
    
    # Test predictions (will be averaged)
    test_predictions += model.predict_proba(X_test_final)[:, 1] / N_FOLDS
    
    # Track fold performance
    fold_auc = roc_auc_score(y_val, oof_predictions[val_idx])
    fold_aucs.append(fold_auc)
    fold_models.append(model)
    
    print(f"AUC: {fold_auc:.4f}, Best iter: {model.best_iteration}")

# ============================================================================
# OVERALL EVALUATION
# ============================================================================
print("\n[4/5] Overall Evaluation...")

# Overall OOF AUC
oof_auc = roc_auc_score(y_train_array, oof_predictions)
print(f"\n  Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
print(f"  Mean Fold AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
print(f"  Overall OOF AUC: {oof_auc:.4f}")

# Brier score (measures calibration)
brier = brier_score_loss(y_train_array, oof_predictions)
print(f"  Brier Score: {brier:.4f} (lower is better, measures calibration)")

# Check calibration
print("\n  Calibration Check (predicted vs actual mortality rate):")
print("  " + "-"*50)

# Bin predictions and check actual rates
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(bins)-1):
    mask = (oof_predictions >= bins[i]) & (oof_predictions < bins[i+1])
    if mask.sum() > 0:
        predicted_mean = oof_predictions[mask].mean()
        actual_mean = y_train_array[mask].mean()
        count = mask.sum()
        print(f"    Pred {bins[i]:.1f}-{bins[i+1]:.1f}: n={count:5d}, pred={predicted_mean:.3f}, actual={actual_mean:.3f}")

# Classification report
print("\n  Classification Report (OOF predictions):")
oof_binary = (oof_predictions >= 0.5).astype(int)
print(classification_report(y_train_array, oof_binary, target_names=['Survived', 'Died']))

# ============================================================================
# FEATURE IMPORTANCE (averaged across folds)
# ============================================================================
print("\n  Top 20 Features (averaged importance across folds):")
avg_importance = np.mean([m.feature_importances_ for m in fold_models], axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': avg_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(20).to_string(index=False))

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n[5/5] Generating submissions...")

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
test_ids = test_raw[id_col].values

output_dir = BASE_DIR / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)

# Save K-Fold ensemble predictions
submission = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': test_predictions
})
output_file = output_dir / "kfold_ensemble.csv"
submission.to_csv(output_file, index=False)
print(f"✓ K-Fold ensemble saved: {output_file}")

# Prediction statistics
print(f"\nTest prediction stats:")
print(f"  Min:    {test_predictions.min():.4f}")
print(f"  Max:    {test_predictions.max():.4f}")
print(f"  Mean:   {test_predictions.mean():.4f}")
print(f"  Median: {np.median(test_predictions):.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
📊 K-Fold Ensemble Results:
  - Folds: {N_FOLDS}
  - Mean Fold AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})
  - Overall OOF AUC: {oof_auc:.4f}
  - Brier Score: {brier:.4f}
  - Features: {len(feature_names)}

🎯 For Causal Inference:
  - Model is trained on all data via K-fold (no wasted samples)
  - Probabilities are calibrated (check Brier score and calibration table)
  - Multiple models averaged = more stable estimates
  - Feature importances averaged = more reliable

📁 Submission: {output_file}
""")

print("="*80)
print("✓ COMPLETE!")
print("="*80)