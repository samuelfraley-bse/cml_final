"""
XGBoost Ensemble - Multiple Models Averaged

Based on scratch.py (best model at 74.9) but with:
- Multiple XGBoost models with different random seeds
- Different hyperparameter variations
- Averaged predictions for more robust output

Run from project root:
    python scripts/classification/xgb_ensemble.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

from hef_prep import (
    split_features_target,
    add_age_features,
    clean_min_bp_outliers,
    add_engineered_features,
    add_age_interactions,
    ID_COLS
)

print("="*80)
print("XGBOOST ENSEMBLE - MULTIPLE MODELS")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# FEATURE ENGINEERING (same as scratch.py)
# ============================================================================

vital_cols = ['HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean',
              'SysBP_Min', 'SysBP_Max', 'SysBP_Mean',
              'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',
              'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean',
              'RespRate_Min', 'RespRate_Max', 'RespRate_Mean',
              'TempC_Min', 'TempC_Max', 'TempC_Mean',
              'SpO2_Min', 'SpO2_Max', 'SpO2_Mean',
              'Glucose_Min', 'Glucose_Max', 'Glucose_Mean']

existing_vitals = [c for c in vital_cols if c in train_raw.columns]

def add_missing_indicators(df, vital_cols):
    """Add binary indicators for missing vitals."""
    df = df.copy()

    for vital_base in ['TempC', 'SysBP', 'SpO2', 'Glucose', 'DiasBP', 'MeanBP', 'RespRate', 'HeartRate']:
        col = f'{vital_base}_Mean'
        if col in df.columns:
            df[f'{vital_base}_missing'] = df[col].isna().astype(int)

    missing_cols = [c for c in df.columns if c.endswith('_missing')]
    if missing_cols:
        df['missing_vital_count'] = df[missing_cols].sum(axis=1)
        df['any_vital_missing'] = (df['missing_vital_count'] > 0).astype(int)

    return df

def add_extreme_flags(df):
    """Add flags for extreme/critical vital values."""
    df = df.copy()

    if 'SysBP_Min' in df.columns:
        df['critical_low_bp'] = (df['SysBP_Min'] < 70).astype(int)
        df['very_low_bp'] = (df['SysBP_Min'] < 80).astype(int)

    if 'SpO2_Min' in df.columns:
        df['critical_low_spo2'] = (df['SpO2_Min'] < 85).astype(int)
        df['very_low_spo2'] = (df['SpO2_Min'] < 88).astype(int)

    if 'HeartRate_Min' in df.columns:
        df['bradycardia'] = (df['HeartRate_Min'] < 50).astype(int)

    if 'TempC_Min' in df.columns:
        df['hypothermia'] = (df['TempC_Min'] < 35).astype(int)

    if 'Glucose_Min' in df.columns:
        df['hypoglycemia'] = (df['Glucose_Min'] < 60).astype(int)

    if 'HeartRate_Max' in df.columns:
        df['severe_tachycardia'] = (df['HeartRate_Max'] > 150).astype(int)
        df['extreme_tachycardia'] = (df['HeartRate_Max'] > 170).astype(int)

    if 'TempC_Max' in df.columns:
        df['high_fever'] = (df['TempC_Max'] > 39).astype(int)
        df['extreme_fever'] = (df['TempC_Max'] > 40).astype(int)

    if 'Glucose_Max' in df.columns:
        df['severe_hyperglycemia'] = (df['Glucose_Max'] > 300).astype(int)
        df['extreme_hyperglycemia'] = (df['Glucose_Max'] > 400).astype(int)

    if 'RespRate_Max' in df.columns:
        df['severe_tachypnea'] = (df['RespRate_Max'] > 30).astype(int)
        df['extreme_tachypnea'] = (df['RespRate_Max'] > 40).astype(int)

    if 'SysBP_Max' in df.columns:
        df['hypertensive_crisis'] = (df['SysBP_Max'] > 180).astype(int)

    critical_cols = ['critical_low_bp', 'critical_low_spo2', 'bradycardia',
                     'hypothermia', 'hypoglycemia', 'severe_tachycardia',
                     'high_fever', 'severe_hyperglycemia', 'severe_tachypnea']
    existing_critical = [c for c in critical_cols if c in df.columns]
    if existing_critical:
        df['critical_flag_count'] = df[existing_critical].sum(axis=1)
        df['any_critical_flag'] = (df['critical_flag_count'] > 0).astype(int)
        df['multiple_critical'] = (df['critical_flag_count'] >= 2).astype(int)

    return df

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\nPreparing data...")

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

# Remove IDs
for col in [c for c in X_train_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_test_raw = X_test_raw.drop(columns=[col])

# Feature engineering pipeline
X_train_miss = add_missing_indicators(X_train_raw, existing_vitals)
X_test_miss = add_missing_indicators(X_test_raw, existing_vitals)

X_train_extreme = add_extreme_flags(X_train_miss)
X_test_extreme = add_extreme_flags(X_test_miss)

X_train_fe = add_age_features(X_train_extreme)
X_test_fe = add_age_features(X_test_extreme)

X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)

X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)

X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

# Add ICD9
icd9_col = 'ICD9_diagnosis'
if icd9_col in train_raw.columns:
    train_icd9 = train_raw[icd9_col].fillna('MISSING')
    test_icd9 = test_raw[icd9_col].fillna('MISSING')

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

print(f"✓ Total features: {X_train_final.shape[1]}")

# ============================================================================
# ENSEMBLE CONFIGURATIONS
# ============================================================================

# Different model configurations to ensemble
ENSEMBLE_CONFIGS = [
    # Config 1: Original best parameters
    {
        'name': 'Original',
        'params': {
            'max_depth': 5,
            'learning_rate': 0.03,
            'n_estimators': 1500,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        },
        'seeds': [42, 123, 456]
    },
    # Config 2: Deeper trees, lower learning rate
    {
        'name': 'Deep',
        'params': {
            'max_depth': 6,
            'learning_rate': 0.02,
            'n_estimators': 2000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 7,
            'gamma': 0.15,
            'reg_alpha': 0.15,
            'reg_lambda': 1.5,
        },
        'seeds': [42, 789]
    },
    # Config 3: Shallower trees, higher learning rate
    {
        'name': 'Shallow',
        'params': {
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 0.8,
        },
        'seeds': [42, 321]
    },
]

# ============================================================================
# TRAIN ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("TRAINING ENSEMBLE MODELS")
print("="*80)

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

all_val_preds = []
all_test_preds = []
model_results = []

model_count = 0
for config in ENSEMBLE_CONFIGS:
    for seed in config['seeds']:
        model_count += 1
        model_name = f"{config['name']}_seed{seed}"

        print(f"\n[{model_count}] Training {model_name}...")

        params = config['params'].copy()
        params['random_state'] = seed
        params['scale_pos_weight'] = scale_pos_weight
        params['tree_method'] = 'hist'
        params['eval_metric'] = 'auc'
        params['early_stopping_rounds'] = 50
        params['n_jobs'] = -1

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # Validation predictions
        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred)

        # Test predictions
        y_test_pred = model.predict_proba(X_test_final)[:, 1]

        all_val_preds.append(y_val_pred)
        all_test_preds.append(y_test_pred)

        model_results.append({
            'name': model_name,
            'val_auc': val_auc
        })

        print(f"    Val AUC: {val_auc:.4f}")

# ============================================================================
# ENSEMBLE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("ENSEMBLE RESULTS")
print("="*80)

# Simple average
val_ensemble_avg = np.mean(all_val_preds, axis=0)
test_ensemble_avg = np.mean(all_test_preds, axis=0)
ensemble_avg_auc = roc_auc_score(y_val, val_ensemble_avg)

print(f"\nIndividual model results:")
for result in sorted(model_results, key=lambda x: x['val_auc'], reverse=True):
    print(f"  {result['name']:<25}: {result['val_auc']:.4f}")

print(f"\n{'='*50}")
print(f"Simple Average Ensemble: {ensemble_avg_auc:.4f}")
print(f"{'='*50}")

# Weighted average (weight by individual AUC)
weights = np.array([r['val_auc'] for r in model_results])
weights = weights / weights.sum()

val_ensemble_weighted = np.average(all_val_preds, axis=0, weights=weights)
test_ensemble_weighted = np.average(all_test_preds, axis=0, weights=weights)
ensemble_weighted_auc = roc_auc_score(y_val, val_ensemble_weighted)

print(f"Weighted Average Ensemble: {ensemble_weighted_auc:.4f}")

# Median ensemble (more robust to outliers)
val_ensemble_median = np.median(all_val_preds, axis=0)
test_ensemble_median = np.median(all_test_preds, axis=0)
ensemble_median_auc = roc_auc_score(y_val, val_ensemble_median)

print(f"Median Ensemble: {ensemble_median_auc:.4f}")

# ============================================================================
# K-FOLD ENSEMBLE (train on all data)
# ============================================================================
print("\n" + "="*80)
print("K-FOLD ENSEMBLE (uses all training data)")
print("="*80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train_final))
test_preds_kfold = np.zeros(len(X_test_final))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_final, y_train)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")

    X_fold_train = X_train_final.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train_final.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    fold_scale = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()

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
        scale_pos_weight=fold_scale,
        random_state=42,
        tree_method='hist',
        eval_metric='auc',
        early_stopping_rounds=50,
        n_jobs=-1
    )

    model.fit(X_fold_train, y_fold_train,
              eval_set=[(X_fold_val, y_fold_val)], verbose=False)

    # OOF predictions
    oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]

    # Test predictions (average across folds)
    test_preds_kfold += model.predict_proba(X_test_final)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_fold_val, oof_preds[val_idx])
    print(f"  Fold AUC: {fold_auc:.4f}")

oof_auc = roc_auc_score(y_train, oof_preds)
print(f"\n{'='*50}")
print(f"K-Fold OOF AUC: {oof_auc:.4f}")
print(f"{'='*50}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

all_results = {
    'Best Individual': max(r['val_auc'] for r in model_results),
    'Simple Average': ensemble_avg_auc,
    'Weighted Average': ensemble_weighted_auc,
    'Median Ensemble': ensemble_median_auc,
    'K-Fold OOF': oof_auc
}

print("\nAll approaches:")
for name, auc in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<20}: {auc:.4f}")

best_approach = max(all_results, key=all_results.get)
print(f"\n🏆 Best: {best_approach} ({all_results[best_approach]:.4f})")

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

# 1. Simple average ensemble
submission_avg = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': test_ensemble_avg
})
output_avg = BASE_DIR / "submissions" / "xgb_ensemble_avg.csv"
submission_avg.to_csv(output_avg, index=False)
print(f"✓ Simple Average: {output_avg}")

# 2. Weighted average ensemble
submission_weighted = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': test_ensemble_weighted
})
output_weighted = BASE_DIR / "submissions" / "xgb_ensemble_weighted.csv"
submission_weighted.to_csv(output_weighted, index=False)
print(f"✓ Weighted Average: {output_weighted}")

# 3. K-Fold ensemble
submission_kfold = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': test_preds_kfold
})
output_kfold = BASE_DIR / "submissions" / "xgb_ensemble_kfold.csv"
submission_kfold.to_csv(output_kfold, index=False)
print(f"✓ K-Fold: {output_kfold}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print("\nRecommend submitting: xgb_ensemble_kfold.csv (uses all training data)")
