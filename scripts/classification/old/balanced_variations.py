"""
Balanced Variations - Explore Around the Best Generalization Config

Balanced won with:
- OOF AUC: 0.8536
- Fold Std: 0.0081
- Gen Score: 0.8455

Now we test variations to find even better generalization.

Run from project root:
    python scripts/classification/balanced_variations.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
print("BALANCED VARIATIONS - PUSH GENERALIZATION HIGHER")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# FEATURE ENGINEERING (same as before)
# ============================================================================

def add_missing_indicators(df):
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

# Prepare data
print("\nPreparing features...")

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

for col in [c for c in X_train_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_test_raw = X_test_raw.drop(columns=[col])

X_train_fe = add_missing_indicators(X_train_raw)
X_test_fe = add_missing_indicators(X_test_raw)
X_train_fe = add_extreme_flags(X_train_fe)
X_test_fe = add_extreme_flags(X_test_fe)
X_train_fe = add_age_features(X_train_fe)
X_test_fe = add_age_features(X_test_fe)
X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)
X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)
X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

# Add ICD9
if 'ICD9_diagnosis' in train_raw.columns:
    train_icd9 = train_raw['ICD9_diagnosis'].fillna('MISSING')
    test_icd9 = test_raw['ICD9_diagnosis'].fillna('MISSING')
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
# VARIATIONS AROUND BALANCED CONFIG
# ============================================================================

configs = {
    # Baseline (the winner)
    'Balanced_Base': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 8,
        'gamma': 0.15,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
    },
    # Slightly more trees
    'Balanced_4000': {
        'max_depth': 4,
        'learning_rate': 0.008,
        'n_estimators': 4000,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 8,
        'gamma': 0.15,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
    },
    # Slightly higher subsample
    'Balanced_Sub80': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.80,
        'colsample_bytree': 0.80,
        'min_child_weight': 8,
        'gamma': 0.15,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
    },
    # Slightly lower subsample
    'Balanced_Sub70': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.70,
        'colsample_bytree': 0.70,
        'min_child_weight': 8,
        'gamma': 0.15,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
    },
    # Higher min_child_weight
    'Balanced_MCW12': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 12,
        'gamma': 0.15,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
    },
    # Lower min_child_weight
    'Balanced_MCW5': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 5,
        'gamma': 0.15,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
    },
    # Higher gamma
    'Balanced_Gamma20': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 8,
        'gamma': 0.20,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
    },
    # Higher reg_lambda
    'Balanced_Lambda2': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 8,
        'gamma': 0.15,
        'reg_alpha': 0.2,
        'reg_lambda': 2.0,
    },
    # Depth 3
    'Balanced_Depth3': {
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 3500,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 8,
        'gamma': 0.15,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
    },
    # Combined tweaks (more conservative)
    'Balanced_Conservative': {
        'max_depth': 4,
        'learning_rate': 0.008,
        'n_estimators': 4000,
        'subsample': 0.70,
        'colsample_bytree': 0.70,
        'min_child_weight': 10,
        'gamma': 0.18,
        'reg_alpha': 0.25,
        'reg_lambda': 1.8,
    },
}

# ============================================================================
# K-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("K-FOLD CROSS-VALIDATION (5 folds)")
print("="*80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

results = []

for name, params in configs.items():
    print(f"\n[{name}]")

    oof_preds = np.zeros(len(X_train_final))
    fold_aucs = []
    test_preds_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_final, y_train)):
        X_fold_train = X_train_final.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_final.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_scale = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()

        model = XGBClassifier(
            **params,
            scale_pos_weight=fold_scale,
            random_state=42,
            tree_method='hist',
            eval_metric='auc',
            early_stopping_rounds=50,
            n_jobs=-1
        )

        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)], verbose=False)

        oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        test_preds_folds.append(model.predict_proba(X_test_final)[:, 1])

    oof_auc = roc_auc_score(y_train, oof_preds)
    fold_std = np.std(fold_aucs)
    test_preds_avg = np.mean(test_preds_folds, axis=0)

    results.append({
        'model': name,
        'oof_auc': oof_auc,
        'fold_std': fold_std,
        'gen_score': oof_auc - fold_std,
        'test_preds': test_preds_avg
    })

    print(f"  OOF AUC: {oof_auc:.4f}, Std: {fold_std:.4f}, Gen: {oof_auc - fold_std:.4f}")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*80)
print("RESULTS - RANKED BY GENERALIZATION SCORE")
print("="*80)

results_df = pd.DataFrame([{
    'model': r['model'],
    'oof_auc': r['oof_auc'],
    'fold_std': r['fold_std'],
    'gen_score': r['gen_score']
} for r in results]).sort_values('gen_score', ascending=False)

print(f"\n{'Model':<25} {'OOF AUC':>10} {'Fold Std':>10} {'Gen Score':>12}")
print("-" * 60)

for _, row in results_df.iterrows():
    print(f"{row['model']:<25} {row['oof_auc']:>10.4f} {row['fold_std']:>10.4f} {row['gen_score']:>12.4f}")

best = results_df.iloc[0]
print(f"\n🏆 Best: {best['model']} (Gen Score: {best['gen_score']:.4f})")

# Compare to previous best
print(f"\nPrevious best (Balanced): Gen Score = 0.8455")
improvement = best['gen_score'] - 0.8455
print(f"Improvement: {improvement:+.4f}")

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS (Top 3)")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

for i, row in results_df.head(3).iterrows():
    model_name = row['model']
    test_preds = [r['test_preds'] for r in results if r['model'] == model_name][0]

    filename = f"bal_{model_name.lower().replace('balanced_', '')}.csv"
    submission = pd.DataFrame({
        'icustay_id': test_raw[id_col].values,
        'HOSPITAL_EXPIRE_FLAG': test_preds
    })
    output_file = BASE_DIR / "submissions" / filename
    submission.to_csv(output_file, index=False)
    print(f"✓ {model_name}: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
