"""
Generalization Search - Find Model with Best Internal Generalization

Instead of tuning on Kaggle, we measure generalization properly:
1. K-fold cross-validation (5 folds)
2. Out-of-fold AUC (every sample predicted when held out)
3. Fold stability (std across folds - lower is better)

The model with highest OOF AUC and lowest variance should generalize best.

Run from project root:
    python scripts/classification/generalization_search.py
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
print("GENERALIZATION SEARCH - FIND BEST INTERNAL GENERALIZATION")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# FEATURE ENGINEERING (same as scratch.py)
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
# MODEL CONFIGURATIONS TO TEST
# ============================================================================

# Scale pos weight (calculate once)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

configs = {
    # Your original best (74.9)
    'Original': {
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
    # MoreTrees (75.1)
    'MoreTrees': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    },
    # HighReg (75.18)
    'HighReg': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'gamma': 0.2,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
    },
    # Even more regularization
    'VeryHighReg': {
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'min_child_weight': 15,
        'gamma': 0.3,
        'reg_alpha': 0.5,
        'reg_lambda': 3.0,
    },
    # Balanced regularization
    'Balanced': {
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
    # Very slow learning
    'VerySlow': {
        'max_depth': 4,
        'learning_rate': 0.005,
        'n_estimators': 5000,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'gamma': 0.2,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
    },
}

# ============================================================================
# K-FOLD CROSS-VALIDATION FOR EACH CONFIG
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

        # OOF predictions
        oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)

        # Test predictions
        test_preds_folds.append(model.predict_proba(X_test_final)[:, 1])

    # Calculate metrics
    oof_auc = roc_auc_score(y_train, oof_preds)
    fold_std = np.std(fold_aucs)
    fold_mean = np.mean(fold_aucs)

    # Average test predictions
    test_preds_avg = np.mean(test_preds_folds, axis=0)

    results.append({
        'model': name,
        'oof_auc': oof_auc,
        'fold_mean': fold_mean,
        'fold_std': fold_std,
        'fold_aucs': fold_aucs,
        'test_preds': test_preds_avg
    })

    print(f"  OOF AUC: {oof_auc:.4f}")
    print(f"  Fold Mean: {fold_mean:.4f} ± {fold_std:.4f}")
    print(f"  Folds: {[f'{x:.4f}' for x in fold_aucs]}")

# ============================================================================
# RESULTS - RANK BY GENERALIZATION
# ============================================================================
print("\n" + "="*80)
print("RESULTS - RANKED BY GENERALIZATION")
print("="*80)

# Sort by OOF AUC (primary) and low std (secondary)
results_df = pd.DataFrame([{
    'model': r['model'],
    'oof_auc': r['oof_auc'],
    'fold_mean': r['fold_mean'],
    'fold_std': r['fold_std']
} for r in results])

# Create a generalization score: high AUC, low variance
results_df['gen_score'] = results_df['oof_auc'] - results_df['fold_std']
results_df = results_df.sort_values('gen_score', ascending=False)

print("\nRanked by Generalization Score (OOF AUC - Fold Std):")
print("-" * 70)
print(f"{'Model':<15} {'OOF AUC':>10} {'Fold Mean':>12} {'Fold Std':>10} {'Gen Score':>12}")
print("-" * 70)

for _, row in results_df.iterrows():
    print(f"{row['model']:<15} {row['oof_auc']:>10.4f} {row['fold_mean']:>12.4f} {row['fold_std']:>10.4f} {row['gen_score']:>12.4f}")

best_model = results_df.iloc[0]['model']
print(f"\n🏆 Best Generalization: {best_model}")

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

# Save top 3 by generalization score
for i, row in results_df.head(3).iterrows():
    model_name = row['model']
    # Find the test preds for this model
    test_preds = [r['test_preds'] for r in results if r['model'] == model_name][0]

    filename = f"gen_{model_name.lower()}.csv"
    submission = pd.DataFrame({
        'icustay_id': test_raw[id_col].values,
        'HOSPITAL_EXPIRE_FLAG': test_preds
    })
    output_file = BASE_DIR / "submissions" / filename
    submission.to_csv(output_file, index=False)
    print(f"✓ {model_name} (Gen Score: {row['gen_score']:.4f}): {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print("\nThe model with highest generalization score should perform best on Kaggle.")
print("Submit the top 1-2 to verify.")
