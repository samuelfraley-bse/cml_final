"""
XGBoost Optimized - Based on MoreTrees Success (75.1)

The MoreTrees config worked best:
- Shallower trees (depth 4)
- Slower learning (0.01)
- More trees (3000)

This script tests variations to push even higher.

Run from project root:
    python scripts/classification/xgb_optimized.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
print("XGBOOST OPTIMIZED - PUSHING PAST 75.1")
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

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

# ============================================================================
# TEST VARIATIONS AROUND WINNING CONFIG
# ============================================================================
print("\n" + "="*80)
print("TESTING VARIATIONS (based on MoreTrees success)")
print("="*80)

configs = {
    # Baseline winning config
    'MoreTrees_Base': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 100,
    },
    # Even slower learning, more trees
    'Slower_5000': {
        'max_depth': 4,
        'learning_rate': 0.005,
        'n_estimators': 5000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 150,
    },
    # Depth 3 (even shallower)
    'Depth3': {
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 4000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 100,
    },
    # More regularization
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
        'early_stopping_rounds': 100,
    },
    # Less regularization (more fitting)
    'LowReg': {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 3,
        'gamma': 0.05,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'early_stopping_rounds': 100,
    },
    # Depth 5 with slow learning
    'Depth5_Slow': {
        'max_depth': 5,
        'learning_rate': 0.008,
        'n_estimators': 4000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 7,
        'gamma': 0.15,
        'reg_alpha': 0.15,
        'reg_lambda': 1.5,
        'early_stopping_rounds': 100,
    },
}

results = []
test_preds = {}

for name, params in configs.items():
    print(f"\n[{name}]")

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist',
        eval_metric='auc',
        n_jobs=-1
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)

    # Train on full data
    model_full = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist',
        eval_metric='auc',
        n_jobs=-1
    )
    model_full.fit(X_train_final, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_test_pred = model_full.predict_proba(X_test_final)[:, 1]
    test_preds[name] = y_test_pred

    results.append({'model': name, 'val_auc': val_auc})
    print(f"  Val AUC: {val_auc:.4f}")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*80)
print("RESULTS")
print("="*80)

results_df = pd.DataFrame(results).sort_values('val_auc', ascending=False)

print("\nModel Rankings:")
for _, row in results_df.iterrows():
    marker = " <-- BEST" if row['val_auc'] == results_df['val_auc'].max() else ""
    print(f"  {row['model']:<20}: {row['val_auc']:.4f}{marker}")

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

for name, preds in test_preds.items():
    filename = f"opt_{name.lower()}.csv"
    submission = pd.DataFrame({
        'icustay_id': test_raw[id_col].values,
        'HOSPITAL_EXPIRE_FLAG': preds
    })
    output_file = BASE_DIR / "submissions" / filename
    submission.to_csv(output_file, index=False)
    print(f"✓ {name}: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print("\nSubmit the top performers to Kaggle to find the best!")
print("Remember: Val AUC doesn't always predict Kaggle score")
