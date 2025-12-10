"""
Blend Best Models - Combine Different Approaches

Hyperparameter tuning is giving diminishing returns.
Let's try blending fundamentally different models:
1. XGBoost variations with different characteristics
2. LightGBM (different algorithm)
3. Weighted blend based on OOF performance

Run from project root:
    python scripts/classification/blend_best.py
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

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available")

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
print("BLEND BEST MODELS - COMBINE DIFFERENT APPROACHES")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# FEATURE ENGINEERING
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
# DEFINE DIVERSE MODELS
# ============================================================================

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    # Your best Kaggle performer
    'XGB_HighReg': {
        'class': XGBClassifier,
        'params': {
            'max_depth': 4,
            'learning_rate': 0.01,
            'n_estimators': 3000,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'gamma': 0.2,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'auc',
            'early_stopping_rounds': 50,
            'n_jobs': -1
        }
    },
    # Best generalization
    'XGB_Balanced': {
        'class': XGBClassifier,
        'params': {
            'max_depth': 4,
            'learning_rate': 0.01,
            'n_estimators': 3000,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 8,
            'gamma': 0.15,
            'reg_alpha': 0.2,
            'reg_lambda': 1.5,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'auc',
            'early_stopping_rounds': 50,
            'n_jobs': -1
        }
    },
    # Different seed for diversity
    'XGB_Seed123': {
        'class': XGBClassifier,
        'params': {
            'max_depth': 4,
            'learning_rate': 0.01,
            'n_estimators': 3000,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 8,
            'gamma': 0.15,
            'reg_alpha': 0.2,
            'reg_lambda': 1.5,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 123,
            'tree_method': 'hist',
            'eval_metric': 'auc',
            'early_stopping_rounds': 50,
            'n_jobs': -1
        }
    },
    # Deeper trees (different perspective)
    'XGB_Deep': {
        'class': XGBClassifier,
        'params': {
            'max_depth': 6,
            'learning_rate': 0.008,
            'n_estimators': 3500,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'gamma': 0.2,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'auc',
            'early_stopping_rounds': 50,
            'n_jobs': -1
        }
    },
}

# Add LightGBM if available
if HAS_LGBM:
    models['LightGBM'] = {
        'class': LGBMClassifier,
        'params': {
            'max_depth': 4,
            'learning_rate': 0.01,
            'n_estimators': 3000,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_samples': 20,
            'reg_alpha': 0.2,
            'reg_lambda': 1.5,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    }

# ============================================================================
# TRAIN ALL MODELS WITH K-FOLD
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODELS WITH K-FOLD CV")
print("="*80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

model_results = {}

for name, config in models.items():
    print(f"\n[{name}]")

    oof_preds = np.zeros(len(X_train_final))
    test_preds_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_final, y_train)):
        X_fold_train = X_train_final.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_final.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Update scale_pos_weight for this fold
        params = config['params'].copy()
        if 'scale_pos_weight' in params:
            params['scale_pos_weight'] = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()

        model = config['class'](**params)

        if 'XGB' in name:
            model.fit(X_fold_train, y_fold_train,
                      eval_set=[(X_fold_val, y_fold_val)], verbose=False)
        else:
            model.fit(X_fold_train, y_fold_train,
                      eval_set=[(X_fold_val, y_fold_val)])

        oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        test_preds_folds.append(model.predict_proba(X_test_final)[:, 1])

    oof_auc = roc_auc_score(y_train, oof_preds)
    test_preds_avg = np.mean(test_preds_folds, axis=0)

    model_results[name] = {
        'oof_preds': oof_preds,
        'test_preds': test_preds_avg,
        'oof_auc': oof_auc
    }

    print(f"  OOF AUC: {oof_auc:.4f}")

# ============================================================================
# CREATE BLENDS
# ============================================================================
print("\n" + "="*80)
print("CREATING BLENDS")
print("="*80)

blends = {}

# Simple average of all
all_oof = np.mean([r['oof_preds'] for r in model_results.values()], axis=0)
all_test = np.mean([r['test_preds'] for r in model_results.values()], axis=0)
blends['Avg_All'] = {
    'oof_auc': roc_auc_score(y_train, all_oof),
    'test_preds': all_test
}

# Average of XGB only
xgb_names = [n for n in model_results.keys() if 'XGB' in n]
xgb_oof = np.mean([model_results[n]['oof_preds'] for n in xgb_names], axis=0)
xgb_test = np.mean([model_results[n]['test_preds'] for n in xgb_names], axis=0)
blends['Avg_XGB'] = {
    'oof_auc': roc_auc_score(y_train, xgb_oof),
    'test_preds': xgb_test
}

# Weighted by OOF AUC
weights = np.array([r['oof_auc'] for r in model_results.values()])
weights = weights / weights.sum()
weighted_oof = np.average([r['oof_preds'] for r in model_results.values()], axis=0, weights=weights)
weighted_test = np.average([r['test_preds'] for r in model_results.values()], axis=0, weights=weights)
blends['Weighted'] = {
    'oof_auc': roc_auc_score(y_train, weighted_oof),
    'test_preds': weighted_test
}

# Top 2 blend (HighReg + Balanced)
top2_oof = (model_results['XGB_HighReg']['oof_preds'] + model_results['XGB_Balanced']['oof_preds']) / 2
top2_test = (model_results['XGB_HighReg']['test_preds'] + model_results['XGB_Balanced']['test_preds']) / 2
blends['Top2_Avg'] = {
    'oof_auc': roc_auc_score(y_train, top2_oof),
    'test_preds': top2_test
}

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print("\nIndividual Models:")
for name, result in sorted(model_results.items(), key=lambda x: x[1]['oof_auc'], reverse=True):
    print(f"  {name:<20}: {result['oof_auc']:.4f}")

print("\nBlends:")
for name, result in sorted(blends.items(), key=lambda x: x[1]['oof_auc'], reverse=True):
    print(f"  {name:<20}: {result['oof_auc']:.4f}")

# Find best overall
all_results = {**{k: v['oof_auc'] for k, v in model_results.items()},
               **{k: v['oof_auc'] for k, v in blends.items()}}
best_name = max(all_results, key=all_results.get)
best_auc = all_results[best_name]

print(f"\n🏆 Best Overall: {best_name} (OOF AUC: {best_auc:.4f})")

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

# Save all blends
for name, result in blends.items():
    filename = f"blend_{name.lower()}.csv"
    submission = pd.DataFrame({
        'icustay_id': test_raw[id_col].values,
        'HOSPITAL_EXPIRE_FLAG': result['test_preds']
    })
    output_file = BASE_DIR / "submissions" / filename
    submission.to_csv(output_file, index=False)
    print(f"✓ {name}: {output_file}")

# Save best individual
best_individual = max(model_results.items(), key=lambda x: x[1]['oof_auc'])
filename = f"best_{best_individual[0].lower()}.csv"
submission = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': best_individual[1]['test_preds']
})
output_file = BASE_DIR / "submissions" / filename
submission.to_csv(output_file, index=False)
print(f"✓ {best_individual[0]}: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
