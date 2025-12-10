"""
LightGBM with Simple Keywords - Test if LightGBM improves on XGBoost

Use the original simpler keyword set that gave 0.8585 OOF AUC.
LightGBM may capture different patterns than XGBoost.

Run from project root:
    python scripts/classification/lgbm_simple_keywords.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import lightgbm as lgb
import re
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
print("LIGHTGBM WITH SIMPLE KEYWORDS")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# SIMPLE KEYWORD EXTRACTION (same as text_features.py)
# ============================================================================

train_dx = train_raw['DIAGNOSIS'].fillna('UNKNOWN').astype(str)
test_dx = test_raw['DIAGNOSIS'].fillna('UNKNOWN').astype(str)

def extract_diagnosis_keywords(diagnosis_series):
    """Extract clinically meaningful keywords from diagnosis text."""
    df = pd.DataFrame(index=diagnosis_series.index)
    dx_lower = diagnosis_series.str.lower()

    # Critical conditions
    df['dx_sepsis'] = dx_lower.str.contains(r'sepsis|septic', na=False).astype(int)
    df['dx_respiratory_failure'] = dx_lower.str.contains(r'respiratory failure|resp.*fail|ards', na=False).astype(int)
    df['dx_cardiac_arrest'] = dx_lower.str.contains(r'cardiac arrest|v-?fib|asystole|code', na=False).astype(int)
    df['dx_shock'] = dx_lower.str.contains(r'\bshock\b|cardiogenic|hypovolemic', na=False).astype(int)
    df['dx_multi_organ'] = dx_lower.str.contains(r'multi.*organ|mods|mof', na=False).astype(int)

    # Respiratory
    df['dx_pneumonia'] = dx_lower.str.contains(r'pneumonia|pna\b', na=False).astype(int)
    df['dx_copd'] = dx_lower.str.contains(r'copd|emphysema|chronic.*obstruct', na=False).astype(int)
    df['dx_intubation'] = dx_lower.str.contains(r'intubat|ventilat|mechanical vent', na=False).astype(int)

    # Cardiac
    df['dx_mi'] = dx_lower.str.contains(r'myocardial infarct|mi\b|stemi|nstemi|heart attack', na=False).astype(int)
    df['dx_chf'] = dx_lower.str.contains(r'heart failure|chf|congestive', na=False).astype(int)
    df['dx_arrhythmia'] = dx_lower.str.contains(r'arrhythm|a-?fib|atrial fib|v-?tach', na=False).astype(int)
    df['dx_cabg'] = dx_lower.str.contains(r'cabg|bypass|coronary artery bypass', na=False).astype(int)

    # Neurological
    df['dx_stroke'] = dx_lower.str.contains(r'stroke|cva|cerebrovascular|intracranial', na=False).astype(int)
    df['dx_altered_mental'] = dx_lower.str.contains(r'altered mental|encephalop|confusion|delirium', na=False).astype(int)

    # GI
    df['dx_gi_bleed'] = dx_lower.str.contains(r'gi bleed|gastrointestinal bleed|melena|hematemesis', na=False).astype(int)
    df['dx_liver'] = dx_lower.str.contains(r'liver|hepat|cirrhosis', na=False).astype(int)
    df['dx_pancreatitis'] = dx_lower.str.contains(r'pancreatit', na=False).astype(int)

    # Renal
    df['dx_renal_failure'] = dx_lower.str.contains(r'renal failure|kidney fail|arf|acute kidney|dialysis', na=False).astype(int)

    # Other
    df['dx_trauma'] = dx_lower.str.contains(r'trauma|fracture|injury|mva|fall', na=False).astype(int)
    df['dx_surgery'] = dx_lower.str.contains(r'post.*op|surgery|surgical|s/p\b', na=False).astype(int)
    df['dx_cancer'] = dx_lower.str.contains(r'cancer|carcinoma|malign|tumor|metasta|leukemia|lymphoma', na=False).astype(int)
    df['dx_infection'] = dx_lower.str.contains(r'infection|infected|abscess|cellulitis', na=False).astype(int)
    df['dx_diabetes'] = dx_lower.str.contains(r'diabet|dka|hyperglyc', na=False).astype(int)
    df['dx_overdose'] = dx_lower.str.contains(r'overdose|toxic|poison|intoxicat', na=False).astype(int)

    # Aggregates
    critical_cols = ['dx_sepsis', 'dx_respiratory_failure', 'dx_cardiac_arrest', 'dx_shock', 'dx_multi_organ']
    df['dx_critical_count'] = df[critical_cols].sum(axis=1)
    df['dx_any_critical'] = (df['dx_critical_count'] > 0).astype(int)

    return df

print("\nExtracting simple keyword features...")
train_dx_features = extract_diagnosis_keywords(train_dx)
test_dx_features = extract_diagnosis_keywords(test_dx)
print(f"Keyword features: {train_dx_features.shape[1]}")

# ============================================================================
# BASE FEATURES
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

# Prepare base features
leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

for col in [c for c in X_train_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_test_raw = X_test_raw.drop(columns=[col])

X_train_base = add_missing_indicators(X_train_raw)
X_test_base = add_missing_indicators(X_test_raw)
X_train_base = add_extreme_flags(X_train_base)
X_test_base = add_extreme_flags(X_test_base)
X_train_base = add_age_features(X_train_base)
X_test_base = add_age_features(X_test_base)
X_train_base = clean_min_bp_outliers(X_train_base)
X_test_base = clean_min_bp_outliers(X_test_base)
X_train_base = add_engineered_features(X_train_base)
X_test_base = add_engineered_features(X_test_base)
X_train_base = add_age_interactions(X_train_base)
X_test_base = add_age_interactions(X_test_base)

# Add ICD9
if 'ICD9_diagnosis' in train_raw.columns:
    train_icd9 = train_raw['ICD9_diagnosis'].fillna('MISSING')
    test_icd9 = test_raw['ICD9_diagnosis'].fillna('MISSING')
    le = LabelEncoder()
    le.fit(pd.concat([train_icd9, test_icd9]))
    X_train_base['ICD9_encoded'] = le.transform(train_icd9)
    X_test_base['ICD9_encoded'] = le.transform(test_icd9)
    train_cat = train_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    test_cat = test_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    le_cat = LabelEncoder()
    le_cat.fit(pd.concat([train_cat, test_cat]))
    X_train_base['ICD9_category'] = le_cat.transform(train_cat)
    X_test_base['ICD9_category'] = le_cat.transform(test_cat)

# Encode categoricals
cat_cols = X_train_base.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_base[col], X_test_base[col]]).astype(str)
    le_c.fit(combined)
    X_train_base[col] = le_c.transform(X_train_base[col].astype(str))
    X_test_base[col] = le_c.transform(X_test_base[col].astype(str))

print(f"Base features: {X_train_base.shape[1]}")

# TF-IDF
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_dx_clean = train_dx.apply(clean_text)
test_dx_clean = test_dx.apply(clean_text)

tfidf = TfidfVectorizer(max_features=50, min_df=10, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(train_dx_clean)
test_tfidf = tfidf.transform(test_dx_clean)

tfidf_cols = [f'tfidf_{term}' for term in tfidf.get_feature_names_out()]
train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_cols, index=train_raw.index)
test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=tfidf_cols, index=test_raw.index)

# Combine all
X_train_full = pd.concat([X_train_base.reset_index(drop=True),
                          train_dx_features.reset_index(drop=True),
                          train_tfidf_df.reset_index(drop=True)], axis=1)
X_test_full = pd.concat([X_test_base.reset_index(drop=True),
                         test_dx_features.reset_index(drop=True),
                         test_tfidf_df.reset_index(drop=True)], axis=1)

print(f"Total features: {X_train_full.shape[1]}")

# ============================================================================
# K-FOLD CV - COMPARE XGB VS LGB VS BLEND
# ============================================================================
print("\n" + "="*80)
print("K-FOLD CROSS-VALIDATION")
print("="*80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

results = []

# XGBoost params
xgb_params = {
    'max_depth': 4,
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'gamma': 0.2,
    'reg_alpha': 0.3,
    'reg_lambda': 2.0,
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'early_stopping_rounds': 50,
    'n_jobs': -1
}

# LightGBM params - tuned for this dataset
lgb_params_list = [
    {
        'name': 'LGB_Default',
        'params': {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.01,
            'n_estimators': 3000,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_samples': 20,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'verbose': -1,
            'n_jobs': -1
        }
    },
    {
        'name': 'LGB_MoreLeaves',
        'params': {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 3000,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_samples': 30,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'verbose': -1,
            'n_jobs': -1
        }
    },
    {
        'name': 'LGB_DART',
        'params': {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'dart',
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.02,
            'n_estimators': 1500,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_samples': 20,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'drop_rate': 0.1,
            'verbose': -1,
            'n_jobs': -1
        }
    }
]

# 1. XGBoost
print("\n[1] XGBoost")
oof_xgb = np.zeros(len(X_train_full))
test_preds_xgb = []
fold_aucs_xgb = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train)):
    X_fold_train = X_train_full.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train_full.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    fold_scale = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()

    model = XGBClassifier(**xgb_params, scale_pos_weight=fold_scale, random_state=42)
    model.fit(X_fold_train, y_fold_train,
              eval_set=[(X_fold_val, y_fold_val)], verbose=False)

    oof_xgb[val_idx] = model.predict_proba(X_fold_val)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, oof_xgb[val_idx])
    fold_aucs_xgb.append(fold_auc)
    test_preds_xgb.append(model.predict_proba(X_test_full)[:, 1])

oof_auc_xgb = roc_auc_score(y_train, oof_xgb)
std_xgb = np.std(fold_aucs_xgb)
print(f"   OOF AUC: {oof_auc_xgb:.4f}, Std: {std_xgb:.4f}")
results.append(('XGBoost', oof_auc_xgb, std_xgb, np.mean(test_preds_xgb, axis=0), oof_xgb))

# 2. LightGBM variants
oof_lgb_best = None
best_lgb_auc = 0
best_lgb_name = None

for lgb_config in lgb_params_list:
    name = lgb_config['name']
    params = lgb_config['params']

    print(f"\n[{name}]")
    oof_lgb = np.zeros(len(X_train_full))
    test_preds_lgb = []
    fold_aucs_lgb = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train)):
        X_fold_train = X_train_full.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_full.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_scale = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()

        model = lgb.LGBMClassifier(**params, scale_pos_weight=fold_scale, random_state=42)

        if params['boosting_type'] == 'dart':
            # DART doesn't support early stopping
            model.fit(X_fold_train, y_fold_train)
        else:
            model.fit(X_fold_train, y_fold_train,
                      eval_set=[(X_fold_val, y_fold_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

        oof_lgb[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, oof_lgb[val_idx])
        fold_aucs_lgb.append(fold_auc)
        test_preds_lgb.append(model.predict_proba(X_test_full)[:, 1])

    oof_auc_lgb = roc_auc_score(y_train, oof_lgb)
    std_lgb = np.std(fold_aucs_lgb)
    print(f"   OOF AUC: {oof_auc_lgb:.4f}, Std: {std_lgb:.4f}")
    results.append((name, oof_auc_lgb, std_lgb, np.mean(test_preds_lgb, axis=0), oof_lgb))

    if oof_auc_lgb > best_lgb_auc:
        best_lgb_auc = oof_auc_lgb
        best_lgb_name = name
        oof_lgb_best = oof_lgb
        test_lgb_best = np.mean(test_preds_lgb, axis=0)

# 3. Blend XGB + Best LGB
print("\n[Blend XGB + Best LGB]")
best_w = 0.5
best_blend_auc = 0
for w in np.arange(0.0, 1.01, 0.05):
    blend = w * oof_xgb + (1-w) * oof_lgb_best
    auc = roc_auc_score(y_train, blend)
    if auc > best_blend_auc:
        best_blend_auc = auc
        best_w = w

print(f"   Optimal: XGB={best_w:.2f}, {best_lgb_name}={1-best_w:.2f}")
print(f"   OOF AUC: {best_blend_auc:.4f}")

test_blend = best_w * np.mean(test_preds_xgb, axis=0) + (1-best_w) * test_lgb_best
results.append(('Blend_XGB_LGB', best_blend_auc, 0, test_blend, None))

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print(f"\n{'Model':<20} {'OOF AUC':>10} {'Fold Std':>10} {'Gen Score':>10}")
print("-" * 55)
for name, auc, std, _, _ in results:
    gen = auc - std if std > 0 else auc
    print(f"{name:<20} {auc:>10.4f} {std:>10.4f} {gen:>10.4f}")

print(f"\nPrevious best (text__both): 0.8585 OOF AUC -> 0.753 Kaggle")
best_result = max(results, key=lambda x: x[1])
diff = best_result[1] - 0.8585
print(f"Best this run: {best_result[0]} = {best_result[1]:.4f} ({diff:+.4f})")

# ============================================================================
# SAVE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("SAVING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

for name, auc, std, preds, _ in results:
    clean_name = name.lower().replace('_', '-')
    filename = f"lgbm_{clean_name}.csv"
    submission = pd.DataFrame({
        'icustay_id': test_raw[id_col].values,
        'HOSPITAL_EXPIRE_FLAG': preds
    })
    output_file = BASE_DIR / "submissions" / filename
    submission.to_csv(output_file, index=False)
    print(f"Saved: {output_file.name} (OOF: {auc:.4f})")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
