"""
Blend Text Model with HighReg Model

Combines:
1. text__both.csv (0.753) - DIAGNOSIS keywords + TF-IDF
2. gen_balanced.csv or scratch best - HighReg vitals model

Different signal sources should be complementary.

Run from project root:
    python scripts/classification/blend_text_highreg.py
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
print("BLEND TEXT + HIGHREG MODELS")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# MODEL 1: HIGHREG (BASE FEATURES ONLY - NO TEXT)
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: HIGHREG (122 BASE FEATURES)")
print("="*80)

# Helper functions
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

# Apply feature engineering
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

# ============================================================================
# MODEL 2: TEXT FEATURES (KEYWORDS + TF-IDF)
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: TEXT FEATURES")
print("="*80)

train_dx = train_raw['DIAGNOSIS'].fillna('UNKNOWN').astype(str)
test_dx = test_raw['DIAGNOSIS'].fillna('UNKNOWN').astype(str)

def extract_diagnosis_keywords(diagnosis_series):
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

train_dx_features = extract_diagnosis_keywords(train_dx)
test_dx_features = extract_diagnosis_keywords(test_dx)

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

# Combine base + text
X_train_text = pd.concat([X_train_base.reset_index(drop=True),
                          train_dx_features.reset_index(drop=True),
                          train_tfidf_df.reset_index(drop=True)], axis=1)
X_test_text = pd.concat([X_test_base.reset_index(drop=True),
                         test_dx_features.reset_index(drop=True),
                         test_tfidf_df.reset_index(drop=True)], axis=1)

print(f"Text model features: {X_train_text.shape[1]}")

# ============================================================================
# TRAIN BOTH MODELS WITH K-FOLD
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODELS WITH K-FOLD CV")
print("="*80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# HighReg params
params = {
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

def train_model(X_train, X_test, name):
    oof_preds = np.zeros(len(X_train))
    test_preds_folds = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_scale = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()

        model = XGBClassifier(**params, scale_pos_weight=fold_scale, random_state=42)
        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)], verbose=False)

        oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        test_preds_folds.append(model.predict_proba(X_test)[:, 1])

    oof_auc = roc_auc_score(y_train, oof_preds)
    return oof_preds, np.mean(test_preds_folds, axis=0), oof_auc

# Train both models
print("\nTraining Model 1 (Base/HighReg)...")
oof_base, test_base, auc_base = train_model(X_train_base, X_test_base, "Base")
print(f"  OOF AUC: {auc_base:.4f}")

print("\nTraining Model 2 (Text)...")
oof_text, test_text, auc_text = train_model(X_train_text, X_test_text, "Text")
print(f"  OOF AUC: {auc_text:.4f}")

# ============================================================================
# BLEND WITH DIFFERENT WEIGHTS
# ============================================================================
print("\n" + "="*80)
print("BLENDING MODELS")
print("="*80)

# Test different blend weights
weights = [
    (0.5, 0.5),   # Equal
    (0.4, 0.6),   # More text
    (0.3, 0.7),   # Much more text
    (0.6, 0.4),   # More base
]

best_blend_auc = 0
best_weights = None
best_test_preds = None

print(f"\n{'Base Weight':>12} {'Text Weight':>12} {'OOF AUC':>10}")
print("-" * 40)

for w_base, w_text in weights:
    oof_blend = w_base * oof_base + w_text * oof_text
    blend_auc = roc_auc_score(y_train, oof_blend)
    print(f"{w_base:>12.1f} {w_text:>12.1f} {blend_auc:>10.4f}")

    if blend_auc > best_blend_auc:
        best_blend_auc = blend_auc
        best_weights = (w_base, w_text)
        best_test_preds = w_base * test_base + w_text * test_text

# Also try optimal weight search
print("\nSearching for optimal weight...")
best_w = 0.5
best_search_auc = 0
for w in np.arange(0.0, 1.01, 0.05):
    oof_blend = w * oof_base + (1-w) * oof_text
    blend_auc = roc_auc_score(y_train, oof_blend)
    if blend_auc > best_search_auc:
        best_search_auc = blend_auc
        best_w = w

print(f"Optimal: Base={best_w:.2f}, Text={1-best_w:.2f} -> OOF AUC: {best_search_auc:.4f}")

# Use optimal weights for final blend
final_test_preds = best_w * test_base + (1-best_w) * test_text

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nIndividual Models:")
print(f"  Base (HighReg): {auc_base:.4f}")
print(f"  Text (Keywords+TF-IDF): {auc_text:.4f}")
print(f"\nBest Blend: {best_search_auc:.4f}")
print(f"  Weights: Base={best_w:.2f}, Text={1-best_w:.2f}")

improvement = best_search_auc - max(auc_base, auc_text)
print(f"\nImprovement over best single: {improvement:+.4f}")

# ============================================================================
# SAVE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("SAVING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

# Save optimal blend
submission = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': final_test_preds
})
output_file = BASE_DIR / "submissions" / "blend_text_highreg.csv"
submission.to_csv(output_file, index=False)
print(f"Saved: {output_file.name}")

# Also save 50/50 blend for comparison
blend_5050 = 0.5 * test_base + 0.5 * test_text
submission_5050 = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': blend_5050
})
output_file_5050 = BASE_DIR / "submissions" / "blend_5050.csv"
submission_5050.to_csv(output_file_5050, index=False)
print(f"Saved: {output_file_5050.name}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
