"""
Text Features Model - Extract Signal from DIAGNOSIS Field

The DIAGNOSIS field has 6,193 unique free-text values that are currently ignored.
This script:
1. Extracts meaningful keywords/phrases from DIAGNOSIS
2. Creates TF-IDF features or keyword flags
3. Combines with best features from scratch.py

Run from project root:
    python scripts/classification/text_features.py
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
print("TEXT FEATURES MODEL - EXTRACT DIAGNOSIS SIGNAL")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# ANALYZE DIAGNOSIS FIELD
# ============================================================================
print("\n" + "="*80)
print("ANALYZING DIAGNOSIS TEXT FIELD")
print("="*80)

train_dx = train_raw['DIAGNOSIS'].fillna('UNKNOWN').astype(str)
test_dx = test_raw['DIAGNOSIS'].fillna('UNKNOWN').astype(str)
y_train_full = train_raw['HOSPITAL_EXPIRE_FLAG']

print(f"\nUnique DIAGNOSIS values: {train_dx.nunique()}")

# Look at mortality by diagnosis
dx_stats = pd.DataFrame({
    'diagnosis': train_dx,
    'expired': y_train_full
}).groupby('diagnosis').agg(
    count=('expired', 'count'),
    mortality=('expired', 'mean')
).reset_index()

# High mortality diagnoses
high_mort_dx = dx_stats[
    (dx_stats['mortality'] > 0.25) &
    (dx_stats['count'] >= 20)
].sort_values('mortality', ascending=False)

print(f"\nHigh mortality diagnoses (>25%, n≥20):")
for _, row in high_mort_dx.head(20).iterrows():
    print(f"  {row['count']:4d} | {row['mortality']:.1%} | {row['diagnosis'][:60]}")

# ============================================================================
# EXTRACT KEYWORD FEATURES FROM DIAGNOSIS
# ============================================================================
print("\n" + "="*80)
print("EXTRACTING KEYWORD FEATURES")
print("="*80)

def extract_diagnosis_keywords(diagnosis_series):
    """Extract clinically meaningful keywords from diagnosis text."""
    df = pd.DataFrame(index=diagnosis_series.index)

    # Convert to lowercase for matching
    dx_lower = diagnosis_series.str.lower()

    # Critical conditions (high mortality)
    df['dx_sepsis'] = dx_lower.str.contains(r'sepsis|septic', na=False).astype(int)
    df['dx_respiratory_failure'] = dx_lower.str.contains(r'respiratory failure|resp.*fail|ards', na=False).astype(int)
    df['dx_cardiac_arrest'] = dx_lower.str.contains(r'cardiac arrest|v-?fib|asystole|code', na=False).astype(int)
    df['dx_shock'] = dx_lower.str.contains(r'\bshock\b|cardiogenic|hypovolemic', na=False).astype(int)
    df['dx_multi_organ'] = dx_lower.str.contains(r'multi.*organ|mods|mof', na=False).astype(int)

    # Respiratory conditions
    df['dx_pneumonia'] = dx_lower.str.contains(r'pneumonia|pna\b', na=False).astype(int)
    df['dx_copd'] = dx_lower.str.contains(r'copd|emphysema|chronic.*obstruct', na=False).astype(int)
    df['dx_intubation'] = dx_lower.str.contains(r'intubat|ventilat|mechanical vent', na=False).astype(int)

    # Cardiac conditions
    df['dx_mi'] = dx_lower.str.contains(r'myocardial infarct|mi\b|stemi|nstemi|heart attack', na=False).astype(int)
    df['dx_chf'] = dx_lower.str.contains(r'heart failure|chf|congestive', na=False).astype(int)
    df['dx_arrhythmia'] = dx_lower.str.contains(r'arrhythm|a-?fib|atrial fib|v-?tach', na=False).astype(int)
    df['dx_cabg'] = dx_lower.str.contains(r'cabg|bypass|coronary artery bypass', na=False).astype(int)

    # Neurological
    df['dx_stroke'] = dx_lower.str.contains(r'stroke|cva|cerebrovascular|intracranial', na=False).astype(int)
    df['dx_altered_mental'] = dx_lower.str.contains(r'altered mental|encephalop|confusion|delirium', na=False).astype(int)

    # GI/Abdominal
    df['dx_gi_bleed'] = dx_lower.str.contains(r'gi bleed|gastrointestinal bleed|melena|hematemesis', na=False).astype(int)
    df['dx_liver'] = dx_lower.str.contains(r'liver|hepat|cirrhosis', na=False).astype(int)
    df['dx_pancreatitis'] = dx_lower.str.contains(r'pancreatit', na=False).astype(int)

    # Renal
    df['dx_renal_failure'] = dx_lower.str.contains(r'renal failure|kidney fail|arf|acute kidney|dialysis', na=False).astype(int)

    # Trauma/Surgery
    df['dx_trauma'] = dx_lower.str.contains(r'trauma|fracture|injury|mva|fall', na=False).astype(int)
    df['dx_surgery'] = dx_lower.str.contains(r'post.*op|surgery|surgical|s/p\b', na=False).astype(int)

    # Cancer
    df['dx_cancer'] = dx_lower.str.contains(r'cancer|carcinoma|malign|tumor|metasta|leukemia|lymphoma', na=False).astype(int)

    # Infections
    df['dx_infection'] = dx_lower.str.contains(r'infection|infected|abscess|cellulitis', na=False).astype(int)

    # Diabetes
    df['dx_diabetes'] = dx_lower.str.contains(r'diabet|dka|hyperglyc', na=False).astype(int)

    # Overdose/Toxicity
    df['dx_overdose'] = dx_lower.str.contains(r'overdose|toxic|poison|intoxicat', na=False).astype(int)

    # Count critical keywords
    critical_cols = ['dx_sepsis', 'dx_respiratory_failure', 'dx_cardiac_arrest',
                     'dx_shock', 'dx_multi_organ']
    df['dx_critical_count'] = df[critical_cols].sum(axis=1)
    df['dx_any_critical'] = (df['dx_critical_count'] > 0).astype(int)

    return df

# Extract keyword features
print("\nExtracting keyword features...")
train_dx_features = extract_diagnosis_keywords(train_dx)
test_dx_features = extract_diagnosis_keywords(test_dx)

# Show keyword prevalence and mortality
print("\nKeyword prevalence and mortality:")
keyword_stats = []
for col in train_dx_features.columns:
    if col.startswith('dx_') and col not in ['dx_critical_count', 'dx_any_critical']:
        mask = train_dx_features[col] == 1
        count = mask.sum()
        if count > 0:
            mortality = y_train_full[mask].mean()
            keyword_stats.append({
                'keyword': col.replace('dx_', ''),
                'count': count,
                'mortality': mortality
            })

keyword_stats = pd.DataFrame(keyword_stats).sort_values('mortality', ascending=False)
print(f"\n{'Keyword':<20} {'Count':>8} {'Mortality':>10}")
print("-" * 40)
for _, row in keyword_stats.iterrows():
    print(f"{row['keyword']:<20} {row['count']:>8} {row['mortality']:>10.1%}")

# ============================================================================
# ALSO TRY TF-IDF (TOP TERMS ONLY)
# ============================================================================
print("\n" + "="*80)
print("TF-IDF FEATURES (TOP 50 TERMS)")
print("="*80)

# Clean and vectorize
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_dx_clean = train_dx.apply(clean_text)
test_dx_clean = test_dx.apply(clean_text)

# TF-IDF with limited features
tfidf = TfidfVectorizer(
    max_features=50,  # Only top 50 terms
    min_df=10,        # Must appear in at least 10 docs
    max_df=0.5,       # Can't be in more than 50% of docs
    ngram_range=(1, 2)  # Unigrams and bigrams
)

train_tfidf = tfidf.fit_transform(train_dx_clean)
test_tfidf = tfidf.transform(test_dx_clean)

# Convert to DataFrame
tfidf_cols = [f'tfidf_{term}' for term in tfidf.get_feature_names_out()]
train_tfidf_df = pd.DataFrame(
    train_tfidf.toarray(),
    columns=tfidf_cols,
    index=train_raw.index
)
test_tfidf_df = pd.DataFrame(
    test_tfidf.toarray(),
    columns=tfidf_cols,
    index=test_raw.index
)

print(f"TF-IDF features: {len(tfidf_cols)}")
print(f"Top terms: {tfidf.get_feature_names_out()[:10]}")

# ============================================================================
# PREPARE FULL FEATURE SET (BASE + TEXT)
# ============================================================================
print("\n" + "="*80)
print("PREPARING FEATURE SET")
print("="*80)

# Helper functions from scratch.py
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

# Add ICD9 (simple encoding)
if 'ICD9_diagnosis' in train_raw.columns:
    train_icd9 = train_raw['ICD9_diagnosis'].fillna('MISSING')
    test_icd9 = test_raw['ICD9_diagnosis'].fillna('MISSING')
    le = LabelEncoder()
    le.fit(pd.concat([train_icd9, test_icd9]))
    X_train_fe['ICD9_encoded'] = le.transform(train_icd9)
    X_test_fe['ICD9_encoded'] = le.transform(test_icd9)

    # Also add ICD9 category (first 3 digits)
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

print(f"Base features: {X_train_fe.shape[1]}")

# ============================================================================
# TEST DIFFERENT FEATURE COMBINATIONS
# ============================================================================
print("\n" + "="*80)
print("TESTING FEATURE COMBINATIONS")
print("="*80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Best params from HighReg
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

def evaluate_features(X_train, X_test, name):
    """Evaluate feature set with K-fold CV."""
    oof_preds = np.zeros(len(X_train))
    test_preds_folds = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_scale = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()

        model = XGBClassifier(
            **params,
            scale_pos_weight=fold_scale,
            random_state=42
        )

        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)], verbose=False)

        oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)

        test_preds_folds.append(model.predict_proba(X_test)[:, 1])

    oof_auc = roc_auc_score(y_train, oof_preds)
    fold_std = np.std(fold_aucs)
    gen_score = oof_auc - fold_std

    return oof_auc, fold_std, gen_score, np.mean(test_preds_folds, axis=0)

# 1. Base only (122 features)
print("\n[1] Base Features Only")
oof_base, std_base, gen_base, preds_base = evaluate_features(X_train_fe, X_test_fe, "Base")
print(f"   Features: {X_train_fe.shape[1]}")
print(f"   OOF AUC: {oof_base:.4f}, Std: {std_base:.4f}, Gen: {gen_base:.4f}")

# 2. Base + Keyword features
print("\n[2] Base + Keyword Features")
X_train_kw = pd.concat([X_train_fe.reset_index(drop=True), train_dx_features.reset_index(drop=True)], axis=1)
X_test_kw = pd.concat([X_test_fe.reset_index(drop=True), test_dx_features.reset_index(drop=True)], axis=1)
oof_kw, std_kw, gen_kw, preds_kw = evaluate_features(X_train_kw, X_test_kw, "Keywords")
print(f"   Features: {X_train_kw.shape[1]}")
print(f"   OOF AUC: {oof_kw:.4f}, Std: {std_kw:.4f}, Gen: {gen_kw:.4f}")

# 3. Base + TF-IDF features
print("\n[3] Base + TF-IDF Features")
X_train_tfidf = pd.concat([X_train_fe.reset_index(drop=True), train_tfidf_df.reset_index(drop=True)], axis=1)
X_test_tfidf = pd.concat([X_test_fe.reset_index(drop=True), test_tfidf_df.reset_index(drop=True)], axis=1)
oof_tfidf, std_tfidf, gen_tfidf, preds_tfidf = evaluate_features(X_train_tfidf, X_test_tfidf, "TF-IDF")
print(f"   Features: {X_train_tfidf.shape[1]}")
print(f"   OOF AUC: {oof_tfidf:.4f}, Std: {std_tfidf:.4f}, Gen: {gen_tfidf:.4f}")

# 4. Base + Keywords + TF-IDF
print("\n[4] Base + Keywords + TF-IDF")
X_train_all = pd.concat([X_train_fe.reset_index(drop=True),
                         train_dx_features.reset_index(drop=True),
                         train_tfidf_df.reset_index(drop=True)], axis=1)
X_test_all = pd.concat([X_test_fe.reset_index(drop=True),
                        test_dx_features.reset_index(drop=True),
                        test_tfidf_df.reset_index(drop=True)], axis=1)
oof_all, std_all, gen_all, preds_all = evaluate_features(X_train_all, X_test_all, "All")
print(f"   Features: {X_train_all.shape[1]}")
print(f"   OOF AUC: {oof_all:.4f}, Std: {std_all:.4f}, Gen: {gen_all:.4f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results = [
    ('Base (122)', X_train_fe.shape[1], oof_base, std_base, gen_base, preds_base),
    ('+ Keywords', X_train_kw.shape[1], oof_kw, std_kw, gen_kw, preds_kw),
    ('+ TF-IDF', X_train_tfidf.shape[1], oof_tfidf, std_tfidf, gen_tfidf, preds_tfidf),
    ('+ Both', X_train_all.shape[1], oof_all, std_all, gen_all, preds_all),
]

print(f"\n{'Model':<15} {'Features':>10} {'OOF AUC':>10} {'Fold Std':>10} {'Gen Score':>10}")
print("-" * 60)
for name, n_feat, oof, std, gen, _ in results:
    print(f"{name:<15} {n_feat:>10} {oof:>10.4f} {std:>10.4f} {gen:>10.4f}")

# Find best
best_idx = np.argmax([r[4] for r in results])  # By gen score
best_name = results[best_idx][0]
best_preds = results[best_idx][5]

print(f"\nBest by Gen Score: {best_name}")

# ============================================================================
# SAVE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("SAVING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

# Save all variations
for name, n_feat, oof, std, gen, preds in results:
    clean_name = name.replace('(', '').replace(')', '').replace(' ', '').replace('+', '_').lower()
    filename = f"text_{clean_name}.csv"
    submission = pd.DataFrame({
        'icustay_id': test_raw[id_col].values,
        'HOSPITAL_EXPIRE_FLAG': preds
    })
    output_file = BASE_DIR / "submissions" / filename
    submission.to_csv(output_file, index=False)
    print(f"Saved: {output_file.name} (Gen: {gen:.4f})")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nSubmit the best performing model to Kaggle.")
