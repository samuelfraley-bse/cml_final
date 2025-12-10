"""
XGBoost V3 - Enhanced Diagnosis Features

Changes from V2:
1. ✓ Enhanced diagnosis features (ALL diagnoses, not just primary):
   - Counts per chapter (respiratory_dx_count, circulatory_dx_count, etc.)
   - Top 2 diagnosis chapters for target encoding (primary + secondary)
   - Proven metrics (total count, diversity, flags)

2. ✓ Core engineered features only (no flags, no temporal):
   - Uses add_engineered_features_core() instead of add_engineered_features()
   - Removes 0.000 importance features

3. ✓ Same regularization as V2 (proven to generalize)

Run from project root:
    python scripts/classification/xgb_v3_enhanced_diagnosis.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

from category_encoders import TargetEncoder
from xgboost import XGBClassifier

# Setup paths
BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

print("="*80)
print("XGBOOST V3 - ENHANCED DIAGNOSIS FEATURES")
print("="*80)
print("\n✨ New in V3:")
print("  ✓ ALL diagnosis codes (not just primary)")
print("  ✓ Counts per chapter (respiratory_dx_count, etc.)")
print("  ✓ Secondary diagnosis for target encoding")
print("  ✓ Core features only (no flags, no temporal)")
print()

# ============================================================================
# HELPER FUNCTIONS (same as V2)
# ============================================================================

def get_icd9_chapter(code):
    """Map ICD9 code to broad chapter"""
    if pd.isna(code) or code == 'MISSING':
        return 'MISSING'
    code_str = str(code).strip().upper()
    if code_str.startswith('V'):
        return 'V_CODES'
    if code_str.startswith('E'):
        return 'E_CODES'
    try:
        code_num = int(code_str.replace('.', '')[:3])
    except (ValueError, IndexError):
        return 'UNKNOWN'
    
    if 1 <= code_num <= 139:
        return 'INFECTIOUS'
    elif 140 <= code_num <= 239:
        return 'NEOPLASMS'
    elif 240 <= code_num <= 279:
        return 'ENDOCRINE'
    elif 280 <= code_num <= 289:
        return 'BLOOD'
    elif 290 <= code_num <= 319:
        return 'MENTAL'
    elif 320 <= code_num <= 389:
        return 'NERVOUS'
    elif 390 <= code_num <= 459:
        return 'CIRCULATORY'
    elif 460 <= code_num <= 519:
        return 'RESPIRATORY'
    elif 520 <= code_num <= 579:
        return 'DIGESTIVE'
    elif 580 <= code_num <= 629:
        return 'GENITOURINARY'
    elif 630 <= code_num <= 679:
        return 'PREGNANCY'
    elif 680 <= code_num <= 709:
        return 'SKIN'
    elif 710 <= code_num <= 739:
        return 'MUSCULOSKELETAL'
    elif 740 <= code_num <= 759:
        return 'CONGENITAL'
    elif 760 <= code_num <= 779:
        return 'PERINATAL'
    elif 780 <= code_num <= 799:
        return 'SYMPTOMS'
    elif 800 <= code_num <= 999:
        return 'INJURY'
    else:
        return 'UNKNOWN'


def load_diagnosis_data(filepath=None):
    """Load MIMIC_diagnoses.csv"""
    if filepath is None:
        base_dir = Path.cwd()
        filepath = base_dir / "data" / "raw" / "MIMIC III dataset HEF" / "extra_data" / "MIMIC_diagnoses.csv"
    filepath = Path(filepath)
    if not filepath.exists():
        base_dir = Path.cwd()
        raise FileNotFoundError(f"MIMIC_diagnoses.csv not found at {filepath}")
    
    print(f"Loading diagnosis data from: {filepath}")
    diagnosis_df = pd.read_csv(filepath)
    diagnosis_df.columns = diagnosis_df.columns.str.upper()
    if 'HADM_ID' in diagnosis_df.columns:
        diagnosis_df = diagnosis_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"  ✓ Loaded {len(diagnosis_df):,} diagnosis records")
    print(f"  ✓ Unique admissions: {diagnosis_df['hadm_id'].nunique():,}")
    return diagnosis_df


def prepare_enhanced_diagnosis_features(main_df, diagnosis_df, hadm_col='hadm_id'):
    """Enhanced diagnosis features - V3 (same as helper file)"""
    df = main_df.copy()
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        print(f"⚠️  Warning: {hadm_col} not found")
        df['primary_diagnosis_chapter'] = 'MISSING'
        df['secondary_diagnosis_chapter'] = 'MISSING'
        df['diagnosis_count'] = 0
        df['num_organ_systems'] = 0
        for chapter in ['INFECTIOUS', 'CIRCULATORY', 'RESPIRATORY', 'DIGESTIVE', 
                       'GENITOURINARY', 'NEOPLASMS', 'INJURY', 'SYMPTOMS']:
            df[f'{chapter.lower()}_dx_count'] = 0
        df['has_respiratory'] = 0
        df['has_cardiac'] = 0
        df['has_infection'] = 0
        return df
    
    diag_df = diagnosis_df.copy()
    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"\n[Enhanced Diagnosis Features - V3]")
    print(f"  Main dataset: {len(df):,} rows")
    
    diag_df['icd9_chapter'] = diag_df['ICD9_CODE'].apply(get_icd9_chapter)
    
    # Primary + Secondary
    primary_dx = diag_df[diag_df['SEQ_NUM'] == 1][['hadm_id', 'icd9_chapter']].copy()
    primary_dx = primary_dx.rename(columns={'icd9_chapter': 'primary_diagnosis_chapter'})
    df = df.merge(primary_dx, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['primary_diagnosis_chapter'] = df['primary_diagnosis_chapter'].fillna('MISSING')
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    secondary_dx = diag_df[diag_df['SEQ_NUM'] == 2][['hadm_id', 'icd9_chapter']].copy()
    secondary_dx = secondary_dx.rename(columns={'icd9_chapter': 'secondary_diagnosis_chapter'})
    df = df.merge(secondary_dx, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['secondary_diagnosis_chapter'] = df['secondary_diagnosis_chapter'].fillna('MISSING')
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    print(f"  ✓ Primary: {df['primary_diagnosis_chapter'].nunique()} chapters")
    print(f"  ✓ Secondary: {df['secondary_diagnosis_chapter'].nunique()} chapters")
    
    # Counts per chapter
    chapter_counts = diag_df.groupby(['hadm_id', 'icd9_chapter']).size().reset_index(name='count')
    chapter_pivot = chapter_counts.pivot(index='hadm_id', columns='icd9_chapter', values='count').fillna(0)
    df = df.merge(chapter_pivot, left_on=hadm_col_actual, right_index=True, how='left')
    
    major_chapters = ['INFECTIOUS', 'CIRCULATORY', 'RESPIRATORY', 'DIGESTIVE', 
                     'GENITOURINARY', 'NEOPLASMS', 'INJURY', 'SYMPTOMS']
    counts_added = 0
    for chapter in major_chapters:
        if chapter in df.columns:
            new_col = f'{chapter.lower()}_dx_count'
            df[new_col] = df[chapter].fillna(0).astype(int)
            df = df.drop(columns=[chapter])
            counts_added += 1
        else:
            df[f'{chapter.lower()}_dx_count'] = 0
            counts_added += 1
    
    print(f"  ✓ Chapter counts: {counts_added} features")
    
    # Total count
    dx_counts = diag_df.groupby('hadm_id').size().reset_index(name='diagnosis_count')
    df = df.merge(dx_counts, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['diagnosis_count'] = df['diagnosis_count'].fillna(0).astype(int)
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    # Diversity
    organ_counts = diag_df.groupby('hadm_id')['icd9_chapter'].nunique().reset_index(name='num_organ_systems')
    df = df.merge(organ_counts, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['num_organ_systems'] = df['num_organ_systems'].fillna(0).astype(int)
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    # Binary flags
    respiratory_admissions = diag_df[diag_df['icd9_chapter'] == 'RESPIRATORY']['hadm_id'].unique()
    cardiac_admissions = diag_df[diag_df['icd9_chapter'] == 'CIRCULATORY']['hadm_id'].unique()
    infection_admissions = diag_df[diag_df['icd9_chapter'] == 'INFECTIOUS']['hadm_id'].unique()
    
    df['has_respiratory'] = df[hadm_col_actual].isin(respiratory_admissions).astype(int)
    df['has_cardiac'] = df[hadm_col_actual].isin(cardiac_admissions).astype(int)
    df['has_infection'] = df[hadm_col_actual].isin(infection_admissions).astype(int)
    
    n_features_total = 2 + counts_added + 1 + 1 + 3
    print(f"  ✓ Total diagnosis features: {n_features_total}")
    
    return df


# ============================================================================
# IMPORT FROM hef_prep.py
# ============================================================================

from hef_prep import (
    split_features_target,
    add_age_features,
    clean_min_bp_outliers,
    add_engineered_features_core,  # NEW: core features only
    TARGET_COL_CLASS,
    ID_COLS
)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOAD RAW DATA")
print("="*80)

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"✓ Train: {train_raw.shape}")
print(f"✓ Test: {test_raw.shape}")

# Save test IDs
test_id_col = None
for col in test_raw.columns:
    if col.lower() == 'icustay_id':
        test_id_col = col
        break
test_ids = test_raw[test_id_col].values

# ============================================================================
# STEP 2: MERGE ENHANCED DIAGNOSIS DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 2: MERGE ENHANCED DIAGNOSIS DATA (V3 - ALL DIAGNOSES)")
print("="*80)

diagnosis_df = load_diagnosis_data()

train_with_dx = prepare_enhanced_diagnosis_features(train_raw, diagnosis_df)
test_with_dx = prepare_enhanced_diagnosis_features(test_raw, diagnosis_df)

print(f"\n✓ After diagnosis merge - Train: {train_with_dx.shape}, Test: {test_with_dx.shape}")

# Remove Diff
if 'Diff' in train_with_dx.columns:
    train_with_dx = train_with_dx.drop(columns=['Diff'])
if 'Diff' in test_with_dx.columns:
    test_with_dx = test_with_dx.drop(columns=['Diff'])

# ============================================================================
# STEP 3: SPLIT FEATURES AND TARGET
# ============================================================================

print("\n" + "="*80)
print("STEP 3: SPLIT FEATURES AND TARGET")
print("="*80)

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]

X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_with_dx,
    test_df=test_with_dx,
    task="class",
    leak_cols=leak_cols,
    id_cols=ID_COLS
)

print(f"✓ Target distribution: {y_train.mean():.3f}")

# ============================================================================
# STEP 4: ADD CORE FEATURES ONLY
# ============================================================================

print("\n" + "="*80)
print("STEP 4: ADD CORE FEATURES (NO FLAGS, NO TEMPORAL)")
print("="*80)

print("\n[4a] Adding age features...")
X_train_age = add_age_features(X_train_raw)
X_test_age = add_age_features(X_test_raw)

# Remove age bins
age_bins = ['age_0_40', 'age_40_50', 'age_50_60', 'age_60_70', 'age_70_80', 'age_80_90', 'age_90_plus']
X_train_age = X_train_age.drop(columns=[c for c in age_bins if c in X_train_age.columns], errors='ignore')
X_test_age = X_test_age.drop(columns=[c for c in age_bins if c in X_test_age.columns], errors='ignore')

print("\n[4b] Cleaning BP outliers...")
X_train_clean = clean_min_bp_outliers(X_train_age)
X_test_clean = clean_min_bp_outliers(X_test_age)

print("\n[4c] Adding CORE engineered features...")
X_train_fe = add_engineered_features_core(X_train_clean)
X_test_fe = add_engineered_features_core(X_test_clean)

print(f"\n✓ Features after core engineering: {X_train_fe.shape[1]}")

X_train_final = X_train_fe
X_test_final = X_test_fe

# ============================================================================
# STEP 5: ENCODE CATEGORICAL FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 5: ENCODE CATEGORICAL FEATURES")
print("="*80)

cat_cols = X_train_final.select_dtypes(include=['object']).columns.tolist()

# Remove diagnosis chapters (will be target encoded)
diagnosis_cat_cols = ['primary_diagnosis_chapter', 'secondary_diagnosis_chapter']
for col in diagnosis_cat_cols:
    if col in cat_cols:
        cat_cols.remove(col)

print(f"✓ Target encoding: {diagnosis_cat_cols}")
print(f"✓ Label encoding: {len(cat_cols)} columns")

if len(cat_cols) > 0:
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_final[col], X_test_final[col]]).astype(str)
        le.fit(combined)
        X_train_final[col] = le.transform(X_train_final[col].astype(str))
        X_test_final[col] = le.transform(X_test_final[col].astype(str))

print(f"\n✓ Final feature count: {X_train_final.shape[1]}")

# ============================================================================
# STEP 6: BUILD PIPELINE
# ============================================================================

print("\n" + "="*80)
print("STEP 6: BUILD PIPELINE (TARGET ENCODE PRIMARY + SECONDARY)")
print("="*80)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

pipeline = Pipeline([
    ('target_encoder', TargetEncoder(
        cols=['primary_diagnosis_chapter', 'secondary_diagnosis_chapter'],  # V3: Encode BOTH
        smoothing=20.0,
        min_samples_leaf=20,
        return_df=True
    )),
    ('classifier', XGBClassifier(
        max_depth=3,
        learning_rate=0.01,
        n_estimators=1000,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        gamma=0.5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist',
        eval_metric='auc',
        #early_stopping_rounds=50,
        n_jobs=-1
    ))
])

print("✓ Pipeline created")

# ============================================================================
# STEP 7: CROSS-VALIDATION
# ============================================================================

print("\n" + "="*80)
print("STEP 7: CROSS-VALIDATION (5-FOLD)")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train_final, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"\n✓ Cross-validation complete!")
print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# STEP 8: TRAIN FINAL MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 8: TRAIN FINAL MODEL WITH EARLY STOPPING")
print("="*80)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)

target_enc = TargetEncoder(
    cols=['primary_diagnosis_chapter', 'secondary_diagnosis_chapter'],
    smoothing=20.0, min_samples_leaf=20, return_df=True
)

X_tr_encoded = target_enc.fit_transform(X_tr, y_tr)
X_val_encoded = target_enc.transform(X_val)
X_test_encoded = target_enc.transform(X_test_final)

model = XGBClassifier(
    max_depth=3, learning_rate=0.01, n_estimators=1000,
    subsample=0.7, colsample_bytree=0.7,
    min_child_weight=10, gamma=0.5,
    reg_alpha=1.0, reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42, tree_method='hist',
    eval_metric='auc', early_stopping_rounds=50, n_jobs=-1
)

print("\n  Training...")
model.fit(
    X_tr_encoded, y_tr,
    eval_set=[(X_tr_encoded, y_tr), (X_val_encoded, y_val)],
    verbose=50
)

print(f"\n✓ Best iteration: {model.best_iteration}")

# ============================================================================
# STEP 9: EVALUATE
# ============================================================================

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

y_val_pred = model.predict_proba(X_val_encoded)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"\n✓ Validation AUC: {val_auc:.4f}")

# ============================================================================
# STEP 10: GENERATE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 10: GENERATE TEST PREDICTIONS")
print("="*80)

y_test_proba = model.predict_proba(X_test_encoded)[:, 1]

# ============================================================================
# STEP 11: CREATE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("STEP 11: CREATE SUBMISSION")
print("="*80)

submission = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': y_test_proba
})

output_dir = BASE_DIR / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "xgb_v3_enhanced_diagnosis.csv"

submission.to_csv(output_file, index=False)
print(f"✓ Submission saved: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY - V3 ENHANCED DIAGNOSIS")
print("="*80)

print(f"\n📊 Performance:")
print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Validation AUC: {val_auc:.4f}")

print(f"\n✨ V3 Enhancements:")
print(f"  ✓ Enhanced diagnosis: ALL codes (not just primary)")
print(f"  ✓ Chapter counts: 8 features (respiratory_dx_count, etc.)")
print(f"  ✓ Secondary diagnosis: target encoded alongside primary")
print(f"  ✓ Core features: No flags, no temporal (cleaner)")

print(f"\n📁 Submission: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)