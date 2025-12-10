"""
XGBoost V5 - Charlson Comorbidity Index + More High-Risk Flags + ICU History

Changes from V5 original:
1. ✓ All V5 features (Charlson Index, 11 high-risk flags, core vitals)
2. ✓ NEW: ICU History Features (10 features)
   - Readmission flags, days since last ICU, frequent flyer metrics
3. ✓ CONFIRMED: Target encoding is inside the Pipeline for cross-validation (leakage-safe).

Run from project root:
    python scripts/classification/xgb_v5_updated.py
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
print("XGBOOST V5 - CHARLSON INDEX + MORE HIGH-RISK FLAGS + ICU HISTORY")
print("="*80)
print("\n✨ New in V5 Updated:")
print("  ✓ All V5 original features")
print("  ✓ NEW: ICU History Features (Readmission, days since last ICU, etc.)")
print("  ✓ Confirmed Target Encoding is leakage-safe (inside Pipeline)")
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


def add_high_risk_diagnosis_flags(main_df, diagnosis_df, hadm_col='hadm_id'):
    """
    Add binary flags for specific high-mortality conditions.
    
    Flags: sepsis, respiratory failure, shock, acute MI, heart failure
    """
    df = main_df.copy()
    
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        print(f"⚠️  Warning: {hadm_col} not found")
        df['has_sepsis'] = 0
        df['has_respiratory_failure'] = 0
        df['has_shock'] = 0
        df['has_acute_mi'] = 0
        df['has_heart_failure'] = 0
        return df
    
    diag_df = diagnosis_df.copy()
    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"\n[High-Risk Diagnosis Flags]")
    
    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].astype(str).str.strip().str.upper()
    
    # Sepsis
    sepsis_codes = (
        diag_df['ICD9_CODE'].str.startswith('038') |
        diag_df['ICD9_CODE'].isin(['99591', '99592', '995.91', '995.92'])
    )
    sepsis_admissions = diag_df[sepsis_codes]['hadm_id'].unique()
    df['has_sepsis'] = df[hadm_col_actual].isin(sepsis_admissions).astype(int)
    print(f"  ✓ Sepsis: {df['has_sepsis'].sum()} patients ({df['has_sepsis'].mean()*100:.1f}%)")
    
    # Respiratory failure
    resp_failure_codes = (
        diag_df['ICD9_CODE'].isin(['51881', '51884', '518.81', '518.84']) |
        diag_df['ICD9_CODE'].str.startswith('967')
    )
    resp_failure_admissions = diag_df[resp_failure_codes]['hadm_id'].unique()
    df['has_respiratory_failure'] = df[hadm_col_actual].isin(resp_failure_admissions).astype(int)
    print(f"  ✓ Respiratory failure: {df['has_respiratory_failure'].sum()} patients ({df['has_respiratory_failure'].mean()*100:.1f}%)")
    
    # Shock
    shock_codes = (
        diag_df['ICD9_CODE'].str.startswith('7855') |
        diag_df['ICD9_CODE'].str.startswith('9980') |
        diag_df['ICD9_CODE'].str.startswith('785.5') |
        diag_df['ICD9_CODE'].str.startswith('998.0')
    )
    shock_admissions = diag_df[shock_codes]['hadm_id'].unique()
    df['has_shock'] = df[hadm_col_actual].isin(shock_admissions).astype(int)
    print(f"  ✓ Shock: {df['has_shock'].sum()} patients ({df['has_shock'].mean()*100:.1f}%)")
    
    # Acute MI
    acute_mi_codes = diag_df['ICD9_CODE'].str.startswith('410')
    acute_mi_admissions = diag_df[acute_mi_codes]['hadm_id'].unique()
    df['has_acute_mi'] = df[hadm_col_actual].isin(acute_mi_admissions).astype(int)
    print(f"  ✓ Acute MI: {df['has_acute_mi'].sum()} patients ({df['has_acute_mi'].mean()*100:.1f}%)")
    
    # Heart failure
    heart_failure_codes = diag_df['ICD9_CODE'].str.startswith('428')
    heart_failure_admissions = diag_df[heart_failure_codes]['hadm_id'].unique()
    df['has_heart_failure'] = df[hadm_col_actual].isin(heart_failure_admissions).astype(int)
    print(f"  ✓ Heart failure: {df['has_heart_failure'].sum()} patients ({df['has_heart_failure'].mean()*100:.1f}%)")
    
    print(f"  ✓ Total high-risk flags: 5")
    
    return df


def calculate_charlson_score(main_df, diagnosis_df, hadm_col='hadm_id'):
    """Calculate Charlson Comorbidity Index - validated mortality predictor"""
    df = main_df.copy()
    
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        print(f"⚠️  Warning: {hadm_col} not found")
        df['charlson_score'] = 0
        return df
    
    diag_df = diagnosis_df.copy()
    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"\n[Charlson Comorbidity Index]")
    
    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].astype(str).str.strip().str.upper()
    
    charlson_scores = {}
    
    for hadm_id in diag_df['hadm_id'].unique():
        codes = diag_df[diag_df['hadm_id'] == hadm_id]['ICD9_CODE'].tolist()
        score = 0
        
        # 1 POINT: MI, CHF, PVD, CVD, Dementia, COPD, CTD, Ulcer, Mild liver
        if any(c.startswith('410') or c in ['412'] for c in codes): score += 1
        if any(c.startswith('428') for c in codes): score += 1
        if any(c.startswith('441') or c.startswith('443') for c in codes): score += 1
        if any(c.startswith('43') for c in codes): score += 1
        if any(c.startswith('290') for c in codes): score += 1
        if any((c.startswith('49') or c.startswith('50')) for c in codes): score += 1
        if any(c.startswith('710') or c.startswith('714') for c in codes): score += 1
        if any(c.startswith('531') or c.startswith('532') or c.startswith('533') or c.startswith('534') for c in codes): score += 1
        
        has_mild_liver = any(c.startswith('5712') or c.startswith('5715') or c.startswith('5716') or c.startswith('5714') for c in codes)
        has_severe_liver = any(c.startswith('5722') or c.startswith('5723') or c.startswith('5724') or c.startswith('5728') for c in codes)
        if has_mild_liver and not has_severe_liver: score += 1
        
        # 2 POINTS: Diabetes with complications, Hemiplegia, Renal, Cancer, Leukemia, Lymphoma
        if any(c.startswith('2504') or c.startswith('2505') or c.startswith('2506') or 
               c.startswith('2507') or c.startswith('2508') or c.startswith('2509') for c in codes): score += 2
        if any(c.startswith('342') or c.startswith('344') for c in codes): score += 2
        if any(c.startswith('582') or c.startswith('583') or c.startswith('585') or c in ['586'] for c in codes): score += 2
        
        has_cancer = any((c.startswith('14') or c.startswith('15') or c.startswith('16') or 
                         c.startswith('17') or c.startswith('18') or c.startswith('19')) for c in codes)
        has_mets = any(c.startswith('196') or c.startswith('197') or c.startswith('198') or c.startswith('199') for c in codes)
        if has_cancer and not has_mets: score += 2
        
        if any(c.startswith('204') or c.startswith('205') or c.startswith('206') or c.startswith('207') or c.startswith('208') for c in codes): score += 2
        if any(c.startswith('200') or c.startswith('201') or c.startswith('202') or c.startswith('203') for c in codes): score += 2
        
        # 3 POINTS: Severe liver
        if has_severe_liver: score += 3
        
        # 6 POINTS: Metastatic cancer, AIDS
        if has_mets: score += 6
        if any(c.startswith('042') or c.startswith('043') or c.startswith('044') for c in codes): score += 6
        
        charlson_scores[hadm_id] = score
    
    df['charlson_score'] = df[hadm_col_actual].map(charlson_scores).fillna(0).astype(int)
    
    print(f"  ✓ Charlson scores calculated")
    print(f"    Mean: {df['charlson_score'].mean():.2f}, Max: {df['charlson_score'].max()}")
    print(f"    0: {(df['charlson_score'] == 0).sum()} | 1-2: {((df['charlson_score'] >= 1) & (df['charlson_score'] <= 2)).sum()} | 3-4: {((df['charlson_score'] >= 3) & (df['charlson_score'] <= 4)).sum()} | ≥5: {(df['charlson_score'] >= 5).sum()}")
    
    return df


def add_additional_high_risk_flags(main_df, diagnosis_df, hadm_col='hadm_id'):
    """Add 6 more high-risk flags: AKI, pneumonia, stroke, PE, ARDS, GI bleeding"""
    df = main_df.copy()
    
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        print(f"⚠️  Warning: {hadm_col} not found")
        df['has_acute_kidney_injury'] = 0
        df['has_pneumonia'] = 0
        df['has_stroke'] = 0
        df['has_pulmonary_embolism'] = 0
        df['has_ards'] = 0
        df['has_gi_bleeding'] = 0
        return df
    
    diag_df = diagnosis_df.copy()
    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"\n[Additional High-Risk Flags]")
    
    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].astype(str).str.strip().str.upper()
    
    # Acute kidney injury
    aki_codes = diag_df['ICD9_CODE'].str.startswith('584')
    aki_admissions = diag_df[aki_codes]['hadm_id'].unique()
    df['has_acute_kidney_injury'] = df[hadm_col_actual].isin(aki_admissions).astype(int)
    print(f"  ✓ Acute kidney injury: {df['has_acute_kidney_injury'].sum()} ({df['has_acute_kidney_injury'].mean()*100:.1f}%)")
    
    # Pneumonia
    pneumonia_codes = (
        diag_df['ICD9_CODE'].isin(['486', '485']) |
        diag_df['ICD9_CODE'].str.startswith('481') |
        diag_df['ICD9_CODE'].str.startswith('482') |
        diag_df['ICD9_CODE'].str.startswith('483')
    )
    pneumonia_admissions = diag_df[pneumonia_codes]['hadm_id'].unique()
    df['has_pneumonia'] = df[hadm_col_actual].isin(pneumonia_admissions).astype(int)
    print(f"  ✓ Pneumonia: {df['has_pneumonia'].sum()} ({df['has_pneumonia'].mean()*100:.1f}%)")
    
    # Stroke
    stroke_codes = diag_df['ICD9_CODE'].str.startswith('43')
    stroke_admissions = diag_df[stroke_codes]['hadm_id'].unique()
    df['has_stroke'] = df[hadm_col_actual].isin(stroke_admissions).astype(int)
    print(f"  ✓ Stroke: {df['has_stroke'].sum()} ({df['has_stroke'].mean()*100:.1f}%)")
    
    # Pulmonary embolism
    pe_codes = diag_df['ICD9_CODE'].str.startswith('4151') | diag_df['ICD9_CODE'].str.startswith('415.1')
    pe_admissions = diag_df[pe_codes]['hadm_id'].unique()
    df['has_pulmonary_embolism'] = df[hadm_col_actual].isin(pe_admissions).astype(int)
    print(f"  ✓ Pulmonary embolism: {df['has_pulmonary_embolism'].sum()} ({df['has_pulmonary_embolism'].mean()*100:.1f}%)")
    
    # ARDS
    ards_codes = diag_df['ICD9_CODE'].isin(['51882', '5185', '518.82', '518.5'])
    ards_admissions = diag_df[ards_codes]['hadm_id'].unique()
    df['has_ards'] = df[hadm_col_actual].isin(ards_admissions).astype(int)
    print(f"  ✓ ARDS: {df['has_ards'].sum()} ({df['has_ards'].mean()*100:.1f}%)")
    
    # GI bleeding
    gi_bleed_codes = diag_df['ICD9_CODE'].str.startswith('578')
    gi_bleed_admissions = diag_df[gi_bleed_codes]['hadm_id'].unique()
    df['has_gi_bleeding'] = df[hadm_col_actual].isin(gi_bleed_admissions).astype(int)
    print(f"  ✓ GI bleeding: {df['has_gi_bleeding'].sum()} ({df['has_gi_bleeding'].mean()*100:.1f}%)")
    
    print(f"  ✓ Total additional flags: 6")
    
    return df


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
    add_icu_history_features,  # ADDED: for ICU history features
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
print("STEP 2: MERGE DIAGNOSIS DATA (V5 - CHARLSON + MORE FLAGS)")
print("="*80)

diagnosis_df = load_diagnosis_data()

train_with_dx = prepare_enhanced_diagnosis_features(train_raw, diagnosis_df)
test_with_dx = prepare_enhanced_diagnosis_features(test_raw, diagnosis_df)

print(f"\n✓ After diagnosis merge - Train: {train_with_dx.shape}, Test: {test_with_dx.shape}")

# Add V4 high-risk diagnosis flags
print("\n[Adding V4 high-risk flags...]")
train_with_dx = add_high_risk_diagnosis_flags(train_with_dx, diagnosis_df)
test_with_dx = add_high_risk_diagnosis_flags(test_with_dx, diagnosis_df)

# NEW V5: Calculate Charlson Comorbidity Index
train_with_dx = calculate_charlson_score(train_with_dx, diagnosis_df)
test_with_dx = calculate_charlson_score(test_with_dx, diagnosis_df)

# NEW V5: Add additional high-risk flags
train_with_dx = add_additional_high_risk_flags(train_with_dx, diagnosis_df)
test_with_dx = add_additional_high_risk_flags(test_with_dx, diagnosis_df)

print(f"\n✓ After all diagnosis features - Train: {train_with_dx.shape}, Test: {test_with_dx.shape}")

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
# Temporary ID list that keeps 'subject_id' for ICU history calculation
ID_COLS_TEMP = [c for c in ID_COLS if c != 'subject_id']

X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_with_dx,
    test_df=test_with_dx,
    task="class",
    leak_cols=leak_cols,
    id_cols=ID_COLS_TEMP  # KEEPING subject_id for add_icu_history_features
)

print(f"✓ Target distribution: {y_train.mean():.3f}")

# ============================================================================
# STEP 3.5: ADD ICU HISTORY FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 3.5: ADD ICU HISTORY FEATURES")
print("="*80)

# Requires 'subject_id' and 'ADMITTIME' which are currently retained
X_train_raw = add_icu_history_features(X_train_raw, subject_col='subject_id', admit_col='ADMITTIME')
X_test_raw = add_icu_history_features(X_test_raw, subject_col='subject_id', admit_col='ADMITTIME')

# Drop subject_id now that history features are calculated
X_train_raw = X_train_raw.drop(columns=['subject_id'])
X_test_raw = X_test_raw.drop(columns=['subject_id'])

print(f"✓ Features after ICU history: {X_train_raw.shape[1]}")

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

# TARGET ENCODER IS INSIDE THE PIPELINE: This correctly prevents data leakage
# during cross-validation, as requested by the professor.
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
        n_estimators=500,  # Lower since no early stopping in CV
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
        # NO early_stopping_rounds here (causes error in CV)
        n_jobs=-1
    ))
])

print("✓ Pipeline created (no early stopping for CV)")

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
output_file = output_dir / "xgb_v5_updated.csv"

submission.to_csv(output_file, index=False)
print(f"✓ Submission saved: {output_file}")

# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importances from final model
importance_df = pd.DataFrame({
    'feature': X_tr_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 30 Most Important Features:")
print(importance_df.head(30).to_string(index=False))

# Diagnosis features analysis
print("\n" + "="*80)
print("DIAGNOSIS FEATURES BREAKDOWN")
print("="*80)

diagnosis_features = [col for col in importance_df['feature'].tolist() if any(x in col.lower() for x in [
    'diagnosis', 'charlson', 'sepsis', 'respiratory_failure', 'shock', 'acute_mi', 
    'heart_failure', 'acute_kidney_injury', 'pneumonia', 'stroke', 'pulmonary_embolism',
    'ards', 'gi_bleeding', 'has_respiratory', 'has_cardiac', 'has_infection',
    'primary_diagnosis', 'secondary_diagnosis'
])]

diag_importance = importance_df[importance_df['feature'].isin(diagnosis_features)]

print(f"\n✨ Diagnosis Features ({len(diag_importance)} total):")
print(diag_importance.to_string(index=False))

diag_importance_sum = diag_importance['importance'].sum()
print(f"\n📈 Total diagnosis importance: {diag_importance_sum:.4f} ({diag_importance_sum/importance_df['importance'].sum()*100:.1f}% of model)")

# Group features by type
print("\n" + "="*80)
print("FEATURE GROUP SUMMARY")
print("="*80)

feature_groups = {
    'High-Risk Flags (V4+V5)': ['sepsis', 'respiratory_failure', 'shock', 'acute_mi', 'heart_failure', 
                                  'acute_kidney_injury', 'pneumonia', 'stroke', 'pulmonary_embolism', 'ards', 'gi_bleeding'],
    'Charlson Score': ['charlson'],
    'Chapter Counts': ['_dx_count'],
    'Chapter Flags': ['has_respiratory', 'has_cardiac', 'has_infection'],
    'Primary/Secondary Dx': ['primary_diagnosis', 'secondary_diagnosis'],
    'Diagnosis Metrics': ['diagnosis_count', 'num_organ_systems'],
    'ICU History': ['icu_stays', 'readmission', 'days_since', 'frequent_flyer'], # ADDED
    'Vitals': ['spo2', 'sysbp', 'resprate', 'temp', 'glucose', 'hr_', 'meanbp', 'diasbp', 'heartrate'],
    'Age': ['age_', 'elderly'],
    'Demographics': ['admission_type', 'marital', 'religion', 'ethnicity', 'insurance', 'careunit'],
    'Engineered': ['shock_index', 'pulse_pressure', 'instability', 'hypoxia_and', 'deficit', 'excess', 'dev_']
}

group_summary = []
for group_name, keywords in feature_groups.items():
    group_features = [f for f in importance_df['feature'].tolist() 
                     if any(kw.lower() in f.lower() for kw in keywords)]
    group_importance = importance_df[importance_df['feature'].isin(group_features)]['importance'].sum()
    group_summary.append({
        'Feature Group': group_name,
        'Count': len(group_features),
        'Total Importance': f"{group_importance:.4f}",
        '% of Model': f"{group_importance/importance_df['importance'].sum()*100:.1f}%"
    })

group_summary_df = pd.DataFrame(group_summary).sort_values('Total Importance', ascending=False)
print("\n")
print(group_summary_df.to_string(index=False))

# Save analysis to file
analysis_dir = BASE_DIR / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)
analysis_file = analysis_dir / "v5_updated_feature_importance.csv"
importance_df.to_csv(analysis_file, index=False)
print(f"\n✓ Full feature importance saved: {analysis_file}")

# Top performers by category
print("\n" + "="*80)
print("TOP PERFORMERS BY CATEGORY")
print("="*80)

# Top diagnosis features
print("\n🏆 Top 5 Diagnosis Features:")
top_diag = diag_importance.head(5)
for _, row in top_diag.iterrows():
    print(f"  {row['feature']:35s} {row['importance']:.6f}")

# Top vital features
vital_features = [f for f in importance_df['feature'].tolist() 
                 if any(v in f.lower() for v in ['spo2', 'sysbp', 'resprate', 'temp', 'glucose', 'hr_', 'heartrate', 'meanb'])]
vital_importance = importance_df[importance_df['feature'].isin(vital_features)]
print("\n🏆 Top 5 Vital Features:")
top_vitals = vital_importance.head(5)
for _, row in top_vitals.iterrows():
    print(f"  {row['feature']:35s} {row['importance']:.6f}")

# Top engineered features
eng_features = [f for f in importance_df['feature'].tolist() 
               if any(e in f.lower() for e in ['shock_index', 'pulse_pressure', 'instability', 'deficit', 'excess', 'dev_'])]
eng_importance = importance_df[importance_df['feature'].isin(eng_features)]
if len(eng_importance) > 0:
    print("\n🏆 Top 5 Engineered Features:")
    top_eng = eng_importance.head(5)
    for _, row in top_eng.iterrows():
        print(f"  {row['feature']:35s} {row['importance']:.6f}")

# Zero importance features
zero_importance = importance_df[importance_df['importance'] == 0.0]
if len(zero_importance) > 0:
    print(f"\n⚠️  Features with ZERO importance ({len(zero_importance)} total):")
    print(f"   Consider removing these in next iteration:")
    for feat in zero_importance['feature'].head(10).tolist():
        print(f"   - {feat}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY - V5 UPDATED (ICU HISTORY INCLUDED)")
print("="*80)

print(f"\n📊 Performance:")
print(f"  Mean CV AUC will be calculated after running the script.")
print(f"  Validation AUC will be calculated after running the script.")

print(f"\n✨ V5 Updated Enhancements:")
print(f"  ✓ Added 10 ICU history features (days since last ICU, readmission flags).")
print(f"  ✓ Confirmed Target Encoding is leakage-safe via Pipeline implementation.")
print(f"  ✓ Retains Charlson Score and 11 High-Risk Flags.")

print(f"\n📁 Submission: xgb_v5_updated.csv")

print("\n" + "="*80)
print("✓ SCRIPT GENERATED!")
print("="*80)