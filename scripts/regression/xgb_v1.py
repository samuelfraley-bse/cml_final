"""
XGBoost LOS Regression V1 - With V5 Diagnosis Features

Applies classification V5 diagnosis approach to LOS regression:
1. ✓ Enhanced diagnosis (ALL codes, counts, primary+secondary)
2. ✓ Charlson Comorbidity Index
3. ✓ V4 high-risk flags (sepsis, resp failure, shock, MI, CHF)
4. ✓ V5 additional flags (AKI, pneumonia, stroke, PE, ARDS, GI bleeding)
5. ✓ Core vital features (no flags, no temporal)

Hypothesis: Diagnosis features predict both mortality AND length of stay!
- Sicker patients (high Charlson) stay longer
- Specific conditions (sepsis, ARDS) require extended care
- Comorbidities increase recovery time

Run from project root:
    python scripts/regression/xgb_los_v1.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from category_encoders import TargetEncoder
from xgboost import XGBRegressor

# Setup paths
BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

print("="*80)
print("XGBOOST LOS V1 - DIAGNOSIS FEATURES")
print("="*80)
print("\n✨ Applying V5 classification features to LOS regression:")
print("  ✓ Enhanced diagnosis (counts, primary+secondary)")
print("  ✓ Charlson Comorbidity Index")
print("  ✓ 11 high-risk condition flags")
print("  ✓ Core vital features")
print()

# ============================================================================
# IMPORT DIAGNOSIS FUNCTIONS FROM CLASSIFICATION
# ============================================================================

# These functions work for BOTH classification and regression!
# Diagnosis codes are assigned at hospital admission, before ICU entry

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
    """Enhanced diagnosis features (same as V5 classification)"""
    df = main_df.copy()
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        print(f"⚠️  Warning: {hadm_col} not found")
        # Create placeholder features
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
    
    print(f"\n[Enhanced Diagnosis Features]")
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


# Add high-risk flags (same V4 + V5 functions)
def add_high_risk_diagnosis_flags(main_df, diagnosis_df, hadm_col='hadm_id'):
    """Add V4 high-risk flags"""
    df = main_df.copy()
    
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        df['has_sepsis'] = 0
        df['has_respiratory_failure'] = 0
        df['has_shock'] = 0
        df['has_acute_mi'] = 0
        df['has_heart_failure'] = 0
        return df
    
    diag_df = diagnosis_df.copy()
    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"\n[V4 High-Risk Flags]")
    
    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].astype(str).str.strip().str.upper()
    
    # Sepsis
    sepsis_codes = (
        diag_df['ICD9_CODE'].str.startswith('038') |
        diag_df['ICD9_CODE'].isin(['99591', '99592', '995.91', '995.92'])
    )
    sepsis_admissions = diag_df[sepsis_codes]['hadm_id'].unique()
    df['has_sepsis'] = df[hadm_col_actual].isin(sepsis_admissions).astype(int)
    print(f"  ✓ Sepsis: {df['has_sepsis'].sum()} ({df['has_sepsis'].mean()*100:.1f}%)")
    
    # Respiratory failure
    resp_failure_codes = (
        diag_df['ICD9_CODE'].isin(['51881', '51884', '518.81', '518.84']) |
        diag_df['ICD9_CODE'].str.startswith('967')
    )
    resp_failure_admissions = diag_df[resp_failure_codes]['hadm_id'].unique()
    df['has_respiratory_failure'] = df[hadm_col_actual].isin(resp_failure_admissions).astype(int)
    print(f"  ✓ Respiratory failure: {df['has_respiratory_failure'].sum()} ({df['has_respiratory_failure'].mean()*100:.1f}%)")
    
    # Shock
    shock_codes = (
        diag_df['ICD9_CODE'].str.startswith('7855') |
        diag_df['ICD9_CODE'].str.startswith('9980') |
        diag_df['ICD9_CODE'].str.startswith('785.5') |
        diag_df['ICD9_CODE'].str.startswith('998.0')
    )
    shock_admissions = diag_df[shock_codes]['hadm_id'].unique()
    df['has_shock'] = df[hadm_col_actual].isin(shock_admissions).astype(int)
    print(f"  ✓ Shock: {df['has_shock'].sum()} ({df['has_shock'].mean()*100:.1f}%)")
    
    # Acute MI
    acute_mi_codes = diag_df['ICD9_CODE'].str.startswith('410')
    acute_mi_admissions = diag_df[acute_mi_codes]['hadm_id'].unique()
    df['has_acute_mi'] = df[hadm_col_actual].isin(acute_mi_admissions).astype(int)
    print(f"  ✓ Acute MI: {df['has_acute_mi'].sum()} ({df['has_acute_mi'].mean()*100:.1f}%)")
    
    # Heart failure
    heart_failure_codes = diag_df['ICD9_CODE'].str.startswith('428')
    heart_failure_admissions = diag_df[heart_failure_codes]['hadm_id'].unique()
    df['has_heart_failure'] = df[hadm_col_actual].isin(heart_failure_admissions).astype(int)
    print(f"  ✓ Heart failure: {df['has_heart_failure'].sum()} ({df['has_heart_failure'].mean()*100:.1f}%)")
    
    print(f"  ✓ Total: 5 flags")
    
    return df


def calculate_charlson_score(main_df, diagnosis_df, hadm_col='hadm_id'):
    """Calculate Charlson Comorbidity Index"""
    df = main_df.copy()
    
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
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
        
        # 1 POINT
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
        
        # 2 POINTS
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
        
        # 3 POINTS
        if has_severe_liver: score += 3
        
        # 6 POINTS
        if has_mets: score += 6
        if any(c.startswith('042') or c.startswith('043') or c.startswith('044') for c in codes): score += 6
        
        charlson_scores[hadm_id] = score
    
    df['charlson_score'] = df[hadm_col_actual].map(charlson_scores).fillna(0).astype(int)
    
    print(f"  ✓ Charlson scores calculated")
    print(f"    Mean: {df['charlson_score'].mean():.2f}, Max: {df['charlson_score'].max()}")
    
    return df


def add_additional_high_risk_flags(main_df, diagnosis_df, hadm_col='hadm_id'):
    """Add V5 additional flags"""
    df = main_df.copy()
    
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
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
    
    print(f"\n[V5 Additional Flags]")
    
    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].astype(str).str.strip().str.upper()
    
    # AKI
    aki_codes = diag_df['ICD9_CODE'].str.startswith('584')
    aki_admissions = diag_df[aki_codes]['hadm_id'].unique()
    df['has_acute_kidney_injury'] = df[hadm_col_actual].isin(aki_admissions).astype(int)
    print(f"  ✓ AKI: {df['has_acute_kidney_injury'].sum()} ({df['has_acute_kidney_injury'].mean()*100:.1f}%)")
    
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
    
    # PE
    pe_codes = diag_df['ICD9_CODE'].str.startswith('4151') | diag_df['ICD9_CODE'].str.startswith('415.1')
    pe_admissions = diag_df[pe_codes]['hadm_id'].unique()
    df['has_pulmonary_embolism'] = df[hadm_col_actual].isin(pe_admissions).astype(int)
    print(f"  ✓ PE: {df['has_pulmonary_embolism'].sum()} ({df['has_pulmonary_embolism'].mean()*100:.1f}%)")
    
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
    
    print(f"  ✓ Total: 6 flags")
    
    return df


# ============================================================================
# IMPORT FROM los_prep_enhanced.py
# ============================================================================

from los_prep_enhanced import (
    load_raw_data,
    split_features_target,
    add_age_features,
    clean_min_bp_outliers,
    add_engineered_features_core,
    TARGET_COL_REG,
    ID_COLS
)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOAD RAW DATA")
print("="*80)

train_raw, test_raw = load_raw_data()

# Save test IDs
test_id_col = None
for col in test_raw.columns:
    if col.lower() == 'icustay_id':
        test_id_col = col
        break
test_ids = test_raw[test_id_col].values

# ============================================================================
# STEP 2: MERGE DIAGNOSIS DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 2: MERGE DIAGNOSIS DATA")
print("="*80)

diagnosis_df = load_diagnosis_data()

train_with_dx = prepare_enhanced_diagnosis_features(train_raw, diagnosis_df)
test_with_dx = prepare_enhanced_diagnosis_features(test_raw, diagnosis_df)

print(f"\n✓ After diagnosis merge - Train: {train_with_dx.shape}, Test: {test_with_dx.shape}")

# Add high-risk flags
train_with_dx = add_high_risk_diagnosis_flags(train_with_dx, diagnosis_df)
test_with_dx = add_high_risk_diagnosis_flags(test_with_dx, diagnosis_df)

# Charlson score
train_with_dx = calculate_charlson_score(train_with_dx, diagnosis_df)
test_with_dx = calculate_charlson_score(test_with_dx, diagnosis_df)

# Additional flags
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

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis", 
             "ADMITTIME", "LOS", "HOSPITAL_EXPIRE_FLAG"]

X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_with_dx,
    test_df=test_with_dx,
    task="reg",
    leak_cols=leak_cols,
    id_cols=ID_COLS
)

# ============================================================================
# STEP 4: ADD CORE FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 4: ADD CORE FEATURES")
print("="*80)

print("\n[4a] Adding age features...")
X_train_age = add_age_features(X_train_raw)
X_test_age = add_age_features(X_test_raw)

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
# STEP 6: BUILD PIPELINE (REGRESSION)
# ============================================================================

print("\n" + "="*80)
print("STEP 6: BUILD PIPELINE (XGBREGRESSOR)")
print("="*80)

pipeline = Pipeline([
    ('target_encoder', TargetEncoder(
        cols=['primary_diagnosis_chapter', 'secondary_diagnosis_chapter'],
        smoothing=20.0,
        min_samples_leaf=20,
        return_df=True
    )),
    ('regressor', XGBRegressor(
        max_depth=5,
        learning_rate=0.03,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        tree_method='hist',
        eval_metric='rmse',
        n_jobs=-1
    ))
])

print("✓ Pipeline created (XGBRegressor)")

# ============================================================================
# STEP 7: CROSS-VALIDATION (REGRESSION)
# ============================================================================

print("\n" + "="*80)
print("STEP 7: CROSS-VALIDATION (5-FOLD)")
print("="*80)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Negative MSE for sklearn compatibility
cv_scores_mse = cross_val_score(pipeline, X_train_final, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
cv_rmse = np.sqrt(-cv_scores_mse)

cv_scores_mae = cross_val_score(pipeline, X_train_final, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
cv_mae = -cv_scores_mae

print(f"\n✓ Cross-validation complete!")
print(f"  Mean CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
print(f"  Mean CV MAE:  {cv_mae.mean():.4f} (+/- {cv_mae.std():.4f})")
print(f"\n  Benchmark: 15.7 RMSE (NN)")

# ============================================================================
# STEP 8: TRAIN FINAL MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 8: TRAIN FINAL MODEL WITH EARLY STOPPING")
print("="*80)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42
)

target_enc = TargetEncoder(
    cols=['primary_diagnosis_chapter', 'secondary_diagnosis_chapter'],
    smoothing=20.0, min_samples_leaf=20, return_df=True
)

X_tr_encoded = target_enc.fit_transform(X_tr, y_tr)
X_val_encoded = target_enc.transform(X_val)
X_test_encoded = target_enc.transform(X_test_final)

model = XGBRegressor(
    max_depth=5, learning_rate=0.03, n_estimators=1000,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.5,
    random_state=42, tree_method='hist',
    eval_metric='rmse', early_stopping_rounds=50, n_jobs=-1
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

y_val_pred = np.maximum(model.predict(X_val_encoded), 0)  # Clip to non-negative
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"\n✓ Validation metrics:")
print(f"  RMSE: {val_rmse:.4f}")
print(f"  MAE:  {val_mae:.4f}")
print(f"  R²:   {val_r2:.4f}")

# ============================================================================
# STEP 10: GENERATE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 10: GENERATE TEST PREDICTIONS")
print("="*80)

y_test_pred = np.maximum(model.predict(X_test_encoded), 0)  # Clip to non-negative

print(f"\nTest predictions:")
print(f"  Min:    {y_test_pred.min():.2f} days")
print(f"  Max:    {y_test_pred.max():.2f} days")
print(f"  Mean:   {y_test_pred.mean():.2f} days")
print(f"  Median: {np.median(y_test_pred):.2f} days")

# ============================================================================
# STEP 11: CREATE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("STEP 11: CREATE SUBMISSION")
print("="*80)

submission = pd.DataFrame({
    'icustay_id': test_ids,
    'LOS': y_test_pred
})

output_dir = BASE_DIR / "submissions" / "regression"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "xgb_los_v1_diagnosis.csv"

submission.to_csv(output_file, index=False)
print(f"✓ Submission saved: {output_file}")

# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

importance_df = pd.DataFrame({
    'feature': X_tr_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 30 Most Important Features:")
print(importance_df.head(30).to_string(index=False))

# Diagnosis features analysis
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

# Save analysis
analysis_dir = BASE_DIR / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)
analysis_file = analysis_dir / "los_v1_feature_importance.csv"
importance_df.to_csv(analysis_file, index=False)
print(f"\n✓ Full feature importance saved: {analysis_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY - LOS V1 WITH DIAGNOSIS FEATURES")
print("="*80)

print(f"\n📊 Performance:")
print(f"  CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
print(f"  Validation RMSE: {val_rmse:.4f}")
print(f"  Validation MAE: {val_mae:.4f}")
print(f"  Validation R²: {val_r2:.4f}")

print(f"\n🎯 vs Benchmarks:")
print(f"  NN Benchmark: 15.7 RMSE")
if val_rmse < 15.7:
    print(f"  ✅ BEAT BENCHMARK! ({val_rmse:.4f} < 15.7)")
else:
    print(f"  Gap: {val_rmse - 15.7:.2f} points")

print(f"\n✨ Features Used:")
print(f"  ✓ Enhanced diagnosis (counts, primary+secondary)")
print(f"  ✓ Charlson Comorbidity Index")
print(f"  ✓ 11 high-risk condition flags")
print(f"  ✓ Core vital features")
print(f"  ✓ Total diagnosis importance: {diag_importance_sum/importance_df['importance'].sum()*100:.1f}%")

print(f"\n📁 Submission: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)