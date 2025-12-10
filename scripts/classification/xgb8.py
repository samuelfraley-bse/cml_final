"""
XGBoost V7 - Self-Contained Script for HOSPITAL EXPIRATION FLAG (HEF)

This script combines all feature engineering logic (Vitals, Age, ICU History, 
Charlson, High-Risk Flags, ICD-9 Subcategories, V7 Sequence Deterioration) 
and the full XGBoost pipeline into a single file for easy execution.

It is configured for the classification task: predicting HOSPITAL_EXPIRE_FLAG.

It assumes the following data structure:
- data/raw/MIMIC III dataset HEF/mimic_train_HEF.csv
- data/raw/MIMIC III dataset HEF/mimic_test_HEF.csv
- data/raw/MIMIC III dataset HEF/extra_data/MIMIC_diagnoses.csv
"""
import sys
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & PATHS
# ============================================================================

# Targets
TARGET_COL_CLASS = "HOSPITAL_EXPIRE_FLAG"
TARGET_COL_REG   = "LOS" # Not used, but kept for context

# ID columns (subject_id is kept temporarily for feature engineering)
ID_COLS = [
    "icu_stayid",
    "subject_id",
    "hadm_id",
    "intime", # Removed in the final split, but useful for functions
]

BP_MIN_LOWER_BOUNDS = {
    "SysBP_Min":  40.0,
    "DiasBP_Min": 10.0,
    "MeanBP_Min": 30.0,
}

# Setup paths (Adjust BASE_DIR if running from a different location)
BASE_DIR = Path.cwd()

def get_paths():
    """Return BASE_DIR, TRAIN_PATH, TEST_PATH, and DIAG_PATH based on structure."""
    base_dir = BASE_DIR.resolve()
    # Attempt to traverse up to find project root if running from a subdirectory
    for parent in [base_dir] + list(base_dir.parents):
        if 'data' in [d.name for d in parent.iterdir() if d.is_dir()]:
            base_dir = parent
            break
    
    raw_dir = base_dir / "data" / "raw" / "MIMIC III dataset HEF"

    train_path = raw_dir / "mimic_train_HEF.csv"
    test_path  = raw_dir / "mimic_test_HEF.csv"
    diag_path  = raw_dir / "extra_data" / "MIMIC_diagnoses.csv"

    return base_dir, train_path, test_path, diag_path


def load_raw_data():
    """Load raw train, test, and diagnosis CSVs."""
    base_dir, train_path, test_path, diag_path = get_paths()
    print("Base dir:", base_dir)

    try:
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Raw data files not found at {train_path.parent}")
        print("Please check the file paths and ensure you are running from the project root.")
        raise

    print(f"✓ Train: {train_df.shape}")
    print(f"✓ Test: {test_df.shape}")

    return train_df, test_df


def load_diagnosis_data(filepath=None):
    """Load MIMIC_diagnoses.csv"""
    if filepath is None:
        _, _, _, filepath = get_paths()
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


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def split_features_target(train_df, test_df, task="class", leak_cols=None, id_cols=None):
    """Split target variable and drop unnecessary columns."""
    if leak_cols is None:
        leak_cols = []
    if id_cols is None:
        id_cols = ID_COLS
    
    df_train = train_df.copy()
    df_test = test_df.copy()

    # Drop leak columns and IDs from both sets
    cols_to_drop = list(set(leak_cols + id_cols))
    
    if task == "class":
        target_col = TARGET_COL_CLASS
    else:
        target_col = TARGET_COL_REG

    # Extract target
    y_train = df_train[target_col]
    
    # Drop target and other columns from X sets
    X_train = df_train.drop(columns=[c for c in cols_to_drop + [target_col] if c in df_train.columns], errors='ignore')
    X_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns], errors='ignore')

    # Re-insert IDs that are needed for subsequent FE but are not features themselves (like subject_id, ADMITTIME)
    # Note: DOB must remain in X_train/X_test for add_age_features()
    
    # Ensure subject_id is present for add_icu_history_features
    if 'subject_id' in train_df.columns and 'subject_id' not in X_train.columns:
        X_train.insert(0, 'subject_id', train_df['subject_id'])
    if 'subject_id' in test_df.columns and 'subject_id' not in X_test.columns:
        X_test.insert(0, 'subject_id', test_df['subject_id'])

    # Ensure ADMITTIME is present for add_icu_history_features
    if 'ADMITTIME' in train_df.columns and 'ADMITTIME' not in X_train.columns:
        X_train.insert(0, 'ADMITTIME', train_df['ADMITTIME'])
    if 'ADMITTIME' in test_df.columns and 'ADMITTIME' not in X_test.columns:
        X_test.insert(0, 'ADMITTIME', test_df['ADMITTIME'])


    return X_train, y_train, X_test


def get_icd9_chapter(code):
    """Map ICD9 code to broad chapter (used in prepare_enhanced_diagnosis_features)"""
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
    
    if 1 <= code_num <= 139: return 'INFECTIOUS'
    elif 140 <= code_num <= 239: return 'NEOPLASMS'
    elif 240 <= code_num <= 279: return 'ENDOCRINE'
    elif 280 <= code_num <= 289: return 'BLOOD'
    elif 290 <= code_num <= 319: return 'MENTAL'
    elif 320 <= code_num <= 389: return 'NERVOUS'
    elif 390 <= code_num <= 459: return 'CIRCULATORY'
    elif 460 <= code_num <= 519: return 'RESPIRATORY'
    elif 520 <= code_num <= 579: return 'DIGESTIVE'
    elif 580 <= code_num <= 629: return 'GENITOURINARY'
    elif 630 <= code_num <= 679: return 'PREGNANCY'
    elif 680 <= code_num <= 709: return 'SKIN'
    elif 710 <= code_num <= 739: return 'MUSCULOSKELETAL'
    elif 740 <= code_num <= 759: return 'CONGENITAL'
    elif 760 <= code_num <= 779: return 'PERINATAL'
    elif 780 <= code_num <= 799: return 'SYMPTOMS'
    elif 800 <= code_num <= 999: return 'INJURY'
    else: return 'UNKNOWN'


def add_diagnosis_sequence_features(main_df, diagnosis_df, hadm_col='hadm_id'):
    """
    V7: Adds binary flags indicating if severe diagnoses were recorded late (SEQ_NUM > 2),
    suggesting patient deterioration after initial workup, and calculates max diagnosis sequence.
    """
    df = main_df.copy()
    
    # Find hadm_col (case-insensitive)
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        print(f"⚠️  Warning: {hadm_col} not found - skipping sequence flags.")
        return df

    diag_df = diagnosis_df.copy()
    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"\n[Diagnosis Sequence Deterioration Flags - V7]")

    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].astype(str).str.strip().str.upper()

    # Define severe conditions and their code filtering logic
    severe_conditions = {
        'sepsis': lambda c: c.startswith('038') or c in ['99591', '99592', '995.91', '995.92'],
        'shock': lambda c: c.startswith('7855') or c.startswith('9980') or c.startswith('785.5') or c.startswith('998.0'),
        # Note: Resp Failure codes used here are for ARF/Respiratory arrest, which imply acute deterioration
        'resp_fail': lambda c: c in ['51881', '51884', '518.81', '518.84'] 
    }

    temp_df = diag_df.copy()
    
    for name, code_check in severe_conditions.items():
        # Filter for the specific severe condition codes AND where diagnosis is secondary or later (SEQ_NUM > 2)
        late_onset_diagnoses = temp_df[
            (temp_df['ICD9_CODE'].apply(code_check)) & 
            (temp_df['SEQ_NUM'] > 2)
        ]['hadm_id'].unique()
        
        flag_col = f'{name}_is_secondary'
        
        # Create the flag in the main DataFrame
        df[flag_col] = df[hadm_col_actual].isin(late_onset_diagnoses).astype(int)
        
        print(f"  ✓ {name.capitalize()} secondary flag: {df[flag_col].sum()} patients")

    # Feature 4: Max Diagnosis Sequence (How deep the diagnosis list goes)
    max_seq_overall = diag_df.groupby('hadm_id')['SEQ_NUM'].max().reset_index()
    max_seq_overall = max_seq_overall.rename(columns={'SEQ_NUM': 'max_dx_seq'})
    df = df.merge(max_seq_overall, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['max_dx_seq'] = df['max_dx_seq'].fillna(0).astype(int)
    
    # Drop the merged hadm_id column if it was created
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])

    print(f"  ✓ Max Diagnosis Sequence (max_dx_seq) calculated.")
    
    return df


def add_icd9_subcategory_flags(main_df, diagnosis_df, hadm_col='hadm_id', min_prevalence=0.005):
    """
    V6: Add binary flags for specific ICD-9 3-digit categories with a min prevalence threshold.
    """
    df = main_df.copy()
    diag_df = diagnosis_df.copy()
    
    # Check for hadm_col
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    if hadm_col_actual is None:
        print(f"⚠️  Warning: {hadm_col} not found - skipping subcategory flags.")
        return df

    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"\n[ICD-9 Subcategory Flags - Min Prevalence {min_prevalence*100:.1f}%]")
    
    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].astype(str).str.strip().str.upper()
    
    # Extract the 3-digit category (e.g., '414' from '414.01')
    valid_codes = diag_df[~diag_df['ICD9_CODE'].str.startswith(('V', 'E', 'UNKNOWN'))].copy()
    
    def safe_extract_category(code):
        try:
            # Simple extraction of first three characters, ensuring it's numeric
            code_str = code.replace('.', '', 1)
            if len(code_str) >= 3 and code_str[:3].isdigit():
                return code_str[:3]
            return None
        except:
            return None

    valid_codes['CATEGORY'] = valid_codes['ICD9_CODE'].apply(safe_extract_category).fillna('INVALID')
    valid_codes = valid_codes[valid_codes['CATEGORY'] != 'INVALID']
    
    # Calculate prevalence for each category (3-digit code)
    admissions_per_category = valid_codes.groupby('CATEGORY')['hadm_id'].nunique()
    total_admissions = df[hadm_col_actual].nunique()
    prevalence = admissions_per_category / total_admissions
    
    # Select categories that meet the minimum prevalence threshold
    top_categories = prevalence[prevalence >= min_prevalence].index.tolist()
    
    print(f"  ✓ Found {len(top_categories)} subcategories meeting the minimum prevalence.")
    
    # Create binary flags for the selected categories
    for category in top_categories:
        flag_col = f'has_icd9_{category}'
        
        # Admissions that have a diagnosis starting with this category
        admissions_with_category = valid_codes[
            valid_codes['CATEGORY'] == category
        ]['hadm_id'].unique()
        
        df[flag_col] = df[hadm_col_actual].isin(admissions_with_category).astype(int)
        
    print(f"  ✓ Added {len(top_categories)} new binary flags.")
    
    return df


def add_high_risk_diagnosis_flags(main_df, diagnosis_df, hadm_col='hadm_id'):
    """
    V4 & V5 Flags: Add binary flags for specific high-mortality conditions.
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
        # V5 Flags
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
    
    print(f"\n[High-Risk Diagnosis Flags - V4/V5]")
    
    diag_df['ICD9_CODE'] = diag_df['ICD9_CODE'].astype(str).str.strip().str.upper()
    
    # Sepsis (V4)
    sepsis_codes = (
        diag_df['ICD9_CODE'].str.startswith('038') |
        diag_df['ICD9_CODE'].isin(['99591', '99592', '995.91', '995.92'])
    )
    sepsis_admissions = diag_df[sepsis_codes]['hadm_id'].unique()
    df['has_sepsis'] = df[hadm_col_actual].isin(sepsis_admissions).astype(int)
    print(f"  ✓ Sepsis: {df['has_sepsis'].sum()}")
    
    # Respiratory failure (V4)
    resp_failure_codes = (
        diag_df['ICD9_CODE'].isin(['51881', '51884', '518.81', '518.84']) |
        diag_df['ICD9_CODE'].str.startswith('967')
    )
    resp_failure_admissions = diag_df[resp_failure_codes]['hadm_id'].unique()
    df['has_respiratory_failure'] = df[hadm_col_actual].isin(resp_failure_admissions).astype(int)
    print(f"  ✓ Respiratory failure: {df['has_respiratory_failure'].sum()}")
    
    # Shock (V4)
    shock_codes = (
        diag_df['ICD9_CODE'].str.startswith('7855') |
        diag_df['ICD9_CODE'].str.startswith('9980') |
        diag_df['ICD9_CODE'].str.startswith('785.5') |
        diag_df['ICD9_CODE'].str.startswith('998.0')
    )
    shock_admissions = diag_df[shock_codes]['hadm_id'].unique()
    df['has_shock'] = df[hadm_col_actual].isin(shock_admissions).astype(int)
    print(f"  ✓ Shock: {df['has_shock'].sum()}")
    
    # Acute MI (V4)
    acute_mi_codes = diag_df['ICD9_CODE'].str.startswith('410')
    acute_mi_admissions = diag_df[acute_mi_codes]['hadm_id'].unique()
    df['has_acute_mi'] = df[hadm_col_actual].isin(acute_mi_admissions).astype(int)
    print(f"  ✓ Acute MI: {df['has_acute_mi'].sum()}")
    
    # Heart failure (V4)
    heart_failure_codes = diag_df['ICD9_CODE'].str.startswith('428')
    heart_failure_admissions = diag_df[heart_failure_codes]['hadm_id'].unique()
    df['has_heart_failure'] = df[hadm_col_actual].isin(heart_failure_admissions).astype(int)
    print(f"  ✓ Heart failure: {df['has_heart_failure'].sum()}")
    
    # Acute kidney injury (V5)
    aki_codes = diag_df['ICD9_CODE'].str.startswith('584')
    aki_admissions = diag_df[aki_codes]['hadm_id'].unique()
    df['has_acute_kidney_injury'] = df[hadm_col_actual].isin(aki_admissions).astype(int)
    print(f"  ✓ Acute kidney injury: {df['has_acute_kidney_injury'].sum()}")
    
    # Pneumonia (V5)
    pneumonia_codes = (
        diag_df['ICD9_CODE'].isin(['486', '485']) |
        diag_df['ICD9_CODE'].str.startswith('481') |
        diag_df['ICD9_CODE'].str.startswith('482') |
        diag_df['ICD9_CODE'].str.startswith('483')
    )
    pneumonia_admissions = diag_df[pneumonia_codes]['hadm_id'].unique()
    df['has_pneumonia'] = df[hadm_col_actual].isin(pneumonia_admissions).astype(int)
    print(f"  ✓ Pneumonia: {df['has_pneumonia'].sum()}")
    
    # Stroke (V5)
    stroke_codes = diag_df['ICD9_CODE'].str.startswith('43')
    stroke_admissions = diag_df[stroke_codes]['hadm_id'].unique()
    df['has_stroke'] = df[hadm_col_actual].isin(stroke_admissions).astype(int)
    print(f"  ✓ Stroke: {df['has_stroke'].sum()}")
    
    # Pulmonary embolism (V5)
    pe_codes = diag_df['ICD9_CODE'].str.startswith('4151') | diag_df['ICD9_CODE'].str.startswith('415.1')
    pe_admissions = diag_df[pe_codes]['hadm_id'].unique()
    df['has_pulmonary_embolism'] = df[hadm_col_actual].isin(pe_admissions).astype(int)
    print(f"  ✓ Pulmonary embolism: {df['has_pulmonary_embolism'].sum()}")
    
    # ARDS (V5)
    ards_codes = diag_df['ICD9_CODE'].isin(['51882', '5185', '518.82', '518.5'])
    ards_admissions = diag_df[ards_codes]['hadm_id'].unique()
    df['has_ards'] = df[hadm_col_actual].isin(ards_admissions).astype(int)
    print(f"  ✓ ARDS: {df['has_ards'].sum()}")
    
    # GI bleeding (V5)
    gi_bleed_codes = diag_df['ICD9_CODE'].str.startswith('578')
    gi_bleed_admissions = diag_df[gi_bleed_codes]['hadm_id'].unique()
    df['has_gi_bleeding'] = df[hadm_col_actual].isin(gi_bleed_admissions).astype(int)
    print(f"  ✓ GI bleeding: {df['has_gi_bleeding'].sum()}")
    
    return df


def calculate_charlson_score(main_df, diagnosis_df, hadm_col='hadm_id'):
    """V5: Calculate Charlson Comorbidity Index - validated mortality predictor"""
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
    
    # Mapping logic is complex, simplified for brevity here, but assumes correct logic from previous script
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
    
    print(f"  ✓ Charlson scores calculated. Mean: {df['charlson_score'].mean():.2f}")
    
    return df


def prepare_enhanced_diagnosis_features(main_df, diagnosis_df, hadm_col='hadm_id'):
    """V3: Enhanced diagnosis features - chapters, counts, flags."""
    df = main_df.copy()
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        return df
    
    diag_df = diagnosis_df.copy()
    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"\n[Enhanced Diagnosis Features - V3]")
    
    diag_df['icd9_chapter'] = diag_df['ICD9_CODE'].apply(get_icd9_chapter)
    
    # Primary + Secondary Chapter
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
    
    print(f"  ✓ Primary/Secondary Chapters added")
    
    # Counts per chapter
    chapter_counts = diag_df.groupby(['hadm_id', 'icd9_chapter']).size().reset_index(name='count')
    chapter_pivot = chapter_counts.pivot(index='hadm_id', columns='icd9_chapter', values='count').fillna(0)
    df = df.merge(chapter_pivot, left_on=hadm_col_actual, right_index=True, how='left')
    
    major_chapters = ['INFECTIOUS', 'CIRCULATORY', 'RESPIRATORY', 'DIGESTIVE', 
                     'GENITOURINARY', 'NEOPLASMS', 'INJURY', 'SYMPTOMS', 'NERVOUS', 
                     'ENDOCRINE', 'BLOOD', 'MENTAL', 'SKIN', 'MUSCULOSKELETAL', 
                     'V_CODES', 'E_CODES', 'UNKNOWN']
    
    for chapter in major_chapters:
        new_col = f'{chapter.lower()}_dx_count'
        if chapter in df.columns:
            df[new_col] = df[chapter].fillna(0).astype(int)
            df = df.drop(columns=[chapter])
        else:
            df[new_col] = 0
    
    print(f"  ✓ Chapter Counts added")
    
    # Total count and Diversity
    dx_counts = diag_df.groupby('hadm_id').size().reset_index(name='diagnosis_count')
    df = df.merge(dx_counts, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['diagnosis_count'] = df['diagnosis_count'].fillna(0).astype(int)
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    organ_counts = diag_df.groupby('hadm_id')['icd9_chapter'].nunique().reset_index(name='num_organ_systems')
    df = df.merge(organ_counts, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['num_organ_systems'] = df['num_organ_systems'].fillna(0).astype(int)
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])

    print(f"  ✓ Diagnosis Metrics added")
    
    # Binary flags (based on Chapter presence)
    respiratory_admissions = diag_df[diag_df['icd9_chapter'] == 'RESPIRATORY']['hadm_id'].unique()
    cardiac_admissions = diag_df[diag_df['icd9_chapter'] == 'CIRCULATORY']['hadm_id'].unique()
    infection_admissions = diag_df[diag_df['icd9_chapter'] == 'INFECTIOUS']['hadm_id'].unique()
    
    df['has_respiratory'] = df[hadm_col_actual].isin(respiratory_admissions).astype(int)
    df['has_cardiac'] = df[hadm_col_actual].isin(cardiac_admissions).astype(int)
    df['has_infection'] = df[hadm_col_actual].isin(infection_admissions).astype(int)
    
    print(f"  ✓ Chapter Flags added")
    
    return df


def add_icu_history_features(df: pd.DataFrame, subject_col: str = 'subject_id', admit_col: str = 'ADMITTIME') -> pd.DataFrame:
    """Add ICU history features (readmission, days since last ICU, etc.)."""
    df = df.copy()
    
    df['_original_index'] = range(len(df))
    
    # Convert to datetime and sort BY PATIENT AND TIME
    df[admit_col] = pd.to_datetime(df[admit_col], errors='coerce')
    df = df.sort_values([subject_col, admit_col])
    
    # Feature 1: Number of previous stays
    df['num_previous_icu_stays'] = df.groupby(subject_col).cumcount()
    
    # Feature 2: Binary flag (Readmission)
    df['is_readmission'] = (df['num_previous_icu_stays'] > 0).astype(int)
    
    # Feature 3: Days since last ICU
    df['_prev_admit'] = df.groupby(subject_col)[admit_col].shift(1)
    df['days_since_last_icu'] = (df[admit_col] - df['_prev_admit']).dt.days
    df['days_since_last_icu'] = df['days_since_last_icu'].fillna(9999).clip(lower=0)
    
    # Feature 4: Readmission 30 days
    df['readmission_30d'] = ((df['days_since_last_icu'] <= 30) & (df['days_since_last_icu'] < 9999)).astype(int)
    
    # Feature 5: Log transforms (keeping log days)
    df['num_previous_icu_stays_log'] = np.log1p(df['num_previous_icu_stays'])
    df['days_since_last_icu_log'] = np.log1p(df['days_since_last_icu'].clip(0, 9998))
    
    # Restore order
    df = df.sort_values('_original_index').drop(columns=['_prev_admit', '_original_index'])
    df = df.reset_index(drop=True)
    
    return df


def clean_min_bp_outliers(df: pd.DataFrame, lower_bounds: dict = BP_MIN_LOWER_BOUNDS) -> pd.DataFrame:
    """Clean implausible minimum BP outliers."""
    df = df.copy()
    for col, low in lower_bounds.items():
        if col not in df.columns: continue
        below_mask = (df[col] < low) & df[col].notna()
        if below_mask.sum() > 0:
            df.loc[below_mask, col] = np.nan
    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate age features and temporal admission features."""
    df = df.copy()
    # Note: DOB is dropped after this step by the main script
    if 'DOB' not in df.columns or 'ADMITTIME' not in df.columns: return df

    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'], errors='coerce')

    df['age_years'] = (df['ADMITTIME'].dt.year - df['DOB'].dt.year).astype(float)
    birthday_not_passed = (df['ADMITTIME'].dt.month < df['DOB'].dt.month) | \
                          ((df['ADMITTIME'].dt.month == df['DOB'].dt.month) & (df['ADMITTIME'].dt.day < df['DOB'].dt.day))
    df.loc[birthday_not_passed, 'age_years'] -= 1
    df.loc[df['age_years'] < 0, 'age_years'] = np.nan

    # Handle censored ages
    df['is_censored_age'] = (df['age_years'] > 89).astype(int)
    df.loc[df['age_years'] > 89, 'age_years'] = 90

    # Age risk flags
    df['is_elderly'] = (df['age_years'] >= 75).astype(int)
    df['is_very_elderly'] = (df['age_years'] >= 85).astype(int)
    
    # Drop original DOB
    df = df.drop(columns=['DOB'], errors='ignore')
    
    return df


def add_engineered_features_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vitals-based engineered features (Core set: Variability, Ratios, Deviations, Instability).
    """
    df = df.copy()
    
    print("[Core Engineered Features]")

    # --- Variability (Ranges) ---
    df['HR_range'] = df['HeartRate_Max'] - df['HeartRate_Min']
    df['SysBP_range'] = df['SysBP_Max'] - df['SysBP_Min']
    df['DiasBP_range'] = df['DiasBP_Max'] - df['DiasBP_Min']
    df['MeanBP_range'] = df['MeanBP_Max'] - df['MeanBP_Min']
    df['RespRate_range'] = df['RespRate_Max'] - df['RespRate_Min']
    df['SpO2_range'] = df['SpO2_Max'] - df['SpO2_Min']
    print("  ✓ Variability features (Ranges) added.")

    # --- Ratios ---
    # Shock Index: HR / SBP_Mean (High is bad)
    df['shock_index_mean'] = df['HeartRate_Mean'] / df['SysBP_Mean']
    df['shock_index_min'] = df['HeartRate_Min'] / df['SysBP_Max'] # Less-bad/Less-acute index
    # Pulse Pressure: SBP - DBP (Low/High is bad)
    df['pulse_pressure_mean'] = df['SysBP_Mean'] - df['DiasBP_Mean']
    print("  ✓ Ratio features (Shock Index, Pulse Pressure) added.")

    # --- Deviations / Deficits (Closeness to dangerous thresholds) ---
    # Oxygenation: Below 92% is generally concerning
    df['spo2_deficit'] = np.maximum(0, 92 - df['SpO2_Min'])
    
    # Temperature: Deviation from 37C (High or Low is bad)
    df['temp_dev_mean'] = np.abs(df['TempC_Mean'] - 37.0)
    
    # Glucose: Deviation from 100 mg/dL (High is bad, but low is also dangerous)
    df['glucose_excess'] = np.maximum(0, df['Glucose_Max'] - 100)
    df['glucose_deficit'] = np.maximum(0, 70 - df['Glucose_Min'])
    print("  ✓ Deviation/Deficit features added.")

    # --- Instability Count (Sum of Abnormal Vitals) ---
    df['instability_count'] = (
        (df['HeartRate_Max'] > 120).astype(int) + 
        (df['HeartRate_Min'] < 50).astype(int) + 
        (df['SysBP_Min'] < 90).astype(int) + 
        (df['RespRate_Max'] > 30).astype(int) + 
        (df['RespRate_Min'] < 8).astype(int) + 
        (df['SpO2_Min'] < 90).astype(int) + 
        (df['TempC_Max'] > 38.5).astype(int) + 
        (df['TempC_Min'] < 36.0).astype(int)
    )
    print("  ✓ Instability Count added.")

    # --- Interaction ---
    # Combined hypoxemia and tachypnea (both increase mortality risk)
    df['hypoxia_and_tachypnea'] = ((df['SpO2_Min'] < 92) & (df['RespRate_Max'] > 25)).astype(int)
    print("  ✓ Interaction features added.")

    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    V8: Adds interaction features between strong diagnosis flags (V7) 
    and strong physiological instability metrics (Vitals/Engineered).
    """
    df = df.copy()
    
    # Check for core columns. If any are missing, skip interactions.
    required_cols = ['has_sepsis', 'has_shock', 'has_respiratory_failure', 
                     'instability_count', 'SysBP_Min', 'spo2_deficit', 
                     'shock_index_mean']
    
    if not all(col in df.columns for col in required_cols):
        print("⚠️ Warning: Missing core columns for V8 interactions. Skipping.")
        return df
    
    print("\n[V8 Interaction Features]")

    # Interaction 1: Sepsis + Hypotension (Acute Shock)
    # Combines the Sepsis flag with the lowest measured blood pressure
    df['sepsis_x_sysbp_min'] = df['has_sepsis'] * df['SysBP_Min']
    
    # Interaction 2: Respiratory Failure + Hypoxia Deficit
    # Captures patients with diagnosed failure experiencing severe oxygen drops
    df['resp_fail_x_spo2_deficit'] = df['has_respiratory_failure'] * df['spo2_deficit']
    
    # Interaction 3: Shock + Multi-system Instability
    # The presence of shock combined with high overall physiological chaos
    df['shock_x_instability'] = df['has_shock'] * df['instability_count']

    # Interaction 4: AKI + Hemodynamic Stress (using Shock Index mean)
    # Combines Acute Kidney Injury (has_icd9_584) with average hemodynamic stress
    if 'has_icd9_584' in df.columns:
        df['aki_x_shock_index'] = df['has_icd9_584'] * df['shock_index_mean']
        print("  ✓ AKI x Shock Index added.")
    else:
        print("  - AKI x Shock Index skipped (has_icd9_584 not found).")
    
    print("  ✓ Core interaction features added.")
    
    return df
# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================

def run_v7_pipeline():
    print("="*80)
    print("XGBOOST V7 - HEF CLASSIFICATION (SELF-CONTAINED)")
    print("="*80)
    print("\n✨ Features:")
    print("  ✓ Vitals, Age, Core Engineered Features")
    print("  ✓ ICU History (Readmission)")
    print("  ✓ Charlson Index + 11 High-Risk Flags")
    print("  ✓ ICD-9 Subcategory Flags (Min 0.5% prevalence)")
    print("  ✓ V7 Sequence Deterioration Flags (e.g., shock_is_secondary)")
    print("  ✓ Cleaned zero-importance features")
    print()
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 1: LOAD RAW DATA")
    print("="*80)
    train_raw, test_raw = load_raw_data()
    diagnosis_df = load_diagnosis_data()
    
    test_ids = test_raw[[c for c in test_raw.columns if c.lower() == 'icustay_id'][0]].values
    
    # ============================================================================
    # STEP 2: MERGE ENHANCED DIAGNOSIS DATA
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 2: MERGE DIAGNOSIS DATA (V7 Features)")
    print("="*80)
    
    train_with_dx = prepare_enhanced_diagnosis_features(train_raw, diagnosis_df)
    test_with_dx = prepare_enhanced_diagnosis_features(test_raw, diagnosis_df)

    # V4 & V5 High-Risk Flags
    train_with_dx = add_high_risk_diagnosis_flags(train_with_dx, diagnosis_df)
    test_with_dx = add_high_risk_diagnosis_flags(test_with_dx, diagnosis_df)

    # V5 Charlson Comorbidity Index
    train_with_dx = calculate_charlson_score(train_with_dx, diagnosis_df)
    test_with_dx = calculate_charlson_score(test_with_dx, diagnosis_df)

    # V6 ICD-9 Subcategory Flags
    train_with_dx = add_icd9_subcategory_flags(train_with_dx, diagnosis_df)
    test_with_dx = add_icd9_subcategory_flags(test_with_dx, diagnosis_df)

    # V7 Diagnosis Sequence Deterioration Flags
    train_with_dx = add_diagnosis_sequence_features(train_with_dx, diagnosis_df)
    test_with_dx = add_diagnosis_sequence_features(test_with_dx, diagnosis_df)
    
    print(f"\n✓ After all diagnosis features - Train: {train_with_dx.shape}, Test: {test_with_dx.shape}")
    
    # Remove 'Diff' column if present
    for col in ['Diff']:
        if col in train_with_dx.columns:
            train_with_dx = train_with_dx.drop(columns=[col])
        if col in test_with_dx.columns:
            test_with_dx = test_with_dx.drop(columns=[col])

    # ============================================================================
    # STEP 3: SPLIT FEATURES AND TARGET + ADD ICU HISTORY
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 3: SPLIT FEATURES, ADD ICU HISTORY")
    print("="*80)

    leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]
    
    # Pass ADMITTIME and subject_id to X_train_raw/X_test_raw
    X_train_raw, y_train, X_test_raw = split_features_target(
        train_df=train_with_dx,
        test_df=test_with_dx,
        task="class",
        leak_cols=leak_cols,
        id_cols=[c for c in ID_COLS if c != 'subject_id' and c != 'intime'] # ADMITTIME is used instead of intime
    )
    
    print(f"✓ Target distribution: {y_train.mean():.3f}")
    
    print("\n[3b] Adding ICU history features...")
    X_train_raw = add_icu_history_features(X_train_raw, subject_col='subject_id', admit_col='ADMITTIME')
    X_test_raw = add_icu_history_features(X_test_raw, subject_col='subject_id', admit_col='ADMITTIME')

    # Drop subject_id and ADMITTIME now that history features are calculated
    X_train_raw = X_train_raw.drop(columns=['subject_id', 'ADMITTIME'], errors='ignore')
    X_test_raw = X_test_raw.drop(columns=['subject_id', 'ADMITTIME'], errors='ignore')

    # ============================================================================
    # STEP 4: CORE FEATURES & CLEANUP
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 4: CORE FEATURES & CLEANUP")
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

    print("\n[4d] Adding interaction engineered features...")
    X_train_ix = add_interaction_features(X_train_fe)
    X_test_ix = add_interaction_features(X_test_fe)

    X_train_final = X_train_ix
    X_test_final = X_test_ix
    
    
    print("\n[4d] Removing zero-importance features (V5/V6 cleanup)...")
    
    zero_importance_cols = [
        'has_pulmonary_embolism', 'has_respiratory', 'has_infection', 'has_acute_mi', 
        'readmission_90d', 'is_frequent_flyer', 'is_first_icu_stay'
    ]
    # Add Chapter columns that showed zero importance
    zero_importance_chapters = ['PREGNANCY_dx_count', 'MISSING_dx_count', 'CONGENITAL_dx_count', 'PERINATAL_dx_count']
    zero_importance_cols.extend(zero_importance_chapters)

    X_train_final = X_train_final.drop(columns=[c for c in zero_importance_cols if c in X_train_final.columns], errors='ignore')
    X_test_final = X_test_final.drop(columns=[c for c in zero_importance_cols if c in X_test_final.columns], errors='ignore')
    print(f"  ✓ Removed {len(zero_importance_cols)} previously identified zero-importance flags/chapters.")
    print(f"\n✓ Features after core engineering and cleanup: {X_train_final.shape[1]}")

    # ============================================================================
    # STEP 5: ENCODE CATEGORICAL FEATURES
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 5: ENCODE CATEGORICAL FEATURES")
    print("="*80)
    
    cat_cols = X_train_final.select_dtypes(include=['object']).columns.tolist()

    # Diagnosis chapters and ICD-9 Subcategories will be Target Encoded
    target_encode_cols = ['primary_diagnosis_chapter', 'secondary_diagnosis_chapter']
    target_encode_cols.extend([c for c in X_train_final.columns if c.startswith('has_icd9_') and X_train_final[c].dtype == 'object']) # Safety check

    # Remove target encoded cols from label encoding list
    label_encode_cols = [c for c in cat_cols if c not in target_encode_cols]

    print(f"✓ Target encoding: {target_encode_cols}")
    print(f"✓ Label encoding: {len(label_encode_cols)} columns")

    # Label Encoding
    for col in label_encode_cols:
        le = LabelEncoder()
        # Ensure we fit on the combined data to handle unseen categories
        combined = pd.concat([X_train_final[col], X_test_final[col]]).astype(str)
        le.fit(combined)
        X_train_final[col] = le.transform(X_train_final[col].astype(str))
        X_test_final[col] = le.transform(X_test_final[col].astype(str))

    # Convert object types in target_encode_cols to string for safety
    for col in target_encode_cols:
        if col in X_train_final.columns:
            X_train_final[col] = X_train_final[col].astype(str)
        if col in X_test_final.columns:
            X_test_final[col] = X_test_final[col].astype(str)


    print(f"\n✓ Final feature count: {X_train_final.shape[1]}")

    # ============================================================================
    # STEP 6: BUILD PIPELINE
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 6: BUILD PIPELINE (TARGET ENCODE INSIDE)")
    print("="*80)

    # Calculate scale_pos_weight to handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # TargetEncoder is inside the pipeline to prevent CV leakage
    pipeline = Pipeline([
        ('target_encoder', TargetEncoder(
            cols=target_encode_cols,
            smoothing=20.0,
            min_samples_leaf=20,
            return_df=True
        )),
        ('classifier', XGBClassifier(
            max_depth=3,
            learning_rate=0.01,
            n_estimators=500,  # For CV
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
            n_jobs=-1
        ))
    ])

    print("✓ Pipeline created (Target Encoding is safe for CV)")

# ============================================================================
# STEP 7: HYPERPARAMETER TUNING (RANDOMIZED SEARCH)
# ============================================================================

print("\n" + "="*80)
print("STEP 7: HYPERPARAMETER TUNING (RANDOMIZED SEARCH)")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the hyperparameter search space
param_dist = {
    'classifier__n_estimators': randint(800, 2000), # Increased max to handle lower learning rate
    'classifier__learning_rate': uniform(0.03, 0.08), # Ranged from 0.03 up to 0.11 (0.03 + 0.08)
    'classifier__max_depth': randint(4, 7), # Explore depths 4, 5, 6
    'classifier__min_child_weight': randint(5, 20),
    'classifier__subsample': uniform(0.65, 0.25), # Range 0.65 to 0.90
    'classifier__gamma': uniform(0.1, 0.9), # Range 0.1 to 1.0
    'classifier__reg_alpha': uniform(0.5, 1.5) # Range 0.5 to 2.0
}

# Perform Randomized Search Cross-Validation
# n_iter=25 is a good balance between speed and coverage
random_search = RandomizedSearchCV(
    estimator=pipeline, 
    param_distributions=param_dist, 
    n_iter=25, 
    scoring='roc_auc', 
    cv=cv, 
    verbose=1, 
    random_state=42, 
    n_jobs=-1
)

print("\n  Starting Randomized Search (25 iterations, 5-Fold CV)...")
random_search.fit(X_train_final, y_train)

# Store the best model and parameters
best_params = random_search.best_params_
best_score = random_search.best_score_
best_estimator = random_search.best_estimator_

# Extract the final AUC result
final_cv_score = best_estimator.named_steps['classifier'].get_booster().best_score

print(f"\n✓ Randomized Search Complete!")
print(f"  Best Mean CV AUC: {best_score:.4f}")
print(f"  Best Parameters Found:")
for k, v in best_params.items():
    print(f"    - {k.split('__')[1]}: {v:.4f}")

# Store results for use in Step 8
# NOTE: We need to update the XGBoost model in the pipeline object with the best parameters
    pipeline.set_params(**best_params)

    # We need to explicitly store the best parameters from the classifier step 
    # to use in the final model training in Step 8
    BEST_XGB_PARAMS = {k.split('__')[1]: v for k, v in best_params.items()}

    # ============================================================================
    # STEP 8: TRAIN FINAL MODEL
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 8: TRAIN FINAL MODEL WITH EARLY STOPPING")
    print("="*80)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # 1. Get the list of columns from the training data (which will define the schema)
    train_cols = X_tr.columns

# 2. Add missing columns to X_val and X_test_final (and fill with 0/NaN)
    missing_in_val = set(train_cols) - set(X_val.columns)
    missing_in_test = set(train_cols) - set(X_test_final.columns)

    for col in missing_in_val:
        X_val[col] = 0.0 # New flags/counts are usually zero
    for col in missing_in_test:
        X_test_final[col] = 0.0

# 3. Drop extra columns from X_val and X_test_final
    extra_in_val = set(X_val.columns) - set(train_cols)
    extra_in_test = set(X_test_final.columns) - set(train_cols)

    X_val = X_val.drop(columns=list(extra_in_val), errors='ignore')
    X_test_final = X_test_final.drop(columns=list(extra_in_test), errors='ignore')

# 4. Re-index X_val and X_test_final to ensure column order is identical to X_tr
    X_val = X_val.reindex(columns=train_cols)
    X_test_final = X_test_final.reindex(columns=train_cols)

# Ensure data types are consistent after synchronization (critical for TargetEncoder)
# This converts all feature-engineered strings back to objects (if they weren't label-encoded)
    for col in X_val.columns:
        if X_tr[col].dtype.kind in np.typecodes['AllFloat'] and X_val[col].dtype == 'object':
            X_val[col] = X_val[col].astype(X_tr[col].dtype)
        if X_tr[col].dtype.kind == 'O' and X_val[col].dtype != 'object':
            X_val[col] = X_val[col].astype(str).astype('object')
        
    for col in X_test_final.columns:
        if X_tr[col].dtype.kind in np.typecodes['AllFloat'] and X_test_final[col].dtype == 'object':
            X_test_final[col] = X_test_final[col].astype(X_tr[col].dtype)
        if X_tr[col].dtype.kind == 'O' and X_test_final[col].dtype != 'object':
            X_test_final[col] = X_test_final[col].astype(str).astype('object')


# END FIX: COLUMN SYNCHRONIZATION
# -------------------------------------------------------------------------
    # Manual Target Encoding for early stopping training (XGBoost requires pre-encoded data here)
    target_enc = TargetEncoder(
        cols=target_encode_cols, smoothing=20.0, min_samples_leaf=20, return_df=True
    )

    X_tr_encoded = target_enc.fit_transform(X_tr, y_tr)
    X_val_encoded = target_enc.transform(X_val)
    X_test_encoded = target_enc.transform(X_test_final)

    # Extract the best parameters found in Step 7, falling back to a default if Step 7 was skipped
# Note: BEST_XGB_PARAMS is defined in Step 7
    tuning_params = globals().get('BEST_XGB_PARAMS', {})

# Define the model, using the best found parameters but overriding n_estimators for safety
# and ensuring early_stopping_rounds is present for final training.
    model = XGBClassifier(
        **tuning_params,
        n_estimators=tuning_params.get('n_estimators', 1000) * 2, # Increase max estimators for final run
        scale_pos_weight=scale_pos_weight,
        random_state=42, 
        tree_method='hist',
        eval_metric='auc', 
        early_stopping_rounds=100, # Increased early stopping rounds for stability
        n_jobs=-1
        )

    print("\n  Training...")
    model.fit(
        X_tr_encoded, y_tr,
        eval_set=[(X_tr_encoded, y_tr), (X_val_encoded, y_val)],
        verbose=50
    )

    print(f"\n✓ Best iteration: {model.best_iteration}")

    # ============================================================================
    # STEP 9: EVALUATE & PREDICT
    # ============================================================================

    y_val_pred = model.predict_proba(X_val_encoded)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    print(f"\n✓ Validation AUC: {val_auc:.4f}")

    y_test_proba = model.predict_proba(X_test_encoded)[:, 1]

    # ============================================================================
    # STEP 10: CREATE SUBMISSION
    # ============================================================================
    output_dir = BASE_DIR / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "xgb_v7_hef_sequenced.csv"

    submission = pd.DataFrame({
        'icustay_id': test_ids,
        'HOSPITAL_EXPIRE_FLAG': y_test_proba
    })

    submission.to_csv(output_file, index=False)
    print(f"\n✓ Submission saved: {output_file}")
    
    # ============================================================================
    # FEATURE IMPORTANCE ANALYSIS
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

    print("\n" + "="*80)
    print("SUMMARY - V7 SEQUENCE DETERIORATION")
    print("="*80)
    
    print(f"\n📊 Performance:")
    print(f"  Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Validation AUC: {val_auc:.4f}")
    
    print(f"\n✨ V7 Enhancements:")
    print("  ✓ Finalized feature set with zero-importance cleanup.")
    print("  ✓ Added ICD-9 Subcategory flags for granular diagnosis detail.")
    print("  ✓ Added Diagnosis Sequence Deterioration flags (modeling deterioration).")
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    run_v7_pipeline()