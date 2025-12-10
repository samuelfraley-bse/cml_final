"""
XGBoost with Diagnosis Features - V2 (Generalization Focus) - FIXED

CRITICAL FIX: Diagnosis features now merged BEFORE split_features_target!

Changes from V1 to improve generalization (reduce CV=0.83 → Kaggle=0.73 gap):
1. Use CHAPTER-LEVEL diagnosis encoding (broader, more generalizable)
2. Higher smoothing (20 vs 5) for target encoding
3. More XGBoost regularization (simpler trees, lower learning rate)
4. Simpler feature set (remove complex interactions that overfit)
5. Early stopping to prevent overfitting
6. Focus on clinically meaningful, generalizable patterns

FIXED: Diagnosis merge order
- Diagnosis features merged into raw data FIRST (deterministic, no leakage)
- THEN split_features_target removes IDs
- Target encoding happens in Pipeline (per professor's requirement)

Professor's requirement: "Target encoding inside pipeline to avoid data leakage"
✓ Diagnosis merge = deterministic (no target used) = OK before pipeline
✓ Target encoding = uses target = MUST be in pipeline

Goal: Reduce overfitting, improve Kaggle score

Run from project root:
    python scripts/classification/xgb_diagnosis_v2_fixed.py
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
print("XGBOOST V2 - GENERALIZATION FOCUS")
print("="*80)
print("\n🎯 Goal: Reduce overfitting (CV 0.83 → Kaggle 0.73 gap)")
print("\n✓ Broader diagnosis categories (chapter level)")
print("✓ Higher target encoding smoothing (20 vs 5)")
print("✓ More XGBoost regularization")
print("✓ Simpler, generalizable features only")
print("✓ Early stopping\n")

# ============================================================================
# HELPER FUNCTIONS FOR DIAGNOSIS FEATURES
# ============================================================================

def get_icd9_chapter(code):
    """
    Map ICD9 code to broad chapter/organ system.
    These are MORE generalizable than 3-digit codes.
    
    Based on: https://en.wikipedia.org/wiki/List_of_ICD-9_codes
    """
    if pd.isna(code) or code == 'MISSING':
        return 'MISSING'
    
    code_str = str(code).strip().upper()
    
    # Handle V and E codes
    if code_str.startswith('V'):
        return 'V_CODES'
    if code_str.startswith('E'):
        return 'E_CODES'
    
    # Convert to numeric
    try:
        code_num = int(code_str.replace('.', '')[:3])
    except (ValueError, IndexError):
        return 'UNKNOWN'
    
    # Map to chapters (broad categories)
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
        raise FileNotFoundError(
            f"MIMIC_diagnoses.csv not found at {filepath}\n"
            f"Expected location: {base_dir / 'data' / 'raw' / 'MIMIC III dataset HEF' / 'extra_data' / 'MIMIC_diagnoses.csv'}"
        )
    
    print(f"Loading diagnosis data from: {filepath}")
    diagnosis_df = pd.read_csv(filepath)
    
    diagnosis_df.columns = diagnosis_df.columns.str.upper()
    if 'HADM_ID' in diagnosis_df.columns:
        diagnosis_df = diagnosis_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"  ✓ Loaded {len(diagnosis_df):,} diagnosis records")
    print(f"  ✓ Unique admissions: {diagnosis_df['hadm_id'].nunique():,}")
    
    return diagnosis_df


def prepare_generalizable_diagnosis_features(main_df, diagnosis_df, hadm_col='hadm_id'):
    """
    Create GENERALIZABLE diagnosis features.
    
    V2 Changes:
    - Use CHAPTER level instead of 3-digit codes (broader)
    - Add comorbidity count (simple, generalizable)
    - Keep domain flags
    - Remove granular features that overfit
    """
    df = main_df.copy()
    
    # Find hadm_col (case-insensitive)
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        print(f"⚠️  Warning: {hadm_col} not found in data")
        df['primary_diagnosis_chapter'] = 'MISSING'
        df['diagnosis_count'] = 0
        df['num_organ_systems'] = 0
        df['has_respiratory'] = 0
        df['has_cardiac'] = 0
        df['has_infection'] = 0
        return df
    
    diag_df = diagnosis_df.copy()
    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    print(f"\n[Preparing generalizable diagnosis features]")
    print(f"  Main dataset: {len(df):,} rows")
    
    # Add chapter to all diagnosis records
    diag_df['icd9_chapter'] = diag_df['ICD9_CODE'].apply(get_icd9_chapter)
    
    # ========================================================================
    # FEATURE 1: Primary diagnosis CHAPTER (broader than 3-digit)
    # ========================================================================
    primary_dx = diag_df[diag_df['SEQ_NUM'] == 1][['hadm_id', 'icd9_chapter']].copy()
    primary_dx = primary_dx.rename(columns={'icd9_chapter': 'primary_diagnosis_chapter'})
    
    df = df.merge(primary_dx, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['primary_diagnosis_chapter'] = df['primary_diagnosis_chapter'].fillna('MISSING')
    
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    n_unique = df['primary_diagnosis_chapter'].nunique()
    print(f"  ✓ Primary diagnosis chapter: {n_unique} categories (broader than 3-digit)")
    print(f"    Categories: {sorted(df['primary_diagnosis_chapter'].unique())}")
    
    # ========================================================================
    # FEATURE 2: Diagnosis count (simple aggregate)
    # ========================================================================
    dx_counts = diag_df.groupby('hadm_id').size().reset_index(name='diagnosis_count')
    df = df.merge(dx_counts, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['diagnosis_count'] = df['diagnosis_count'].fillna(0).astype(int)
    
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    print(f"  ✓ Diagnosis count: mean={df['diagnosis_count'].mean():.1f}")
    
    # ========================================================================
    # FEATURE 3: Number of unique organ systems affected (comorbidity)
    # ========================================================================
    organ_counts = diag_df.groupby('hadm_id')['icd9_chapter'].nunique().reset_index(name='num_organ_systems')
    df = df.merge(organ_counts, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['num_organ_systems'] = df['num_organ_systems'].fillna(0).astype(int)
    
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    print(f"  ✓ Organ systems affected: mean={df['num_organ_systems'].mean():.1f}")
    
    # ========================================================================
    # FEATURE 4-6: Domain-specific flags (clinically meaningful)
    # ========================================================================
    respiratory_admissions = diag_df[diag_df['icd9_chapter'] == 'RESPIRATORY']['hadm_id'].unique()
    cardiac_admissions = diag_df[diag_df['icd9_chapter'] == 'CIRCULATORY']['hadm_id'].unique()
    infection_admissions = diag_df[diag_df['icd9_chapter'] == 'INFECTIOUS']['hadm_id'].unique()
    
    df['has_respiratory'] = df[hadm_col_actual].isin(respiratory_admissions).astype(int)
    df['has_cardiac'] = df[hadm_col_actual].isin(cardiac_admissions).astype(int)
    df['has_infection'] = df[hadm_col_actual].isin(infection_admissions).astype(int)
    
    print(f"  ✓ Respiratory: {df['has_respiratory'].sum()} ({df['has_respiratory'].mean()*100:.1f}%)")
    print(f"  ✓ Cardiac: {df['has_cardiac'].sum()} ({df['has_cardiac'].mean()*100:.1f}%)")
    print(f"  ✓ Infection: {df['has_infection'].sum()} ({df['has_infection'].mean()*100:.1f}%)")
    
    print(f"\n  ✓ Total: 6 features (1 categorical for target encoding, 5 numeric)")
    
    return df


# ============================================================================
# IMPORT FROM hef_prep.py (ONLY SIMPLE FUNCTIONS)
# ============================================================================

from hef_prep import (
    split_features_target,
    add_age_features,
    clean_min_bp_outliers,
    add_engineered_features,
    TARGET_COL_CLASS,
    ID_COLS
)

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOAD RAW DATA")
print("="*80)

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"✓ Train: {train_raw.shape}")
print(f"✓ Test: {test_raw.shape}")

# Save test IDs BEFORE any processing
test_id_col = None
for col in test_raw.columns:
    if col.lower() == 'icustay_id':
        test_id_col = col
        break

test_ids = test_raw[test_id_col].values
print(f"✓ Test IDs saved: {len(test_ids)} IDs")

# ============================================================================
# STEP 2: LOAD AND MERGE DIAGNOSIS DATA (BEFORE split_features_target!)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: MERGE DIAGNOSIS DATA (DETERMINISTIC - NO LEAKAGE)")
print("="*80)

diagnosis_df = load_diagnosis_data()

# Merge diagnosis features into RAW data (before split)
# This is safe because it's deterministic - no target information used
print("\nMerging diagnosis features into train data...")
train_with_dx = prepare_generalizable_diagnosis_features(train_raw, diagnosis_df)

print("\nMerging diagnosis features into test data...")
test_with_dx = prepare_generalizable_diagnosis_features(test_raw, diagnosis_df)

print(f"\n✓ Train after diagnosis merge: {train_with_dx.shape}")
print(f"✓ Test after diagnosis merge: {test_with_dx.shape}")

# Verify diagnosis features were created
diag_features = ['primary_diagnosis_chapter', 'diagnosis_count', 'num_organ_systems', 
                 'has_respiratory', 'has_cardiac', 'has_infection']
print(f"\n✓ Verifying diagnosis features:")
for feat in diag_features:
    if feat in train_with_dx.columns:
        if feat == 'primary_diagnosis_chapter':
            n_unique = train_with_dx[feat].nunique()
            print(f"  ✓ {feat}: {n_unique} unique values")
        else:
            n_nonzero = (train_with_dx[feat] != 0).sum()
            mean_val = train_with_dx[feat].mean()
            print(f"  ✓ {feat}: mean={mean_val:.2f}, non-zero={n_nonzero}")
    else:
        print(f"  ❌ {feat}: NOT FOUND!")

# ============================================================================
# STEP 3: SPLIT FEATURES AND TARGET (THIS REMOVES hadm_id)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: SPLIT FEATURES AND TARGET")
print("="*80)

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]

X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_with_dx,  # Use data with diagnosis already merged
    test_df=test_with_dx,     # Use data with diagnosis already merged
    task="class",
    leak_cols=leak_cols,
    id_cols=ID_COLS
)

print(f"✓ Target distribution: {y_train.mean():.3f} (death rate)")
print(f"✓ Diagnosis features already included: primary_diagnosis_chapter, diagnosis_count, etc.")

# ============================================================================
# STEP 4: ADD OTHER GENERALIZABLE FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 4: ADD OTHER FEATURES (AGE, VITALS)")
print("="*80)

print("\n[4a] Adding age features...")
X_train_age = add_age_features(X_train_raw)
X_test_age = add_age_features(X_test_raw)

# REMOVE COMPLEX AGE FEATURES that likely overfit
age_features_to_remove = [
    'age_0_40', 'age_40_50', 'age_50_60', 'age_60_70', 'age_70_80', 'age_80_90', 'age_90_plus'
]
X_train_age = X_train_age.drop(columns=[c for c in age_features_to_remove if c in X_train_age.columns], errors='ignore')
X_test_age = X_test_age.drop(columns=[c for c in age_features_to_remove if c in X_test_age.columns], errors='ignore')
print(f"  ✓ Removed {len(age_features_to_remove)} overfitting age bins")

print("\n[4b] Cleaning BP outliers...")
X_train_clean = clean_min_bp_outliers(X_train_age)
X_test_clean = clean_min_bp_outliers(X_test_age)

print("\n[4c] Adding SIMPLE engineered vital features only...")
X_train_fe = add_engineered_features(X_train_clean)
X_test_fe = add_engineered_features(X_test_clean)

# REMOVE COMPLEX INTERACTIONS that likely overfit
complex_features_to_remove = [
    # Age interactions
    'age_x_instability', 'age_x_temp_dev', 'age_x_shock_index', 'age_x_spo2_deficit',
    'age_x_spo2_min', 'age_x_sysbp_min', 'elderly_and_hypotensive', 'elderly_and_hypoxic',
    'very_elderly_and_tachy',
    # Complex composite scores
    'organ_dysfunction_score', 'cardiovascular_stress_score', 'multi_organ_dysfunction',
    'shock_state', 'resp_failure_risk', 'age_adjusted_critical'
]

removed_count = 0
for feat in complex_features_to_remove:
    if feat in X_train_fe.columns:
        X_train_fe = X_train_fe.drop(columns=[feat])
        X_test_fe = X_test_fe.drop(columns=[feat])
        removed_count += 1

print(f"  ✓ Removed {removed_count} complex interaction features")
print(f"  ✓ Features after simplification: {X_train_fe.shape[1]} columns")

X_train_final = X_train_fe
X_test_final = X_test_fe

# ============================================================================
# STEP 5: ENCODE CATEGORICAL FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 5: ENCODE CATEGORICAL FEATURES")
print("="*80)

cat_cols = X_train_final.select_dtypes(include=['object']).columns.tolist()

if 'primary_diagnosis_chapter' in cat_cols:
    cat_cols.remove('primary_diagnosis_chapter')
    print(f"✓ 'primary_diagnosis_chapter' will be target encoded in pipeline")

if len(cat_cols) > 0:
    print(f"✓ Label encoding {len(cat_cols)} categorical columns")
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_final[col], X_test_final[col]]).astype(str)
        le.fit(combined)
        X_train_final[col] = le.transform(X_train_final[col].astype(str))
        X_test_final[col] = le.transform(X_test_final[col].astype(str))

print(f"\n✓ Final feature count: {X_train_final.shape[1]} (simplified from V1)")

# ============================================================================
# STEP 6: BUILD REGULARIZED PIPELINE
# ============================================================================

print("\n" + "="*80)
print("STEP 6: BUILD REGULARIZED PIPELINE")
print("="*80)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"✓ Class imbalance: {scale_pos_weight:.2f}")

print("\n🎯 V2 Changes for Generalization:")
print("  1. Target encoding smoothing: 5 → 20 (more conservative)")
print("  2. XGBoost max_depth: 5 → 3 (simpler trees)")
print("  3. Learning rate: 0.03 → 0.01 (more conservative)")
print("  4. Regularization increased (alpha, lambda, gamma)")
print("  5. More subsampling (0.85 → 0.7)")
print("  6. Early stopping enabled")

pipeline = Pipeline([
    ('target_encoder', TargetEncoder(
        cols=['primary_diagnosis_chapter'],
        smoothing=20.0,  # INCREASED from 5.0 (more conservative)
        min_samples_leaf=20,  # INCREASED from 10
        return_df=True
    )),
    ('classifier', XGBClassifier(
        max_depth=3,              # REDUCED from 5 (simpler trees)
        learning_rate=0.01,       # REDUCED from 0.03 (more conservative)
        n_estimators=1000,        # INCREASED (with early stopping)
        subsample=0.7,            # REDUCED from 0.85 (more bagging)
        colsample_bytree=0.7,     # REDUCED from 0.85
        min_child_weight=10,      # INCREASED from 5 (more regularization)
        gamma=0.5,                # INCREASED from 0.1 (more conservative splits)
        reg_alpha=1.0,            # INCREASED from 0.1 (L1 regularization)
        reg_lambda=2.0,           # INCREASED from 1.0 (L2 regularization)
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist',
        eval_metric='auc',
        n_jobs=-1
    ))
])

print("\n✓ Pipeline created with heavy regularization")

# ============================================================================
# STEP 7: CROSS-VALIDATION
# ============================================================================

print("\n" + "="*80)
print("STEP 7: CROSS-VALIDATION (5-FOLD)")
print("="*80)

print("Running 5-fold stratified cross-validation...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipeline, 
    X_train_final, 
    y_train, 
    cv=cv, 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

print(f"✓ Cross-validation complete!")
print(f"\n  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"\n  📊 Expected: CV score should be LOWER than V1 (0.83)")
print(f"     This is GOOD - means less overfitting!")
print(f"     Kaggle score should IMPROVE (generalization)")

# ============================================================================
# STEP 8: TRAIN WITH EARLY STOPPING
# ============================================================================

print("\n" + "="*80)
print("STEP 8: TRAIN FINAL MODEL WITH EARLY STOPPING")
print("="*80)

# Create validation set for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print(f"  Train: {X_tr.shape[0]:,} samples")
print(f"  Validation: {X_val.shape[0]:,} samples")

# Fit target encoder on training set
target_enc = TargetEncoder(
    cols=['primary_diagnosis_chapter'],
    smoothing=20.0,
    min_samples_leaf=20,
    return_df=True
)

X_tr_encoded = target_enc.fit_transform(X_tr, y_tr)
X_val_encoded = target_enc.transform(X_val)
X_test_encoded = target_enc.transform(X_test_final)

# Train with early stopping
model = XGBClassifier(
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
    early_stopping_rounds=50,  # MOVED HERE (newer XGBoost API)
    n_jobs=-1
)

print("\n  Training with early stopping...")
model.fit(
    X_tr_encoded, y_tr,
    eval_set=[(X_tr_encoded, y_tr), (X_val_encoded, y_val)],
    verbose=50
)

print(f"\n  ✓ Training stopped at iteration: {model.best_iteration}")
print(f"  ✓ Best validation AUC: {model.best_score:.4f}")

# ============================================================================
# STEP 9: EVALUATE
# ============================================================================

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

y_val_pred = model.predict_proba(X_val_encoded)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)

print(f"\n✓ Final validation AUC: {val_auc:.4f}")

# ============================================================================
# STEP 10: GENERATE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 10: GENERATE TEST PREDICTIONS")
print("="*80)

y_test_proba = model.predict_proba(X_test_encoded)[:, 1]

print(f"✓ Predictions generated for {len(y_test_proba)} test samples")
print(f"  Prediction stats:")
print(f"    Min:  {y_test_proba.min():.4f}")
print(f"    Mean: {y_test_proba.mean():.4f}")
print(f"    Max:  {y_test_proba.max():.4f}")

# ============================================================================
# STEP 11: CREATE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("STEP 11: CREATE SUBMISSION FILE")
print("="*80)

submission = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': y_test_proba
})

output_dir = BASE_DIR / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "xgb_diagnosis_v2_generalizable.csv"

submission.to_csv(output_file, index=False)

print(f"✓ Submission file created: {output_file}")
print(f"\nSubmission preview:")
print(submission.head(10))

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY - V2 GENERALIZATION FOCUS")
print("="*80)

print(f"\n📊 Performance:")
print(f"  Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Best iteration: {model.best_iteration}")

print(f"\n🎯 Changes from V1:")
print(f"  ✓ Diagnosis encoding: 3-digit codes → chapter level (broader)")
print(f"  ✓ Smoothing: 5 → 20 (more conservative)")
print(f"  ✓ Max depth: 5 → 3 (simpler trees)")
print(f"  ✓ Learning rate: 0.03 → 0.01 (more conservative)")
print(f"  ✓ Regularization: increased across all parameters")
print(f"  ✓ Features: removed complex interactions")
print(f"  ✓ Early stopping: enabled")

print(f"\n📈 Expected Outcome:")
print(f"  CV score might be slightly LOWER than V1 (0.83)")
print(f"  BUT Kaggle score should IMPROVE (less overfitting!)")
print(f"  Goal: Close the CV→Kaggle gap")

print(f"\n🔍 Diagnosis Features Check:")
diag_features = ['primary_diagnosis_chapter', 'diagnosis_count', 'num_organ_systems', 
                 'has_respiratory', 'has_cardiac', 'has_infection']
importance_df = pd.DataFrame({
    'feature': X_tr_encoded.columns,
    'importance': model.feature_importances_
})
diag_importance = importance_df[importance_df['feature'].isin(diag_features)].sort_values('importance', ascending=False)
if len(diag_importance) > 0:
    print(f"  Diagnosis feature importance:")
    for _, row in diag_importance.iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:.6f}")
    if diag_importance['importance'].sum() > 0:
        print(f"  ✓ Diagnosis features ARE contributing!")
    else:
        print(f"  ⚠️  Diagnosis features have ZERO importance - something may be wrong")
else:
    print(f"  ❌ No diagnosis features found in model!")

print(f"\n📁 Files Created:")
print(f"  {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE! Submit to Kaggle and compare!")
print("="*80)