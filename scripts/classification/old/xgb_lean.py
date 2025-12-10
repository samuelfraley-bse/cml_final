"""
XGBoost Classification - LEAN VERSION

This version adds ONLY the top-performing new features to the baseline:
- age_adjusted_critical (#1 in v2)
- is_emergency + emergency interactions (#2, #9, #12)
- critical_illness_score (#3)
- respiratory_distress_score (#5)

Target: Beat 74.6 score with ~90 features instead of 109

Run from project root:
    python scripts/classification/xgb_lean.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Add notebooks/HEF to path
BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

from hef_prep import (
    split_features_target,
    add_age_features,
    clean_min_bp_outliers, 
    add_engineered_features,
    TARGET_COL_CLASS,
    ID_COLS
)

print("="*80)
print("XGBOOST LEAN VERSION - SELECTED TOP FEATURES ONLY")
print("="*80)


def add_lean_age_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ONLY the age interactions that proved valuable.
    Removes low-value features like age bins and weak interactions.
    """
    if 'age_years' not in df.columns:
        return df
    
    df = df.copy()
    
    # TOP PERFORMERS from previous model (keeping these)
    if 'instability_count' in df.columns:
        df['age_x_instability'] = df['age_years'] * df['instability_count']
    
    if 'temp_dev_mean' in df.columns:
        df['age_x_temp_dev'] = df['age_years'] * df['temp_dev_mean']
    
    if 'shock_index_mean' in df.columns:
        df['age_x_shock_index'] = df['age_years'] * df['shock_index_mean']
    
    if 'spo2_deficit' in df.columns:
        df['age_x_spo2_deficit'] = df['age_years'] * df['spo2_deficit']
    
    if 'SpO2_Min' in df.columns:
        df['age_x_spo2_min'] = df['age_years'] * df['SpO2_Min']
    
    if 'SysBP_Min' in df.columns:
        df['age_x_sysbp_min'] = df['age_years'] * df['SysBP_Min']
    
    # ELDERLY FLAGS (keeping - they worked)
    df['is_elderly'] = (df['age_years'] >= 75).astype(int)
    df['is_very_elderly'] = (df['age_years'] >= 85).astype(int)
    
    # ELDERLY × CONDITIONS (keeping top performers only)
    if 'hypotension_flag' in df.columns:
        df['elderly_and_hypotensive'] = (
            (df['is_elderly'] == 1) & (df['hypotension_flag'] == 1)
        ).astype(int)
    
    if 'hypoxia_flag' in df.columns:
        df['elderly_and_hypoxic'] = (
            (df['is_elderly'] == 1) & (df['hypoxia_flag'] == 1)
        ).astype(int)
    
    if 'tachy_flag' in df.columns:
        df['very_elderly_and_tachy'] = (
            (df['is_very_elderly'] == 1) & (df['tachy_flag'] == 1)
        ).astype(int)
    
    print(f"✓ Lean age interactions added")
    return df


def add_lean_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ONLY the composite scores that proved valuable.
    - critical_illness_score (#3)
    - age_adjusted_critical (#1!) 
    - respiratory_distress_score (#5)
    
    Removes: organ_dysfunction_score, cardiovascular_stress_score, 
             multi_organ_dysfunction, shock_state, resp_failure_risk
    """
    df = df.copy()
    
    # CRITICAL ILLNESS SCORE (ranked #3)
    critical_score = pd.Series(0.0, index=df.index)
    
    if 'shock_index_mean' in df.columns:
        critical_score += df['shock_index_mean'] * 10
    
    if 'spo2_deficit' in df.columns:
        critical_score += df['spo2_deficit']
    
    if 'instability_count' in df.columns:
        critical_score += df['instability_count'] * 2
    
    if 'temp_dev_mean' in df.columns:
        critical_score += df['temp_dev_mean'] * 2
    
    df['critical_illness_score'] = critical_score
    
    # AGE-ADJUSTED CRITICAL SCORE (ranked #1!)
    if 'age_years' in df.columns:
        df['age_adjusted_critical'] = df['critical_illness_score'] * (df['age_years'] / 50)
    
    # RESPIRATORY DISTRESS SCORE (ranked #5)
    resp_score = pd.Series(0.0, index=df.index)
    
    if 'spo2_deficit' in df.columns:
        resp_score += df['spo2_deficit']
    
    if 'tachypnea_flag' in df.columns:
        resp_score += df['tachypnea_flag'] * 5
    
    if 'RespRate_Mean' in df.columns:
        resp_score += (df['RespRate_Mean'] - 16).clip(lower=0) * 0.5
    
    df['respiratory_distress_score'] = resp_score
    
    print(f"✓ Lean composite scores added (3 features)")
    return df


def add_lean_admission_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ONLY the admission interactions that proved valuable.
    - is_emergency (#2)
    - emergency_and_elderly (#9)
    - emergency_x_shock (#12)
    
    Removes: is_urgent (0% importance), is_elective, emergency_and_hypotensive, etc.
    """
    df = df.copy()
    
    if 'ADMISSION_TYPE' not in df.columns:
        print(f"⚠️  ADMISSION_TYPE not found")
        return df
    
    if df['ADMISSION_TYPE'].dtype == 'object':
        # EMERGENCY FLAG (ranked #2)
        df['is_emergency'] = (df['ADMISSION_TYPE'].str.upper() == 'EMERGENCY').astype(int)
        
        # EMERGENCY × ELDERLY (ranked #9)
        if 'is_elderly' in df.columns:
            df['emergency_and_elderly'] = (
                (df['is_emergency'] == 1) & (df['is_elderly'] == 1)
            ).astype(int)
        
        # EMERGENCY × SHOCK (ranked #12)
        if 'shock_index_mean' in df.columns:
            df['emergency_x_shock'] = df['is_emergency'] * df['shock_index_mean']
        
        # EMERGENCY × UNSTABLE (ranked #73 but still useful)
        if 'instability_count' in df.columns:
            df['emergency_and_unstable'] = (
                (df['is_emergency'] == 1) & (df['instability_count'] >= 2)
            ).astype(int)
        
        print(f"✓ Lean admission interactions added (4 features)")
    else:
        print(f"⚠️  ADMISSION_TYPE already encoded")
    
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

# Load data
print("\n[1/8] Loading raw data...")
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")
print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

# Check for ICD9
print("\n[2/8] Checking for ICD9_diagnosis column...")
icd9_col = None
for col in train_raw.columns:
    if 'ICD9' in col.upper() and 'DIAG' in col.upper():
        icd9_col = col
        break
if icd9_col:
    print(f"✓ Found ICD9 column: '{icd9_col}'")

# Split features
print("\n[3/8] Splitting features and target...")
leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]

X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw,
    test_df=test_raw,
    task="class",
    leak_cols=leak_cols,
    id_cols=ID_COLS
)

# Remove remaining IDs
id_cols_to_check = ['icustay_id', 'subject_id', 'hadm_id']
remaining_ids_train = [c for c in X_train_raw.columns if c.lower() in id_cols_to_check]
remaining_ids_test = [c for c in X_test_raw.columns if c.lower() in id_cols_to_check]
if remaining_ids_train:
    X_train_raw = X_train_raw.drop(columns=remaining_ids_train)
if remaining_ids_test:
    X_test_raw = X_test_raw.drop(columns=remaining_ids_test)

# Add age features
print("\n[4/8] Adding age features...")
X_train_age = add_age_features(X_train_raw)
X_test_age = add_age_features(X_test_raw)

# Clean BP
print("\n[5/8] Cleaning BP outliers...")
X_train_clean = clean_min_bp_outliers(X_train_age)
X_test_clean = clean_min_bp_outliers(X_test_age)

# Add engineered features
print("\n[6/8] Adding engineered vital features...")
X_train_fe = add_engineered_features(X_train_clean)
X_test_fe = add_engineered_features(X_test_clean)
print(f"  After vitals engineering: {X_train_fe.shape[1]} columns")

# Add LEAN age interactions (not all of them!)
print("\n[7/8] Adding LEAN feature set...")
X_train_lean = add_lean_age_interactions(X_train_fe)
X_test_lean = add_lean_age_interactions(X_test_fe)

X_train_lean = add_lean_composite_scores(X_train_lean)
X_test_lean = add_lean_composite_scores(X_test_lean)

X_train_lean = add_lean_admission_interactions(X_train_lean)
X_test_lean = add_lean_admission_interactions(X_test_lean)

print(f"  After lean features: {X_train_lean.shape[1]} columns")

# Add ICD9
print("\n[8/8] Processing ICD9 and encoding...")
if icd9_col:
    train_icd9 = train_raw[icd9_col].copy().fillna('MISSING')
    test_icd9 = test_raw[icd9_col].copy().fillna('MISSING')
    
    le = LabelEncoder()
    all_codes = pd.concat([train_icd9, test_icd9])
    le.fit(all_codes)
    
    X_train_lean['ICD9_encoded'] = le.transform(train_icd9)
    X_test_lean['ICD9_encoded'] = le.transform(test_icd9)
    
    train_icd9_cat = train_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    test_icd9_cat = test_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    
    le_cat = LabelEncoder()
    all_cats = pd.concat([train_icd9_cat, test_icd9_cat])
    le_cat.fit(all_cats)
    
    X_train_lean['ICD9_category'] = le_cat.transform(train_icd9_cat)
    X_test_lean['ICD9_category'] = le_cat.transform(test_icd9_cat)

X_train_final = X_train_lean
X_test_final = X_test_lean

# Encode categoricals
cat_cols = X_train_final.select_dtypes(include=['object']).columns.tolist()
if len(cat_cols) > 0:
    print(f"  Encoding {len(cat_cols)} categorical columns")
    for col in cat_cols:
        le_cat = LabelEncoder()
        combined = pd.concat([X_train_final[col], X_test_final[col]]).astype(str)
        le_cat.fit(combined)
        X_train_final[col] = le_cat.transform(X_train_final[col].astype(str))
        X_test_final[col] = le_cat.transform(X_test_final[col].astype(str))

print(f"\n✓ FINAL FEATURE COUNT: {X_train_final.shape[1]} features")
print(f"  (Baseline was 85, bloated was 109, lean target ~90)")

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n" + "="*80)
print("TRAINING XGBOOST MODEL")
print("="*80)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print(f"\n  Train: {X_tr.shape[0]} samples, Valid: {X_val.shape[0]} samples")

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

model = XGBClassifier(
    max_depth=5,
    learning_rate=0.03,
    n_estimators=1500,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    early_stopping_rounds=50,
    n_jobs=-1
)

print("\n  Training...")
model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=50
)

print(f"\n  ✓ Training complete! Best iteration: {model.best_iteration}")

# ============================================================================
# EVALUATE
# ============================================================================
print("\n" + "="*80)
print("EVALUATION")
print("="*80)

y_val_pred = model.predict_proba(X_val)[:, 1]
y_test_pred = model.predict_proba(X_test_final)[:, 1]

val_auc = roc_auc_score(y_val, y_val_pred)
print(f"\n✓ Validation AUC: {val_auc:.4f}")

y_val_binary = (y_val_pred >= 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_val, y_val_binary, target_names=['Survived', 'Died']))

# Feature importance
print("\nTop 25 Most Important Features:")
importance_df = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(25).to_string(index=False))

# ============================================================================
# SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
test_ids = test_raw[id_col].values

submission = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': y_test_pred
})

output_dir = BASE_DIR / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "xgboost_lean.csv"

submission.to_csv(output_file, index=False)

print(f"\n✓ Submission saved: {output_file}")
print(f"\nPrediction stats: Min={y_test_pred.min():.4f}, Max={y_test_pred.max():.4f}, Mean={y_test_pred.mean():.4f}")

print(f"\n📊 Final Model Summary:")
print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Features used: {X_train_final.shape[1]}")
print(f"  Best iteration: {model.best_iteration}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)