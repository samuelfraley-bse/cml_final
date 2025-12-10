"""
XGBoost with DIAGNOSIS Text Features

We were dropping the DIAGNOSIS column thinking it was leakage - IT'S NOT!
It's the free-text admission diagnosis, known at admission time.

This script adds keyword extraction from DIAGNOSIS to capture:
- High mortality: SEPSIS, HEMORRHAGE, ARREST, FAILURE, etc.
- Low mortality: CHEST PAIN, DKA, SYNCOPE, etc.

Run from project root:
    python scripts/classification/xgb_with_diagnosis.py
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
    add_age_interactions,
    TARGET_COL_CLASS,
    ID_COLS
)

print("="*80)
print("XGBOOST WITH DIAGNOSIS TEXT FEATURES")
print("="*80)


def extract_diagnosis_features(df: pd.DataFrame, diagnosis_col: str = 'DIAGNOSIS') -> pd.DataFrame:
    """
    Extract binary features from free-text DIAGNOSIS column.
    
    Creates flags for high-risk and low-risk diagnosis keywords.
    """
    df = df.copy()
    
    # Get diagnosis text, uppercase for matching
    diag = df[diagnosis_col].fillna('').str.upper()
    
    # =========================================================================
    # HIGH MORTALITY DIAGNOSES
    # =========================================================================
    
    # Sepsis / Infection (very high mortality)
    df['diag_sepsis'] = diag.str.contains('SEPSIS|SEPTIC', regex=True).astype(int)
    
    # Cardiac arrest / failure
    df['diag_cardiac_arrest'] = diag.str.contains('CARDIAC ARREST|ARREST', regex=True).astype(int)
    df['diag_heart_failure'] = diag.str.contains('HEART FAILURE|CHF|CONGESTIVE', regex=True).astype(int)
    
    # Respiratory failure
    df['diag_resp_failure'] = diag.str.contains('RESPIRATORY FAILURE|RESP FAILURE|ARDS', regex=True).astype(int)
    
    # Hemorrhage / Bleeding (brain)
    df['diag_intracranial_hemorrhage'] = diag.str.contains('INTRACRANIAL HEMORRHAGE|SUBARACHNOID|SUBDURAL|ICH|SAH', regex=True).astype(int)
    
    # Stroke
    df['diag_stroke'] = diag.str.contains('STROKE|CVA|CEREBROVASCULAR', regex=True).astype(int)
    
    # GI Bleed
    df['diag_gi_bleed'] = diag.str.contains('GI BLEED|GASTROINTESTINAL BLEED|UPPER GI|LOWER GI', regex=True).astype(int)
    
    # Acute kidney/renal failure
    df['diag_renal_failure'] = diag.str.contains('RENAL FAILURE|KIDNEY FAILURE|AKI|ACUTE KIDNEY', regex=True).astype(int)
    
    # Shock
    df['diag_shock'] = diag.str.contains('SHOCK|HYPOTENSION', regex=True).astype(int)
    
    # Liver failure
    df['diag_liver_failure'] = diag.str.contains('LIVER FAILURE|HEPATIC FAILURE|CIRRHOSIS', regex=True).astype(int)
    
    # Pneumonia (common, moderate mortality)
    df['diag_pneumonia'] = diag.str.contains('PNEUMONIA|PNA', regex=True).astype(int)
    
    # Trauma
    df['diag_trauma'] = diag.str.contains('TRAUMA|BLUNT|MOTOR VEHICLE|MVA|FALL', regex=True).astype(int)
    
    # Acute coronary syndrome / MI
    df['diag_acs'] = diag.str.contains('CORONARY|MYOCARDIAL|STEMI|NSTEMI|MI |HEART ATTACK|ACS', regex=True).astype(int)
    
    # Altered mental status (can be serious)
    df['diag_altered_mental'] = diag.str.contains('ALTERED MENTAL|AMS|ENCEPHALOPATHY|CONFUSION', regex=True).astype(int)
    
    # Pancreatitis
    df['diag_pancreatitis'] = diag.str.contains('PANCREATITIS', regex=True).astype(int)
    
    # =========================================================================
    # LOWER MORTALITY DIAGNOSES
    # =========================================================================
    
    # Chest pain (often rule-out, lower mortality)
    df['diag_chest_pain'] = diag.str.contains('CHEST PAIN', regex=True).astype(int)
    
    # DKA (treatable)
    df['diag_dka'] = diag.str.contains('DIABETIC KETOACIDOSIS|DKA', regex=True).astype(int)
    
    # Seizure
    df['diag_seizure'] = diag.str.contains('SEIZURE|EPILEP', regex=True).astype(int)
    
    # Syncope
    df['diag_syncope'] = diag.str.contains('SYNCOPE|FAINTING', regex=True).astype(int)
    
    # Overdose
    df['diag_overdose'] = diag.str.contains('OVERDOSE|INTOXICATION|DRUG', regex=True).astype(int)
    
    # Asthma/COPD (manageable)
    df['diag_copd_asthma'] = diag.str.contains('ASTHMA|COPD|CHRONIC OBSTRUCTIVE', regex=True).astype(int)
    
    # UTI
    df['diag_uti'] = diag.str.contains('URINARY TRACT|UTI|PYELONEPHRITIS', regex=True).astype(int)
    
    # =========================================================================
    # SURGICAL FLAGS
    # =========================================================================
    df['diag_surgery'] = diag.str.contains('SURGERY|SURGICAL|/SDA|REPLACEMENT|BYPASS|CABG|GRAFT', regex=True).astype(int)
    df['diag_transplant'] = diag.str.contains('TRANSPLANT', regex=True).astype(int)
    
    # =========================================================================
    # AGGREGATE FEATURES
    # =========================================================================
    
    # Count of high-risk flags
    high_risk_cols = ['diag_sepsis', 'diag_cardiac_arrest', 'diag_resp_failure', 
                      'diag_intracranial_hemorrhage', 'diag_shock', 'diag_liver_failure',
                      'diag_renal_failure']
    df['diag_high_risk_count'] = df[high_risk_cols].sum(axis=1)
    
    # Any high-risk diagnosis
    df['diag_any_high_risk'] = (df['diag_high_risk_count'] > 0).astype(int)
    
    # Count total diagnosis flags
    diag_cols = [c for c in df.columns if c.startswith('diag_') and c not in ['diag_high_risk_count', 'diag_any_high_risk']]
    df['diag_flag_count'] = df[diag_cols].sum(axis=1)
    
    return df


# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/7] Loading raw data...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")
print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# CHECK DIAGNOSIS COLUMN
# ============================================================================
print("\n[2/7] Checking DIAGNOSIS column...")

if 'DIAGNOSIS' in train_raw.columns:
    print(f"  ✓ DIAGNOSIS found!")
    print(f"  Unique diagnoses: {train_raw['DIAGNOSIS'].nunique()}")
    print(f"  Missing: {train_raw['DIAGNOSIS'].isna().sum()}")
else:
    print("  ❌ DIAGNOSIS not found!")
    sys.exit(1)

# Check ICD9 too
icd9_col = None
for col in train_raw.columns:
    if 'ICD9' in col.upper() and 'DIAG' in col.upper():
        icd9_col = col
        break

# ============================================================================
# SPLIT FEATURES - NOW KEEPING DIAGNOSIS!
# ============================================================================
print("\n[3/7] Splitting features (keeping DIAGNOSIS for feature extraction)...")

# Only drop true leakage - NOT diagnosis!
leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "ICD9_diagnosis"]  # Removed DIAGNOSIS!

X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

# Remove IDs
id_cols_to_check = ['icustay_id', 'subject_id', 'hadm_id']
for col in [c for c in X_train_raw.columns if c.lower() in id_cols_to_check]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in id_cols_to_check]:
    X_test_raw = X_test_raw.drop(columns=[col])

print(f"  X_train shape: {X_train_raw.shape}")

# ============================================================================
# EXTRACT DIAGNOSIS FEATURES
# ============================================================================
print("\n[4/7] Extracting DIAGNOSIS text features...")

X_train_diag = extract_diagnosis_features(X_train_raw, 'DIAGNOSIS')
X_test_diag = extract_diagnosis_features(X_test_raw, 'DIAGNOSIS')

# Now drop the raw DIAGNOSIS column (we've extracted features from it)
X_train_diag = X_train_diag.drop(columns=['DIAGNOSIS'])
X_test_diag = X_test_diag.drop(columns=['DIAGNOSIS'])

# Count how many diagnosis features we added
diag_features = [c for c in X_train_diag.columns if c.startswith('diag_')]
print(f"  ✓ Created {len(diag_features)} diagnosis features")
print(f"  Features: {diag_features}")

# Quick stats
print(f"\n  Diagnosis feature prevalence (train):")
for feat in diag_features[:10]:
    pct = X_train_diag[feat].mean() * 100
    print(f"    {feat}: {pct:.1f}%")

# ============================================================================
# STANDARD FEATURE ENGINEERING
# ============================================================================
print("\n[5/7] Adding standard engineered features...")

X_train_fe = add_age_features(X_train_diag)
X_test_fe = add_age_features(X_test_diag)

X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)

X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)

X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

# ============================================================================
# ADD ICD9 ENCODED
# ============================================================================
print("\n[6/7] Adding ICD9 encoded features...")

if icd9_col:
    train_icd9 = train_raw[icd9_col].copy().fillna('MISSING')
    test_icd9 = test_raw[icd9_col].copy().fillna('MISSING')
    
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
    print(f"  ✓ Added ICD9 features")

# ============================================================================
# ENCODE CATEGORICALS
# ============================================================================
print("\n[7/7] Encoding categorical columns...")

cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
print(f"  Found {len(cat_cols)} categorical columns: {cat_cols}")

for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le_c.fit(combined)
    X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))

X_train_final = X_train_fe
X_test_final = X_test_fe

print(f"\n✓ FINAL FEATURE COUNT: {X_train_final.shape[1]} features")
print(f"  (Previous best had 85 features, now we have {X_train_final.shape[1]})")

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

# Use tuned params from previous run
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
print("\nTop 30 Most Important Features:")
importance_df = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(30).to_string(index=False))

# ============================================================================
# DIAGNOSIS FEATURE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSIS FEATURE ANALYSIS")
print("="*80)

diag_importance = importance_df[importance_df['feature'].str.startswith('diag_')]
print(f"\nAll {len(diag_importance)} diagnosis features ranked:\n")
for idx, row in diag_importance.iterrows():
    rank = list(importance_df['feature']).index(row['feature']) + 1
    pct = row['importance'] / importance_df['importance'].sum() * 100
    print(f"  {row['feature']:<35} Rank: #{rank:3d}  Importance: {row['importance']:.6f}  ({pct:.2f}%)")

total_diag_importance = diag_importance['importance'].sum()
total_importance = importance_df['importance'].sum()
print(f"\n  Total DIAGNOSIS contribution: {(total_diag_importance/total_importance*100):.2f}%")

# ============================================================================
# GENERATE SUBMISSION
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
output_file = output_dir / "xgboost_with_diagnosis.csv"

submission.to_csv(output_file, index=False)

print(f"\n✓ Submission saved: {output_file}")
print(f"\nPrediction stats: Min={y_test_pred.min():.4f}, Max={y_test_pred.max():.4f}, Mean={y_test_pred.mean():.4f}")

print(f"\n📊 Final Model Summary:")
print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Features used: {X_train_final.shape[1]}")
print(f"  Diagnosis features: {len(diag_features)}")
print(f"  Best iteration: {model.best_iteration}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)