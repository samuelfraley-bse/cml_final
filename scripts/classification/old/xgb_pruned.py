"""
XGBoost with ALL Categorical Features - PRUNED VERSION

Same as xgb_full_categorical.py but removes all 26 zero-importance features
identified from the first run.

Run from project root:
    python scripts/classification/xgb_pruned.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
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
print("XGBOOST WITH ALL CATEGORICALS - PRUNED (No Zero-Importance Features)")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# FEATURES TO REMOVE (zero importance from previous run)
# ============================================================================
ZERO_IMPORTANCE_FEATURES = [
    'extreme_hyperglycemia',
    'severe_tachypnea',
    'extreme_tachypnea',
    'hypertensive_crisis',
    'hypoglycemia',
    'DiasBP_missing',
    'HeartRate_missing',
    'RespRate_missing',
    'MeanBP_missing',
    'any_vital_missing',
    'TempC_missing',
    'SysBP_missing',
    'SpO2_missing',
    'Glucose_missing',
    'severe_hyperglycemia',
    'extreme_fever',
    'severe_tachycardia',
    'extreme_tachycardia',
    'very_low_spo2',
    'critical_low_spo2',
    'hyperglycemia_flag',
    'age_90_plus',
    'age_0_40',
    'hypoxia_flag',
    'is_young',
    'very_elderly_and_tachy'
]

print(f"\nWill remove {len(ZERO_IMPORTANCE_FEATURES)} zero-importance features")

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS (PRUNED)
# ============================================================================

def add_missing_indicators_pruned(df, vital_cols):
    """Add only the missing indicator that matters: missing_vital_count."""
    df = df.copy()

    # We still need to count missing, but don't create individual flags
    missing_count = 0
    for vital_base in ['TempC', 'SysBP', 'SpO2', 'Glucose', 'DiasBP', 'MeanBP', 'RespRate', 'HeartRate']:
        col = f'{vital_base}_Mean'
        if col in df.columns:
            missing_count += df[col].isna().astype(int)

    df['missing_vital_count'] = missing_count
    # Skip any_vital_missing - it was zero importance

    return df

def add_extreme_flags_pruned(df):
    """Add only the extreme flags that had importance > 0."""
    df = df.copy()

    # KEEP: critical_low_bp, very_low_bp
    if 'SysBP_Min' in df.columns:
        df['critical_low_bp'] = (df['SysBP_Min'] < 70).astype(int)
        df['very_low_bp'] = (df['SysBP_Min'] < 80).astype(int)

    # SKIP: critical_low_spo2, very_low_spo2 (zero importance)

    # KEEP: bradycardia
    if 'HeartRate_Min' in df.columns:
        df['bradycardia'] = (df['HeartRate_Min'] < 50).astype(int)

    # KEEP: hypothermia
    if 'TempC_Min' in df.columns:
        df['hypothermia'] = (df['TempC_Min'] < 35).astype(int)

    # SKIP: hypoglycemia (zero importance)

    # SKIP: severe_tachycardia, extreme_tachycardia (zero importance)

    # KEEP: high_fever
    if 'TempC_Max' in df.columns:
        df['high_fever'] = (df['TempC_Max'] > 39).astype(int)
        # SKIP: extreme_fever (zero importance)

    # SKIP: severe_hyperglycemia, extreme_hyperglycemia (zero importance)
    # SKIP: severe_tachypnea, extreme_tachypnea (zero importance)
    # SKIP: hypertensive_crisis (zero importance)

    # Count critical flags (only the ones we kept)
    critical_cols = ['critical_low_bp', 'bradycardia', 'hypothermia', 'high_fever']
    existing_critical = [c for c in critical_cols if c in df.columns]
    if existing_critical:
        df['critical_flag_count'] = df[existing_critical].sum(axis=1)
        df['any_critical_flag'] = (df['critical_flag_count'] > 0).astype(int)
        df['multiple_critical'] = (df['critical_flag_count'] >= 2).astype(int)

    return df

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\n" + "="*80)
print("PREPARING DATA (PRUNED VERSION)")
print("="*80)

vital_cols = ['HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean',
              'SysBP_Min', 'SysBP_Max', 'SysBP_Mean',
              'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',
              'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean',
              'RespRate_Min', 'RespRate_Max', 'RespRate_Mean',
              'TempC_Min', 'TempC_Max', 'TempC_Mean',
              'SpO2_Min', 'SpO2_Max', 'SpO2_Mean',
              'Glucose_Min', 'Glucose_Max', 'Glucose_Mean']

existing_vitals = [c for c in vital_cols if c in train_raw.columns]

# Standard prep
leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis", "LOS"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

# Remove IDs
for col in [c for c in X_train_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_test_raw = X_test_raw.drop(columns=[col])

# Add PRUNED missing indicators
X_train_miss = add_missing_indicators_pruned(X_train_raw, existing_vitals)
X_test_miss = add_missing_indicators_pruned(X_test_raw, existing_vitals)

# Add PRUNED extreme flags
X_train_extreme = add_extreme_flags_pruned(X_train_miss)
X_test_extreme = add_extreme_flags_pruned(X_test_miss)

# Standard feature engineering
X_train_fe = add_age_features(X_train_extreme)
X_test_fe = add_age_features(X_test_extreme)

X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)

X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)

X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

# ============================================================================
# REMOVE ZERO-IMPORTANCE FEATURES FROM hef_prep
# ============================================================================
print("\nRemoving zero-importance features from hef_prep...")

features_to_drop = [f for f in ZERO_IMPORTANCE_FEATURES if f in X_train_fe.columns]
X_train_fe = X_train_fe.drop(columns=features_to_drop, errors='ignore')
X_test_fe = X_test_fe.drop(columns=features_to_drop, errors='ignore')
print(f"  Dropped {len(features_to_drop)} features")

# ============================================================================
# ADD ICD9 FEATURES
# ============================================================================
print("\nAdding ICD9 diagnosis features...")

icd9_col = 'ICD9_diagnosis'
if icd9_col in train_raw.columns:
    train_icd9 = train_raw[icd9_col].fillna('MISSING')
    test_icd9 = test_raw[icd9_col].fillna('MISSING')

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
    print(f"  ✓ ICD9_encoded: {len(le.classes_)} unique codes")
    print(f"  ✓ ICD9_category: {len(le_cat.classes_)} unique categories")

# ============================================================================
# ADD ALL CATEGORICAL FEATURES
# ============================================================================
print("\nAdding categorical features...")

categorical_features = ['FIRST_CAREUNIT', 'INSURANCE', 'ETHNICITY', 'MARITAL_STATUS', 'RELIGION', 'GENDER', 'ADMISSION_TYPE']

for cat_col in categorical_features:
    if cat_col in X_train_fe.columns:
        le_c = LabelEncoder()
        train_vals = X_train_fe[cat_col].fillna('MISSING').astype(str)
        test_vals = X_test_fe[cat_col].fillna('MISSING').astype(str)
        combined = pd.concat([train_vals, test_vals])
        le_c.fit(combined)
        X_train_fe[cat_col] = le_c.transform(train_vals)
        X_test_fe[cat_col] = le_c.transform(test_vals)
        print(f"  ✓ {cat_col}: {len(le_c.classes_)} categories")

# ============================================================================
# CATEGORICAL INTERACTIONS
# ============================================================================
print("\nCreating categorical interactions...")

if 'FIRST_CAREUNIT' in X_train_fe.columns and 'critical_flag_count' in X_train_fe.columns:
    X_train_fe['careunit_x_critical'] = X_train_fe['FIRST_CAREUNIT'] * X_train_fe['critical_flag_count']
    X_test_fe['careunit_x_critical'] = X_test_fe['FIRST_CAREUNIT'] * X_test_fe['critical_flag_count']
    print("  ✓ careunit_x_critical")

if 'FIRST_CAREUNIT' in X_train_fe.columns and 'ADMISSION_TYPE' in X_train_fe.columns:
    X_train_fe['admission_careunit'] = X_train_fe['ADMISSION_TYPE'] * 10 + X_train_fe['FIRST_CAREUNIT']
    X_test_fe['admission_careunit'] = X_test_fe['ADMISSION_TYPE'] * 10 + X_test_fe['FIRST_CAREUNIT']
    print("  ✓ admission_careunit")

# Encode any remaining object columns
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le_c.fit(combined)
    X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))

X_train_final = X_train_fe
X_test_final = X_test_fe

print(f"\n✓ Total features: {X_train_final.shape[1]} (was 124, removed {124 - X_train_final.shape[1]})")

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

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

model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

y_val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"\n✓ Validation AUC: {val_auc:.4f}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)

importance_df = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 30 features:")
print(importance_df.head(30).to_string(index=False))

# Check for any remaining zero-importance
zero_imp = importance_df[importance_df['importance'] == 0]['feature'].tolist()
print(f"\nRemaining zero-importance features: {len(zero_imp)}")
if zero_imp:
    for f in zero_imp:
        print(f"  - {f}")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

y_test_pred = model.predict_proba(X_test_final)[:, 1]

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
submission = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': y_test_pred
})

output_file = BASE_DIR / "submissions" / "xgb_pruned.csv"
submission.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# Save feature importance
importance_file = BASE_DIR / "submissions" / "feature_importance_pruned.csv"
importance_df.to_csv(importance_file, index=False)
print(f"✓ Feature importance saved: {importance_file}")

print(f"\n" + "="*80)
print("SUMMARY - PRUNED MODEL")
print("="*80)
print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Total features: {X_train_final.shape[1]} (removed 26 zero-importance)")
print(f"  Remaining zero-importance: {len(zero_imp)}")
print(f"  Submission: {output_file}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"  Full model (124 features): Val AUC = 0.8501")
print(f"  Pruned model ({X_train_final.shape[1]} features): Val AUC = {val_auc:.4f}")
print(f"  Difference: {val_auc - 0.8501:+.4f}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
