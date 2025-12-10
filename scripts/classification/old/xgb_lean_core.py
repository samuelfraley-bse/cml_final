"""
XGBoost Lean Core Model

Only ~50-60 high-value features:
- Removes all redundant/zero-importance features
- Keeps only proven performers
- Better sample-to-feature ratio for generalization

Run from project root:
    python scripts/classification/xgb_lean_core.py
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
    ID_COLS
)

print("="*80)
print("XGBOOST LEAN CORE MODEL (~50-60 features)")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# LEAN FEATURE ENGINEERING
# ============================================================================

def add_lean_features(df):
    """Add only proven high-value features."""
    df = df.copy()

    # ===== AGE (MIMIC-safe calculation) =====
    if 'DOB' in df.columns and 'ADMITTIME' in df.columns:
        df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
        df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'], errors='coerce')

        admit_year = df['ADMITTIME'].dt.year
        admit_month = df['ADMITTIME'].dt.month
        admit_day = df['ADMITTIME'].dt.day
        dob_year = df['DOB'].dt.year
        dob_month = df['DOB'].dt.month
        dob_day = df['DOB'].dt.day

        df['age_years'] = admit_year - dob_year
        birthday_not_passed = (admit_month < dob_month) | \
                              ((admit_month == dob_month) & (admit_day < dob_day))
        df.loc[birthday_not_passed, 'age_years'] -= 1
        df['age_years'] = df['age_years'].clip(0, 120)

        # Only keep the most useful age features
        df['is_elderly'] = (df['age_years'] >= 75).astype(int)

        # Drop datetime columns
        df = df.drop(columns=['DOB', 'ADMITTIME'], errors='ignore')

    # ===== VITAL RANGES (only most predictive) =====
    if 'SpO2_Max' in df.columns and 'SpO2_Min' in df.columns:
        df['SpO2_range'] = df['SpO2_Max'] - df['SpO2_Min']

    if 'TempC_Max' in df.columns and 'TempC_Min' in df.columns:
        df['TempC_range'] = df['TempC_Max'] - df['TempC_Min']

    # ===== COMPOSITE SCORES (top performers) =====
    # Shock index
    if 'HeartRate_Mean' in df.columns and 'SysBP_Mean' in df.columns:
        df['shock_index_mean'] = df['HeartRate_Mean'] / df['SysBP_Mean'].replace(0, np.nan)

    # SpO2 deficit
    if 'SpO2_Min' in df.columns:
        df['spo2_deficit'] = (92 - df['SpO2_Min']).clip(lower=0)

    # Temperature deviation
    if 'TempC_Mean' in df.columns:
        df['temp_dev_mean'] = (df['TempC_Mean'] - 37.0).abs()

    # Glucose excess
    if 'Glucose_Max' in df.columns:
        df['glucose_excess'] = (df['Glucose_Max'] - 180).clip(lower=0)

    # Instability count
    instability = pd.Series(0, index=df.index)
    if 'SysBP_Min' in df.columns:
        instability += (df['SysBP_Min'] < 90).astype(int)
    if 'HeartRate_Max' in df.columns:
        instability += (df['HeartRate_Max'] > 100).astype(int)
    if 'SpO2_Min' in df.columns:
        instability += (df['SpO2_Min'] < 92).astype(int)
    if 'RespRate_Max' in df.columns:
        instability += (df['RespRate_Max'] > 24).astype(int)
    df['instability_count'] = instability

    # ===== CRITICAL FLAGS (only the best one) =====
    critical_count = pd.Series(0, index=df.index)

    if 'SysBP_Min' in df.columns:
        critical_count += (df['SysBP_Min'] < 80).astype(int)
    if 'HeartRate_Min' in df.columns:
        critical_count += (df['HeartRate_Min'] < 50).astype(int)
    if 'TempC_Min' in df.columns:
        critical_count += (df['TempC_Min'] < 35).astype(int)
    if 'TempC_Max' in df.columns:
        critical_count += (df['TempC_Max'] > 39).astype(int)

    df['multiple_critical'] = (critical_count >= 2).astype(int)

    # ===== AGE INTERACTIONS (top 3 only) =====
    if 'age_years' in df.columns:
        if 'instability_count' in df.columns:
            df['age_x_instability'] = df['age_years'] * df['instability_count']
        if 'temp_dev_mean' in df.columns:
            df['age_x_temp_dev'] = df['age_years'] * df['temp_dev_mean']
        if 'shock_index_mean' in df.columns:
            df['age_x_shock_index'] = df['age_years'] * df['shock_index_mean']

    # ===== ELDERLY INTERACTIONS (top 2) =====
    if 'is_elderly' in df.columns:
        if 'SysBP_Min' in df.columns:
            df['elderly_and_hypotensive'] = df['is_elderly'] * (df['SysBP_Min'] < 90).astype(int)
        if 'TempC_Max' in df.columns:
            df['elderly_and_fever'] = df['is_elderly'] * (df['TempC_Max'] > 38.5).astype(int)

    return df

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\nPreparing lean feature set...")

# Keep only essential columns initially
leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis", "LOS", "Diff"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

# Remove IDs
for col in [c for c in X_train_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_test_raw = X_test_raw.drop(columns=[col])

# Add lean features
X_train_fe = add_lean_features(X_train_raw)
X_test_fe = add_lean_features(X_test_raw)

# ===== ADD ICD9 (category only - more generalizable) =====
icd9_col = 'ICD9_diagnosis'
if icd9_col in train_raw.columns:
    train_icd9 = train_raw[icd9_col].fillna('MISSING')
    test_icd9 = test_raw[icd9_col].fillna('MISSING')

    # Only category (first 3 chars) - drops high-cardinality encoded version
    train_cat = train_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    test_cat = test_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    le_cat = LabelEncoder()
    le_cat.fit(pd.concat([train_cat, test_cat]))
    X_train_fe['ICD9_category'] = le_cat.transform(train_cat)
    X_test_fe['ICD9_category'] = le_cat.transform(test_cat)
    print(f"  ICD9_category: {len(le_cat.classes_)} categories")

# ===== SELECT ONLY LEAN FEATURES =====
# Define exactly which features to keep
LEAN_FEATURES = [
    # Raw vitals - Min and Mean only (drop Max - correlated with range)
    'HeartRate_Min', 'HeartRate_Mean',
    'SysBP_Min', 'SysBP_Mean',
    'DiasBP_Min', 'DiasBP_Mean',
    'MeanBP_Min', 'MeanBP_Mean',
    'RespRate_Min', 'RespRate_Mean',
    'TempC_Min', 'TempC_Mean',
    'SpO2_Min', 'SpO2_Mean',
    'Glucose_Min', 'Glucose_Mean',

    # Best ranges
    'SpO2_range', 'TempC_range',

    # Age
    'age_years', 'is_elderly',

    # Composites
    'shock_index_mean', 'spo2_deficit', 'temp_dev_mean',
    'glucose_excess', 'instability_count',

    # Critical flag (only the best)
    'multiple_critical',

    # Age interactions (top 3)
    'age_x_instability', 'age_x_temp_dev', 'age_x_shock_index',

    # Elderly interactions (top 2)
    'elderly_and_hypotensive', 'elderly_and_fever',

    # Categoricals (most predictive only)
    'ADMISSION_TYPE', 'GENDER',

    # ICD9
    'ICD9_category',
]

# Filter to only lean features that exist
existing_lean = [f for f in LEAN_FEATURES if f in X_train_fe.columns]
missing_lean = [f for f in LEAN_FEATURES if f not in X_train_fe.columns]

if missing_lean:
    print(f"\n  Warning: Missing features: {missing_lean}")

X_train_lean = X_train_fe[existing_lean].copy()
X_test_lean = X_test_fe[existing_lean].copy()

# Encode remaining categoricals
cat_cols = X_train_lean.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_lean[col], X_test_lean[col]]).astype(str)
    le_c.fit(combined)
    X_train_lean[col] = le_c.transform(X_train_lean[col].astype(str))
    X_test_lean[col] = le_c.transform(X_test_lean[col].astype(str))
    print(f"  {col}: {len(le_c.classes_)} categories")

X_train_final = X_train_lean
X_test_final = X_test_lean

print(f"\n✓ Total features: {X_train_final.shape[1]}")
print(f"  Samples per feature: {len(X_train_final) / X_train_final.shape[1]:.0f}")

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

print("\nAll features ranked:")
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{i:3}. {row['feature']:<30} {row['importance']:.4f}")

# Check for zero importance
zero_imp = importance_df[importance_df['importance'] == 0]
if len(zero_imp) > 0:
    print(f"\n⚠ Features with zero importance: {len(zero_imp)}")
else:
    print(f"\n✓ All features have non-zero importance")

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

output_file = BASE_DIR / "submissions" / "xgb_lean_core.csv"
submission.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# Save feature importance
importance_file = BASE_DIR / "submissions" / "feature_importance_lean_core.csv"
importance_df.to_csv(importance_file, index=False)
print(f"✓ Feature importance: {importance_file}")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Total features: {X_train_final.shape[1]}")
print(f"  Samples per feature: {len(X_train_final) / X_train_final.shape[1]:.0f}")
print(f"  Submission: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
