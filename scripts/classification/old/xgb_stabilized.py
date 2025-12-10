"""
XGBoost with Stabilized Features

Removes redundant/correlated features to reduce instability:
- Keep only ONE critical flag feature (multiple_critical)
- Remove highly correlated vitals (keep only most predictive)
- Remove features that add noise

Run from project root:
    python scripts/classification/xgb_stabilized.py
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
print("XGBOOST WITH STABILIZED FEATURES")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# STABILIZED FEATURE ENGINEERING
# ============================================================================

vital_cols = ['HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean',
              'SysBP_Min', 'SysBP_Max', 'SysBP_Mean',
              'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',
              'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean',
              'RespRate_Min', 'RespRate_Max', 'RespRate_Mean',
              'TempC_Min', 'TempC_Max', 'TempC_Mean',
              'SpO2_Min', 'SpO2_Max', 'SpO2_Mean',
              'Glucose_Min', 'Glucose_Max', 'Glucose_Mean']

existing_vitals = [c for c in vital_cols if c in train_raw.columns]

def add_missing_indicators_stable(df):
    """Add only the useful missing indicator."""
    df = df.copy()

    # Only count missing vitals - individual flags had 0 importance
    missing_count = 0
    for vital_base in ['TempC', 'SysBP', 'SpO2', 'Glucose', 'DiasBP', 'MeanBP', 'RespRate', 'HeartRate']:
        col = f'{vital_base}_Mean'
        if col in df.columns:
            missing_count += df[col].isna().astype(int)

    df['missing_vital_count'] = missing_count
    return df

def add_extreme_flags_stable(df):
    """Add ONLY non-redundant critical flags."""
    df = df.copy()

    # Individual flags that feed into critical_flag_count
    critical_conditions = []

    # Blood pressure - keep only very_low_bp (more common, better signal)
    if 'SysBP_Min' in df.columns:
        df['very_low_bp'] = (df['SysBP_Min'] < 80).astype(int)
        critical_conditions.append('very_low_bp')

    # Skip SpO2 critical flags - they had 0 importance

    # Bradycardia
    if 'HeartRate_Min' in df.columns:
        df['bradycardia'] = (df['HeartRate_Min'] < 50).astype(int)
        critical_conditions.append('bradycardia')

    # Hypothermia
    if 'TempC_Min' in df.columns:
        df['hypothermia'] = (df['TempC_Min'] < 35).astype(int)
        critical_conditions.append('hypothermia')

    # High fever
    if 'TempC_Max' in df.columns:
        df['high_fever'] = (df['TempC_Max'] > 39).astype(int)
        critical_conditions.append('high_fever')

    # Create ONLY multiple_critical (the best performer)
    # Skip critical_flag_count and any_critical_flag (redundant)
    if critical_conditions:
        critical_sum = df[critical_conditions].sum(axis=1)
        df['multiple_critical'] = (critical_sum >= 2).astype(int)

    return df

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\nPreparing data with stabilized features...")

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

# Remove IDs
for col in [c for c in X_train_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_test_raw = X_test_raw.drop(columns=[col])

# Stabilized feature engineering
X_train_fe = add_missing_indicators_stable(X_train_raw)
X_test_fe = add_missing_indicators_stable(X_test_raw)

X_train_fe = add_extreme_flags_stable(X_train_fe)
X_test_fe = add_extreme_flags_stable(X_test_fe)

X_train_fe = add_age_features(X_train_fe)
X_test_fe = add_age_features(X_test_fe)

X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)

X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)

X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

# Add ICD9
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

# ============================================================================
# REMOVE REDUNDANT/LOW-VALUE FEATURES
# ============================================================================
print("\nRemoving redundant and low-value features...")

# Features to remove (redundant or zero importance)
FEATURES_TO_REMOVE = [
    # Redundant critical flags (keeping only multiple_critical)
    'critical_flag_count',
    'any_critical_flag',
    'critical_low_bp',  # very_low_bp is better
    'critical_low_spo2',
    'very_low_spo2',

    # Zero importance from previous analysis
    'extreme_hyperglycemia',
    'severe_tachypnea',
    'extreme_tachypnea',
    'hypertensive_crisis',
    'hypoglycemia',
    'severe_hyperglycemia',
    'extreme_fever',
    'severe_tachycardia',
    'extreme_tachycardia',
    'hyperglycemia_flag',
    'hypoxia_flag',

    # Low-value age bins (continuous age_years is better)
    'age_90_plus',
    'age_0_40',
    'is_young',

    # Redundant missing indicators
    'any_vital_missing',
    'DiasBP_missing',
    'HeartRate_missing',
    'RespRate_missing',
    'MeanBP_missing',
    'TempC_missing',
    'SysBP_missing',
    'SpO2_missing',
    'Glucose_missing',

    # Low-value interactions
    'very_elderly_and_tachy',
]

removed = 0
for feat in FEATURES_TO_REMOVE:
    if feat in X_train_fe.columns:
        X_train_fe = X_train_fe.drop(columns=[feat])
        X_test_fe = X_test_fe.drop(columns=[feat])
        removed += 1

print(f"  Removed {removed} redundant/low-value features")

# Encode remaining categoricals
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le_c.fit(combined)
    X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))

X_train_final = X_train_fe
X_test_final = X_test_fe

print(f"\n✓ Total features: {X_train_final.shape[1]}")

# ============================================================================
# CHECK FOR MULTICOLLINEARITY
# ============================================================================
print("\n" + "="*80)
print("CORRELATION CHECK (Top correlated pairs)")
print("="*80)

# Calculate correlation matrix
corr_matrix = X_train_final.corr().abs()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.9:
            high_corr_pairs.append({
                'feat1': corr_matrix.columns[i],
                'feat2': corr_matrix.columns[j],
                'corr': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print("\nHighly correlated pairs (>0.9):")
    for pair in sorted(high_corr_pairs, key=lambda x: x['corr'], reverse=True)[:10]:
        print(f"  {pair['feat1']:<25} - {pair['feat2']:<25}: {pair['corr']:.3f}")
else:
    print("\nNo highly correlated pairs found (>0.9)")

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
print("FEATURE IMPORTANCE (Stabilized)")
print("="*80)

importance_df = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 25 features:")
for i, (_, row) in enumerate(importance_df.head(25).iterrows()):
    print(f"{i+1:>3}. {row['feature']:<30} {row['importance']:.4f}")

# Check if multiple_critical is still dominant
top_feat = importance_df.iloc[0]
print(f"\nTop feature concentration: {top_feat['feature']} at {top_feat['importance']:.1%}")

if top_feat['importance'] < 0.15:
    print("✓ Good - importance is more distributed")
else:
    print("⚠ Still concentrated in top feature")

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

output_file = BASE_DIR / "submissions" / "xgb_stabilized.csv"
submission.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# Save feature importance
importance_file = BASE_DIR / "submissions" / "feature_importance_stabilized.csv"
importance_df.to_csv(importance_file, index=False)
print(f"✓ Feature importance: {importance_file}")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Total features: {X_train_final.shape[1]}")
print(f"  Submission: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
