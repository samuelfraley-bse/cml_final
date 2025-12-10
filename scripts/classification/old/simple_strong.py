"""
Simple Strong Model - Focus on What Actually Predicts

Based on raw data analysis:
1. Use top correlated vitals (not overengineered)
2. Properly encode high-signal categoricals (FIRST_CAREUNIT, ADMISSION_TYPE)
3. Simple, interpretable features
4. ~40-50 features max

Run from project root:
    python scripts/classification/simple_strong.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

from hef_prep import split_features_target, ID_COLS

print("="*80)
print("SIMPLE STRONG MODEL - FOCUS ON REAL PREDICTORS")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# SIMPLE FEATURE ENGINEERING - Only What Matters
# ============================================================================

def create_simple_features(df, is_train=True):
    """Create only high-value features."""
    df = df.copy()

    # ===== AGE (simple, MIMIC-safe) =====
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'], errors='coerce')

    df['age_years'] = df['ADMITTIME'].dt.year - df['DOB'].dt.year
    # Simple adjustment
    df['age_years'] = df['age_years'].clip(0, 120)

    # ===== KEY VITAL FEATURES =====
    # Top predictors by correlation

    # SpO2 (strongest predictor)
    if 'SpO2_Min' in df.columns:
        df['spo2_deficit'] = (95 - df['SpO2_Min']).clip(lower=0)

    # Blood pressure
    if 'SysBP_Min' in df.columns and 'SysBP_Max' in df.columns:
        df['bp_range'] = df['SysBP_Max'] - df['SysBP_Min']

    # Shock index (HR/SBP) - clinically meaningful
    if 'HeartRate_Mean' in df.columns and 'SysBP_Mean' in df.columns:
        df['shock_index'] = df['HeartRate_Mean'] / df['SysBP_Mean'].replace(0, np.nan)

    # Temperature deviation
    if 'TempC_Mean' in df.columns:
        df['temp_deviation'] = (df['TempC_Mean'] - 37.0).abs()

    # ===== SIMPLE CRITICAL FLAGS =====
    # Just count how many systems are failing
    critical_count = pd.Series(0, index=df.index)

    if 'SpO2_Min' in df.columns:
        critical_count += (df['SpO2_Min'] < 90).astype(int)
    if 'SysBP_Min' in df.columns:
        critical_count += (df['SysBP_Min'] < 90).astype(int)
    if 'HeartRate_Max' in df.columns:
        critical_count += (df['HeartRate_Max'] > 120).astype(int)
    if 'RespRate_Max' in df.columns:
        critical_count += (df['RespRate_Max'] > 30).astype(int)
    if 'TempC_Max' in df.columns:
        critical_count += (df['TempC_Max'] > 38.5).astype(int)

    df['critical_systems'] = critical_count
    df['multiple_organ_stress'] = (critical_count >= 2).astype(int)

    # ===== AGE INTERACTION (just one, the best) =====
    df['age_x_critical'] = df['age_years'] * df['critical_systems']

    return df

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\nPreparing simple feature set...")

# Create features
train_fe = create_simple_features(train_raw.copy())
test_fe = create_simple_features(test_raw.copy())

# Select features to use
FEATURE_COLS = [
    # Raw vitals (the actual predictors)
    'HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean',
    'SysBP_Min', 'SysBP_Max', 'SysBP_Mean',
    'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',
    'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean',
    'RespRate_Min', 'RespRate_Max', 'RespRate_Mean',
    'TempC_Min', 'TempC_Max', 'TempC_Mean',
    'SpO2_Min', 'SpO2_Max', 'SpO2_Mean',
    'Glucose_Min', 'Glucose_Max', 'Glucose_Mean',

    # Engineered (simple, high-value)
    'age_years',
    'spo2_deficit',
    'bp_range',
    'shock_index',
    'temp_deviation',
    'critical_systems',
    'multiple_organ_stress',
    'age_x_critical',

    # Categoricals (encode these properly)
    'ADMISSION_TYPE',
    'GENDER',
    'FIRST_CAREUNIT',
    'INSURANCE',
    'ETHNICITY',
]

# Get target
y_train = train_raw['HOSPITAL_EXPIRE_FLAG']

# Select features
X_train = train_fe[[c for c in FEATURE_COLS if c in train_fe.columns]].copy()
X_test = test_fe[[c for c in FEATURE_COLS if c in test_fe.columns]].copy()

# ===== ENCODE CATEGORICALS PROPERLY =====
print("\nEncoding categoricals...")

cat_cols = ['ADMISSION_TYPE', 'GENDER', 'FIRST_CAREUNIT', 'INSURANCE', 'ETHNICITY']

for col in cat_cols:
    if col in X_train.columns:
        le = LabelEncoder()
        combined = pd.concat([X_train[col].fillna('MISSING').astype(str),
                              X_test[col].fillna('MISSING').astype(str)])
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].fillna('MISSING').astype(str))
        X_test[col] = le.transform(X_test[col].fillna('MISSING').astype(str))
        print(f"  {col}: {len(le.classes_)} categories")

# Add ICD9 category (just the first 3 digits)
if 'ICD9_diagnosis' in train_raw.columns:
    train_icd = train_raw['ICD9_diagnosis'].fillna('MISSING').apply(
        lambda x: x[:3] if x != 'MISSING' else 'MISSING'
    )
    test_icd = test_raw['ICD9_diagnosis'].fillna('MISSING').apply(
        lambda x: x[:3] if x != 'MISSING' else 'MISSING'
    )
    le_icd = LabelEncoder()
    le_icd.fit(pd.concat([train_icd, test_icd]))
    X_train['ICD9_category'] = le_icd.transform(train_icd)
    X_test['ICD9_category'] = le_icd.transform(test_icd)
    print(f"  ICD9_category: {len(le_icd.classes_)} categories")

print(f"\n✓ Total features: {X_train.shape[1]}")

# ============================================================================
# K-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("K-FOLD CROSS-VALIDATION")
print("="*80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Model config (balanced, which worked well)
params = {
    'max_depth': 4,
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 8,
    'gamma': 0.15,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'early_stopping_rounds': 50,
    'n_jobs': -1
}

oof_preds = np.zeros(len(X_train))
test_preds_folds = []
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    fold_scale = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()

    model = XGBClassifier(
        **params,
        scale_pos_weight=fold_scale,
        random_state=42
    )

    model.fit(X_fold_train, y_fold_train,
              eval_set=[(X_fold_val, y_fold_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, oof_preds[val_idx])
    fold_aucs.append(fold_auc)

    test_preds_folds.append(model.predict_proba(X_test)[:, 1])

    print(f"  Fold {fold+1}: {fold_auc:.4f}")

oof_auc = roc_auc_score(y_train, oof_preds)
fold_std = np.std(fold_aucs)
gen_score = oof_auc - fold_std

print(f"\n{'='*50}")
print(f"OOF AUC: {oof_auc:.4f}")
print(f"Fold Std: {fold_std:.4f}")
print(f"Gen Score: {gen_score:.4f}")
print(f"{'='*50}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)

# Train one final model for importance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
final_model = XGBClassifier(**params, scale_pos_weight=scale_pos_weight, random_state=42)
final_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nAll features ranked:")
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{i:2}. {row['feature']:<25} {row['importance']:.4f}")

# ============================================================================
# COMPARE TO PREVIOUS
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nPrevious best (122 features):")
print(f"  OOF AUC: 0.8537")
print(f"  Gen Score: 0.8455")

print(f"\nThis model ({X_train.shape[1]} features):")
print(f"  OOF AUC: {oof_auc:.4f}")
print(f"  Gen Score: {gen_score:.4f}")

diff = oof_auc - 0.8537
print(f"\nDifference: {diff:+.4f}")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

test_preds = np.mean(test_preds_folds, axis=0)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
submission = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': test_preds
})

output_file = BASE_DIR / "submissions" / "simple_strong.csv"
submission.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
