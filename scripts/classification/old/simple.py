"""
Best Model + Log Transforms + Scaling + Calibration

Adding what we haven't tried:
1. Log transform on skewed features (Glucose, ranges)
2. Standard scaling on all numeric features
3. Probability calibration (Platt scaling)

Based on xgb_with_missing_extreme.py (our 74.9 best)

Run from project root:
    python scripts/classification/xgb_log_scaled_calibrated.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
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

print("="*70)
print("BEST MODEL + LOG TRANSFORMS + SCALING + CALIBRATION")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")
print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# FEATURE ENGINEERING (same as best model)
# ============================================================================
print("\n[2/7] Feature engineering (same as best model)...")

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

# Add missing indicators BEFORE imputation
vital_cols = ['HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'TempC', 'SpO2', 'Glucose']
for vital in vital_cols:
    col = f'{vital}_Mean'
    if col in X_train_raw.columns:
        X_train_raw[f'{vital}_missing'] = X_train_raw[col].isna().astype(int)
        X_test_raw[f'{vital}_missing'] = X_test_raw[col].isna().astype(int)

# Count missing
missing_cols = [c for c in X_train_raw.columns if '_missing' in c]
X_train_raw['missing_vital_count'] = X_train_raw[missing_cols].sum(axis=1)
X_test_raw['missing_vital_count'] = X_test_raw[missing_cols].sum(axis=1)

# Standard feature engineering
X_train_fe = add_age_features(X_train_raw)
X_test_fe = add_age_features(X_test_raw)

X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)

X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)

X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

# ============================================================================
# ADD EXTREME VALUE FLAGS (from best model)
# ============================================================================
print("\n[3/7] Adding extreme value flags...")

def add_extreme_flags(df):
    """Add critical/extreme value flags."""
    # Critical lows
    if 'SysBP_Min' in df.columns:
        df['critical_low_bp'] = (df['SysBP_Min'] < 70).astype(int)
        df['very_low_bp'] = (df['SysBP_Min'] < 80).astype(int)
    if 'SpO2_Min' in df.columns:
        df['critical_low_spo2'] = (df['SpO2_Min'] < 85).astype(int)
        df['very_low_spo2'] = (df['SpO2_Min'] < 88).astype(int)
    if 'HeartRate_Min' in df.columns:
        df['bradycardia'] = (df['HeartRate_Min'] < 50).astype(int)
    if 'TempC_Min' in df.columns:
        df['hypothermia'] = (df['TempC_Min'] < 35).astype(int)
    if 'Glucose_Min' in df.columns:
        df['hypoglycemia'] = (df['Glucose_Min'] < 60).astype(int)
    
    # Critical highs
    if 'HeartRate_Max' in df.columns:
        df['severe_tachycardia'] = (df['HeartRate_Max'] > 150).astype(int)
        df['extreme_tachycardia'] = (df['HeartRate_Max'] > 170).astype(int)
    if 'TempC_Max' in df.columns:
        df['high_fever'] = (df['TempC_Max'] > 39).astype(int)
    if 'Glucose_Max' in df.columns:
        df['severe_hyperglycemia'] = (df['Glucose_Max'] > 300).astype(int)
    if 'RespRate_Max' in df.columns:
        df['severe_tachypnea'] = (df['RespRate_Max'] > 30).astype(int)
        df['extreme_tachypnea'] = (df['RespRate_Max'] > 40).astype(int)
    if 'SysBP_Max' in df.columns:
        df['hypertensive_crisis'] = (df['SysBP_Max'] > 180).astype(int)
    
    # Aggregate
    critical_cols = ['critical_low_bp', 'very_low_bp', 'critical_low_spo2', 
                     'very_low_spo2', 'bradycardia', 'hypothermia', 'hypoglycemia',
                     'severe_tachycardia', 'extreme_tachycardia', 'high_fever',
                     'severe_hyperglycemia', 'severe_tachypnea', 'extreme_tachypnea',
                     'hypertensive_crisis']
    existing = [c for c in critical_cols if c in df.columns]
    df['critical_flag_count'] = df[existing].sum(axis=1)
    df['any_critical_flag'] = (df['critical_flag_count'] > 0).astype(int)
    df['multiple_critical'] = (df['critical_flag_count'] >= 2).astype(int)
    
    return df

X_train_fe = add_extreme_flags(X_train_fe)
X_test_fe = add_extreme_flags(X_test_fe)

# ============================================================================
# ADD LOG TRANSFORMS (NEW!)
# ============================================================================
print("\n[4/7] Adding log transforms on skewed features...")

def add_log_transforms(df):
    """Log transform skewed features."""
    # Glucose is often right-skewed
    for col in ['Glucose_Mean', 'Glucose_Max', 'Glucose_Min']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
    
    # Ranges can be skewed
    range_cols = [c for c in df.columns if '_range' in c.lower()]
    for col in range_cols:
        df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
    
    # Shock index
    if 'shock_index_mean' in df.columns:
        df['shock_index_log'] = np.log1p(df['shock_index_mean'].clip(lower=0))
    
    # Age (slight right skew in elderly)
    if 'age_years' in df.columns:
        df['age_log'] = np.log1p(df['age_years'].clip(lower=0))
    
    return df

X_train_fe = add_log_transforms(X_train_fe)
X_test_fe = add_log_transforms(X_test_fe)

log_features = [c for c in X_train_fe.columns if '_log' in c]
print(f"  Log features added: {len(log_features)}")

# ============================================================================
# ADD ICD9
# ============================================================================
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

# Encode categoricals
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le_c.fit(combined)
    X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))

# Fill NaN
X_train_fe = X_train_fe.fillna(X_train_fe.median())
X_test_fe = X_test_fe.fillna(X_train_fe.median())

print(f"\n✓ Total features before scaling: {X_train_fe.shape[1]}")

# ============================================================================
# STANDARD SCALING (NEW!)
# ============================================================================
print("\n[5/7] Applying standard scaling...")

# Identify numeric columns (excluding binary flags)
binary_cols = [c for c in X_train_fe.columns if X_train_fe[c].nunique() <= 2]
numeric_cols = [c for c in X_train_fe.columns if c not in binary_cols]

print(f"  Binary columns (not scaled): {len(binary_cols)}")
print(f"  Numeric columns (scaled): {len(numeric_cols)}")

scaler = StandardScaler()
X_train_fe[numeric_cols] = scaler.fit_transform(X_train_fe[numeric_cols])
X_test_fe[numeric_cols] = scaler.transform(X_test_fe[numeric_cols])

X_train_final = X_train_fe
X_test_final = X_test_fe
y_train_arr = y_train.values

print(f"\n✓ Final features: {X_train_final.shape[1]}")

# ============================================================================
# K-FOLD TRAINING WITH CALIBRATION
# ============================================================================
print("\n[6/7] K-Fold training with calibration...")

N_FOLDS = 5
kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_uncalibrated = np.zeros(len(X_train_final))
oof_calibrated = np.zeros(len(X_train_final))
test_uncalibrated = np.zeros(len(X_test_final))
test_calibrated = np.zeros(len(X_test_final))

X_train_arr = X_train_final.values
X_test_arr = X_test_final.values

scale_pos_weight = (y_train_arr == 0).sum() / (y_train_arr == 1).sum()

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_arr, y_train_arr)):
    X_tr_fold = X_train_arr[train_idx]
    y_tr_fold = y_train_arr[train_idx]
    X_val_fold = X_train_arr[val_idx]
    y_val_fold = y_train_arr[val_idx]
    
    # Split train into train + calibration
    X_tr_inner, X_cal_inner, y_tr_inner, y_cal_inner = train_test_split(
        X_tr_fold, y_tr_fold, test_size=0.15, random_state=42+fold, stratify=y_tr_fold
    )
    
    # Train base model
    base = XGBClassifier(
        max_depth=5, learning_rate=0.03, n_estimators=500,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=42+fold,
        tree_method='hist', eval_metric='auc', n_jobs=-1, verbosity=0
    )
    base.fit(X_tr_inner, y_tr_inner)
    
    # Calibrate
    calibrated = CalibratedClassifierCV(base, method='sigmoid', cv='prefit')
    calibrated.fit(X_cal_inner, y_cal_inner)
    
    # Predictions
    oof_uncalibrated[val_idx] = base.predict_proba(X_val_fold)[:, 1]
    oof_calibrated[val_idx] = calibrated.predict_proba(X_val_fold)[:, 1]
    
    test_uncalibrated += base.predict_proba(X_test_arr)[:, 1] / N_FOLDS
    test_calibrated += calibrated.predict_proba(X_test_arr)[:, 1] / N_FOLDS
    
    fold_auc_uncal = roc_auc_score(y_val_fold, oof_uncalibrated[val_idx])
    fold_auc_cal = roc_auc_score(y_val_fold, oof_calibrated[val_idx])
    print(f"  Fold {fold+1}: Uncal AUC={fold_auc_uncal:.4f}, Cal AUC={fold_auc_cal:.4f}")

# Overall metrics
oof_auc_uncal = roc_auc_score(y_train_arr, oof_uncalibrated)
oof_auc_cal = roc_auc_score(y_train_arr, oof_calibrated)
oof_brier_uncal = brier_score_loss(y_train_arr, oof_uncalibrated)
oof_brier_cal = brier_score_loss(y_train_arr, oof_calibrated)

print(f"\n  Overall OOF Uncalibrated: AUC={oof_auc_uncal:.4f}, Brier={oof_brier_uncal:.4f}")
print(f"  Overall OOF Calibrated:   AUC={oof_auc_cal:.4f}, Brier={oof_brier_cal:.4f}")

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n[7/7] Generating submissions...")

output_dir = BASE_DIR / "submissions"

# Uncalibrated
sub_uncal = pd.DataFrame({
    'icustay_id': test_raw['icustay_id'].values,
    'HOSPITAL_EXPIRE_FLAG': test_uncalibrated
})
sub_uncal.to_csv(output_dir / "xgb_log_scaled_uncalibrated.csv", index=False)
print(f"✓ Saved: xgb_log_scaled_uncalibrated.csv")

# Calibrated
sub_cal = pd.DataFrame({
    'icustay_id': test_raw['icustay_id'].values,
    'HOSPITAL_EXPIRE_FLAG': test_calibrated
})
sub_cal.to_csv(output_dir / "xgb_log_scaled_calibrated.csv", index=False)
print(f"✓ Saved: xgb_log_scaled_calibrated.csv")

print(f"""
📊 Summary:
  Features: {X_train_final.shape[1]}
  Log transforms: {len(log_features)} features
  Scaling: StandardScaler on {len(numeric_cols)} numeric features
  
  OOF Uncalibrated: AUC={oof_auc_uncal:.4f}, Brier={oof_brier_uncal:.4f}
  OOF Calibrated:   AUC={oof_auc_cal:.4f}, Brier={oof_brier_cal:.4f}
  
  Submit BOTH to compare:
  - xgb_log_scaled_uncalibrated.csv
  - xgb_log_scaled_calibrated.csv
""")

print("="*70)
print("✓ COMPLETE!")
print("="*70)