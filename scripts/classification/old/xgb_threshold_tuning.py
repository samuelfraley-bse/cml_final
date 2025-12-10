"""
XGBoost with Threshold Tuning

Based on scratch.py (best model at 74.9) but with:
- Threshold optimization for predictions
- Calibration analysis
- Multiple threshold strategies

Note: For AUC metric, threshold tuning doesn't directly help since AUC
measures ranking quality across ALL thresholds. However, probability
calibration can help if the metric involves calibration.

Run from project root:
    python scripts/classification/xgb_threshold_tuning.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
print("XGBOOST WITH THRESHOLD TUNING & CALIBRATION")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# FEATURE ENGINEERING (same as scratch.py)
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

def add_missing_indicators(df, vital_cols):
    """Add binary indicators for missing vitals."""
    df = df.copy()

    for vital_base in ['TempC', 'SysBP', 'SpO2', 'Glucose', 'DiasBP', 'MeanBP', 'RespRate', 'HeartRate']:
        col = f'{vital_base}_Mean'
        if col in df.columns:
            df[f'{vital_base}_missing'] = df[col].isna().astype(int)

    missing_cols = [c for c in df.columns if c.endswith('_missing')]
    if missing_cols:
        df['missing_vital_count'] = df[missing_cols].sum(axis=1)
        df['any_vital_missing'] = (df['missing_vital_count'] > 0).astype(int)

    return df

def add_extreme_flags(df):
    """Add flags for extreme/critical vital values."""
    df = df.copy()

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

    if 'HeartRate_Max' in df.columns:
        df['severe_tachycardia'] = (df['HeartRate_Max'] > 150).astype(int)
        df['extreme_tachycardia'] = (df['HeartRate_Max'] > 170).astype(int)

    if 'TempC_Max' in df.columns:
        df['high_fever'] = (df['TempC_Max'] > 39).astype(int)
        df['extreme_fever'] = (df['TempC_Max'] > 40).astype(int)

    if 'Glucose_Max' in df.columns:
        df['severe_hyperglycemia'] = (df['Glucose_Max'] > 300).astype(int)
        df['extreme_hyperglycemia'] = (df['Glucose_Max'] > 400).astype(int)

    if 'RespRate_Max' in df.columns:
        df['severe_tachypnea'] = (df['RespRate_Max'] > 30).astype(int)
        df['extreme_tachypnea'] = (df['RespRate_Max'] > 40).astype(int)

    if 'SysBP_Max' in df.columns:
        df['hypertensive_crisis'] = (df['SysBP_Max'] > 180).astype(int)

    critical_cols = ['critical_low_bp', 'critical_low_spo2', 'bradycardia',
                     'hypothermia', 'hypoglycemia', 'severe_tachycardia',
                     'high_fever', 'severe_hyperglycemia', 'severe_tachypnea']
    existing_critical = [c for c in critical_cols if c in df.columns]
    if existing_critical:
        df['critical_flag_count'] = df[existing_critical].sum(axis=1)
        df['any_critical_flag'] = (df['critical_flag_count'] > 0).astype(int)
        df['multiple_critical'] = (df['critical_flag_count'] >= 2).astype(int)

    return df

# ============================================================================
# PREPARE DATA (same as scratch.py)
# ============================================================================
print("\nPreparing data...")

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

# Feature engineering pipeline
X_train_miss = add_missing_indicators(X_train_raw, existing_vitals)
X_test_miss = add_missing_indicators(X_test_raw, existing_vitals)

X_train_extreme = add_extreme_flags(X_train_miss)
X_test_extreme = add_extreme_flags(X_test_miss)

X_train_fe = add_age_features(X_train_extreme)
X_test_fe = add_age_features(X_test_extreme)

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

# Encode categoricals
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le_c.fit(combined)
    X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))

X_train_final = X_train_fe
X_test_final = X_test_fe

print(f"✓ Total features: {X_train_final.shape[1]}")

# ============================================================================
# TRAIN BASE MODEL
# ============================================================================
print("\n" + "="*80)
print("TRAINING BASE MODEL")
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
print(f"\n✓ Base Model Validation AUC: {val_auc:.4f}")

# ============================================================================
# CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CALIBRATION ANALYSIS")
print("="*80)

# Check how well probabilities are calibrated
prob_true, prob_pred = calibration_curve(y_val, y_val_pred, n_bins=10, strategy='uniform')

print("\nCalibration curve (predicted vs actual):")
print(f"{'Bin':>5} {'Predicted':>12} {'Actual':>10} {'Diff':>10}")
print("-" * 40)
for i, (pred, true) in enumerate(zip(prob_pred, prob_true)):
    diff = pred - true
    print(f"{i+1:>5} {pred:>12.3f} {true:>10.3f} {diff:>+10.3f}")

# Calculate calibration error
calibration_error = np.mean(np.abs(prob_pred - prob_true))
print(f"\nMean Calibration Error: {calibration_error:.4f}")

# ============================================================================
# PROBABILITY TRANSFORMATIONS
# ============================================================================
print("\n" + "="*80)
print("TESTING PROBABILITY TRANSFORMATIONS")
print("="*80)

def test_transformation(y_true, y_pred, name, transform_func):
    """Test a probability transformation and report AUC."""
    y_transformed = transform_func(y_pred)
    auc = roc_auc_score(y_true, y_transformed)
    return auc

# Different transformations to try
transformations = {
    'Original': lambda x: x,
    'Power 0.8': lambda x: np.power(x, 0.8),
    'Power 0.9': lambda x: np.power(x, 0.9),
    'Power 1.1': lambda x: np.power(x, 1.1),
    'Power 1.2': lambda x: np.power(x, 1.2),
    'Log transform': lambda x: np.log1p(x) / np.log1p(1),
    'Sqrt': lambda x: np.sqrt(x),
    'Square': lambda x: np.power(x, 2),
}

print("\nTransformation results (Validation AUC):")
print("-" * 40)
best_transform = 'Original'
best_auc = val_auc

for name, func in transformations.items():
    try:
        auc = test_transformation(y_val, y_val_pred, name, func)
        marker = " <-- BEST" if auc > best_auc else ""
        print(f"{name:<20}: {auc:.4f}{marker}")
        if auc > best_auc:
            best_auc = auc
            best_transform = name
    except Exception as e:
        print(f"{name:<20}: Error - {e}")

print(f"\nBest transformation: {best_transform} (AUC: {best_auc:.4f})")

# ============================================================================
# PLATT SCALING (Sigmoid Calibration)
# ============================================================================
print("\n" + "="*80)
print("PLATT SCALING (Sigmoid Calibration)")
print("="*80)

# Retrain with calibration
model_for_calib = XGBClassifier(
    max_depth=5,
    learning_rate=0.03,
    n_estimators=500,  # Fewer since we'll calibrate
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

# Use CalibratedClassifierCV for Platt scaling
calibrated_model = CalibratedClassifierCV(model_for_calib, method='sigmoid', cv=5)
calibrated_model.fit(X_train_final, y_train)

y_val_calibrated = calibrated_model.predict_proba(X_val)[:, 1]
calibrated_auc = roc_auc_score(y_val, y_val_calibrated)
print(f"\n✓ Calibrated Model Validation AUC: {calibrated_auc:.4f}")
print(f"  Improvement: {calibrated_auc - val_auc:+.4f}")

# ============================================================================
# ISOTONIC CALIBRATION
# ============================================================================
print("\n" + "="*80)
print("ISOTONIC CALIBRATION")
print("="*80)

calibrated_model_iso = CalibratedClassifierCV(model_for_calib, method='isotonic', cv=5)
calibrated_model_iso.fit(X_train_final, y_train)

y_val_isotonic = calibrated_model_iso.predict_proba(X_val)[:, 1]
isotonic_auc = roc_auc_score(y_val, y_val_isotonic)
print(f"\n✓ Isotonic Calibrated AUC: {isotonic_auc:.4f}")
print(f"  Improvement: {isotonic_auc - val_auc:+.4f}")

# ============================================================================
# SUMMARY AND BEST MODEL SELECTION
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

results = {
    'Base Model': val_auc,
    f'Best Transform ({best_transform})': best_auc,
    'Platt Scaling': calibrated_auc,
    'Isotonic Calibration': isotonic_auc
}

print("\nAll approaches:")
for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<30}: {auc:.4f}")

best_approach = max(results, key=results.get)
best_score = results[best_approach]

print(f"\n🏆 Best approach: {best_approach} (AUC: {best_score:.4f})")

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

# 1. Base model predictions
y_test_base = model.predict_proba(X_test_final)[:, 1]
submission_base = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': y_test_base
})
output_base = BASE_DIR / "submissions" / "xgb_base_threshold.csv"
submission_base.to_csv(output_base, index=False)
print(f"✓ Base model: {output_base}")

# 2. Best transformation
if best_transform != 'Original':
    y_test_transformed = transformations[best_transform](y_test_base)
    submission_transform = pd.DataFrame({
        'icustay_id': test_raw[id_col].values,
        'HOSPITAL_EXPIRE_FLAG': y_test_transformed
    })
    output_transform = BASE_DIR / "submissions" / "xgb_transformed.csv"
    submission_transform.to_csv(output_transform, index=False)
    print(f"✓ Transformed ({best_transform}): {output_transform}")

# 3. Platt scaling
y_test_platt = calibrated_model.predict_proba(X_test_final)[:, 1]
submission_platt = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': y_test_platt
})
output_platt = BASE_DIR / "submissions" / "xgb_platt_calibrated.csv"
submission_platt.to_csv(output_platt, index=False)
print(f"✓ Platt calibrated: {output_platt}")

# 4. Isotonic calibration
y_test_isotonic = calibrated_model_iso.predict_proba(X_test_final)[:, 1]
submission_isotonic = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': y_test_isotonic
})
output_isotonic = BASE_DIR / "submissions" / "xgb_isotonic_calibrated.csv"
submission_isotonic.to_csv(output_isotonic, index=False)
print(f"✓ Isotonic calibrated: {output_isotonic}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print("\nRecommendation: Try submitting the calibrated versions to Kaggle")
print("If AUC is the metric, calibration may not help much, but worth testing.")
