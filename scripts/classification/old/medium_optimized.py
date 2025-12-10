"""
Medium Optimized Model - Sweet Spot Between Simple and Complex

Based on learnings:
- 38 features: 0.8514 OOF (too simple)
- 122 features: 0.8537 OOF (too complex)
- Target: ~60-70 features

Key changes:
1. Keep critical flags (they work!)
2. Simplify ICD9 to major disease groups (~17 categories)
3. Remove confirmed zero-importance features
4. Keep top age interactions

Run from project root:
    python scripts/classification/medium_optimized.py
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
print("MEDIUM OPTIMIZED MODEL (~60-70 features)")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# ICD9 MAJOR DISEASE CATEGORIES
# ============================================================================

def get_icd9_major_category(code):
    """Map ICD9 code to major disease category (~17 groups)."""
    if pd.isna(code) or code == 'MISSING':
        return 'MISSING'

    code = str(code).strip()

    # Handle V codes (supplementary)
    if code.startswith('V'):
        return 'V_SUPPLEMENTARY'

    # Handle E codes (external causes)
    if code.startswith('E'):
        return 'E_EXTERNAL'

    # Get first 3 digits for numeric codes
    try:
        num = int(code[:3])
    except:
        return 'OTHER'

    # ICD-9-CM Major Categories
    if 1 <= num <= 139:
        return 'INFECTIOUS'
    elif 140 <= num <= 239:
        return 'NEOPLASMS'
    elif 240 <= num <= 279:
        return 'ENDOCRINE'
    elif 280 <= num <= 289:
        return 'BLOOD'
    elif 290 <= num <= 319:
        return 'MENTAL'
    elif 320 <= num <= 389:
        return 'NERVOUS'
    elif 390 <= num <= 459:
        return 'CIRCULATORY'
    elif 460 <= num <= 519:
        return 'RESPIRATORY'
    elif 520 <= num <= 579:
        return 'DIGESTIVE'
    elif 580 <= num <= 629:
        return 'GENITOURINARY'
    elif 630 <= num <= 679:
        return 'PREGNANCY'
    elif 680 <= num <= 709:
        return 'SKIN'
    elif 710 <= num <= 739:
        return 'MUSCULOSKELETAL'
    elif 740 <= num <= 759:
        return 'CONGENITAL'
    elif 760 <= num <= 779:
        return 'PERINATAL'
    elif 780 <= num <= 799:
        return 'SYMPTOMS'
    elif 800 <= num <= 999:
        return 'INJURY'
    else:
        return 'OTHER'

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_medium_features(df):
    """Create medium complexity feature set."""
    df = df.copy()

    # ===== AGE =====
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

    # Age groups (just 2, not 7)
    df['is_elderly'] = (df['age_years'] >= 75).astype(int)

    # ===== VITAL RANGES =====
    for vital in ['HeartRate', 'SysBP', 'SpO2', 'TempC', 'RespRate', 'Glucose']:
        min_col = f'{vital}_Min'
        max_col = f'{vital}_Max'
        if min_col in df.columns and max_col in df.columns:
            df[f'{vital}_range'] = df[max_col] - df[min_col]

    # ===== COMPOSITE SCORES =====
    # Shock index
    if 'HeartRate_Mean' in df.columns and 'SysBP_Mean' in df.columns:
        df['shock_index_mean'] = df['HeartRate_Mean'] / df['SysBP_Mean'].replace(0, np.nan)

    # SpO2 deficit
    if 'SpO2_Min' in df.columns:
        df['spo2_deficit'] = (95 - df['SpO2_Min']).clip(lower=0)

    # Temperature deviation
    if 'TempC_Mean' in df.columns:
        df['temp_dev_mean'] = (df['TempC_Mean'] - 37.0).abs()

    # Glucose excess
    if 'Glucose_Max' in df.columns:
        df['glucose_excess'] = (df['Glucose_Max'] - 180).clip(lower=0)

    # ===== CRITICAL FLAGS (keep these - they work!) =====
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

    # Critical flags
    critical_count = pd.Series(0, index=df.index)
    if 'SysBP_Min' in df.columns:
        df['very_low_bp'] = (df['SysBP_Min'] < 80).astype(int)
        critical_count += df['very_low_bp']
    if 'HeartRate_Min' in df.columns:
        df['bradycardia'] = (df['HeartRate_Min'] < 50).astype(int)
        critical_count += df['bradycardia']
    if 'TempC_Min' in df.columns:
        df['hypothermia'] = (df['TempC_Min'] < 35).astype(int)
        critical_count += df['hypothermia']
    if 'TempC_Max' in df.columns:
        df['high_fever'] = (df['TempC_Max'] > 39).astype(int)
        critical_count += df['high_fever']

    df['critical_flag_count'] = critical_count
    df['any_critical_flag'] = (critical_count > 0).astype(int)
    df['multiple_critical'] = (critical_count >= 2).astype(int)

    # ===== TOP AGE INTERACTIONS (just 3) =====
    df['age_x_instability'] = df['age_years'] * df['instability_count']
    if 'temp_dev_mean' in df.columns:
        df['age_x_temp_dev'] = df['age_years'] * df['temp_dev_mean']
    if 'shock_index_mean' in df.columns:
        df['age_x_shock_index'] = df['age_years'] * df['shock_index_mean']

    # ===== ELDERLY INTERACTIONS (top 2) =====
    if 'SysBP_Min' in df.columns:
        df['elderly_and_hypotensive'] = df['is_elderly'] * (df['SysBP_Min'] < 90).astype(int)
    if 'TempC_Max' in df.columns:
        df['elderly_and_fever'] = df['is_elderly'] * (df['TempC_Max'] > 38.5).astype(int)

    return df

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\nPreparing medium feature set...")

train_fe = create_medium_features(train_raw.copy())
test_fe = create_medium_features(test_raw.copy())

# Get target
y_train = train_raw['HOSPITAL_EXPIRE_FLAG']

# Select features
FEATURE_COLS = [
    # Raw vitals
    'HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean',
    'SysBP_Min', 'SysBP_Max', 'SysBP_Mean',
    'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',
    'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean',
    'RespRate_Min', 'RespRate_Max', 'RespRate_Mean',
    'TempC_Min', 'TempC_Max', 'TempC_Mean',
    'SpO2_Min', 'SpO2_Max', 'SpO2_Mean',
    'Glucose_Min', 'Glucose_Max', 'Glucose_Mean',

    # Ranges
    'HeartRate_range', 'SysBP_range', 'SpO2_range',
    'TempC_range', 'RespRate_range', 'Glucose_range',

    # Age
    'age_years', 'is_elderly',

    # Composites
    'shock_index_mean', 'spo2_deficit', 'temp_dev_mean', 'glucose_excess',
    'instability_count',

    # Critical flags (keep all - they work)
    'very_low_bp', 'bradycardia', 'hypothermia', 'high_fever',
    'critical_flag_count', 'any_critical_flag', 'multiple_critical',

    # Age interactions
    'age_x_instability', 'age_x_temp_dev', 'age_x_shock_index',
    'elderly_and_hypotensive', 'elderly_and_fever',

    # Categoricals
    'ADMISSION_TYPE', 'GENDER', 'FIRST_CAREUNIT', 'INSURANCE', 'ETHNICITY',
]

X_train = train_fe[[c for c in FEATURE_COLS if c in train_fe.columns]].copy()
X_test = test_fe[[c for c in FEATURE_COLS if c in test_fe.columns]].copy()

# ===== ENCODE CATEGORICALS =====
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

# ===== ICD9 HIGH-RISK FLAGS (instead of 548 categories) =====
print("\n  Analyzing ICD9 mortality rates...")

# Calculate mortality rate for each ICD9 code
train_icd = train_raw['ICD9_diagnosis'].fillna('MISSING')
test_icd = test_raw['ICD9_diagnosis'].fillna('MISSING')

icd_mortality = train_raw.groupby(train_icd)['HOSPITAL_EXPIRE_FLAG'].agg(['mean', 'count'])
icd_mortality.columns = ['mortality', 'count']

# High-risk codes: mortality > 20% AND at least 50 samples
high_risk_codes = icd_mortality[(icd_mortality['mortality'] > 0.20) &
                                 (icd_mortality['count'] >= 50)].index.tolist()

# Very high-risk codes: mortality > 30% AND at least 30 samples
very_high_risk_codes = icd_mortality[(icd_mortality['mortality'] > 0.30) &
                                      (icd_mortality['count'] >= 30)].index.tolist()

print(f"  High-risk ICD9 codes (>20% mortality, n>=50): {len(high_risk_codes)}")
print(f"  Very high-risk codes (>30% mortality, n>=30): {len(very_high_risk_codes)}")

# Create binary flags
X_train['icd9_high_risk'] = train_icd.isin(high_risk_codes).astype(int)
X_test['icd9_high_risk'] = test_icd.isin(high_risk_codes).astype(int)

X_train['icd9_very_high_risk'] = train_icd.isin(very_high_risk_codes).astype(int)
X_test['icd9_very_high_risk'] = test_icd.isin(very_high_risk_codes).astype(int)

# Also add major category (17 groups) as backup
train_icd_major = train_raw['ICD9_diagnosis'].apply(get_icd9_major_category)
test_icd_major = test_raw['ICD9_diagnosis'].apply(get_icd9_major_category)

le_icd = LabelEncoder()
le_icd.fit(pd.concat([train_icd_major, test_icd_major]))
X_train['ICD9_major'] = le_icd.transform(train_icd_major)
X_test['ICD9_major'] = le_icd.transform(test_icd_major)

print(f"  ICD9_major: {len(le_icd.classes_)} categories")

# Show top high-risk codes
print("\n  Top high-risk ICD9 codes:")
top_risky = icd_mortality[(icd_mortality['count'] >= 50)].sort_values('mortality', ascending=False).head(10)
for code, row in top_risky.iterrows():
    print(f"    {code:<10}: {row['mortality']:.1%} mortality ({int(row['count'])} samples)")

print(f"\n✓ Total features: {X_train.shape[1]}")

# ============================================================================
# K-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("K-FOLD CROSS-VALIDATION")
print("="*80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

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

    model = XGBClassifier(**params, scale_pos_weight=fold_scale, random_state=42)
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
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nSimple (38 features):  OOF=0.8514, Gen=0.8431")
print(f"Complex (122 features): OOF=0.8537, Gen=0.8455")
print(f"Medium ({X_train.shape[1]} features):  OOF={oof_auc:.4f}, Gen={gen_score:.4f}")

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

output_file = BASE_DIR / "submissions" / "medium_optimized.csv"
submission.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
