"""
KNN Model for HEF Prediction

K-Nearest Neighbors approach - the "demanding" benchmark is KNN.
Uses same features as scratch.py but with KNN classifier.

Key differences from tree models:
- Distance-based (finds similar patients)
- Requires feature scaling
- All features weighted equally
- Can capture different patterns

Run from project root:
    python scripts/classification/knn_model.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
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
print("KNN MODEL FOR HEF PREDICTION")
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
# PREPARE DATA
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

# Feature engineering
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
# PREPROCESSING FOR KNN
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING FOR KNN")
print("="*80)

# KNN requires:
# 1. No missing values
# 2. Scaled features (distance-based)

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train_final),
    columns=X_train_final.columns,
    index=X_train_final.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test_final),
    columns=X_test_final.columns,
    index=X_test_final.index
)

print(f"✓ Imputed missing values (median strategy)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"✓ Scaled features (StandardScaler)")

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# ============================================================================
# FIND OPTIMAL K
# ============================================================================
print("\n" + "="*80)
print("FINDING OPTIMAL K")
print("="*80)

k_values = [3, 5, 7, 9, 11, 15, 21, 31, 51, 71, 101]
results = []

for k in k_values:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',  # Weight by inverse distance
        metric='euclidean',
        n_jobs=-1
    )
    knn.fit(X_tr, y_tr)
    y_val_pred = knn.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    results.append({'k': k, 'val_auc': val_auc})
    print(f"  K={k:3d}: Val AUC = {val_auc:.4f}")

# Find best K
best_result = max(results, key=lambda x: x['val_auc'])
best_k = best_result['k']
best_auc = best_result['val_auc']

print(f"\n✓ Best K: {best_k} (Val AUC: {best_auc:.4f})")

# ============================================================================
# TRAIN FINAL MODEL WITH BEST K
# ============================================================================
print("\n" + "="*80)
print("TRAINING FINAL KNN MODEL")
print("="*80)

# Try different weight schemes
print("\nTesting weight schemes:")

# Uniform weights
knn_uniform = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='uniform',
    metric='euclidean',
    n_jobs=-1
)
knn_uniform.fit(X_tr, y_tr)
y_val_uniform = knn_uniform.predict_proba(X_val)[:, 1]
auc_uniform = roc_auc_score(y_val, y_val_uniform)
print(f"  Uniform weights: {auc_uniform:.4f}")

# Distance weights
knn_distance = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='distance',
    metric='euclidean',
    n_jobs=-1
)
knn_distance.fit(X_tr, y_tr)
y_val_distance = knn_distance.predict_proba(X_val)[:, 1]
auc_distance = roc_auc_score(y_val, y_val_distance)
print(f"  Distance weights: {auc_distance:.4f}")

# Use best weight scheme
if auc_uniform > auc_distance:
    final_model = knn_uniform
    final_auc = auc_uniform
    weight_scheme = 'uniform'
else:
    final_model = knn_distance
    final_auc = auc_distance
    weight_scheme = 'distance'

print(f"\n✓ Using {weight_scheme} weights (Val AUC: {final_auc:.4f})")

# ============================================================================
# TRAIN ON FULL DATA
# ============================================================================
print("\n" + "="*80)
print("TRAINING ON FULL DATA")
print("="*80)

final_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    weights=weight_scheme,
    metric='euclidean',
    n_jobs=-1
)
final_knn.fit(X_train_scaled, y_train)

print(f"✓ Trained KNN with K={best_k}, weights={weight_scheme}")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

y_test_pred = final_knn.predict_proba(X_test_scaled)[:, 1]

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
submission = pd.DataFrame({
    'icustay_id': test_raw[id_col].values,
    'HOSPITAL_EXPIRE_FLAG': y_test_pred
})

output_file = BASE_DIR / "submissions" / "knn_model.csv"
submission.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"  Model: KNN")
print(f"  K: {best_k}")
print(f"  Weights: {weight_scheme}")
print(f"  Validation AUC: {final_auc:.4f}")
print(f"  Features: {X_train_final.shape[1]}")
print(f"  Submission: {output_file}")

print("\nComparison:")
print(f"  XGBoost (scratch.py): 74.9 Kaggle")
print(f"  KNN (this model):     {final_auc:.4f} Val AUC -> ??? Kaggle")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
