"""
Find Best Model - Clean Start

1. Load features from scratch.py (your best at 74.9)
2. Select top N features by importance
3. Test multiple models
4. Generate submissions for top performers

Run from project root:
    python scripts/classification/find_best_model.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

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
print("FIND BEST MODEL - CLEAN START")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# FEATURE ENGINEERING (exact same as scratch.py)
# ============================================================================

existing_vitals = ['HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean',
                   'SysBP_Min', 'SysBP_Max', 'SysBP_Mean',
                   'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',
                   'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean',
                   'RespRate_Min', 'RespRate_Max', 'RespRate_Mean',
                   'TempC_Min', 'TempC_Max', 'TempC_Mean',
                   'SpO2_Min', 'SpO2_Max', 'SpO2_Mean',
                   'Glucose_Min', 'Glucose_Max', 'Glucose_Mean']

def add_missing_indicators(df):
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

# Prepare data
print("\nPreparing features (same as scratch.py)...")

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw, test_df=test_raw, task="class",
    leak_cols=leak_cols, id_cols=ID_COLS
)

for col in [c for c in X_train_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_train_raw = X_train_raw.drop(columns=[col])
for col in [c for c in X_test_raw.columns if c.lower() in ['icustay_id', 'subject_id', 'hadm_id']]:
    X_test_raw = X_test_raw.drop(columns=[col])

X_train_fe = add_missing_indicators(X_train_raw)
X_test_fe = add_missing_indicators(X_test_raw)
X_train_fe = add_extreme_flags(X_train_fe)
X_test_fe = add_extreme_flags(X_test_fe)
X_train_fe = add_age_features(X_train_fe)
X_test_fe = add_age_features(X_test_fe)
X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)
X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)
X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

# Add ICD9
if 'ICD9_diagnosis' in train_raw.columns:
    train_icd9 = train_raw['ICD9_diagnosis'].fillna('MISSING')
    test_icd9 = test_raw['ICD9_diagnosis'].fillna('MISSING')
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

print(f"✓ Total features: {X_train_fe.shape[1]}")

# ============================================================================
# STEP 1: GET FEATURE IMPORTANCES FROM BASELINE MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 1: BASELINE MODEL & FEATURE IMPORTANCE")
print("="*80)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_fe, y_train, test_size=0.2, stratify=y_train, random_state=42
)

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

baseline = XGBClassifier(
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

baseline.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
y_val_pred = baseline.predict_proba(X_val)[:, 1]
baseline_auc = roc_auc_score(y_val, y_val_pred)

print(f"Baseline (all {X_train_fe.shape[1]} features): Val AUC = {baseline_auc:.4f}")

# Get feature importance
importance_df = pd.DataFrame({
    'feature': X_train_fe.columns,
    'importance': baseline.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
    print(f"  {i:2}. {row['feature']:<30} {row['importance']:.4f}")

# ============================================================================
# STEP 2: TEST DIFFERENT FEATURE COUNTS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: OPTIMAL FEATURE COUNT")
print("="*80)

feature_counts = [30, 50, 70, 90, 110, 122]  # 122 = all
results_by_count = []

for n_features in feature_counts:
    top_features = importance_df.head(n_features)['feature'].tolist()

    X_tr_subset = X_tr[top_features]
    X_val_subset = X_val[top_features]

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

    model.fit(X_tr_subset, y_tr, eval_set=[(X_val_subset, y_val)], verbose=False)
    y_pred = model.predict_proba(X_val_subset)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    results_by_count.append({'n_features': n_features, 'val_auc': auc})
    print(f"  {n_features:3} features: Val AUC = {auc:.4f}")

best_count = max(results_by_count, key=lambda x: x['val_auc'])
print(f"\n✓ Best: {best_count['n_features']} features (Val AUC: {best_count['val_auc']:.4f})")

# Use best feature count
optimal_features = importance_df.head(best_count['n_features'])['feature'].tolist()

# ============================================================================
# STEP 3: TEST DIFFERENT MODELS WITH OPTIMAL FEATURES
# ============================================================================
print("\n" + "="*80)
print(f"STEP 3: MODEL COMPARISON ({best_count['n_features']} features)")
print("="*80)

X_tr_opt = X_tr[optimal_features]
X_val_opt = X_val[optimal_features]
X_train_opt = X_train_fe[optimal_features]
X_test_opt = X_test_fe[optimal_features]

models = {}
test_preds = {}

# Model 1: XGBoost Default
models['XGB_Default'] = XGBClassifier(
    max_depth=5, learning_rate=0.03, n_estimators=1500,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight, random_state=42,
    tree_method='hist', eval_metric='auc', early_stopping_rounds=50, n_jobs=-1
)

# Model 2: XGBoost More Trees
models['XGB_MoreTrees'] = XGBClassifier(
    max_depth=4, learning_rate=0.01, n_estimators=3000,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight, random_state=42,
    tree_method='hist', eval_metric='auc', early_stopping_rounds=100, n_jobs=-1
)

# Model 3: XGBoost Higher Reg
models['XGB_HighReg'] = XGBClassifier(
    max_depth=5, learning_rate=0.03, n_estimators=1500,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
    gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight, random_state=42,
    tree_method='hist', eval_metric='auc', early_stopping_rounds=50, n_jobs=-1
)

# Model 4: LightGBM
if HAS_LGBM:
    models['LightGBM'] = LGBMClassifier(
        max_depth=5, learning_rate=0.03, n_estimators=1500,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1
    )

results = []

for name, model in models.items():
    print(f"\n[{name}]")

    if 'XGB' in name:
        model.fit(X_tr_opt, y_tr, eval_set=[(X_val_opt, y_val)], verbose=False)
    else:
        model.fit(X_tr_opt, y_tr, eval_set=[(X_val_opt, y_val)])

    y_val_pred = model.predict_proba(X_val_opt)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)

    # Train on full data for submission
    if 'XGB' in name:
        model.fit(X_train_opt, y_train, eval_set=[(X_val_opt, y_val)], verbose=False)
    else:
        model.fit(X_train_opt, y_train)

    y_test_pred = model.predict_proba(X_test_opt)[:, 1]
    test_preds[name] = y_test_pred

    results.append({'model': name, 'val_auc': val_auc})
    print(f"  Val AUC: {val_auc:.4f}")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*80)
print("RESULTS")
print("="*80)

results_df = pd.DataFrame(results).sort_values('val_auc', ascending=False)

print("\nModel Rankings:")
for _, row in results_df.iterrows():
    marker = " <-- BEST" if row['val_auc'] == results_df['val_auc'].max() else ""
    print(f"  {row['model']:<20}: {row['val_auc']:.4f}{marker}")

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

for name, preds in test_preds.items():
    filename = f"best_{name.lower()}.csv"
    submission = pd.DataFrame({
        'icustay_id': test_raw[id_col].values,
        'HOSPITAL_EXPIRE_FLAG': preds
    })
    output_file = BASE_DIR / "submissions" / filename
    submission.to_csv(output_file, index=False)
    print(f"✓ {name}: {output_file}")

# Also save feature list
feature_file = BASE_DIR / "submissions" / "optimal_features.txt"
with open(feature_file, 'w') as f:
    f.write(f"Optimal feature count: {best_count['n_features']}\n\n")
    for i, feat in enumerate(optimal_features, 1):
        imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
        f.write(f"{i:3}. {feat}: {imp:.4f}\n")
print(f"✓ Feature list: {feature_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"\nBest model: {results_df.iloc[0]['model']}")
print(f"Val AUC: {results_df.iloc[0]['val_auc']:.4f}")
print(f"Features: {best_count['n_features']}")
