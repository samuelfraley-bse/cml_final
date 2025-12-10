"""
Lean Core Features - Model Comparison

Tests multiple models and tuning approaches on the same lean feature set:
- XGBoost (default)
- XGBoost (tuned)
- LightGBM
- CatBoost
- Random Forest
- Logistic Regression (baseline)

Run from project root:
    python scripts/classification/lean_model_comparison.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed - skipping")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed - skipping")

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

from hef_prep import (
    split_features_target,
    ID_COLS
)

print("="*80)
print("LEAN CORE FEATURES - MODEL COMPARISON")
print("="*80)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"\nTrain: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# LEAN FEATURE ENGINEERING (same as xgb_lean_core.py)
# ============================================================================

def add_lean_features(df):
    """Add only proven high-value features."""
    df = df.copy()

    # Age calculation
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
        df['is_elderly'] = (df['age_years'] >= 75).astype(int)
        df = df.drop(columns=['DOB', 'ADMITTIME'], errors='ignore')

    # Vital ranges
    if 'SpO2_Max' in df.columns and 'SpO2_Min' in df.columns:
        df['SpO2_range'] = df['SpO2_Max'] - df['SpO2_Min']
    if 'TempC_Max' in df.columns and 'TempC_Min' in df.columns:
        df['TempC_range'] = df['TempC_Max'] - df['TempC_Min']

    # Composite scores
    if 'HeartRate_Mean' in df.columns and 'SysBP_Mean' in df.columns:
        df['shock_index_mean'] = df['HeartRate_Mean'] / df['SysBP_Mean'].replace(0, np.nan)
    if 'SpO2_Min' in df.columns:
        df['spo2_deficit'] = (92 - df['SpO2_Min']).clip(lower=0)
    if 'TempC_Mean' in df.columns:
        df['temp_dev_mean'] = (df['TempC_Mean'] - 37.0).abs()
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

    # Critical flag
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

    # Age interactions
    if 'age_years' in df.columns:
        if 'instability_count' in df.columns:
            df['age_x_instability'] = df['age_years'] * df['instability_count']
        if 'temp_dev_mean' in df.columns:
            df['age_x_temp_dev'] = df['age_years'] * df['temp_dev_mean']
        if 'shock_index_mean' in df.columns:
            df['age_x_shock_index'] = df['age_years'] * df['shock_index_mean']

    # Elderly interactions
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

# Add ICD9 category
icd9_col = 'ICD9_diagnosis'
if icd9_col in train_raw.columns:
    train_icd9 = train_raw[icd9_col].fillna('MISSING')
    test_icd9 = test_raw[icd9_col].fillna('MISSING')
    train_cat = train_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    test_cat = test_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    le_cat = LabelEncoder()
    le_cat.fit(pd.concat([train_cat, test_cat]))
    X_train_fe['ICD9_category'] = le_cat.transform(train_cat)
    X_test_fe['ICD9_category'] = le_cat.transform(test_cat)

# Select lean features
LEAN_FEATURES = [
    'HeartRate_Min', 'HeartRate_Mean',
    'SysBP_Min', 'SysBP_Mean',
    'DiasBP_Min', 'DiasBP_Mean',
    'MeanBP_Min', 'MeanBP_Mean',
    'RespRate_Min', 'RespRate_Mean',
    'TempC_Min', 'TempC_Mean',
    'SpO2_Min', 'SpO2_Mean',
    'Glucose_Min', 'Glucose_Mean',
    'SpO2_range', 'TempC_range',
    'age_years', 'is_elderly',
    'shock_index_mean', 'spo2_deficit', 'temp_dev_mean',
    'glucose_excess', 'instability_count',
    'multiple_critical',
    'age_x_instability', 'age_x_temp_dev', 'age_x_shock_index',
    'elderly_and_hypotensive', 'elderly_and_fever',
    'ADMISSION_TYPE', 'GENDER',
    'ICD9_category',
]

existing_lean = [f for f in LEAN_FEATURES if f in X_train_fe.columns]
X_train_lean = X_train_fe[existing_lean].copy()
X_test_lean = X_test_fe[existing_lean].copy()

# Encode categoricals
cat_cols = X_train_lean.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_lean[col], X_test_lean[col]]).astype(str)
    le_c.fit(combined)
    X_train_lean[col] = le_c.transform(X_train_lean[col].astype(str))
    X_test_lean[col] = le_c.transform(X_test_lean[col].astype(str))

X_train_final = X_train_lean
X_test_final = X_test_lean

print(f"✓ Total features: {X_train_final.shape[1]}")

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

models = {}

# 1. XGBoost - Default (your current params)
models['XGB_Default'] = XGBClassifier(
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

# 2. XGBoost - Conservative (less overfitting)
models['XGB_Conservative'] = XGBClassifier(
    max_depth=4,
    learning_rate=0.02,
    n_estimators=2000,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=10,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    early_stopping_rounds=50,
    n_jobs=-1
)

# 3. XGBoost - Aggressive (more complexity)
models['XGB_Aggressive'] = XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.05,
    reg_alpha=0.05,
    reg_lambda=0.5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    early_stopping_rounds=50,
    n_jobs=-1
)

# 4. LightGBM
if HAS_LGBM:
    models['LightGBM'] = LGBMClassifier(
        max_depth=5,
        learning_rate=0.03,
        n_estimators=1500,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

# 5. CatBoost
if HAS_CATBOOST:
    models['CatBoost'] = CatBoostClassifier(
        depth=5,
        learning_rate=0.03,
        iterations=1500,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False,
        auto_class_weights='Balanced'
    )

# 6. Random Forest
models['RandomForest'] = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 7. Gradient Boosting (sklearn)
models['GradientBoosting'] = GradientBoostingClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    min_samples_split=10,
    random_state=42
)

# 8. Logistic Regression (baseline)
models['LogisticRegression'] = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# ============================================================================
# TRAIN AND EVALUATE ALL MODELS
# ============================================================================
print("\n" + "="*80)
print("TRAINING AND EVALUATING MODELS")
print("="*80)

results = []
test_predictions = {}

for name, model in models.items():
    print(f"\n[{name}]")

    try:
        # Train
        if 'XGB' in name:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif name == 'LightGBM':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        elif name == 'LogisticRegression':
            # Scale features for logistic regression
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test_final)
            model.fit(X_tr_scaled, y_tr)
            y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
            y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_tr, y_tr)

        # Predict
        if name != 'LogisticRegression':
            y_val_pred = model.predict_proba(X_val)[:, 1]
            y_test_pred = model.predict_proba(X_test_final)[:, 1]

        # Evaluate
        val_auc = roc_auc_score(y_val, y_val_pred)

        results.append({
            'model': name,
            'val_auc': val_auc
        })
        test_predictions[name] = y_test_pred

        print(f"  Val AUC: {val_auc:.4f}")

    except Exception as e:
        print(f"  Error: {e}")
        results.append({
            'model': name,
            'val_auc': 0.0
        })

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results).sort_values('val_auc', ascending=False)

print("\nModel Rankings:")
print("-" * 50)
for i, row in results_df.iterrows():
    marker = " <-- BEST" if row['val_auc'] == results_df['val_auc'].max() else ""
    print(f"  {row['model']:<25}: {row['val_auc']:.4f}{marker}")

best_model = results_df.iloc[0]['model']
best_auc = results_df.iloc[0]['val_auc']

print(f"\nBest Model: {best_model} (Val AUC: {best_auc:.4f})")

# ============================================================================
# GENERATE SUBMISSIONS FOR TOP MODELS
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]

# Save top 3 models
for i, row in results_df.head(3).iterrows():
    model_name = row['model']
    if model_name in test_predictions:
        filename = f"lean_{model_name.lower().replace(' ', '_')}.csv"
        submission = pd.DataFrame({
            'icustay_id': test_raw[id_col].values,
            'HOSPITAL_EXPIRE_FLAG': test_predictions[model_name]
        })
        output_file = BASE_DIR / "submissions" / filename
        submission.to_csv(output_file, index=False)
        print(f"✓ {model_name}: {output_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
