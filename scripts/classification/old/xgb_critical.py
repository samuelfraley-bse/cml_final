"""
Clean Model Based on Correlation Analysis

Built from scratch using only features with proven correlation to mortality.

Top correlations:
1. SpO2_Min: -0.234 (LOW oxygen = death)
2. SysBP_Min: -0.195 (LOW BP = death)  
3. MeanBP_Min: -0.176
4. RespRate_Mean: +0.175 (HIGH resp = death)
5. SpO2_Mean: -0.157
6. TempC_Min: -0.137 (LOW temp = death - sepsis sign)
7. HeartRate_Max: +0.130 (HIGH HR = death)

Run from project root:
    python scripts/classification/clean_model.py
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

print("="*80)
print("CLEAN MODEL - BASED ON CORRELATION ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")
print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

y_train = train_raw['HOSPITAL_EXPIRE_FLAG'].values

# ============================================================================
# BUILD FEATURES BASED ON CORRELATION ANALYSIS
# ============================================================================
print("\n[2/5] Building features from correlation analysis...")

def build_features(df, original_df):
    """
    Build features based on correlation analysis.
    Only include features with meaningful correlation to mortality.
    """
    features = pd.DataFrame(index=df.index)
    
    # =========================================================================
    # TIER 1: STRONGEST CORRELATIONS (|r| > 0.15)
    # =========================================================================
    
    # SpO2_Min: -0.234 (STRONGEST!)
    features['SpO2_Min'] = df['SpO2_Min']
    features['SpO2_Mean'] = df['SpO2_Mean']
    
    # SysBP_Min: -0.195
    features['SysBP_Min'] = df['SysBP_Min']
    features['SysBP_Mean'] = df['SysBP_Mean']
    
    # MeanBP_Min: -0.176
    features['MeanBP_Min'] = df['MeanBP_Min']
    features['MeanBP_Mean'] = df['MeanBP_Mean']
    
    # RespRate: +0.175 (HIGH = bad)
    features['RespRate_Mean'] = df['RespRate_Mean']
    features['RespRate_Max'] = df['RespRate_Max']
    features['RespRate_Min'] = df['RespRate_Min']
    
    # =========================================================================
    # TIER 2: GOOD CORRELATIONS (|r| > 0.10)
    # =========================================================================
    
    # DiasBP_Min: -0.138
    features['DiasBP_Min'] = df['DiasBP_Min']
    features['DiasBP_Mean'] = df['DiasBP_Mean']
    
    # TempC_Min: -0.137 (LOW temp = sepsis/shock)
    features['TempC_Min'] = df['TempC_Min']
    features['TempC_Mean'] = df['TempC_Mean']
    
    # HeartRate_Max: +0.130
    features['HeartRate_Max'] = df['HeartRate_Max']
    features['HeartRate_Mean'] = df['HeartRate_Mean']
    
    # Glucose: +0.105
    features['Glucose_Mean'] = df['Glucose_Mean']
    features['Glucose_Max'] = df['Glucose_Max']
    features['Glucose_Min'] = df['Glucose_Min']
    
    # =========================================================================
    # TIER 3: WEAKER BUT USEFUL (|r| > 0.04)
    # =========================================================================
    features['SpO2_Max'] = df['SpO2_Max']
    features['TempC_Max'] = df['TempC_Max']
    features['HeartRate_Min'] = df['HeartRate_Min']
    features['SysBP_Max'] = df['SysBP_Max']
    features['DiasBP_Max'] = df['DiasBP_Max']
    features['MeanBP_Max'] = df['MeanBP_Max']
    
    # =========================================================================
    # DERIVED FEATURES - Based on top correlations
    # =========================================================================
    
    # SpO2 deficit (how far below 92% - clinical threshold)
    features['spo2_deficit'] = (92 - df['SpO2_Min']).clip(lower=0)
    
    # Severe hypoxia flag
    features['severe_hypoxia'] = (df['SpO2_Min'] < 88).astype(int)
    features['critical_hypoxia'] = (df['SpO2_Min'] < 85).astype(int)
    
    # BP flags (based on SysBP_Min being #2 correlation)
    features['hypotension'] = (df['SysBP_Min'] < 90).astype(int)
    features['severe_hypotension'] = (df['SysBP_Min'] < 80).astype(int)
    features['critical_hypotension'] = (df['SysBP_Min'] < 70).astype(int)
    
    # Shock index (HR/SBP) - combines two correlated features
    features['shock_index'] = df['HeartRate_Mean'] / df['SysBP_Mean'].replace(0, np.nan)
    features['shock_index_max'] = df['HeartRate_Max'] / df['SysBP_Min'].replace(0, np.nan)
    
    # Respiratory distress (RespRate was #4 correlation)
    features['tachypnea'] = (df['RespRate_Mean'] > 22).astype(int)
    features['severe_tachypnea'] = (df['RespRate_Max'] > 30).astype(int)
    features['critical_tachypnea'] = (df['RespRate_Max'] > 35).astype(int)
    
    # Temperature - hypothermia is bad (sepsis sign)
    features['hypothermia'] = (df['TempC_Min'] < 36).astype(int)
    features['severe_hypothermia'] = (df['TempC_Min'] < 35).astype(int)
    features['fever'] = (df['TempC_Max'] > 38.3).astype(int)
    
    # Tachycardia
    features['tachycardia'] = (df['HeartRate_Max'] > 100).astype(int)
    features['severe_tachycardia'] = (df['HeartRate_Max'] > 120).astype(int)
    features['extreme_tachycardia'] = (df['HeartRate_Max'] > 150).astype(int)
    
    # Glucose
    features['hyperglycemia'] = (df['Glucose_Max'] > 180).astype(int)
    features['severe_hyperglycemia'] = (df['Glucose_Max'] > 300).astype(int)
    features['hypoglycemia'] = (df['Glucose_Min'] < 70).astype(int)
    
    # =========================================================================
    # CRITICAL FLAG COUNT (proven to be #1 feature!)
    # =========================================================================
    critical_cols = [
        'critical_hypoxia', 'severe_hypoxia',
        'critical_hypotension', 'severe_hypotension',
        'severe_hypothermia', 'hypothermia',
        'critical_tachypnea', 'severe_tachypnea',
        'extreme_tachycardia', 'severe_tachycardia',
        'severe_hyperglycemia', 'hypoglycemia'
    ]
    existing_critical = [c for c in critical_cols if c in features.columns]
    features['critical_count'] = features[existing_critical].sum(axis=1)
    features['multiple_critical'] = (features['critical_count'] >= 2).astype(int)
    features['severe_critical'] = (features['critical_count'] >= 3).astype(int)
    
    # =========================================================================
    # VITAL INSTABILITY (ranges)
    # =========================================================================
    features['SpO2_range'] = df['SpO2_Max'] - df['SpO2_Min']
    features['SysBP_range'] = df['SysBP_Max'] - df['SysBP_Min']
    features['HR_range'] = df['HeartRate_Max'] - df['HeartRate_Min']
    features['Temp_range'] = df['TempC_Max'] - df['TempC_Min']
    features['RespRate_range'] = df['RespRate_Max'] - df['RespRate_Min']
    
    # Instability count
    features['high_spo2_instability'] = (features['SpO2_range'] > 10).astype(int)
    features['high_bp_instability'] = (features['SysBP_range'] > 50).astype(int)
    features['high_hr_instability'] = (features['HR_range'] > 40).astype(int)
    
    features['instability_count'] = (
        features['high_spo2_instability'] + 
        features['high_bp_instability'] + 
        features['high_hr_instability']
    )
    
    # =========================================================================
    # AGE (from DOB and ADMITTIME)
    # =========================================================================
    if 'DOB' in df.columns and 'ADMITTIME' in df.columns:
        dob = pd.to_datetime(df['DOB'])
        admit = pd.to_datetime(df['ADMITTIME'])
        features['age_years'] = admit.dt.year - dob.dt.year
        features['is_elderly'] = (features['age_years'] >= 75).astype(int)
        features['is_very_elderly'] = (features['age_years'] >= 85).astype(int)
        
        # Age interactions with top correlates
        features['age_x_spo2_deficit'] = features['age_years'] * features['spo2_deficit']
        features['age_x_critical'] = features['age_years'] * features['critical_count']
        features['elderly_and_critical'] = (
            (features['is_elderly'] == 1) & (features['multiple_critical'] == 1)
        ).astype(int)
    
    # =========================================================================
    # GLUCOSE MISSING FLAG (44% mortality when missing!)
    # =========================================================================
    features['glucose_missing'] = original_df['Glucose_Mean'].isna().astype(int)
    
    # =========================================================================
    # CATEGORICAL FEATURES
    # =========================================================================
    cat_cols = ['GENDER', 'ADMISSION_TYPE', 'FIRST_CAREUNIT', 'INSURANCE', 'ETHNICITY', 'MARITAL_STATUS']
    for col in cat_cols:
        if col in df.columns:
            features[col] = df[col]
    
    # ICD9 category
    if 'ICD9_diagnosis' in original_df.columns:
        features['ICD9_category'] = original_df['ICD9_diagnosis'].fillna('MISSING').apply(
            lambda x: str(x)[:3] if x != 'MISSING' else 'MISSING'
        )
    
    return features


# Build features
X_train = build_features(train_raw, train_raw)
X_test = build_features(test_raw, test_raw)

print(f"  Features built: {X_train.shape[1]}")

# ============================================================================
# ENCODE CATEGORICALS
# ============================================================================
print("\n[3/5] Encoding categoricals...")

cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]]).astype(str).fillna('MISSING')
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str).fillna('MISSING'))
    X_test[col] = le.transform(X_test[col].astype(str).fillna('MISSING'))

# Fill NaN with median
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

print(f"✓ Final features: {X_train.shape[1]}")

# ============================================================================
# K-FOLD TRAINING
# ============================================================================
print("\n[4/5] K-Fold training...")

N_FOLDS = 5
kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X_train))
test_predictions = np.zeros(len(X_test))
fold_aucs = []
all_importances = []

X_train_arr = X_train.values
X_test_arr = X_test.values

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_arr, y_train)):
    X_tr, X_val = X_train_arr[train_idx], X_train_arr[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
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
        random_state=42 + fold,
        tree_method='hist',
        eval_metric='auc',
        early_stopping_rounds=50,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
    test_predictions += model.predict_proba(X_test_arr)[:, 1] / N_FOLDS
    all_importances.append(model.feature_importances_)
    
    fold_auc = roc_auc_score(y_val, oof_predictions[val_idx])
    fold_aucs.append(fold_auc)
    print(f"  Fold {fold+1}: AUC = {fold_auc:.4f}")

oof_auc = roc_auc_score(y_train, oof_predictions)
print(f"\n  Mean Fold AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
print(f"  Overall OOF AUC: {oof_auc:.4f}")

# ============================================================================
# FEATURE IMPORTANCE (averaged)
# ============================================================================
print("\n" + "="*80)
print("TOP 30 FEATURES (averaged across folds)")
print("="*80)

avg_importance = np.mean(all_importances, axis=0)
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': avg_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(30).to_string(index=False))

# Show correlations vs importance
print("\n\nCorrelation vs Importance (sanity check):")
print("-"*60)
raw_correlations = {
    'SpO2_Min': -0.234, 'SysBP_Min': -0.195, 'MeanBP_Min': -0.176,
    'RespRate_Mean': 0.175, 'SpO2_Mean': -0.157, 'TempC_Min': -0.137,
    'HeartRate_Max': 0.130, 'Glucose_Mean': 0.105
}
for feat, corr in raw_correlations.items():
    if feat in importance_df['feature'].values:
        imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
        rank = list(importance_df['feature']).index(feat) + 1
        print(f"  {feat:20s}: corr={corr:+.3f}, imp={imp:.4f}, rank=#{rank}")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n[5/5] Generating submission...")

submission = pd.DataFrame({
    'icustay_id': test_raw['icustay_id'].values,
    'HOSPITAL_EXPIRE_FLAG': test_predictions
})

output_file = BASE_DIR / "submissions" / "clean_model.csv"
submission.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

print(f"""
📊 Summary:
  Features: {X_train.shape[1]}
  OOF AUC: {oof_auc:.4f}
  Mean Fold AUC: {np.mean(fold_aucs):.4f}
  
  Built from scratch using correlation analysis:
  - Focused on SpO2_Min (#1), SysBP_Min (#2), RespRate (#4)
  - Critical flags based on actual clinical thresholds
  - Age interactions with top predictors
  - Glucose missing flag (44% mortality!)
""")

print("="*80)
print("✓ COMPLETE!")
print("="*80)