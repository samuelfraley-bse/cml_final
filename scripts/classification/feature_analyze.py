"""
Feature Analysis and Investigation Script

This script analyzes your features to understand:
1. Which features are most important
2. Which features are redundant (correlated)
3. Train vs test distribution differences (explains CV→Kaggle gap)
4. Missing value patterns
5. Feature interactions
6. Potential leakage

Run from project root:
    python scripts/analysis/feature_analysis.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
from category_encoders import TargetEncoder

# Setup
BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

print("="*80)
print("FEATURE ANALYSIS AND INVESTIGATION")
print("="*80)

# ============================================================================
# LOAD YOUR BASELINE MODEL DATA
# ============================================================================

print("\n[Step 1/7] Loading baseline model data...")

# Import helper functions (same as your baseline)
def get_icd9_chapter(code):
    """Map ICD9 code to broad chapter"""
    if pd.isna(code) or code == 'MISSING':
        return 'MISSING'
    code_str = str(code).strip().upper()
    if code_str.startswith('V'):
        return 'V_CODES'
    if code_str.startswith('E'):
        return 'E_CODES'
    try:
        code_num = int(code_str.replace('.', '')[:3])
    except (ValueError, IndexError):
        return 'UNKNOWN'
    
    if 1 <= code_num <= 139:
        return 'INFECTIOUS'
    elif 140 <= code_num <= 239:
        return 'NEOPLASMS'
    elif 240 <= code_num <= 279:
        return 'ENDOCRINE'
    elif 280 <= code_num <= 289:
        return 'BLOOD'
    elif 290 <= code_num <= 319:
        return 'MENTAL'
    elif 320 <= code_num <= 389:
        return 'NERVOUS'
    elif 390 <= code_num <= 459:
        return 'CIRCULATORY'
    elif 460 <= code_num <= 519:
        return 'RESPIRATORY'
    elif 520 <= code_num <= 579:
        return 'DIGESTIVE'
    elif 580 <= code_num <= 629:
        return 'GENITOURINARY'
    elif 630 <= code_num <= 679:
        return 'PREGNANCY'
    elif 680 <= code_num <= 709:
        return 'SKIN'
    elif 710 <= code_num <= 739:
        return 'MUSCULOSKELETAL'
    elif 740 <= code_num <= 759:
        return 'CONGENITAL'
    elif 760 <= code_num <= 779:
        return 'PERINATAL'
    elif 780 <= code_num <= 799:
        return 'SYMPTOMS'
    elif 800 <= code_num <= 999:
        return 'INJURY'
    else:
        return 'UNKNOWN'

def load_diagnosis_data(filepath=None):
    """Load MIMIC_diagnoses.csv"""
    if filepath is None:
        base_dir = Path.cwd()
        filepath = base_dir / "data" / "raw" / "MIMIC III dataset HEF" / "extra_data" / "MIMIC_diagnoses.csv"
    filepath = Path(filepath)
    diagnosis_df = pd.read_csv(filepath)
    diagnosis_df.columns = diagnosis_df.columns.str.upper()
    if 'HADM_ID' in diagnosis_df.columns:
        diagnosis_df = diagnosis_df.rename(columns={'HADM_ID': 'hadm_id'})
    return diagnosis_df

def prepare_diagnosis_features(main_df, diagnosis_df, hadm_col='hadm_id'):
    """
    Create GENERALIZABLE diagnosis features (same as V2).
    """
    df = main_df.copy()
    
    # Find hadm_col (case-insensitive)
    hadm_col_actual = None
    for col in df.columns:
        if col.upper() == hadm_col.upper():
            hadm_col_actual = col
            break
    
    if hadm_col_actual is None:
        print(f"⚠️  Warning: {hadm_col} not found in data")
        df['primary_diagnosis_chapter'] = 'MISSING'
        df['diagnosis_count'] = 0
        df['num_organ_systems'] = 0
        df['has_respiratory'] = 0
        df['has_cardiac'] = 0
        df['has_infection'] = 0
        return df
    
    diag_df = diagnosis_df.copy()
    if 'hadm_id' not in diag_df.columns and 'HADM_ID' in diag_df.columns:
        diag_df = diag_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    # Add chapter to all diagnosis records
    diag_df['icd9_chapter'] = diag_df['ICD9_CODE'].apply(get_icd9_chapter)
    
    # Feature 1: Primary diagnosis CHAPTER
    primary_dx = diag_df[diag_df['SEQ_NUM'] == 1][['hadm_id', 'icd9_chapter']].copy()
    primary_dx = primary_dx.rename(columns={'icd9_chapter': 'primary_diagnosis_chapter'})
    
    df = df.merge(primary_dx, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['primary_diagnosis_chapter'] = df['primary_diagnosis_chapter'].fillna('MISSING')
    
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    # Feature 2: Diagnosis count
    dx_counts = diag_df.groupby('hadm_id').size().reset_index(name='diagnosis_count')
    df = df.merge(dx_counts, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['diagnosis_count'] = df['diagnosis_count'].fillna(0).astype(int)
    
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    # Feature 3: Number of unique organ systems
    organ_counts = diag_df.groupby('hadm_id')['icd9_chapter'].nunique().reset_index(name='num_organ_systems')
    df = df.merge(organ_counts, left_on=hadm_col_actual, right_on='hadm_id', how='left')
    df['num_organ_systems'] = df['num_organ_systems'].fillna(0).astype(int)
    
    if 'hadm_id' in df.columns and hadm_col_actual != 'hadm_id':
        df = df.drop(columns=['hadm_id'])
    
    # Feature 4-6: Domain flags
    respiratory_admissions = diag_df[diag_df['icd9_chapter'] == 'RESPIRATORY']['hadm_id'].unique()
    cardiac_admissions = diag_df[diag_df['icd9_chapter'] == 'CIRCULATORY']['hadm_id'].unique()
    infection_admissions = diag_df[diag_df['icd9_chapter'] == 'INFECTIOUS']['hadm_id'].unique()
    
    df['has_respiratory'] = df[hadm_col_actual].isin(respiratory_admissions).astype(int)
    df['has_cardiac'] = df[hadm_col_actual].isin(cardiac_admissions).astype(int)
    df['has_infection'] = df[hadm_col_actual].isin(infection_admissions).astype(int)
    
    return df

# Import from hef_prep
from hef_prep import (
    split_features_target,
    add_age_features,
    clean_min_bp_outliers,
    add_engineered_features,
    TARGET_COL_CLASS,
    ID_COLS
)

# Load data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"✓ Train raw: {train_raw.shape}")
print(f"✓ Test raw: {test_raw.shape}")

# ============================================================================
# CRITICAL FIX: Merge diagnosis BEFORE split_features_target
# ============================================================================

print("\n[Loading and merging diagnosis data...]")
diagnosis_df = load_diagnosis_data()

# Merge diagnosis features into RAW data (before split)
print("Merging diagnosis into train...")
train_raw = prepare_diagnosis_features(train_raw, diagnosis_df)

print("Merging diagnosis into test...")
test_raw = prepare_diagnosis_features(test_raw, diagnosis_df)

print(f"✓ After diagnosis merge - Train: {train_raw.shape}, Test: {test_raw.shape}")

# Remove "Diff" column (just used for date adjustment, no predictive value)
if 'Diff' in train_raw.columns:
    train_raw = train_raw.drop(columns=['Diff'])
    print("✓ Removed 'Diff' column from train")
if 'Diff' in test_raw.columns:
    test_raw = test_raw.drop(columns=['Diff'])
    print("✓ Removed 'Diff' column from test")

# ============================================================================
# NOW split features (this removes hadm_id)
# ============================================================================

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]
X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw,
    test_df=test_raw,
    task="class",
    leak_cols=leak_cols,
    id_cols=ID_COLS
)

# Add features (same as baseline)
X_train_age = add_age_features(X_train_raw)
X_test_age = add_age_features(X_test_raw)

# Remove age bins
age_bins = ['age_0_40', 'age_40_50', 'age_50_60', 'age_60_70', 'age_70_80', 'age_80_90', 'age_90_plus']
X_train_age = X_train_age.drop(columns=[c for c in age_bins if c in X_train_age.columns], errors='ignore')
X_test_age = X_test_age.drop(columns=[c for c in age_bins if c in X_test_age.columns], errors='ignore')

X_train_clean = clean_min_bp_outliers(X_train_age)
X_test_clean = clean_min_bp_outliers(X_test_age)

X_train_fe = add_engineered_features(X_train_clean)
X_test_fe = add_engineered_features(X_test_clean)

# Remove complex features
complex_features = [
    'age_x_instability', 'age_x_temp_dev', 'age_x_shock_index', 'age_x_spo2_deficit',
    'age_x_spo2_min', 'age_x_sysbp_min', 'elderly_and_hypotensive', 'elderly_and_hypoxic',
    'very_elderly_and_tachy', 'organ_dysfunction_score', 'cardiovascular_stress_score',
    'multi_organ_dysfunction', 'shock_state', 'resp_failure_risk', 'age_adjusted_critical'
]
X_train_fe = X_train_fe.drop(columns=[c for c in complex_features if c in X_train_fe.columns], errors='ignore')
X_test_fe = X_test_fe.drop(columns=[c for c in complex_features if c in X_test_fe.columns], errors='ignore')

# Encode categoricals
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
if 'primary_diagnosis_chapter' in cat_cols:
    cat_cols.remove('primary_diagnosis_chapter')

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le.fit(combined)
    X_train_fe[col] = le.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le.transform(X_test_fe[col].astype(str))

# Target encode
target_enc = TargetEncoder(cols=['primary_diagnosis_chapter'], smoothing=20.0, min_samples_leaf=20, return_df=True)
X_train_encoded = target_enc.fit_transform(X_train_fe, y_train)
X_test_encoded = target_enc.transform(X_test_fe)

print(f"✓ Data loaded: {X_train_encoded.shape[1]} features")

# ============================================================================
# ANALYSIS 1: FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("[Step 2/7] FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Train simple model
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    max_depth=3, learning_rate=0.01, n_estimators=300,
    subsample=0.7, colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight,
    random_state=42, n_jobs=-1
)
model.fit(X_train_encoded, y_train, verbose=False)

# Get feature importance
importance_df = pd.DataFrame({
    'feature': X_train_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 30 Most Important Features:")
print(importance_df.head(30).to_string(index=False))

print("\nBottom 20 Least Important Features (consider removing):")
print(importance_df.tail(20).to_string(index=False))

# Save to file
output_dir = BASE_DIR / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)
importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
print(f"\n✓ Saved to: {output_dir / 'feature_importance.csv'}")

# ============================================================================
# ANALYSIS 2: FEATURE CORRELATION
# ============================================================================

print("\n" + "="*80)
print("[Step 3/7] FEATURE CORRELATION ANALYSIS")
print("="*80)

# Calculate correlations
corr_matrix = X_train_encoded.corr().abs()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.8:  # Threshold for high correlation
            high_corr_pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

if len(high_corr_pairs) > 0:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
    print(f"\n⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.8):")
    print(high_corr_df.to_string(index=False))
    print("\n💡 Consider removing one feature from each pair to reduce redundancy")
    high_corr_df.to_csv(output_dir / "high_correlations.csv", index=False)
else:
    print("\n✓ No highly correlated features found (>0.8 threshold)")

# ============================================================================
# ANALYSIS 3: TRAIN VS TEST DISTRIBUTION
# ============================================================================

print("\n" + "="*80)
print("[Step 4/7] TRAIN VS TEST DISTRIBUTION ANALYSIS")
print("="*80)
print("This helps explain why CV score (0.83) differs from Kaggle score (0.73)")

# Kolmogorov-Smirnov test for distribution differences
distribution_diffs = []

for col in X_train_encoded.columns:
    train_vals = X_train_encoded[col].dropna()
    test_vals = X_test_encoded[col].dropna()
    
    if len(train_vals) > 0 and len(test_vals) > 0:
        statistic, pvalue = ks_2samp(train_vals, test_vals)
        distribution_diffs.append({
            'feature': col,
            'ks_statistic': statistic,
            'p_value': pvalue,
            'train_mean': train_vals.mean(),
            'test_mean': test_vals.mean(),
            'mean_diff': abs(train_vals.mean() - test_vals.mean())
        })

dist_df = pd.DataFrame(distribution_diffs).sort_values('ks_statistic', ascending=False)

print("\nTop 20 features with LARGEST train/test distribution differences:")
print("(High KS statistic = distributions are very different)")
print(dist_df.head(20)[['feature', 'ks_statistic', 'train_mean', 'test_mean']].to_string(index=False))

print("\n💡 Features with large distribution shifts may not generalize well!")
print("   Consider removing or transforming these features.")

dist_df.to_csv(output_dir / "train_test_distributions.csv", index=False)
print(f"\n✓ Saved to: {output_dir / 'train_test_distributions.csv'}")

# ============================================================================
# ANALYSIS 4: MISSING VALUE PATTERNS
# ============================================================================

print("\n" + "="*80)
print("[Step 5/7] MISSING VALUE ANALYSIS")
print("="*80)

# Calculate missing percentages
train_missing = (X_train_encoded.isnull().sum() / len(X_train_encoded) * 100).sort_values(ascending=False)
test_missing = (X_test_encoded.isnull().sum() / len(X_test_encoded) * 100).sort_values(ascending=False)

missing_df = pd.DataFrame({
    'feature': train_missing.index,
    'train_missing_pct': train_missing.values,
    'test_missing_pct': test_missing[train_missing.index].values
})
missing_df['missing_diff'] = abs(missing_df['train_missing_pct'] - missing_df['test_missing_pct'])
missing_df = missing_df[missing_df['train_missing_pct'] > 0].sort_values('train_missing_pct', ascending=False)

if len(missing_df) > 0:
    print(f"\nFeatures with missing values:")
    print(missing_df.head(20).to_string(index=False))
    
    # Check if missingness is informative
    print("\n💡 Checking if missingness is predictive...")
    for feat in missing_df.head(5)['feature']:
        if feat in X_train_encoded.columns:
            missing_indicator = X_train_encoded[feat].isnull().astype(int)
            if missing_indicator.sum() > 10:
                death_rate_missing = y_train[missing_indicator == 1].mean()
                death_rate_not_missing = y_train[missing_indicator == 0].mean()
                diff = abs(death_rate_missing - death_rate_not_missing)
                if diff > 0.05:
                    print(f"  {feat}: Death rate when missing={death_rate_missing:.3f}, not missing={death_rate_not_missing:.3f} (diff={diff:.3f})")
    
    missing_df.to_csv(output_dir / "missing_values.csv", index=False)
else:
    print("\n✓ No missing values found")

# ============================================================================
# ANALYSIS 5: TARGET CORRELATION
# ============================================================================

print("\n" + "="*80)
print("[Step 6/7] TARGET CORRELATION ANALYSIS")
print("="*80)

# Calculate correlation with target
target_corr = X_train_encoded.corrwith(y_train).abs().sort_values(ascending=False)
target_corr_df = pd.DataFrame({
    'feature': target_corr.index,
    'abs_correlation': target_corr.values
})

print("\nTop 20 features most correlated with mortality:")
print(target_corr_df.head(20).to_string(index=False))

print("\nBottom 20 features least correlated with mortality (consider removing):")
# Exclude Diff from display
target_corr_bottom = target_corr_df[target_corr_df['feature'] != 'Diff'].tail(20)
print(target_corr_bottom.to_string(index=False))

# Potential leakage detection
print("\n⚠️  Checking for potential leakage (correlation > 0.5):")
potential_leakage = target_corr_df[target_corr_df['abs_correlation'] > 0.5]
if len(potential_leakage) > 0:
    print(potential_leakage.to_string(index=False))
    print("\n💡 Extremely high correlations might indicate leakage - investigate these!")
else:
    print("✓ No features with suspiciously high correlation (>0.5)")

target_corr_df.to_csv(output_dir / "target_correlation.csv", index=False)

# ============================================================================
# ANALYSIS 6: FEATURE GROUP PERFORMANCE
# ============================================================================

print("\n" + "="*80)
print("[Step 7/7] FEATURE GROUP PERFORMANCE")
print("="*80)
print("Testing which groups of features contribute most to performance\n")

# Define feature groups
feature_groups = {
    'diagnosis': ['primary_diagnosis_chapter', 'diagnosis_count', 'num_organ_systems', 
                  'has_respiratory', 'has_cardiac', 'has_infection'],
    'age': [c for c in X_train_encoded.columns if 'age' in c.lower()],
    'vitals_raw': [c for c in X_train_encoded.columns if any(v in c for v in ['HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'TempC', 'SpO2', 'Glucose'])],
    'vitals_engineered': [c for c in X_train_encoded.columns if any(v in c.lower() for v in ['shock', 'pulse_pressure', 'instability', 'deviation', 'critical'])],
    'demographic': [c for c in X_train_encoded.columns if any(d in c.lower() for d in ['gender', 'insurance', 'religion', 'marital', 'ethnicity', 'admission'])],
    'temporal': [c for c in X_train_encoded.columns if any(t in c.lower() for t in ['hour', 'day', 'month', 'weekend', 'night', 'winter'])]
}

# Test each group
group_scores = []
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for group_name, features in feature_groups.items():
    available_features = [f for f in features if f in X_train_encoded.columns]
    if len(available_features) == 0:
        continue
    
    X_group = X_train_encoded[available_features]
    
    model_group = XGBClassifier(
        max_depth=3, learning_rate=0.05, n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1
    )
    
    scores = cross_val_score(model_group, X_group, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    group_scores.append({
        'feature_group': group_name,
        'num_features': len(available_features),
        'cv_auc': scores.mean(),
        'cv_std': scores.std()
    })
    print(f"  {group_name:20s}: {len(available_features):3d} features, AUC={scores.mean():.4f} (+/- {scores.std():.4f})")

# Test all features combined
all_scores = cross_val_score(model, X_train_encoded, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
group_scores.append({
    'feature_group': 'ALL_FEATURES',
    'num_features': X_train_encoded.shape[1],
    'cv_auc': all_scores.mean(),
    'cv_std': all_scores.std()
})
print(f"  {'ALL_FEATURES':20s}: {X_train_encoded.shape[1]:3d} features, AUC={all_scores.mean():.4f} (+/- {all_scores.std():.4f})")

group_scores_df = pd.DataFrame(group_scores).sort_values('cv_auc', ascending=False)
group_scores_df.to_csv(output_dir / "feature_group_performance.csv", index=False)

print(f"\n✓ Saved to: {output_dir / 'feature_group_performance.csv'}")

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n📊 Key Findings:")
print(f"  1. Total features: {X_train_encoded.shape[1]}")
print(f"  2. Features with >0 importance: {(importance_df['importance'] > 0).sum()}")
print(f"  3. Features with <0.001 importance: {(importance_df['importance'] < 0.001).sum()} (consider removing)")
print(f"  4. Highly correlated pairs: {len(high_corr_pairs)}")
print(f"  5. Features with large train/test drift: {(dist_df['ks_statistic'] > 0.1).sum()}")

print("\n💡 Recommendations:")

# Recommendation 1: Remove low importance features
low_importance = importance_df[importance_df['importance'] < 0.001]
if len(low_importance) > 10:
    print(f"\n  1. Remove {len(low_importance)} low-importance features:")
    print(f"     {', '.join(low_importance.head(10)['feature'].tolist())}...")

# Recommendation 2: Handle correlated features
if len(high_corr_pairs) > 0:
    print(f"\n  2. Remove redundant features from {len(high_corr_pairs)} correlated pairs")

# Recommendation 3: Distribution drift
high_drift = dist_df[dist_df['ks_statistic'] > 0.15]
if len(high_drift) > 0:
    print(f"\n  3. Investigate {len(high_drift)} features with large train/test distribution differences:")
    print(f"     {', '.join(high_drift.head(5)['feature'].tolist())}")
    print(f"     These may explain CV (0.83) vs Kaggle (0.73) gap!")

# Recommendation 4: Best feature groups
print(f"\n  4. Most predictive feature groups (use these as foundation):")
for _, row in group_scores_df.head(3).iterrows():
    print(f"     - {row['feature_group']}: AUC={row['cv_auc']:.4f}")

print("\n📁 All analysis files saved to: " + str(output_dir))
print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)