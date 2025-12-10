"""
XGBoost Classification with ICD9_diagnosis + AGE Features - Using sklearn API

This version:
- Calculates age from DOB and ADMITTIME
- Adds age interactions with vitals
- Adds temporal admission features
- Uses XGBClassifier (sklearn wrapper)

Run from project root:
    python scripts/classification/xgb_dx_age.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Add notebooks/HEF to path to import hef_prep functions
BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

from hef_prep import (
    split_features_target,
    add_age_features,
    clean_min_bp_outliers, 
    add_engineered_features,
    add_age_interactions,
    add_composite_scores,
    add_admission_interactions,
    TARGET_COL_CLASS,
    ID_COLS
)

print("="*80)
print("XGBOOST WITH ICD9 + AGE FEATURES (sklearn API)")
print("="*80)

# ============================================================================
# STEP 1: Load raw data
# ============================================================================
print("\n[1/7] Loading raw data...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"

train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"  Train: {train_raw.shape}")
print(f"  Test: {test_raw.shape}")

# ============================================================================
# STEP 2: Check for ICD9_diagnosis column
# ============================================================================
print("\n[2/7] Checking for ICD9_diagnosis column...")

icd9_col = None
for col in train_raw.columns:
    if 'ICD9' in col.upper() and 'DIAG' in col.upper():
        icd9_col = col
        break

if icd9_col:
    print(f"✓ Found ICD9 column: '{icd9_col}'")
    n_train = len(train_raw)
    n_missing_train = train_raw[icd9_col].isna().sum()
    n_unique_train = train_raw[icd9_col].nunique()
    
    print(f"  Total: {n_train}, Missing: {n_missing_train}, Unique codes: {n_unique_train}")
else:
    print("⚠️  No ICD9_diagnosis column found!")

# ============================================================================
# STEP 3: Split features and target (KEEP DOB and ADMITTIME for age calc)
# ============================================================================
print("\n[3/7] Splitting features and target...")

# CRITICAL: Don't drop DOB and ADMITTIME - we need them for age!
# DOD (Date of Death) = LEAKAGE (only exists if patient died)
# DEATHTIME = LEAKAGE (same as DOD, more specific)
# DISCHTIME = LEAKAGE (knows outcome)
# DIAGNOSIS and ICD9_diagnosis = dropped separately (added back encoded later)
leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]

X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw,
    test_df=test_raw,
    task="class",
    leak_cols=leak_cols,
    id_cols=ID_COLS
)

# Remove any remaining ID columns (case-insensitive)
id_cols_to_check = ['icustay_id', 'subject_id', 'hadm_id']
remaining_ids_train = [c for c in X_train_raw.columns if c.lower() in id_cols_to_check]
remaining_ids_test = [c for c in X_test_raw.columns if c.lower() in id_cols_to_check]

if remaining_ids_train:
    X_train_raw = X_train_raw.drop(columns=remaining_ids_train)
if remaining_ids_test:
    X_test_raw = X_test_raw.drop(columns=remaining_ids_test)

print(f"✓ Features split. Positive rate (death): {y_train.mean():.3%}")

# ============================================================================
# STEP 4: Add AGE features (calculates age, then drops DOB/ADMITTIME)
# ============================================================================
print("\n[4/7] Adding age features...")

X_train_with_age = add_age_features(X_train_raw)
X_test_with_age = add_age_features(X_test_raw)

print(f"✓ Shape after age features: Train {X_train_with_age.shape}, Test {X_test_with_age.shape}")

# ============================================================================
# STEP 5: Clean BP outliers
# ============================================================================
print("\n[5/7] Cleaning BP outliers...")

X_train_clean = clean_min_bp_outliers(X_train_with_age)
X_test_clean = clean_min_bp_outliers(X_test_with_age)

# ============================================================================
# STEP 6: Add engineered vital features
# ============================================================================
print("\n[6/7] Adding engineered vital features...")

X_train_base = add_engineered_features(X_train_clean)
X_test_base = add_engineered_features(X_test_clean)

print(f"✓ Base features prepared: {X_train_base.shape[1]} columns")

# ============================================================================
# STEP 7: Add age interactions
# ============================================================================
print("\n[7/11] Adding age interaction features...")

X_train_with_interactions = add_age_interactions(X_train_base)
X_test_with_interactions = add_age_interactions(X_test_base)

print(f"✓ After age interactions: {X_train_with_interactions.shape[1]} columns")

# ============================================================================
# STEP 8: Add composite severity scores
# ============================================================================
print("\n[8/11] Adding composite severity scores...")

X_train_with_composites = add_composite_scores(X_train_with_interactions)
X_test_with_composites = add_composite_scores(X_test_with_interactions)

print(f"✓ After composite scores: {X_train_with_composites.shape[1]} columns")

# ============================================================================
# STEP 9: Add admission interactions (BEFORE encoding!)
# ============================================================================
print("\n[9/11] Adding admission interaction features...")

X_train_with_admit = add_admission_interactions(X_train_with_composites)
X_test_with_admit = add_admission_interactions(X_test_with_composites)

print(f"✓ After admission interactions: {X_train_with_admit.shape[1]} columns")

# ============================================================================
# STEP 10: Process ICD9_diagnosis and add to features
# ============================================================================
print("\n[10/11] Processing ICD9_diagnosis column...")

if icd9_col:
    train_icd9 = train_raw[icd9_col].copy().fillna('MISSING')
    test_icd9 = test_raw[icd9_col].copy().fillna('MISSING')
    
    # Label encode full ICD9 codes
    le = LabelEncoder()
    all_codes = pd.concat([train_icd9, test_icd9])
    le.fit(all_codes)
    
    X_train_with_admit['ICD9_encoded'] = le.transform(train_icd9)
    X_test_with_admit['ICD9_encoded'] = le.transform(test_icd9)
    
    # Add ICD9 category (first 3 digits)
    train_icd9_category = train_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    test_icd9_category = test_icd9.apply(lambda x: x[:3] if x != 'MISSING' else 'MISSING')
    
    le_category = LabelEncoder()
    all_categories = pd.concat([train_icd9_category, test_icd9_category])
    le_category.fit(all_categories)
    
    X_train_with_admit['ICD9_category'] = le_category.transform(train_icd9_category)
    X_test_with_admit['ICD9_category'] = le_category.transform(test_icd9_category)
    
    print(f"  ✓ Added ICD9_encoded and ICD9_category features")
    print(f"  Total features now: {X_train_with_admit.shape[1]}")

X_train_final = X_train_with_admit
X_test_final = X_test_with_admit

# ============================================================================
# STEP 11: Encode any remaining categorical columns
# ============================================================================
print("\n[11/11] Encoding categorical columns...")

cat_cols = X_train_final.select_dtypes(include=['object']).columns.tolist()

if len(cat_cols) > 0:
    print(f"  Found {len(cat_cols)} categorical columns")
    
    for col in cat_cols:
        le_cat = LabelEncoder()
        combined = pd.concat([X_train_final[col], X_test_final[col]]).astype(str)
        le_cat.fit(combined)
        
        X_train_final[col] = le_cat.transform(X_train_final[col].astype(str))
        X_test_final[col] = le_cat.transform(X_test_final[col].astype(str))
    
    print(f"  ✓ All categorical columns encoded")

print(f"\n✓ FINAL FEATURE COUNT: {X_train_final.shape[1]} features")

# ============================================================================
# STEP 12: Train XGBoost model (sklearn API)
# ============================================================================
print("\n[12/12] Training XGBoost model with sklearn API...")

# Create train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print(f"  Train: {X_tr.shape[0]} samples, Valid: {X_val.shape[0]} samples")

# Calculate scale_pos_weight
scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

# XGBClassifier with same parameters
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

print("\n  Training...")
model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=50
)

print(f"\n  ✓ Training complete!")
print(f"  Best iteration: {model.best_iteration}")

# ============================================================================
# Evaluate and show feature importance
# ============================================================================
print("\n" + "="*80)
print("EVALUATION & FEATURE IMPORTANCE")
print("="*80)

# Predictions
y_val_pred = model.predict_proba(X_val)[:, 1]
y_test_pred = model.predict_proba(X_test_final)[:, 1]

# Calculate metrics
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"\n✓ Validation AUC: {val_auc:.4f}")

# Binary predictions
y_val_binary = (y_val_pred >= 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_val, y_val_binary, target_names=['Survived', 'Died']))

# Feature importance
print("\nTop 30 Most Important Features:")
importance_df = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=False)

# Show top 30
print(importance_df.head(30).to_string(index=False))

# Analyze AGE features specifically
print("\n" + "="*80)
print("AGE FEATURE ANALYSIS")
print("="*80)

age_features = importance_df[importance_df['feature'].str.contains('age', case=False)]

if len(age_features) > 0:
    print(f"\nFound {len(age_features)} age-related features:\n")
    for idx, row in age_features.iterrows():
        rank = list(importance_df.index).index(idx) + 1
        pct = (row['importance'] / importance_df['importance'].sum() * 100)
        print(f"  {row['feature']:<30} Rank: #{rank:3d}  Importance: {row['importance']:.6f}  ({pct:.2f}%)")
    
    # Overall age contribution
    age_total_importance = age_features['importance'].sum()
    total_importance = importance_df['importance'].sum()
    print(f"\n  Total AGE contribution: {(age_total_importance/total_importance*100):.2f}% of all features")

# Check ICD9 features
if icd9_col:
    print("\n" + "="*80)
    print("ICD9 FEATURE ANALYSIS")
    print("="*80)
    
    icd9_features = importance_df[importance_df['feature'].str.contains('ICD9', case=False)]
    
    if len(icd9_features) > 0:
        for idx, row in icd9_features.iterrows():
            rank = list(importance_df.index).index(idx) + 1
            print(f"\n  {row['feature']}:")
            print(f"    - Rank: #{rank} out of {len(importance_df)}")
            print(f"    - Importance: {row['importance']:.6f}")
            print(f"    - % of total: {(row['importance'] / importance_df['importance'].sum() * 100):.2f}%")
    
    # Overall ICD9 contribution
    icd9_total_importance = icd9_features['importance'].sum()
    total_importance = importance_df['importance'].sum()
    print(f"\n  Total ICD9 contribution: {(icd9_total_importance/total_importance*100):.2f}% of all features")

# ============================================================================
# Generate submission file
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
test_ids = test_raw[id_col].values

submission = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': y_test_pred
})

output_dir = BASE_DIR / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "xgboost_v2_with_composites.csv"

submission.to_csv(output_file, index=False)

print(f"\n✓ Submission saved: {output_file}")
print(f"\nPrediction statistics:")
print(f"  Min:    {y_test_pred.min():.4f}")
print(f"  Max:    {y_test_pred.max():.4f}")
print(f"  Mean:   {y_test_pred.mean():.4f}")
print(f"  Median: {np.median(y_test_pred):.4f}")

# ============================================================================
# Analyze composite score features
# ============================================================================
print("\n" + "="*80)
print("COMPOSITE SCORE ANALYSIS")
print("="*80)

composite_keywords = ['score', 'dysfunction', 'shock_state', 'failure_risk', 'critical', 'distress']
composite_features = importance_df[importance_df['feature'].str.contains('|'.join(composite_keywords), case=False)]

if len(composite_features) > 0:
    print(f"\nFound {len(composite_features)} composite score features:\n")
    for idx, row in composite_features.iterrows():
        rank = list(importance_df.index).index(idx) + 1
        pct = (row['importance'] / importance_df['importance'].sum() * 100)
        print(f"  {row['feature']:<35} Rank: #{rank:3d}  Importance: {row['importance']:.6f}  ({pct:.2f}%)")
    
    comp_total = composite_features['importance'].sum()
    print(f"\n  Total COMPOSITE contribution: {(comp_total/importance_df['importance'].sum()*100):.2f}%")

# ============================================================================
# Analyze admission interaction features
# ============================================================================
print("\n" + "="*80)
print("ADMISSION INTERACTION ANALYSIS")
print("="*80)

admit_keywords = ['emergency', 'urgent', 'elective', 'is_emergency', 'is_urgent', 'is_elective']
admit_features = importance_df[importance_df['feature'].str.contains('|'.join(admit_keywords), case=False)]

if len(admit_features) > 0:
    print(f"\nFound {len(admit_features)} admission-related features:\n")
    for idx, row in admit_features.iterrows():
        rank = list(importance_df.index).index(idx) + 1
        pct = (row['importance'] / importance_df['importance'].sum() * 100)
        print(f"  {row['feature']:<35} Rank: #{rank:3d}  Importance: {row['importance']:.6f}  ({pct:.2f}%)")
    
    admit_total = admit_features['importance'].sum()
    print(f"\n  Total ADMISSION contribution: {(admit_total/importance_df['importance'].sum()*100):.2f}%")

print(f"\n📊 Final Model Summary:")
print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Features used: {X_train_final.shape[1]}")
print(f"  Best iteration: {model.best_iteration}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)