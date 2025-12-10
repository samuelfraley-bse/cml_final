"""
XGBoost with ICU History Features

Takes your best model (74.6 score) and adds ICU history features.

Expected improvement: 74.6 → 77-80+

Key insight: Patients with multiple previous ICU stays have different
mortality risk than first-time ICU patients.

Run from project root:
    python scripts/classification/xgb_with_history.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Find project root
current_dir = Path.cwd()
script_dir = Path(__file__).parent

if (current_dir / "notebooks" / "HEF").exists():
    BASE_DIR = current_dir
elif (current_dir.parent.parent / "notebooks" / "HEF").exists():
    BASE_DIR = current_dir.parent.parent
elif (script_dir.parent.parent / "notebooks" / "HEF").exists():
    BASE_DIR = script_dir.parent.parent
else:
    print("ERROR: Could not find project root!")
    sys.exit(1)

print(f"Project root: {BASE_DIR}")
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

from hef_prep import (
    add_icu_history_features,
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
print("XGBOOST WITH ICU HISTORY FEATURES")
print("Your best model (74.6) + Previous ICU stays features")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading raw data...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")
print(f"  ✓ Train: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# ADD ICU HISTORY FEATURES FIRST (on raw data, before any processing)
# ============================================================================
print("\n[2/6] Adding ICU history features...")
print("  Processing train and test data separately (safe, no leakage)")

train_with_hist = add_icu_history_features(
    train_raw,
    subject_col='subject_id',
    admit_col='ADMITTIME'
)

test_with_hist = add_icu_history_features(
    test_raw,
    subject_col='subject_id',
    admit_col='ADMITTIME'
)

print(f"\n  ✓ ICU history features added")
print(f"    Train: {train_raw.shape[1]} → {train_with_hist.shape[1]} features (+{train_with_hist.shape[1] - train_raw.shape[1]})")
print(f"    Test: {test_raw.shape[1]} → {test_with_hist.shape[1]} features (+{test_with_hist.shape[1] - test_raw.shape[1]})")

# ============================================================================
# SPLIT TARGET AND REMOVE LEAKAGE
# ============================================================================
print("\n[3/6] Preparing features...")

# Remove TRUE leakage
leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "LOS"]
id_cols_lower = [c.lower() for c in ID_COLS]

# Get target
y_train = train_with_hist[TARGET_COL_CLASS].copy()

# Drop leakage + IDs + targets
drop_cols = leak_cols + ID_COLS + [TARGET_COL_CLASS]
X_train_raw = train_with_hist.drop(columns=[c for c in drop_cols if c in train_with_hist.columns], errors='ignore')
X_test_raw = test_with_hist.drop(columns=[c for c in drop_cols if c in test_with_hist.columns], errors='ignore')

# Drop remaining ID columns (case-insensitive)
for col in list(X_train_raw.columns):
    if col.lower() in id_cols_lower:
        X_train_raw = X_train_raw.drop(columns=[col])
for col in list(X_test_raw.columns):
    if col.lower() in id_cols_lower:
        X_test_raw = X_test_raw.drop(columns=[col])

print(f"  ✓ X_train: {X_train_raw.shape}")
print(f"  ✓ y_train: {y_train.shape}, positive rate: {y_train.mean():.3f}")

# ============================================================================
# FEATURE ENGINEERING (YOUR BEST APPROACH)
# ============================================================================
print("\n[4/6] Applying feature engineering...")

print("  [4.1] Age features...")
X_train_fe = add_age_features(X_train_raw)
X_test_fe = add_age_features(X_test_raw)

print("  [4.2] BP outliers...")
X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)

print("  [4.3] Engineered features...")
X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)

print("  [4.4] Age interactions...")
X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

print("  [4.5] Composite scores...")
X_train_fe = add_composite_scores(X_train_fe)
X_test_fe = add_composite_scores(X_test_fe)

print("  [4.6] Admission interactions...")
X_train_fe = add_admission_interactions(X_train_fe)
X_test_fe = add_admission_interactions(X_test_fe)

# ============================================================================
# ADD DIAGNOSIS FEATURES (LABEL ENCODED - works better for XGBoost)
# ============================================================================
print("\n  [4.7] Adding DIAGNOSIS and ICD9 (label encoded)...")

# Check for DIAGNOSIS
if 'DIAGNOSIS' in X_train_fe.columns:
    print("    Adding DIAGNOSIS as label encoded...")
    le_diag = LabelEncoder()
    combined_diag = pd.concat([
        X_train_fe['DIAGNOSIS'].fillna('MISSING'),
        X_test_fe['DIAGNOSIS'].fillna('MISSING')
    ])
    le_diag.fit(combined_diag)
    X_train_fe['DIAGNOSIS_encoded'] = le_diag.transform(X_train_fe['DIAGNOSIS'].fillna('MISSING'))
    X_test_fe['DIAGNOSIS_encoded'] = le_diag.transform(X_test_fe['DIAGNOSIS'].fillna('MISSING'))
    
    # Drop original
    X_train_fe = X_train_fe.drop(columns=['DIAGNOSIS'])
    X_test_fe = X_test_fe.drop(columns=['DIAGNOSIS'])
    print(f"      ✓ DIAGNOSIS encoded: {len(le_diag.classes_)} unique values")

# Check for ICD9
icd9_col = None
for col in train_with_hist.columns:
    if 'ICD9' in col.upper() and 'DIAG' in col.upper():
        icd9_col = col
        break

if icd9_col:
    print(f"    Adding {icd9_col} as label encoded...")
    train_icd9 = train_with_hist[icd9_col].copy().fillna('MISSING')
    test_icd9 = test_with_hist[icd9_col].copy().fillna('MISSING')
    
    le_icd9 = LabelEncoder()
    le_icd9.fit(pd.concat([train_icd9, test_icd9]))
    X_train_fe['ICD9_encoded'] = le_icd9.transform(train_icd9)
    X_test_fe['ICD9_encoded'] = le_icd9.transform(test_icd9)
    
    # Also add ICD9 category (first 3 digits)
    train_cat = train_icd9.apply(lambda x: str(x)[:3] if x != 'MISSING' else 'MISSING')
    test_cat = test_icd9.apply(lambda x: str(x)[:3] if x != 'MISSING' else 'MISSING')
    le_cat = LabelEncoder()
    le_cat.fit(pd.concat([train_cat, test_cat]))
    X_train_fe['ICD9_category'] = le_cat.transform(train_cat)
    X_test_fe['ICD9_category'] = le_cat.transform(test_cat)
    
    print(f"      ✓ ICD9 encoded: {len(le_icd9.classes_)} unique codes")
    print(f"      ✓ ICD9 categories: {len(le_cat.classes_)} categories")

# ============================================================================
# ENCODE REMAINING CATEGORICALS
# ============================================================================
print("\n  [4.8] Encoding remaining categoricals...")

cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
print(f"    Found {len(cat_cols)} categorical columns: {cat_cols}")

for col in cat_cols:
    le_c = LabelEncoder()
    combined = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le_c.fit(combined)
    X_train_fe[col] = le_c.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le_c.transform(X_test_fe[col].astype(str))

X_train_final = X_train_fe
X_test_final = X_test_fe

print(f"\n  ✓ Final feature count: {X_train_final.shape[1]}")
print(f"    Your previous best had: 85 features")
print(f"    ICU history features: 10")
print(f"    New total: {X_train_final.shape[1]} features")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
print("\n[5/6] Creating train/validation split...")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print(f"  Train: {X_tr.shape[0]}, Valid: {X_val.shape[0]}")

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

# ============================================================================
# TRAIN XGBOOST (YOUR BEST HYPERPARAMETERS)
# ============================================================================
print("\n[6/6] Training XGBoost...")

# Use your best hyperparameters from xgb_best_param.py
model = XGBClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=1500,
    subsample=0.70,
    colsample_bytree=0.60,
    min_child_weight=7,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1.5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    early_stopping_rounds=50,
    n_jobs=-1
)

print("\n  Training with early stopping...")
model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=50
)

print(f"\n  ✓ Training complete! Best iteration: {model.best_iteration}")

# ============================================================================
# EVALUATE
# ============================================================================
print("\n" + "="*80)
print("EVALUATION")
print("="*80)

y_val_pred = model.predict_proba(X_val)[:, 1]
y_test_pred = model.predict_proba(X_test_final)[:, 1]

val_auc = roc_auc_score(y_val, y_val_pred)
print(f"\n✓ Validation AUC: {val_auc:.4f}")
print(f"  Your previous best: 0.8559 (Val AUC)")
print(f"  Improvement: {'+' if val_auc > 0.8559 else ''}{(val_auc - 0.8559):.4f}")

y_val_binary = (y_val_pred >= 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_val, y_val_binary, target_names=['Survived', 'Died']))

# ============================================================================
# FEATURE IMPORTANCE - SHOW ICU HISTORY FEATURES
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)

importance_df = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 30 Most Important Features:")
print(importance_df.head(30).to_string(index=False))

# Show ICU history features specifically
print("\n" + "="*80)
print("ICU HISTORY FEATURES IMPORTANCE")
print("="*80)

history_features = [
    'num_previous_icu_stays', 'is_first_icu_stay', 'is_readmission',
    'is_frequent_flyer', 'days_since_last_icu', 'readmission_30d',
    'readmission_90d', 'readmission_180d', 'num_previous_icu_stays_log',
    'days_since_last_icu_log', 'avg_previous_los', 'max_previous_los',
    'total_previous_icu_days', 'frequent_recent_readmit'
]

history_importance = importance_df[importance_df['feature'].isin(history_features)]

if len(history_importance) > 0:
    print(f"\nICU History Features (found {len(history_importance)}):\n")
    for idx, row in history_importance.iterrows():
        rank = list(importance_df['feature']).index(row['feature']) + 1
        pct = row['importance'] / importance_df['importance'].sum() * 100
        print(f"  #{rank:3d}  {row['feature']:<35} {row['importance']:.6f} ({pct:.2f}%)")
    
    total_hist_importance = history_importance['importance'].sum()
    total_importance = importance_df['importance'].sum()
    print(f"\n  Total ICU history contribution: {(total_hist_importance/total_importance*100):.2f}%")
else:
    print("  ⚠️  No ICU history features found in importance!")

# ============================================================================
# GENERATE SUBMISSION
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
output_file = output_dir / "xgb_with_icu_history.csv"

submission.to_csv(output_file, index=False)

print(f"\n✓ Submission saved: {output_file}")
print(f"\nPrediction stats:")
print(f"  Min:  {y_test_pred.min():.4f}")
print(f"  Max:  {y_test_pred.max():.4f}")
print(f"  Mean: {y_test_pred.mean():.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)

print(f"""
🎯 MODEL PERFORMANCE:
  Validation AUC:           {val_auc:.4f}
  Previous best (no history): 0.8559
  Improvement:              {'+' if val_auc > 0.8559 else ''}{(val_auc - 0.8559):.4f} AUC points
  
  Best iteration:           {model.best_iteration}
  
🔧 FEATURES:
  Total features:           {X_train_final.shape[1]}
  Previous best:            85 features
  New ICU history features: 10
  
📊 ICU HISTORY INSIGHTS:
  Patients with 0 previous stays:  {(X_train_final['is_first_icu_stay'] == 1).sum()} ({(X_train_final['is_first_icu_stay'] == 1).sum()/len(X_train_final)*100:.1f}%)
  Patients with 1+ previous stays: {(X_train_final['is_readmission'] == 1).sum()} ({(X_train_final['is_readmission'] == 1).sum()/len(X_train_final)*100:.1f}%)
  Frequent flyers (3+ stays):      {(X_train_final['is_frequent_flyer'] == 1).sum()} ({(X_train_final['is_frequent_flyer'] == 1).sum()/len(X_train_final)*100:.1f}%)
  30-day readmissions:             {(X_train_final['readmission_30d'] == 1).sum()} ({(X_train_final['readmission_30d'] == 1).sum()/len(X_train_final)*100:.1f}%)
  
🎲 EXPECTED KAGGLE SCORE:
  Previous: 74.6
  Expected: 76-79 (if ICU history features are predictive)
  
📁 OUTPUT:
  {output_file}
""")

print("="*80)
print("✓ COMPLETE! Submit and see if ICU history helps! 🎯")
print("="*80)