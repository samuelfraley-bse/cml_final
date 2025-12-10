"""
Logistic Regression with Simple ICD9 One-Hot Encoding

Strategy:
- Keep ONLY the most important/common ICD9 codes
- One-hot encode them (creates ~50-100 binary features)
- Add your age features (proven winners)
- Use Logistic Regression (like your classmate who got 80)
- Keep it simple and avoid overfitting

Expected: 78-81 score

Run from project root:
    python scripts/classification/logreg_simple_icd9.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
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
    add_age_features,
    clean_min_bp_outliers, 
    add_engineered_features,
    add_age_interactions,
    TARGET_COL_CLASS,
    ID_COLS
)

print("="*80)
print("LOGISTIC REGRESSION - SIMPLE ICD9 ONE-HOT")
print("Keeping it simple: Top ICD9 codes + Age features")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/7] Loading raw data...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")
print(f"  ✓ Train: {train_raw.shape}, Test: {test_raw.shape}")

# ============================================================================
# PREPARE ICD9 - KEEP ONLY TOP N CODES
# ============================================================================
print("\n[2/7] Processing ICD9 codes...")

# Find ICD9 column
icd9_col = None
for col in train_raw.columns:
    if 'ICD9' in col.upper() and 'DIAG' in col.upper():
        icd9_col = col
        break

if icd9_col:
    print(f"  Found ICD9 column: '{icd9_col}'")
    print(f"  Total unique codes: {train_raw[icd9_col].nunique()}")
    
    # Strategy: Keep top 50 most common ICD9 codes
    top_n = 50
    top_icd9_codes = train_raw[icd9_col].value_counts().head(top_n).index.tolist()
    
    coverage_train = (train_raw[icd9_col].isin(top_icd9_codes).sum() / len(train_raw) * 100)
    print(f"\n  Keeping top {top_n} ICD9 codes")
    print(f"  Coverage: {coverage_train:.1f}% of training data")
    
    # Create simplified ICD9 column: top codes + "OTHER"
    train_raw['ICD9_simplified'] = train_raw[icd9_col].apply(
        lambda x: x if x in top_icd9_codes else 'OTHER'
    )
    test_raw['ICD9_simplified'] = test_raw[icd9_col].apply(
        lambda x: x if x in top_icd9_codes else 'OTHER'
    )
    
    print(f"  Final categories: {train_raw['ICD9_simplified'].nunique()} (top {top_n} + OTHER)")
else:
    print("  ⚠️  No ICD9 column found")
    train_raw['ICD9_simplified'] = 'MISSING'
    test_raw['ICD9_simplified'] = 'MISSING'

# ============================================================================
# SPLIT TARGET AND REMOVE LEAKAGE
# ============================================================================
print("\n[3/7] Preparing features...")

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "LOS"]
id_cols_lower = [c.lower() for c in ID_COLS]

# Get target
y_train = train_raw[TARGET_COL_CLASS].copy()
print(f"  ✓ Target: {y_train.shape}, mortality rate: {y_train.mean():.3f}")

# Drop leakage + IDs + targets
drop_cols = leak_cols + ID_COLS + [TARGET_COL_CLASS]
X_train_raw = train_raw.drop(columns=[c for c in drop_cols if c in train_raw.columns], errors='ignore')
X_test_raw = test_raw.drop(columns=[c for c in drop_cols if c in test_raw.columns], errors='ignore')

# Drop remaining ID columns
for col in list(X_train_raw.columns):
    if col.lower() in id_cols_lower:
        X_train_raw = X_train_raw.drop(columns=[col])
for col in list(X_test_raw.columns):
    if col.lower() in id_cols_lower:
        X_test_raw = X_test_raw.drop(columns=[col])

print(f"  ✓ X_train: {X_train_raw.shape}")

# ============================================================================
# ADD AGE FEATURES (YOUR WINNING STRATEGY)
# ============================================================================
print("\n[4/7] Adding age features...")

X_train_fe = add_age_features(X_train_raw)
X_test_fe = add_age_features(X_test_raw)

X_train_fe = clean_min_bp_outliers(X_train_fe)
X_test_fe = clean_min_bp_outliers(X_test_fe)

# Add basic engineered features (vitals only, no complex stuff)
X_train_fe = add_engineered_features(X_train_fe)
X_test_fe = add_engineered_features(X_test_fe)

# Add age interactions (these were top features!)
X_train_fe = add_age_interactions(X_train_fe)
X_test_fe = add_age_interactions(X_test_fe)

print(f"  ✓ After age engineering: {X_train_fe.shape}")

# ============================================================================
# IDENTIFY FEATURE TYPES
# ============================================================================
print("\n[5/7] Identifying feature types...")

# Get numeric features
numeric_features = X_train_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Get categorical features (should just be ICD9_simplified + maybe a few others)
categorical_features = X_train_fe.select_dtypes(include=['object']).columns.tolist()

# Make sure ICD9_simplified is categorical
if 'ICD9_simplified' in X_train_fe.columns and 'ICD9_simplified' not in categorical_features:
    categorical_features.append('ICD9_simplified')
    if 'ICD9_simplified' in numeric_features:
        numeric_features.remove('ICD9_simplified')

print(f"  Numeric features: {len(numeric_features)}")
print(f"  Categorical features: {len(categorical_features)}")
print(f"  Categorical columns: {categorical_features}")

if 'ICD9_simplified' in categorical_features:
    n_icd9_cats = X_train_fe['ICD9_simplified'].nunique()
    print(f"\n  ICD9_simplified will create {n_icd9_cats} one-hot features")

# ============================================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================================
print("\n[6/7] Creating preprocessing pipeline...")

# Numeric: impute + scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: impute + one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print(f"  ✓ Pipeline created")
print(f"  Estimated features after one-hot: ~{len(numeric_features) + n_icd9_cats + 20}")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
print("\n[7/7] Training Logistic Regression with GridSearch...")

X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_fe, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print(f"  Train: {X_train.shape[0]}, Valid: {X_val.shape[0]}")

# Create pipeline
logreg_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# Grid search for optimal C
param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    estimator=logreg_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print(f"\n  Running GridSearchCV...")
grid_search.fit(X_train, y_train_split)

print(f"\n  ✓ GridSearchCV complete!")
print(f"  Best C: {grid_search.best_params_['model__C']}")
print(f"  Best CV AUC: {grid_search.best_score_:.4f}")

# Show all CV results
print(f"\n  All CV results:")
for params, score in zip(grid_search.cv_results_['params'], 
                         grid_search.cv_results_['mean_test_score']):
    print(f"    C={params['model__C']:6} → CV AUC: {score:.4f}")

# ============================================================================
# EVALUATE
# ============================================================================
print("\n" + "="*80)
print("VALIDATION EVALUATION")
print("="*80)

best_model = grid_search.best_estimator_

y_val_pred = best_model.predict(X_val)
y_val_proba = best_model.predict_proba(X_val)[:, 1]

val_auc = roc_auc_score(y_val, y_val_proba)
print(f"\n✓ Validation AUC: {val_auc:.4f}")
print(f"  Your XGBoost best: 0.8559")
print(f"  Her LogReg: ~0.865-0.870")
print(f"  This model: {val_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Survived', 'Died']))

# ============================================================================
# RETRAIN ON FULL DATA
# ============================================================================
print("\n" + "="*80)
print("RETRAINING ON FULL TRAINING DATA")
print("="*80)

print(f"  Retraining on all {len(y_train)} samples...")
best_model.fit(X_train_fe, y_train)
print("  ✓ Complete!")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

y_test_proba = best_model.predict_proba(X_test_fe)[:, 1]

print(f"  Test predictions:")
print(f"    Min:  {y_test_proba.min():.4f}")
print(f"    Max:  {y_test_proba.max():.4f}")
print(f"    Mean: {y_test_proba.mean():.4f}")

# Get test IDs
id_col = [c for c in test_raw.columns if c.lower() == 'icustay_id'][0]
test_ids = test_raw[id_col].values

submission = pd.DataFrame({
    'icustay_id': test_ids,
    'HOSPITAL_EXPIRE_FLAG': y_test_proba
})

output_dir = BASE_DIR / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "logreg_simple_icd9.csv"

submission.to_csv(output_file, index=False)

print(f"\n✓ Submission saved: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)

print(f"""
🎯 MODEL PERFORMANCE:
  Validation AUC:        {val_auc:.4f}
  Best C parameter:      {grid_search.best_params_['model__C']}
  5-Fold CV AUC:         {grid_search.best_score_:.4f}

🔧 FEATURES (SIMPLE!):
  Age features:          ~25 (age, bins, interactions)
  Vital signs:           ~30 (engineered features)
  ICD9 one-hot:          ~{n_icd9_cats} (top 50 codes + OTHER)
  Other categoricals:    ~10
  
  Total estimated:       ~{len(numeric_features) + n_icd9_cats + 20} features
  
💡 WHY THIS SHOULD WORK:
  1. ✓ Logistic Regression (like her winning model)
  2. ✓ ICD9 one-hot encoded (strong signal, manageable size)
  3. ✓ Your age features (proven to help)
  4. ✓ Simple and focused (less overfitting)
  5. ✓ Proper scaling and regularization

🎲 EXPECTED KAGGLE SCORE:
  Conservative: 76-78
  Optimistic:   78-80
  
  Your current best: 74.6
  Her score:         80.0
  
📁 OUTPUT:
  {output_file}

🔑 KEY DIFFERENCES FROM PREVIOUS ATTEMPTS:
  - NO ICU history (too weak, caused problems)
  - NO full DIAGNOSIS text (just ICD9 codes)
  - YES to top 50 ICD9 codes (manageable + predictive)
  - YES to your age engineering (your advantage)
  - Simple Logistic Regression (proven to work)
""")

print("="*80)
print("✓ COMPLETE! This should be a strong submission! 🎯")
print("="*80)