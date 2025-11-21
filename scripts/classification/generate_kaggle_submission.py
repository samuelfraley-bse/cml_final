"""
Kaggle Submission Generator for HEF Dataset
Run this script from your project root directory where your data folder is located

Usage:
    python generate_kaggle_submissions.py

Requirements:
    - mimic_train_HEF.csv and mimic_test_HEF.csv in data/raw/MIMIC III dataset HEF/
    - sklearn, pandas, numpy installed
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# ============================================
# CONFIGURATION
# ============================================

# Adjust these paths to match your directory structure
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
OUTPUT_DIR = BASE_DIR / "submissions"

TRAIN_FILE = "mimic_train_HEF.csv"
TEST_FILE = "mimic_test_HEF.csv"

# Columns to drop (leakage)
LEAK_COLS = [
    "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS", 
    "DOB", "DEATHTIME", "DISCHTIME", "DOD", 
    "LOS", "HOSPITAL_EXPIRE_FLAG"
]

ID_COLS = ["icustay_id", "subject_id", "hadm_id"]

# ============================================
# DATA PREPARATION
# ============================================

def load_data():
    """Load train and test data"""
    print("Loading data...")
    print(f"  Train: {DATA_DIR / TRAIN_FILE}")
    print(f"  Test: {DATA_DIR / TEST_FILE}")
    
    train_df = pd.read_csv(DATA_DIR / TRAIN_FILE)
    test_df = pd.read_csv(DATA_DIR / TEST_FILE)
    
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    
    return train_df, test_df

def clean_bp_outliers(df):
    """Remove implausible blood pressure readings"""
    df = df.copy()
    
    bp_thresholds = {
        "SysBP_Min": 40.0,
        "DiasBP_Min": 10.0,
        "MeanBP_Min": 30.0,
    }
    
    for col, threshold in bp_thresholds.items():
        if col in df.columns:
            mask = (df[col] < threshold) & df[col].notna()
            n_removed = mask.sum()
            if n_removed > 0:
                print(f"  Removing {n_removed} implausible {col} values (< {threshold})")
                df.loc[mask, col] = np.nan
    
    return df

def prepare_features_target(train_df, test_df):
    """Split into X, y and handle column drops"""
    print("\nPreparing features and target...")
    
    # Find target column (case-insensitive)
    target_col = None
    for col in train_df.columns:
        if col.lower() == "hospital_expire_flag":
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("HOSPITAL_EXPIRE_FLAG not found in training data")
    
    # Get y
    y = train_df[target_col].copy()
    print(f"  Target: {target_col}")
    print(f"  Positive rate: {y.mean():.3f}")
    
    # Drop columns from X (case-insensitive)
    def get_drop_cols(df, cols_to_drop):
        cols_lower = [c.lower() for c in cols_to_drop]
        return [col for col in df.columns if col.lower() in cols_lower]
    
    drop_train = get_drop_cols(train_df, ID_COLS + LEAK_COLS + ["LOS"])
    drop_test = get_drop_cols(test_df, ID_COLS + LEAK_COLS)
    
    X = train_df.drop(columns=drop_train)
    X_test = test_df.drop(columns=drop_test)
    
    print(f"  X shape: {X.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    return X, y, X_test

# ============================================
# PREPROCESSING
# ============================================

def create_preprocessor(X_train):
    """Create sklearn preprocessing pipeline"""
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\n  Numeric features: {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
    ])
    
    return preprocessor

# ============================================
# MODEL TRAINING
# ============================================

def train_random_forest(X_train, y_train, preprocessor):
    """Train Random Forest with best hyperparameters from your tests"""
    print("\nTraining Random Forest...")
    print("  (Using best params from validation: 0.8181 AUC)")
    
    rf = RandomForestClassifier(
        n_estimators=203,
        max_depth=None,
        min_samples_split=7,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('clf', rf),
    ])
    
    pipeline.fit(X_train, y_train)
    print("  ✓ Random Forest trained")
    
    return pipeline

def train_gradient_boosting(X_train, y_train, preprocessor):
    """Train Gradient Boosting"""
    print("\nTraining Gradient Boosting...")
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
    )
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('clf', gb),
    ])
    
    pipeline.fit(X_train, y_train)
    print("  ✓ Gradient Boosting trained")
    
    return pipeline

def train_extra_trees(X_train, y_train, preprocessor):
    """Train Extra Trees"""
    print("\nTraining Extra Trees...")
    
    from sklearn.ensemble import ExtraTreesClassifier
    
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=7,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('clf', et),
    ])
    
    pipeline.fit(X_train, y_train)
    print("  ✓ Extra Trees trained")
    
    return pipeline

# ============================================
# SUBMISSION GENERATION
# ============================================

def generate_submissions():
    """Main function to generate all submissions"""
    print("="*70)
    print("KAGGLE SUBMISSION GENERATOR")
    print("="*70)
    
    # Load and prepare data
    train_df, test_df = load_data()
    train_df = clean_bp_outliers(train_df)
    test_df = clean_bp_outliers(test_df)
    X, y, X_test = prepare_features_target(train_df, test_df)
    
    # Train/validation split
    print("\nCreating train/validation split...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Valid: {len(X_valid)} samples")
    
    # Create preprocessor
    preprocessor = create_preprocessor(X_train)
    
    # Train models
    models = {}
    predictions = {}
    
    # Model 1: Random Forest (your best)
    rf_model = train_random_forest(X_train, y_train, preprocessor)
    models['rf'] = rf_model
    
    # Validate RF
    y_valid_proba = rf_model.predict_proba(X_valid)[:, 1]
    rf_auc = roc_auc_score(y_valid, y_valid_proba)
    print(f"  Validation AUC: {rf_auc:.4f}")
    
    # Predict on test
    predictions['rf'] = rf_model.predict_proba(X_test)[:, 1]
    
    # Model 2: Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train, preprocessor)
    models['gb'] = gb_model
    
    # Validate GB
    y_valid_proba_gb = gb_model.predict_proba(X_valid)[:, 1]
    gb_auc = roc_auc_score(y_valid, y_valid_proba_gb)
    print(f"  Validation AUC: {gb_auc:.4f}")
    
    # Predict on test
    predictions['gb'] = gb_model.predict_proba(X_test)[:, 1]
    
    # Model 3: Extra Trees
    try:
        et_model = train_extra_trees(X_train, y_train, preprocessor)
        models['et'] = et_model
        
        # Validate ET
        y_valid_proba_et = et_model.predict_proba(X_valid)[:, 1]
        et_auc = roc_auc_score(y_valid, y_valid_proba_et)
        print(f"  Validation AUC: {et_auc:.4f}")
        
        # Predict on test
        predictions['et'] = et_model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"  Skipping Extra Trees: {e}")
        et_auc = 0
    
    # Ensemble (weighted by validation AUC)
    print("\nCreating ensemble...")
    total_auc = rf_auc + gb_auc + et_auc
    if total_auc > 0:
        weights = {
            'rf': rf_auc / total_auc,
            'gb': gb_auc / total_auc,
            'et': et_auc / total_auc if et_auc > 0 else 0,
        }
        
        ensemble_pred = np.zeros(len(X_test))
        for model_name, weight in weights.items():
            if model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
        
        predictions['ensemble'] = ensemble_pred
        
        print("  Weights:")
        for name, weight in weights.items():
            if weight > 0:
                print(f"    {name}: {weight:.3f}")
    
    # Save submissions
    print("\n" + "="*70)
    print("SAVING SUBMISSIONS")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get icustay_id from test data
    id_col = [c for c in test_df.columns if c.lower() == 'icustay_id'][0]
    icustay_ids = test_df[id_col]
    
    saved_files = []
    for name, proba in predictions.items():
        submission = pd.DataFrame({
            'icustay_id': icustay_ids,
            'HOSPITAL_EXPIRE_FLAG': proba
        })
        
        filename = f"submission_{name}.csv"
        filepath = OUTPUT_DIR / filename
        submission.to_csv(filepath, index=False)
        
        print(f"\n{name.upper()}:")
        print(f"  File: {filepath}")
        print(f"  Stats: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}")
        
        saved_files.append(filepath)
    
    # Print recommendations
    print("\n" + "="*70)
    print("SUBMISSION STRATEGY")
    print("="*70)
    print("\nRecommended submission order (you have 20/day):")
    print("\n1. submission_rf.csv")
    print("   → Best validation AUC in your tests (0.8181)")
    print("\n2. submission_ensemble.csv")
    print("   → Weighted combination of all models")
    print("\n3. submission_gb.csv")
    print("   → Alternative strong model")
    
    if 'et' in predictions:
        print("\n4. submission_et.csv")
        print("   → Extra Trees for diversity")
    
    print("\nSubmit #1 first, wait for score, then decide on next steps!")
    print("="*70)
    
    return saved_files

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    try:
        submissions = generate_submissions()
        print("\n✓ SUCCESS! Submissions ready.")
        print(f"\nSaved {len(submissions)} submission files to: {OUTPUT_DIR}")
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nMake sure your data files are in the correct location:")
        print(f"  {DATA_DIR / TRAIN_FILE}")
        print(f"  {DATA_DIR / TEST_FILE}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()