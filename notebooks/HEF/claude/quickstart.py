"""
Quick Start Script - Get Kaggle Submission ASAP
Run this to quickly generate competitive predictions
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
BASE_DIR = Path.cwd().parents[1] if 'notebooks' in str(Path.cwd()) else Path.cwd()
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


def quick_ensemble_prediction():
    """
    Fastest path to a good Kaggle submission:
    1. Load data (no FE - it hurt performance)
    2. Simple preprocessing
    3. Train 3 diverse models
    4. Soft voting ensemble
    5. Generate submission
    """
    
    print("="*60)
    print("QUICK ENSEMBLE PREDICTOR")
    print("="*60)
    
    # 1. Load data without FE
    print("\n[1/6] Loading data...")
    from hef_prep import prepare_data
    
    X, y, X_test = prepare_data(
        task="class",
        leak_cols=[
            "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS", 
            "DOB", "DEATHTIME", "DISCHTIME", "DOD", 
            "LOS", "HOSPITAL_EXPIRE_FLAG"
        ],
        apply_fe=False,  # Raw features performed better
    )
    
    # 2. Train/valid split
    print("[2/6] Creating train/validation split...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Setup preprocessing
    print("[3/6] Setting up preprocessing...")
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
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
    
    # 4. Create ensemble
    print("[4/6] Training ensemble (this may take a few minutes)...")
    
    # Model 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    
    # Model 2: Gradient Boosting
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
    
    # Model 3: Random Forest with different params
    rf2 = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=123,  # different seed for diversity
    )
    
    # Voting ensemble
    voting = VotingClassifier(
        estimators=[
            ('rf1', rf),
            ('gb', gb),
            ('rf2', rf2),
        ],
        voting='soft',
        weights=[2, 2, 1],  # RF gets more weight
        n_jobs=-1,
    )
    
    # Full pipeline
    ensemble_pipe = Pipeline([
        ('preprocess', preprocessor),
        ('ensemble', voting),
    ])
    
    # Fit
    print("   - Fitting Random Forest 1...")
    print("   - Fitting Gradient Boosting...")
    print("   - Fitting Random Forest 2...")
    print("   - Training voting ensemble...")
    ensemble_pipe.fit(X_train, y_train)
    
    # 5. Evaluate
    print("[5/6] Evaluating...")
    y_train_proba = ensemble_pipe.predict_proba(X_train)[:, 1]
    y_valid_proba = ensemble_pipe.predict_proba(X_valid)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_proba)
    valid_auc = roc_auc_score(y_valid, y_valid_proba)
    
    print(f"\n   Train AUC: {train_auc:.4f}")
    print(f"   Valid AUC: {valid_auc:.4f}")
    print(f"   Overfit: {train_auc - valid_auc:.4f}")
    
    # 6. Generate submission
    print("[6/6] Generating Kaggle submission...")
    test_proba = ensemble_pipe.predict_proba(X_test)[:, 1]
    
    submission = pd.DataFrame({
        'HOSPITAL_EXPIRE_FLAG': test_proba
    })
    
    # Save
    output_path = BASE_DIR / "data" / "submissions" / "quick_ensemble_submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ Submission saved to: {output_path}")
    print(f"✓ Rows: {len(submission)}")
    print(f"\nPrediction statistics:")
    print(f"   Min:    {test_proba.min():.4f}")
    print(f"   Max:    {test_proba.max():.4f}")
    print(f"   Mean:   {test_proba.mean():.4f}")
    print(f"   Median: {np.median(test_proba):.4f}")
    
    print("\n" + "="*60)
    print("DONE! Upload to Kaggle:")
    print(f"  {output_path}")
    print("="*60)
    
    return ensemble_pipe, submission


def quick_xgboost_prediction():
    """
    Alternative: Use XGBoost (often better than RF on tabular data)
    Requires: pip install xgboost
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Run: pip install xgboost")
        return None, None
    
    print("\n" + "="*60)
    print("QUICK XGBOOST PREDICTOR")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data...")
    from hef_prep import prepare_data
    
    X, y, X_test = prepare_data(
        task="class",
        leak_cols=[
            "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS", 
            "DOB", "DEATHTIME", "DISCHTIME", "DOD", 
            "LOS", "HOSPITAL_EXPIRE_FLAG"
        ],
        apply_fe=False,
    )
    
    # Split
    print("[2/5] Creating split...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Preprocessing
    print("[3/5] Preprocessing...")
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
    ])
    
    # XGBoost model
    print("[4/5] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=7.9,  # 88.8 / 11.2 for class balance
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    
    xgb_pipe = Pipeline([
        ('preprocess', preprocessor),
        ('xgb', xgb_model),
    ])
    
    xgb_pipe.fit(X_train, y_train)
    
    # Evaluate
    y_train_proba = xgb_pipe.predict_proba(X_train)[:, 1]
    y_valid_proba = xgb_pipe.predict_proba(X_valid)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_proba)
    valid_auc = roc_auc_score(y_valid, y_valid_proba)
    
    print(f"\n   Train AUC: {train_auc:.4f}")
    print(f"   Valid AUC: {valid_auc:.4f}")
    
    # Generate submission
    print("[5/5] Generating submission...")
    test_proba = xgb_pipe.predict_proba(X_test)[:, 1]
    
    submission = pd.DataFrame({
        'HOSPITAL_EXPIRE_FLAG': test_proba
    })
    
    output_path = BASE_DIR / "data" / "submissions" / "xgboost_submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ XGBoost submission saved to: {output_path}")
    print("="*60)
    
    return xgb_pipe, submission


if __name__ == "__main__":
    print("Starting quick prediction script...\n")
    
    # Run ensemble
    ensemble_model, ensemble_submission = quick_ensemble_prediction()
    
    # Optionally run XGBoost
    print("\n\nWould you like to also try XGBoost? (y/n)")
    print("(Skip for now if XGBoost not installed)")
    
    # Auto-run XGBoost if available
    try:
        import xgboost
        print("\nXGBoost detected, running XGBoost predictor...\n")
        xgb_model, xgb_submission = quick_xgboost_prediction()
    except ImportError:
        print("\nXGBoost not installed, skipping.")
        print("To install: pip install xgboost")
    
    print("\n✓✓✓ All done! Check your submissions folder. ✓✓✓")