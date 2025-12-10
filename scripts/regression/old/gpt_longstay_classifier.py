"""
Long-Stay Classifier for ICU LOS (Diagnostic Script)

Goal:
    - Predict whether a patient will have a "long stay" in ICU.
    - Use this as a diagnostic tool to see which features are useful
      for long-stay prediction and how predictable long stays are at all.

What it does:
    1. Uses prepare_data() to load leak-free LOS data.
    2. Creates a binary target: long_stay = (LOS > THRESHOLD_DAYS).
    3. Builds a preprocessing + GradientBoostingClassifier pipeline.
    4. Evaluates:
        - Prevalence of long stays
        - ROC AUC
        - PR AUC (more informative for imbalanced data)
        - Confusion matrix at a chosen decision threshold
    5. Computes permutation feature importances on the original columns
       to show which features matter most for long-stay prediction.

Run from project root:
    python scripts/regression/long_stay_classifier.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------- Path setup -------------------- #
SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "regression":
    BASE_DIR = SCRIPT_DIR.parents[1]
    sys.path.append(str(BASE_DIR / "src"))
    sys.path.append(str(SCRIPT_DIR))
else:
    BASE_DIR = Path.cwd()
    sys.path.append(str(BASE_DIR / "src"))

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

from los_prep import prepare_data

print("=" * 80)
print("LONG-STAY CLASSIFIER FOR ICU LOS (DIAGNOSTIC)")
print("=" * 80)

# -------------------- 1. Load data -------------------- #

# Only genuine leak columns – keep diagnosis if present
LEAK_COLS = [
    "ADMITTIME",
    "DOB",
    "DEATHTIME",
    "DISCHTIME",
    "DOD",
    "LOS",
    "HOSPITAL_EXPIRE_FLAG",
]

print("\n[1/5] Loading data (with feature engineering)...")
X, y, X_test = prepare_data(
    leak_cols=LEAK_COLS,
    apply_fe=True,   # use your usual FE setup
)

y = pd.Series(y)

print(f"  Train shape: {X.shape}")
print(f"  LOS range: [{y.min():.2f}, {y.max():.2f}] days")
print(f"  LOS mean:   {y.mean():.2f} days")
print(f"  LOS median: {y.median():.2f} days")

# -------------------- 2. Define long-stay label -------------------- #
# You can tweak this threshold; 7 or 14 are common choices.
THRESHOLD_DAYS = 7.0

print(f"\n[2/5] Creating long-stay label (LOS > {THRESHOLD_DAYS} days)...")
y_long = (y > THRESHOLD_DAYS).astype(int)

prevalence = y_long.mean()
n_long = int(y_long.sum())
n_total = len(y_long)

print(f"  Long-stay threshold: {THRESHOLD_DAYS} days")
print(f"  Long-stay patients: {n_long} / {n_total} ({prevalence * 100:.2f}%)")

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_long,
    test_size=0.2,
    random_state=42,
    stratify=y_long,   # preserve class balance
)

print(f"\n  X_train shape: {X_train.shape}")
print(f"  X_valid shape: {X_valid.shape}")

# -------------------- 3. Preprocessing + model pipeline -------------------- #
print("\n[3/5] Setting up preprocessing and classifier pipeline...")

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"  Numeric columns:    {len(num_cols)}")
print(f"  Categorical columns:{len(cat_cols)}")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            cat_cols,
        ),
    ]
)

clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ]
)

print("  Fitting pipeline...")
pipeline.fit(X_train, y_train)

# -------------------- 4. Evaluation -------------------- #
print("\n[4/5] Evaluating long-stay classifier...")

# Predicted probabilities for the positive class (long stay)
proba_valid = pipeline.predict_proba(X_valid)[:, 1]

# ROC AUC and PR AUC
roc_auc = roc_auc_score(y_valid, proba_valid)
pr_auc = average_precision_score(y_valid, proba_valid)

print(f"\n  ROC AUC: {roc_auc:.3f}")
print(f"  PR AUC:  {pr_auc:.3f} (more informative for imbalanced data)")

# Choose a decision threshold for classification diagnostics
DECISION_THRESHOLD = 0.5  # can tweak, or tune based on ROC/PR
y_valid_pred = (proba_valid >= DECISION_THRESHOLD).astype(int)

cm = confusion_matrix(y_valid, y_valid_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Decision threshold: {DECISION_THRESHOLD:.2f}")
print("  Confusion matrix (rows=true, cols=pred):")
print("      pred=0    pred=1")
print(f"  true=0   {cm[0,0]:7d} {cm[0,1]:7d}")
print(f"  true=1   {cm[1,0]:7d} {cm[1,1]:7d}")

print("\n  Classification report:")
print(classification_report(y_valid, y_valid_pred, digits=3))

# -------------------- 5. Permutation feature importance -------------------- #
print("\n[5/5] Computing permutation feature importance (this may take a bit)...")

# Permutation importance on original columns (pipeline handles preprocessing internally)
result = permutation_importance(
    pipeline,
    X_valid,
    y_valid,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",
)

importances = result.importances_mean
stds = result.importances_std

feature_names = X_valid.columns.to_list()

# Build a DataFrame for sorting
imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": importances,
    "importance_std": stds,
})

imp_df = imp_df.sort_values("importance_mean", ascending=False)

TOP_N = 20
print(f"\nTop {TOP_N} features by permutation importance (ROC AUC drop):")
for i, row in imp_df.head(TOP_N).iterrows():
    print(f"  {row['feature']:<30s}  importance={row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

print("\n" + "=" * 80)
print("DONE – Long-stay classifier diagnostics complete.")
print("=" * 80)
print("\nKey things to look at:")
print("  - ROC AUC / PR AUC: are long stays predictable at all?")
print("  - Confusion matrix: how many long stays are missed at 0.5 threshold?")
print("  - Top features: which variables matter most for long-stay prediction?")
print("=" * 80)
