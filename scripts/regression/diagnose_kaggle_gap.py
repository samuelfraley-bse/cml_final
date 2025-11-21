"""
Diagnose the Validation vs Kaggle RMSE Gap

Validation RMSE: ~4.7
Kaggle RMSE: ~20.8

Why is there such a huge gap? Let's investigate:
1. Train/Test distribution differences
2. Prediction range issues
3. Feature coverage in test set
4. Potential scaling/transformation issues

Run: python scripts/regression/diagnose_kaggle_gap.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

print("=" * 70)
print("DIAGNOSING VALIDATION vs KAGGLE RMSE GAP")
print("=" * 70)

# Load data
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
train_df = pd.read_csv(DATA_DIR / "mimic_train_LOS.csv")
test_df = pd.read_csv(DATA_DIR / "mimic_test_LOS.csv")

print(f"\nTrain: {len(train_df)} rows")
print(f"Test: {len(test_df)} rows")

# =============================================================================
# SECTION 1: TARGET DISTRIBUTION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: TARGET (LOS) DISTRIBUTION")
print("=" * 70)

los = train_df['LOS']
print(f"\nTraining LOS statistics:")
print(f"  Mean:   {los.mean():.2f}")
print(f"  Median: {los.median():.2f}")
print(f"  Std:    {los.std():.2f}")
print(f"  Min:    {los.min():.2f}")
print(f"  Max:    {los.max():.2f}")

print(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(los, p):.2f}")

print(f"\nExtreme values:")
print(f"  LOS > 10 days: {(los > 10).sum()} ({(los > 10).mean()*100:.1f}%)")
print(f"  LOS > 20 days: {(los > 20).sum()} ({(los > 20).mean()*100:.1f}%)")
print(f"  LOS > 30 days: {(los > 30).sum()} ({(los > 30).mean()*100:.1f}%)")

# =============================================================================
# SECTION 2: CHECK SUBMISSION FILES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: SUBMISSION FILE ANALYSIS")
print("=" * 70)

submission_dir = BASE_DIR / "submissions" / "regression"
submission_files = list(submission_dir.glob("*.csv"))

print(f"\nFound {len(submission_files)} submission files")

# Check a few key submissions
key_submissions = [
    "submission_los_ensemble_weighted.csv",
    "submission_los_gb_single.csv",
    "submission_los_gb_enhanced_v2.csv",
]

for fname in key_submissions:
    fpath = submission_dir / fname
    if fpath.exists():
        sub_df = pd.read_csv(fpath)
        preds = sub_df['LOS']
        print(f"\n{fname}:")
        print(f"  Shape: {sub_df.shape}")
        print(f"  Columns: {list(sub_df.columns)}")
        print(f"  Predictions - min: {preds.min():.2f}, max: {preds.max():.2f}, mean: {preds.mean():.2f}")

        # Check for issues
        if preds.min() < 0:
            print(f"  ⚠ WARNING: Negative predictions!")
        if preds.max() < 10:
            print(f"  ⚠ WARNING: Max prediction very low compared to actual LOS range!")

# =============================================================================
# SECTION 3: TRAIN VS TEST DISTRIBUTION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: TRAIN vs TEST FEATURE DISTRIBUTIONS")
print("=" * 70)

# Check key numerical features
numerical_cols = ['HeartRate_Mean', 'SysBP_Mean', 'DiasBP_Mean', 'TempC_Mean',
                  'RespRate_Mean', 'Glucose_Mean', 'GCS', 'SOFA', 'SAPSII']

print("\nNumerical feature comparison (mean ± std):")
print(f"{'Feature':20} {'Train':>20} {'Test':>20} {'Diff%':>10}")
print("-" * 72)

for col in numerical_cols:
    if col in train_df.columns and col in test_df.columns:
        train_mean = train_df[col].mean()
        train_std = train_df[col].std()
        test_mean = test_df[col].mean()
        test_std = test_df[col].std()

        diff_pct = abs(train_mean - test_mean) / train_mean * 100 if train_mean != 0 else 0

        train_str = f"{train_mean:.1f} ± {train_std:.1f}"
        test_str = f"{test_mean:.1f} ± {test_std:.1f}"

        warning = "⚠" if diff_pct > 10 else ""
        print(f"{col:20} {train_str:>20} {test_str:>20} {diff_pct:>8.1f}% {warning}")

# Check categorical distributions
print("\n\nCategorical feature comparison:")
categorical_cols = ['GENDER', 'ADMISSION_TYPE', 'INSURANCE', 'FIRST_CAREUNIT']

for col in categorical_cols:
    if col in train_df.columns and col in test_df.columns:
        print(f"\n{col}:")
        train_dist = train_df[col].value_counts(normalize=True)
        test_dist = test_df[col].value_counts(normalize=True)

        all_cats = set(train_dist.index) | set(test_dist.index)

        for cat in sorted(all_cats):
            train_pct = train_dist.get(cat, 0) * 100
            test_pct = test_dist.get(cat, 0) * 100
            diff = abs(train_pct - test_pct)
            warning = "⚠" if diff > 5 else ""
            print(f"  {str(cat):15} train: {train_pct:5.1f}%  test: {test_pct:5.1f}%  {warning}")

# =============================================================================
# SECTION 4: ICD9 DISTRIBUTION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: ICD9 DIAGNOSIS DISTRIBUTION")
print("=" * 70)

# Check if high-LOS ICD9 codes appear in test
high_los_icd9 = ['51884', '5770', '430', '44101', '0380', '51881', '0389', '431']

print("\nHigh-LOS ICD9 codes in train vs test:")
for code in high_los_icd9:
    train_count = (train_df['ICD9_diagnosis'] == code).sum()
    test_count = (test_df['ICD9_diagnosis'] == code).sum()
    train_pct = train_count / len(train_df) * 100
    test_pct = test_count / len(test_df) * 100
    print(f"  {code}: train {train_count} ({train_pct:.2f}%)  test {test_count} ({test_pct:.2f}%)")

# =============================================================================
# SECTION 5: EXTREME VALUE ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: EXTREME VALUES & OUTLIERS")
print("=" * 70)

# What if Kaggle test has many extreme LOS values?
print("\nIf test set has more extreme LOS values, our model would underpredict badly")
print("Our models predict max ~17-19 days, but training LOS goes up to", los.max(), "days")

# Analyze model's ability to predict extremes
print("\nLet's check our validation holdout to see how we handle extremes:")

# Quick split to check
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Simple model on just numerical features
num_cols = [c for c in train_df.columns if train_df[c].dtype in ['int64', 'float64']]
num_cols = [c for c in num_cols if c not in ['icustay_id', 'subject_id', 'hadm_id', 'LOS', 'Diff']]

X = train_df[num_cols].fillna(0)
y = train_df['LOS']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple model
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print(f"\nValidation predictions vs actual:")
print(f"  Actual range: [{y_val.min():.2f}, {y_val.max():.2f}]")
print(f"  Predicted range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

# Error by LOS bucket
print("\nRMSE by actual LOS bucket:")
for low, high in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100)]:
    mask = (y_val >= low) & (y_val < high)
    if mask.sum() > 10:
        rmse = np.sqrt(np.mean((y_val[mask] - y_pred[mask])**2))
        count = mask.sum()
        pct = mask.sum() / len(y_val) * 100
        print(f"  LOS {low:3}-{high:3}: RMSE={rmse:6.2f} (n={count:4}, {pct:5.1f}%)")

# =============================================================================
# SECTION 6: POSSIBLE ROOT CAUSES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: POSSIBLE ROOT CAUSES")
print("=" * 70)

print("""
Hypotheses for the ~4.7 vs ~20.8 RMSE gap:

1. EXTREME VALUES
   - Training LOS goes up to {:.0f} days
   - Our models cap predictions around 17-19 days
   - If test has patients with LOS > 20 days, we'd have huge errors
   - Example: predicting 15 for a 50-day stay = error of 35!

2. TRAIN/TEST DISTRIBUTION SHIFT
   - Test set might have different patient mix
   - Different severity distribution
   - Different diagnoses distribution

3. OVERFITTING TO VALIDATION SPLIT
   - Our random train/val split might not match Kaggle's test distribution
   - Kaggle might have stratified differently

4. METRIC INTERPRETATION
   - Are we using the same metric as Kaggle?
   - RMSE vs MSE vs other?

5. MISSING IMPORTANT SIGNAL
   - Maybe there's a critical feature we're not using well
   - Or we're losing information in preprocessing
""".format(los.max()))

# =============================================================================
# SECTION 7: RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("""
To investigate further:

1. CHECK KAGGLE METRIC
   - Confirm it's RMSE, not MSE or another metric
   - sqrt(MSE) ≈ 4.5 would give MSE ≈ 20.8!

2. TRY LOG-TRANSFORMED TARGET
   - Train on log(LOS), predict, then exp()
   - Handles extreme values better

3. CLIP PREDICTIONS
   - Don't let predictions go negative
   - Maybe cap at reasonable max (e.g., 30 days)

4. QUANTILE REGRESSION
   - Predict median instead of mean
   - Less sensitive to outliers

5. TWO-STAGE MODEL
   - First predict if LOS > X days
   - Then predict actual LOS

Would you like me to create a script to try any of these approaches?
""")

print("\n" + "=" * 70)
