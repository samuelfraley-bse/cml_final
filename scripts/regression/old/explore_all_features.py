"""
Explore ALL Features for LOS Prediction
Check categorical variables, find hidden patterns, and identify any missing signals.

Run: python scripts/regression/explore_all_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("COMPREHENSIVE FEATURE EXPLORATION")
print("=" * 70)

# Load data
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
train_df = pd.read_csv(DATA_DIR / "mimic_train_LOS.csv")

print(f"\nLoaded {len(train_df)} records")
los = train_df['LOS']

# =============================================================================
# SECTION 1: ALL CATEGORICAL VARIABLES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: CATEGORICAL VARIABLES - LOS BY CATEGORY")
print("=" * 70)

categorical_cols = ['GENDER', 'ADMISSION_TYPE', 'INSURANCE', 'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY', 'FIRST_CAREUNIT']

for col in categorical_cols:
    print(f"\n{col}:")
    print("-" * 50)

    # Get value counts and LOS stats
    results = []
    for val in train_df[col].dropna().unique():
        mask = train_df[col] == val
        count = mask.sum()
        avg_los = train_df.loc[mask, 'LOS'].mean()
        std_los = train_df.loc[mask, 'LOS'].std()
        results.append((val, count, avg_los, std_los))

    # Sort by avg LOS
    results.sort(key=lambda x: x[2], reverse=True)

    # Calculate range
    if results:
        max_los = max(r[2] for r in results)
        min_los = min(r[2] for r in results)
        range_los = max_los - min_los

        print(f"  LOS range across categories: {range_los:.2f} days")
        print(f"  (from {min_los:.2f} to {max_los:.2f})")
        print()

        # Show all if < 10, otherwise top/bottom
        if len(results) <= 10:
            for val, count, avg_los, std_los in results:
                val_str = str(val)[:35]
                print(f"  {val_str:35} n={count:5}, avg_LOS={avg_los:.2f} ±{std_los:.2f}")
        else:
            print(f"  Top 5 (longest LOS):")
            for val, count, avg_los, std_los in results[:5]:
                val_str = str(val)[:35]
                print(f"    {val_str:35} n={count:5}, avg_LOS={avg_los:.2f}")
            print(f"  Bottom 5 (shortest LOS):")
            for val, count, avg_los, std_los in results[-5:]:
                val_str = str(val)[:35]
                print(f"    {val_str:35} n={count:5}, avg_LOS={avg_los:.2f}")

# =============================================================================
# SECTION 2: NUMERICAL FEATURES - CORRELATIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: ALL NUMERICAL CORRELATIONS WITH LOS")
print("=" * 70)

numerical_cols = train_df.select_dtypes(include=[np.number]).columns
correlations = []

for col in numerical_cols:
    if col != 'LOS' and col not in ['icustay_id', 'subject_id', 'hadm_id']:
        corr = train_df[col].corr(los)
        if not np.isnan(corr):
            correlations.append((col, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nAll numerical correlations (sorted by |correlation|):")
for col, corr in correlations:
    direction = "+" if corr > 0 else "-"
    strength = ""
    if abs(corr) > 0.1:
        strength = "**"
    elif abs(corr) > 0.05:
        strength = "*"
    print(f"  {direction} {col:30} {corr:+.4f} {strength}")

# =============================================================================
# SECTION 3: WHICH CATEGORICAL HAS MOST PREDICTIVE POWER?
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: CATEGORICAL PREDICTIVE POWER (variance in mean LOS)")
print("=" * 70)

print("\nMeasuring how much each categorical variable explains LOS variance:")
print("(Higher = more predictive potential)")

cat_power = []
for col in categorical_cols:
    # Calculate between-group variance vs total variance
    group_means = train_df.groupby(col)['LOS'].mean()
    group_sizes = train_df.groupby(col)['LOS'].count()

    # Weighted variance of group means
    grand_mean = los.mean()
    between_var = ((group_means - grand_mean) ** 2 * group_sizes).sum() / len(train_df)
    total_var = los.var()

    # Eta-squared (proportion of variance explained)
    eta_squared = between_var / total_var if total_var > 0 else 0

    # Range of means
    range_los = group_means.max() - group_means.min()

    cat_power.append((col, eta_squared, range_los, len(group_means)))

# Sort by eta-squared
cat_power.sort(key=lambda x: x[1], reverse=True)

print("\nRanked by predictive power (eta-squared):")
for col, eta_sq, range_los, n_groups in cat_power:
    print(f"  {col:20} η²={eta_sq:.4f}  range={range_los:.2f} days  ({n_groups} groups)")

# =============================================================================
# SECTION 4: ICD9 DIAGNOSIS DEEP DIVE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: ICD9 DIAGNOSIS - FINDING HIGH-LOS CODES")
print("=" * 70)

# Get ICD9 codes with enough samples
icd9_stats = train_df.groupby('ICD9_diagnosis').agg({
    'LOS': ['mean', 'std', 'count']
}).reset_index()
icd9_stats.columns = ['ICD9', 'avg_los', 'std_los', 'count']

# Filter for codes with at least 50 patients
icd9_common = icd9_stats[icd9_stats['count'] >= 50].copy()
icd9_common = icd9_common.sort_values('avg_los', ascending=False)

print(f"\nICD9 codes with n≥50 patients: {len(icd9_common)}")

print("\nTop 15 ICD9 codes with LONGEST LOS:")
for _, row in icd9_common.head(15).iterrows():
    print(f"  {row['ICD9']:10} avg_LOS={row['avg_los']:.2f} ±{row['std_los']:.2f} (n={int(row['count'])})")

print("\nTop 10 ICD9 codes with SHORTEST LOS:")
for _, row in icd9_common.tail(10).iterrows():
    print(f"  {row['ICD9']:10} avg_LOS={row['avg_los']:.2f} ±{row['std_los']:.2f} (n={int(row['count'])})")

# Check ICD9 category (first 3 digits)
print("\n" + "-" * 50)
print("ICD9 CATEGORIES (first 3 digits):")

train_df['icd9_cat'] = train_df['ICD9_diagnosis'].astype(str).str[:3]
cat_stats = train_df.groupby('icd9_cat').agg({
    'LOS': ['mean', 'count']
}).reset_index()
cat_stats.columns = ['category', 'avg_los', 'count']
cat_stats = cat_stats[cat_stats['count'] >= 100].sort_values('avg_los', ascending=False)

print(f"\nCategories with n≥100 patients: {len(cat_stats)}")
print("\nTop 10 categories with longest LOS:")
for _, row in cat_stats.head(10).iterrows():
    print(f"  {row['category']:5} avg_LOS={row['avg_los']:.2f} (n={int(row['count'])})")

# =============================================================================
# SECTION 5: DIAGNOSIS TEXT PATTERNS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: DIAGNOSIS TEXT - KEYWORD ANALYSIS")
print("=" * 70)

# Keywords to check
keywords = [
    'SEPSIS', 'PNEUMONIA', 'HEMORRHAGE', 'STROKE', 'CARDIAC', 'HEART',
    'RESPIRATORY', 'FAILURE', 'CANCER', 'TRANSPLANT', 'SURGERY',
    'TRAUMA', 'FRACTURE', 'OVERDOSE', 'ARREST', 'SHOCK', 'INFECTION',
    'PANCREATITIS', 'LIVER', 'KIDNEY', 'DIALYSIS', 'VENTILATOR',
    'INTUBATION', 'BLEED', 'GI ', 'COPD', 'CHF', 'MI ', 'CAD',
    'DKA', 'DIABETIC', 'ALTERED', 'SEIZURE', 'ENCEPHALOPATHY'
]

diag_upper = train_df['DIAGNOSIS'].fillna('').str.upper()

keyword_results = []
for kw in keywords:
    mask = diag_upper.str.contains(kw)
    count = mask.sum()
    if count >= 30:  # Need enough samples
        avg_los = train_df.loc[mask, 'LOS'].mean()
        keyword_results.append((kw, count, avg_los))

# Sort by avg LOS
keyword_results.sort(key=lambda x: x[2], reverse=True)

print("\nKeywords in DIAGNOSIS text (n≥30):")
print("Sorted by average LOS:")
for kw, count, avg_los in keyword_results:
    diff = avg_los - los.mean()
    direction = "↑" if diff > 0.3 else "↓" if diff < -0.3 else " "
    print(f"  {direction} {kw:20} n={count:5}, avg_LOS={avg_los:.2f} ({diff:+.2f} vs mean)")

# =============================================================================
# SECTION 6: INTERACTION ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: POTENTIAL INTERACTIONS")
print("=" * 70)

print("\nFirst Care Unit × Admission Type:")
for unit in ['MICU', 'SICU', 'CCU', 'CSRU', 'TSICU']:
    for atype in ['EMERGENCY', 'ELECTIVE']:
        mask = (train_df['FIRST_CAREUNIT'] == unit) & (train_df['ADMISSION_TYPE'] == atype)
        count = mask.sum()
        if count >= 50:
            avg_los = train_df.loc[mask, 'LOS'].mean()
            print(f"  {unit:6} + {atype:10} n={count:5}, avg_LOS={avg_los:.2f}")

# =============================================================================
# SECTION 7: SUMMARY & RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

print("\n1. MOST PREDICTIVE CATEGORICAL VARIABLES:")
for col, eta_sq, range_los, _ in cat_power[:3]:
    print(f"   - {col}: η²={eta_sq:.4f}, range={range_los:.2f} days")

print("\n2. TOP NUMERICAL CORRELATIONS:")
for col, corr in correlations[:5]:
    print(f"   - {col}: r={corr:+.4f}")

print("\n3. HIGH-VALUE ICD9 CODES TO FLAG:")
high_los_codes = icd9_common[icd9_common['avg_los'] > 5]['ICD9'].tolist()[:10]
print(f"   Codes with avg LOS > 5 days: {high_los_codes}")

print("\n4. VALUABLE DIAGNOSIS KEYWORDS:")
high_los_kw = [kw for kw, count, avg_los in keyword_results if avg_los > 4.5][:10]
print(f"   Keywords with avg LOS > 4.5 days: {high_los_kw}")

print("\n5. FEATURES CURRENTLY NOT USED WELL:")
print("   - FIRST_CAREUNIT: Different ICU types have different LOS patterns")
print("   - ICD9 codes: Could add more specific code flags")
print("   - ETHNICITY: Some groups have notably different LOS")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
