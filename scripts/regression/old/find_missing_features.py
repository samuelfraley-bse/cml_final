"""
Find Missing Features and Engineering Opportunities

We need to close a 5-point gap. Let's systematically check:
1. All columns we have but aren't using
2. Feature interactions we haven't tried
3. Non-linear transformations
4. Ratio features
5. Anything else in the data

Run: python scripts/regression/find_missing_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("FINDING MISSING FEATURES")
print("=" * 70)

# Load data
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
train_df = pd.read_csv(DATA_DIR / "mimic_train_LOS.csv")

print(f"\nLoaded {len(train_df)} records with {len(train_df.columns)} columns")

los = train_df['LOS']

# =============================================================================
# SECTION 1: ALL COLUMNS AND THEIR USAGE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: ALL COLUMNS IN DATASET")
print("=" * 70)

print("\nAll columns:")
for i, col in enumerate(train_df.columns):
    dtype = train_df[col].dtype
    nulls = train_df[col].isna().sum()
    null_pct = nulls / len(train_df) * 100

    # Correlation with LOS if numerical
    if dtype in ['int64', 'float64']:
        corr = train_df[col].corr(los)
        corr_str = f"corr={corr:+.3f}" if not np.isnan(corr) else "corr=NaN"
    else:
        corr_str = f"categorical"

    print(f"  {i+1:2}. {col:25} {str(dtype):10} nulls={null_pct:5.1f}% {corr_str}")

# =============================================================================
# SECTION 2: UNUSED NUMERICAL FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: CORRELATION OF ALL NUMERICAL FEATURES WITH LOS")
print("=" * 70)

numerical_cols = train_df.select_dtypes(include=[np.number]).columns
correlations = []

for col in numerical_cols:
    if col not in ['icustay_id', 'subject_id', 'hadm_id', 'LOS']:
        corr = train_df[col].corr(los)
        if not np.isnan(corr):
            correlations.append((col, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nAll numerical correlations with LOS (sorted by |corr|):")
for col, corr in correlations:
    strength = "***" if abs(corr) > 0.15 else "**" if abs(corr) > 0.1 else "*" if abs(corr) > 0.05 else ""
    print(f"  {col:30} {corr:+.4f} {strength}")

# =============================================================================
# SECTION 3: FEATURE INTERACTIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: TESTING FEATURE INTERACTIONS")
print("=" * 70)

# Key features to try interactions with
key_features = ['SOFA', 'SAPSII', 'GCS', 'HeartRate_Mean', 'RespRate_Mean',
                'SysBP_Mean', 'TempC_Mean', 'Glucose_Mean']

print("\nTesting interactions (multiplication):")
interaction_results = []

for i, f1 in enumerate(key_features):
    for f2 in key_features[i+1:]:
        if f1 in train_df.columns and f2 in train_df.columns:
            interaction = train_df[f1] * train_df[f2]
            corr = interaction.corr(los)
            if not np.isnan(corr) and abs(corr) > 0.05:
                interaction_results.append((f"{f1} * {f2}", corr))

interaction_results.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nTop interactions (|corr| > 0.05):")
for name, corr in interaction_results[:15]:
    print(f"  {name:40} {corr:+.4f}")

# =============================================================================
# SECTION 4: RATIO FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: TESTING RATIO FEATURES")
print("=" * 70)

ratio_results = []

# Blood pressure ratios
if all(col in train_df.columns for col in ['SysBP_Mean', 'DiasBP_Mean']):
    pp = train_df['SysBP_Mean'] - train_df['DiasBP_Mean']  # Pulse pressure
    ratio_results.append(('Pulse Pressure (SysBP - DiasBP)', pp.corr(los)))

if all(col in train_df.columns for col in ['HeartRate_Mean', 'SysBP_Mean']):
    shock_idx = train_df['HeartRate_Mean'] / train_df['SysBP_Mean']  # Shock index
    ratio_results.append(('Shock Index (HR/SysBP)', shock_idx.corr(los)))

# Max/Min ratios (instability)
vital_pairs = [
    ('HeartRate', 'HR'), ('SysBP', 'SBP'), ('RespRate', 'RR'),
    ('TempC', 'Temp'), ('Glucose', 'Gluc')
]

for vital, abbrev in vital_pairs:
    max_col = f'{vital}_Max'
    min_col = f'{vital}_Min'
    mean_col = f'{vital}_Mean'

    if max_col in train_df.columns and min_col in train_df.columns:
        # Range
        range_feat = train_df[max_col] - train_df[min_col]
        ratio_results.append((f'{abbrev} Range (Max-Min)', range_feat.corr(los)))

        # Coefficient of variation proxy
        if mean_col in train_df.columns:
            cv = range_feat / (train_df[mean_col] + 0.001)
            ratio_results.append((f'{abbrev} CV (Range/Mean)', cv.corr(los)))

# SOFA/SAPSII interactions
if 'SOFA' in train_df.columns and 'SAPSII' in train_df.columns:
    ratio_results.append(('SOFA + SAPSII', (train_df['SOFA'] + train_df['SAPSII']).corr(los)))
    ratio_results.append(('SOFA * SAPSII', (train_df['SOFA'] * train_df['SAPSII']).corr(los)))
    ratio_results.append(('SOFA / (SAPSII+1)', (train_df['SOFA'] / (train_df['SAPSII'] + 1)).corr(los)))

# GCS interactions
if 'GCS' in train_df.columns:
    ratio_results.append(('15 - GCS (severity)', (15 - train_df['GCS']).corr(los)))
    if 'SOFA' in train_df.columns:
        ratio_results.append(('SOFA * (15-GCS)', (train_df['SOFA'] * (15 - train_df['GCS'])).corr(los)))

ratio_results.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)

print("\nRatio/Derived features:")
for name, corr in ratio_results:
    if not np.isnan(corr):
        strength = "**" if abs(corr) > 0.1 else "*" if abs(corr) > 0.05 else ""
        print(f"  {name:40} {corr:+.4f} {strength}")

# =============================================================================
# SECTION 5: NON-LINEAR TRANSFORMATIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: NON-LINEAR TRANSFORMATIONS")
print("=" * 70)

transform_results = []

# Try log, sqrt, square for top correlated features
top_features = ['RespRate_Max', 'TempC_Max', 'HeartRate_Max', 'SOFA', 'SAPSII']

for col in top_features:
    if col in train_df.columns:
        vals = train_df[col].dropna()
        vals_positive = vals[vals > 0]

        if len(vals_positive) > 100:
            # Original
            orig_corr = vals.corr(los[vals.index])

            # Log
            log_vals = np.log1p(vals_positive)
            log_corr = log_vals.corr(los[vals_positive.index])

            # Sqrt
            sqrt_vals = np.sqrt(vals_positive)
            sqrt_corr = sqrt_vals.corr(los[vals_positive.index])

            # Square
            sq_vals = vals ** 2
            sq_corr = sq_vals.corr(los[vals.index])

            transform_results.append((col, orig_corr, log_corr, sqrt_corr, sq_corr))

print("\nNon-linear transformations:")
print(f"{'Feature':20} {'Original':>10} {'Log':>10} {'Sqrt':>10} {'Square':>10}")
print("-" * 65)
for col, orig, log, sqrt, sq in transform_results:
    best = max([abs(orig), abs(log), abs(sqrt), abs(sq)])
    best_name = 'orig' if abs(orig) == best else 'log' if abs(log) == best else 'sqrt' if abs(sqrt) == best else 'sq'
    print(f"{col:20} {orig:+10.4f} {log:+10.4f} {sqrt:+10.4f} {sq:+10.4f}  best={best_name}")

# =============================================================================
# SECTION 6: CATEGORICAL ENCODING OPPORTUNITIES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: CATEGORICAL TARGET ENCODING")
print("=" * 70)

categorical_cols = ['GENDER', 'ADMISSION_TYPE', 'INSURANCE', 'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY', 'FIRST_CAREUNIT']

print("\nTarget encoding (mean LOS per category):")
for col in categorical_cols:
    if col in train_df.columns:
        # Calculate mean LOS per category
        target_enc = train_df.groupby(col)['LOS'].mean()
        encoded = train_df[col].map(target_enc)
        corr = encoded.corr(los)

        # Range of means
        range_los = target_enc.max() - target_enc.min()

        print(f"\n{col}:")
        print(f"  Target encoded correlation: {corr:+.4f}")
        print(f"  Range of category means: {range_los:.2f} days")

        # Show categories
        for cat, mean_los in target_enc.sort_values(ascending=False).head(5).items():
            count = (train_df[col] == cat).sum()
            print(f"    {str(cat)[:30]:30} avg_LOS={mean_los:.2f} (n={count})")

# =============================================================================
# SECTION 7: POLYNOMIAL FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: POLYNOMIAL COMBINATIONS")
print("=" * 70)

# Test if combining top features polynomially helps
top_3 = ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']

if all(col in train_df.columns for col in top_3):
    # Sum
    sum_feat = train_df[top_3[0]] + train_df[top_3[1]] + train_df[top_3[2]]
    print(f"\nSum of top 3: corr = {sum_feat.corr(los):+.4f}")

    # Product
    prod_feat = train_df[top_3[0]] * train_df[top_3[1]] * train_df[top_3[2]]
    print(f"Product of top 3: corr = {prod_feat.corr(los):+.4f}")

    # Mean
    mean_feat = (train_df[top_3[0]] + train_df[top_3[1]] + train_df[top_3[2]]) / 3
    print(f"Mean of top 3: corr = {mean_feat.corr(los):+.4f}")

# =============================================================================
# SECTION 8: BINNING/DISCRETIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: BINNING KEY FEATURES")
print("=" * 70)

print("\nTrying to bin continuous features:")

for col in ['SOFA', 'SAPSII', 'RespRate_Mean', 'HeartRate_Mean']:
    if col in train_df.columns:
        # Try different number of bins
        for n_bins in [3, 5, 10]:
            binned = pd.qcut(train_df[col], q=n_bins, duplicates='drop', labels=False)
            corr = binned.corr(los)
            print(f"  {col:20} {n_bins} bins: corr = {corr:+.4f}")

# =============================================================================
# SECTION 9: RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("""
Based on this analysis, here are potential improvements:

1. HIGHEST POTENTIAL NEW FEATURES:
   - RespRate_Max has highest correlation - make sure we're using it
   - TempC_Max correlation 0.13 - are we using this?
   - Vital sign ranges (Max - Min) show instability

2. RATIO FEATURES TO ADD:
   - Shock Index (HR/SysBP)
   - Pulse Pressure
   - CV proxies (Range/Mean)

3. SEVERITY COMBINATIONS:
   - SOFA + SAPSII combined score
   - SOFA * (15-GCS) - combines multiple severity measures

4. TARGET ENCODING:
   - Encode FIRST_CAREUNIT by mean LOS
   - Encode ETHNICITY by mean LOS

5. CHECK IF WE'RE USING:
   - All _Max and _Min columns
   - HOSPITAL_EXPIRE_FLAG (corr 0.11 with long-stay!)

6. POSSIBLE ISSUES:
   - Are we accidentally dropping important columns?
   - Is preprocessing removing signal?
""")

print("=" * 70)
