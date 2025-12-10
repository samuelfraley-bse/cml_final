"""
Investigate Long-Stay Patients

Why is it so hard to predict long stays (>10 days)?
- Are there features that distinguish them?
- Or is there high variance even within similar feature profiles?

Run: python scripts/regression/investigate_long_stay.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("=" * 70)
print("INVESTIGATING LONG-STAY PATIENTS")
print("=" * 70)

# Load data
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
train_df = pd.read_csv(DATA_DIR / "mimic_train_LOS.csv")

print(f"\nLoaded {len(train_df)} records")

# Define threshold
THRESHOLD = 10
los = train_df['LOS']
is_long = los > THRESHOLD

print(f"\nThreshold: {THRESHOLD} days")
print(f"Long stay (>{THRESHOLD} days): {is_long.sum()} ({is_long.mean()*100:.1f}%)")
print(f"Short stay (≤{THRESHOLD} days): {(~is_long).sum()} ({(~is_long).mean()*100:.1f}%)")

# =============================================================================
# SECTION 1: BASIC STATS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOS DISTRIBUTION")
print("=" * 70)

print("\nShort-stay patients (≤10 days):")
short_los = los[~is_long]
print(f"  Mean: {short_los.mean():.2f}, Median: {short_los.median():.2f}")
print(f"  Std: {short_los.std():.2f}")
print(f"  Range: [{short_los.min():.2f}, {short_los.max():.2f}]")

print("\nLong-stay patients (>10 days):")
long_los = los[is_long]
print(f"  Mean: {long_los.mean():.2f}, Median: {long_los.median():.2f}")
print(f"  Std: {long_los.std():.2f}")
print(f"  Range: [{long_los.min():.2f}, {long_los.max():.2f}]")

# =============================================================================
# SECTION 2: NUMERICAL FEATURES - DO THEY DIFFER?
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: NUMERICAL FEATURES - SHORT vs LONG STAY")
print("=" * 70)

# Key numerical features
numerical_cols = [
    'HeartRate_Mean', 'SysBP_Mean', 'DiasBP_Mean', 'MeanBP_Mean',
    'RespRate_Mean', 'TempC_Mean', 'SpO2_Mean', 'Glucose_Mean',
    'GCS', 'SOFA', 'SAPSII',
    'HeartRate_Max', 'HeartRate_Min',
    'SysBP_Max', 'SysBP_Min',
    'RespRate_Max', 'Glucose_Max'
]

print("\nComparing means (short vs long stay):")
print(f"{'Feature':20} {'Short':>10} {'Long':>10} {'Diff':>10} {'p-value':>10} {'Sig':>5}")
print("-" * 70)

significant_features = []

for col in numerical_cols:
    if col in train_df.columns:
        short_vals = train_df.loc[~is_long, col].dropna()
        long_vals = train_df.loc[is_long, col].dropna()

        if len(short_vals) > 0 and len(long_vals) > 0:
            short_mean = short_vals.mean()
            long_mean = long_vals.mean()
            diff = long_mean - short_mean
            diff_pct = diff / short_mean * 100 if short_mean != 0 else 0

            # T-test
            t_stat, p_value = stats.ttest_ind(short_vals, long_vals)

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

            if p_value < 0.05:
                significant_features.append((col, diff_pct, p_value))

            print(f"{col:20} {short_mean:10.2f} {long_mean:10.2f} {diff_pct:+9.1f}% {p_value:10.4f} {sig:>5}")

print(f"\nSignificant features (p<0.05): {len(significant_features)}")

# =============================================================================
# SECTION 3: CATEGORICAL FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: CATEGORICAL FEATURES - SHORT vs LONG STAY")
print("=" * 70)

categorical_cols = ['GENDER', 'ADMISSION_TYPE', 'INSURANCE', 'FIRST_CAREUNIT']

for col in categorical_cols:
    print(f"\n{col}:")
    print(f"{'Category':20} {'Short%':>10} {'Long%':>10} {'Long/Short':>12}")
    print("-" * 55)

    short_dist = train_df.loc[~is_long, col].value_counts(normalize=True)
    long_dist = train_df.loc[is_long, col].value_counts(normalize=True)

    all_cats = set(short_dist.index) | set(long_dist.index)

    for cat in sorted(all_cats):
        short_pct = short_dist.get(cat, 0) * 100
        long_pct = long_dist.get(cat, 0) * 100
        ratio = long_pct / short_pct if short_pct > 0 else 0

        flag = "↑" if ratio > 1.3 else "↓" if ratio < 0.7 else ""
        print(f"{str(cat):20} {short_pct:9.1f}% {long_pct:9.1f}% {ratio:11.2f}x {flag}")

# =============================================================================
# SECTION 4: DIAGNOSIS - THE KEY?
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: ICD9 DIAGNOSIS - SHORT vs LONG STAY")
print("=" * 70)

# Get ICD9 distribution for long-stay patients
icd9_long = train_df.loc[is_long, 'ICD9_diagnosis'].value_counts()
icd9_short = train_df.loc[~is_long, 'ICD9_diagnosis'].value_counts()

print("\nTop 15 ICD9 codes for LONG-STAY patients:")
for code, count in icd9_long.head(15).items():
    total = (train_df['ICD9_diagnosis'] == code).sum()
    long_rate = count / total * 100 if total > 0 else 0
    avg_los = train_df[train_df['ICD9_diagnosis'] == code]['LOS'].mean()
    print(f"  {code:10} n={count:4} ({long_rate:5.1f}% are long-stay), avg_LOS={avg_los:.1f}")

# Find codes with highest proportion of long stays
print("\nICD9 codes with HIGHEST % long-stay (n≥30):")
icd9_analysis = []
for code in train_df['ICD9_diagnosis'].unique():
    mask = train_df['ICD9_diagnosis'] == code
    total = mask.sum()
    if total >= 30:
        long_count = (mask & is_long).sum()
        long_rate = long_count / total
        avg_los = train_df.loc[mask, 'LOS'].mean()
        icd9_analysis.append((code, total, long_rate, avg_los))

icd9_analysis.sort(key=lambda x: x[2], reverse=True)

for code, total, long_rate, avg_los in icd9_analysis[:15]:
    print(f"  {code:10} {long_rate*100:5.1f}% long-stay (n={total:4}), avg_LOS={avg_los:.1f}")

# =============================================================================
# SECTION 5: DIAGNOSIS TEXT KEYWORDS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: DIAGNOSIS KEYWORDS - SHORT vs LONG STAY")
print("=" * 70)

diag_upper = train_df['DIAGNOSIS'].fillna('').str.upper()

keywords = [
    'SEPSIS', 'PNEUMONIA', 'HEMORRHAGE', 'STROKE', 'CARDIAC', 'HEART',
    'RESPIRATORY', 'FAILURE', 'TRANSPLANT', 'TRAUMA', 'ARREST', 'SHOCK',
    'PANCREATITIS', 'LIVER', 'KIDNEY', 'DIALYSIS', 'INTUBATION',
    'OVERDOSE', 'DIABETIC', 'DKA', 'GI BLEED', 'CANCER'
]

print(f"\n{'Keyword':20} {'Short%':>10} {'Long%':>10} {'Ratio':>10}")
print("-" * 55)

keyword_results = []
for kw in keywords:
    mask = diag_upper.str.contains(kw)
    short_pct = (mask & ~is_long).sum() / (~is_long).sum() * 100
    long_pct = (mask & is_long).sum() / is_long.sum() * 100
    ratio = long_pct / short_pct if short_pct > 0 else 0

    keyword_results.append((kw, short_pct, long_pct, ratio))

# Sort by ratio
keyword_results.sort(key=lambda x: x[3], reverse=True)

for kw, short_pct, long_pct, ratio in keyword_results:
    flag = "↑↑" if ratio > 2 else "↑" if ratio > 1.5 else "↓" if ratio < 0.7 else ""
    print(f"{kw:20} {short_pct:9.1f}% {long_pct:9.1f}% {ratio:9.2f}x {flag}")

# =============================================================================
# SECTION 6: SEVERITY SCORES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: SEVERITY SCORES DISTRIBUTION")
print("=" * 70)

for score in ['SOFA', 'SAPSII', 'GCS']:
    if score in train_df.columns:
        print(f"\n{score}:")

        # Distribution by score bucket
        if score == 'GCS':
            bins = [0, 8, 12, 15]
            labels = ['Severe (3-8)', 'Moderate (9-12)', 'Mild (13-15)']
        elif score == 'SOFA':
            bins = [0, 4, 8, 12, 25]
            labels = ['Low (0-4)', 'Med (5-8)', 'High (9-12)', 'VHigh (13+)']
        else:  # SAPSII
            bins = [0, 30, 50, 70, 200]
            labels = ['Low (0-30)', 'Med (31-50)', 'High (51-70)', 'VHigh (71+)']

        score_groups = pd.cut(train_df[score], bins=bins, labels=labels)

        for label in labels:
            mask = score_groups == label
            if mask.sum() > 0:
                total = mask.sum()
                long_in_group = (mask & is_long).sum()
                long_rate = long_in_group / total * 100
                avg_los = train_df.loc[mask, 'LOS'].mean()
                print(f"  {label:20} n={total:5}, {long_rate:5.1f}% long-stay, avg_LOS={avg_los:.1f}")

# =============================================================================
# SECTION 7: CORRELATION OF ALL FEATURES WITH LONG-STAY FLAG
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: FEATURE CORRELATIONS WITH LONG-STAY FLAG")
print("=" * 70)

# Get all numerical columns
all_numerical = train_df.select_dtypes(include=[np.number]).columns
correlations = []

for col in all_numerical:
    if col not in ['icustay_id', 'subject_id', 'hadm_id', 'LOS', 'Diff']:
        corr = train_df[col].corr(is_long.astype(int))
        if not np.isnan(corr):
            correlations.append((col, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nTop 20 features correlated with long-stay:")
for col, corr in correlations[:20]:
    direction = "+" if corr > 0 else "-"
    strength = "**" if abs(corr) > 0.1 else "*" if abs(corr) > 0.05 else ""
    print(f"  {direction} {col:30} {corr:+.4f} {strength}")

# =============================================================================
# SECTION 8: VARIANCE ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: VARIANCE WITHIN GROUPS")
print("=" * 70)

print("\nCoefficient of Variation (std/mean) for key features:")
print("Higher CV = more variance = harder to predict")
print()

for col in ['SOFA', 'SAPSII', 'HeartRate_Mean', 'RespRate_Mean']:
    if col in train_df.columns:
        short_cv = train_df.loc[~is_long, col].std() / train_df.loc[~is_long, col].mean()
        long_cv = train_df.loc[is_long, col].std() / train_df.loc[is_long, col].mean()
        print(f"{col:20} Short CV={short_cv:.3f}, Long CV={long_cv:.3f}")

# LOS variance
print(f"\nLOS Coefficient of Variation:")
print(f"  Short-stay: CV={short_los.std()/short_los.mean():.3f}")
print(f"  Long-stay:  CV={long_los.std()/long_los.mean():.3f}")

# =============================================================================
# SECTION 9: CONCLUSION
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

# Find best predictors
best_corr = correlations[0] if correlations else ("None", 0)

print(f"""
KEY FINDINGS:

1. BEST PREDICTOR OF LONG-STAY:
   {best_corr[0]}: correlation = {best_corr[1]:+.4f}

2. SIGNIFICANT NUMERICAL DIFFERENCES:""")

for feat, diff_pct, p in significant_features[:5]:
    direction = "higher" if diff_pct > 0 else "lower"
    print(f"   - {feat}: {abs(diff_pct):.1f}% {direction} in long-stay")

print(f"""
3. DIAGNOSIS PATTERNS:
   Keywords more common in long-stay:""")
for kw, _, _, ratio in keyword_results[:5]:
    print(f"   - {kw}: {ratio:.1f}x more common")

print(f"""
4. THE CORE PROBLEM:
   - Maximum correlation with long-stay is only {abs(best_corr[1]):.3f}
   - Even the best features barely distinguish short vs long stay
   - High variance within both groups
   - This is why the classifier gets only 8% F1 score

5. POSSIBLE EXPLANATIONS:
   - Long stays are caused by complications that aren't in initial data
   - The features we have are from ICU admission, not ongoing care
   - Many short-stay patients have similar profiles to long-stay
   - It's inherently unpredictable without time-series data
""")

print("=" * 70)
