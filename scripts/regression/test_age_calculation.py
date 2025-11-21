"""
Test Age Calculation from MIMIC-III Data
Just to verify the age calculation is correct before using in models.

Run: python scripts/regression/test_age_calculation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("AGE CALCULATION TEST")
print("=" * 70)

# Load data
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
train_df = pd.read_csv(DATA_DIR / "mimic_train_LOS.csv")

print(f"\nLoaded {len(train_df)} records")

# Look at the raw date columns
print("\n" + "=" * 70)
print("RAW DATE COLUMNS")
print("=" * 70)

print("\nDOB sample values:")
print(train_df['DOB'].head(10))

print("\nADMITTIME sample values:")
print(train_df['ADMITTIME'].head(10))

print("\nDiff sample values:")
print(train_df['Diff'].head(10))

# Convert to datetime
print("\n" + "=" * 70)
print("CALCULATING AGE")
print("=" * 70)

dob = pd.to_datetime(train_df['DOB'], errors='coerce')
admit = pd.to_datetime(train_df['ADMITTIME'], errors='coerce')

print(f"\nDOB range: {dob.min()} to {dob.max()}")
print(f"ADMITTIME range: {admit.min()} to {admit.max()}")

# Method 1: Calculate year difference directly (avoids overflow)
print("\nMethod: Year difference (handles MIMIC date shifting)")

age_years = admit.dt.year - dob.dt.year

# Adjust for people who haven't had birthday yet this year
birthday_passed = (
    (admit.dt.month > dob.dt.month) |
    ((admit.dt.month == dob.dt.month) & (admit.dt.day >= dob.dt.day))
)
age_years = age_years - (~birthday_passed).astype(int)

print(f"\nBefore capping:")
print(f"  Min age:  {age_years.min()} years")
print(f"  Max age:  {age_years.max()} years")
print(f"  Mean age: {age_years.mean():.1f} years")

# Check for patients with shifted DOB (>89 years old in MIMIC)
print(f"\n  Patients with age > 100: {(age_years > 100).sum()}")
print(f"  Patients with age > 200: {(age_years > 200).sum()}")
print(f"  Patients with age > 300: {(age_years > 300).sum()}")

# Cap at 90 (MIMIC privacy protection)
age_years_capped = age_years.clip(upper=90)

print(f"\nAfter capping at 90:")
print(f"  Min age:  {age_years_capped.min()} years")
print(f"  Max age:  {age_years_capped.max()} years")
print(f"  Mean age: {age_years_capped.mean():.1f} years")
print(f"  Median:   {age_years_capped.median():.1f} years")
print(f"  Std:      {age_years_capped.std():.1f} years")

# Distribution
print("\n" + "=" * 70)
print("AGE DISTRIBUTION")
print("=" * 70)

print(f"\n  <18:    {(age_years_capped < 18).sum():5} ({(age_years_capped < 18).mean()*100:.1f}%)")
print(f"  18-45:  {((age_years_capped >= 18) & (age_years_capped < 45)).sum():5} ({((age_years_capped >= 18) & (age_years_capped < 45)).mean()*100:.1f}%)")
print(f"  45-65:  {((age_years_capped >= 45) & (age_years_capped < 65)).sum():5} ({((age_years_capped >= 45) & (age_years_capped < 65)).mean()*100:.1f}%)")
print(f"  65-75:  {((age_years_capped >= 65) & (age_years_capped < 75)).sum():5} ({((age_years_capped >= 65) & (age_years_capped < 75)).mean()*100:.1f}%)")
print(f"  75-90:  {(age_years_capped >= 75).sum():5} ({(age_years_capped >= 75).mean()*100:.1f}%)")

# Correlation with LOS
print("\n" + "=" * 70)
print("CORRELATION WITH LOS")
print("=" * 70)

corr = age_years_capped.corr(train_df['LOS'])
print(f"\nCorrelation (age vs LOS): {corr:.4f}")

# Average LOS by age group
print("\nAverage LOS by age group:")
age_groups = pd.cut(age_years_capped, bins=[0, 45, 65, 75, 90], labels=['<45', '45-65', '65-75', '75+'])

for group in ['<45', '45-65', '65-75', '75+']:
    mask = age_groups == group
    if mask.sum() > 0:
        avg_los = train_df.loc[mask, 'LOS'].mean()
        count = mask.sum()
        print(f"  {group:10} (n={count:5}): avg_LOS = {avg_los:.2f} days")

# Show some examples
print("\n" + "=" * 70)
print("SAMPLE RECORDS (first 10)")
print("=" * 70)

sample_df = pd.DataFrame({
    'DOB': train_df['DOB'].head(10),
    'ADMITTIME': train_df['ADMITTIME'].head(10),
    'Calculated_Age': age_years_capped.head(10),
    'LOS': train_df['LOS'].head(10)
})
print(sample_df.to_string())

# Check for any issues
print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print(f"\nNull ages: {age_years_capped.isna().sum()}")
print(f"Negative ages: {(age_years_capped < 0).sum()}")
print(f"Zero ages: {(age_years_capped == 0).sum()}")

if (age_years_capped < 0).sum() > 0:
    print("\nWARNING: Found negative ages!")
    neg_idx = age_years_capped[age_years_capped < 0].index[:5]
    print(train_df.loc[neg_idx, ['DOB', 'ADMITTIME']])

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if age_years_capped.mean() > 50 and age_years_capped.mean() < 70:
    print("\n✓ Age calculation looks reasonable!")
    print(f"  Mean age of {age_years_capped.mean():.1f} years is typical for ICU patients.")
else:
    print("\n⚠ Age calculation may need review.")
    print(f"  Mean age of {age_years_capped.mean():.1f} years seems unusual.")

if abs(corr) < 0.05:
    print(f"\n⚠ Low correlation with LOS ({corr:.4f})")
    print("  Age may not be very predictive, but still worth including.")
elif corr > 0:
    print(f"\n✓ Positive correlation with LOS ({corr:.4f})")
    print("  Older patients tend to have longer stays.")
else:
    print(f"\n? Negative correlation with LOS ({corr:.4f})")
    print("  Younger patients tend to have longer stays (unusual).")

print("\n" + "=" * 70)
