"""
Test ADMITTIME Feature Extraction from MIMIC-III Data
Extract hour of admission, day of week, etc. and check correlation with LOS.

Run: python scripts/regression/test_admittime_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("ADMITTIME FEATURES TEST")
print("=" * 70)

# Load data
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
train_df = pd.read_csv(DATA_DIR / "mimic_train_LOS.csv")

print(f"\nLoaded {len(train_df)} records")

# Convert ADMITTIME to datetime
print("\n" + "=" * 70)
print("PARSING ADMITTIME")
print("=" * 70)

admit = pd.to_datetime(train_df['ADMITTIME'], errors='coerce')

print(f"\nSample ADMITTIME values:")
print(train_df['ADMITTIME'].head(10))

print(f"\nParsed datetime range:")
print(f"  Min: {admit.min()}")
print(f"  Max: {admit.max()}")
print(f"  Nulls: {admit.isna().sum()}")

# Extract time features
print("\n" + "=" * 70)
print("EXTRACTING TIME FEATURES")
print("=" * 70)

# Hour of admission
admit_hour = admit.dt.hour
print(f"\nAdmission Hour:")
print(f"  Range: {admit_hour.min()} - {admit_hour.max()}")

# Day of week (0=Monday, 6=Sunday)
admit_dow = admit.dt.dayofweek
dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Weekend flag
is_weekend = (admit_dow >= 5).astype(int)

# Night admission (7pm - 7am)
is_night = ((admit_hour >= 19) | (admit_hour < 7)).astype(int)

# Month
admit_month = admit.dt.month

print(f"\nFeatures extracted:")
print(f"  - admit_hour (0-23)")
print(f"  - admit_dow (0-6, Monday-Sunday)")
print(f"  - is_weekend (0/1)")
print(f"  - is_night (0/1, 7pm-7am)")
print(f"  - admit_month (1-12)")

# Analyze distributions
print("\n" + "=" * 70)
print("FEATURE DISTRIBUTIONS")
print("=" * 70)

# Hour distribution
print("\nAdmission by Hour (top 5):")
hour_counts = admit_hour.value_counts().sort_index()
for hour in sorted(hour_counts.nlargest(5).index):
    count = hour_counts[hour]
    pct = count / len(train_df) * 100
    print(f"  {hour:02d}:00 - {count:5} ({pct:.1f}%)")

# Day of week distribution
print("\nAdmission by Day of Week:")
for i, day in enumerate(dow_names):
    count = (admit_dow == i).sum()
    avg_los = train_df[admit_dow == i]['LOS'].mean()
    print(f"  {day:10} n={count:5}, avg_LOS={avg_los:.2f}")

# Weekend vs weekday
print("\nWeekend vs Weekday:")
for val, name in [(0, 'Weekday'), (1, 'Weekend')]:
    count = (is_weekend == val).sum()
    avg_los = train_df[is_weekend == val]['LOS'].mean()
    pct = count / len(train_df) * 100
    print(f"  {name:10} n={count:5} ({pct:.1f}%), avg_LOS={avg_los:.2f}")

# Night vs day
print("\nNight (7pm-7am) vs Day (7am-7pm):")
for val, name in [(0, 'Day'), (1, 'Night')]:
    count = (is_night == val).sum()
    avg_los = train_df[is_night == val]['LOS'].mean()
    pct = count / len(train_df) * 100
    print(f"  {name:10} n={count:5} ({pct:.1f}%), avg_LOS={avg_los:.2f}")

# Month distribution
print("\nAdmission by Month (avg LOS):")
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for i in range(1, 13):
    count = (admit_month == i).sum()
    avg_los = train_df[admit_month == i]['LOS'].mean()
    print(f"  {month_names[i-1]:3} n={count:5}, avg_LOS={avg_los:.2f}")

# Correlations with LOS
print("\n" + "=" * 70)
print("CORRELATIONS WITH LOS")
print("=" * 70)

los = train_df['LOS']

correlations = {
    'admit_hour': admit_hour.corr(los),
    'admit_dow': admit_dow.corr(los),
    'is_weekend': is_weekend.corr(los),
    'is_night': is_night.corr(los),
    'admit_month': admit_month.corr(los),
}

print("\nCorrelation with LOS:")
for feat, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
    direction = "+" if corr > 0 else "-"
    print(f"  {direction} {feat:15} {corr:+.4f}")

# Detailed hour analysis
print("\n" + "=" * 70)
print("DETAILED HOUR ANALYSIS")
print("=" * 70)

print("\nAverage LOS by admission hour:")
hour_los = []
for hour in range(24):
    mask = admit_hour == hour
    if mask.sum() > 0:
        avg_los = train_df[mask]['LOS'].mean()
        count = mask.sum()
        hour_los.append((hour, avg_los, count))

# Sort by avg LOS to find patterns
hour_los_sorted = sorted(hour_los, key=lambda x: x[1], reverse=True)

print("\nTop 5 hours with longest LOS:")
for hour, avg_los, count in hour_los_sorted[:5]:
    print(f"  {hour:02d}:00 - avg_LOS={avg_los:.2f} (n={count})")

print("\nTop 5 hours with shortest LOS:")
for hour, avg_los, count in hour_los_sorted[-5:]:
    print(f"  {hour:02d}:00 - avg_LOS={avg_los:.2f} (n={count})")

# Sample records
print("\n" + "=" * 70)
print("SAMPLE RECORDS")
print("=" * 70)

sample_df = pd.DataFrame({
    'ADMITTIME': train_df['ADMITTIME'].head(10),
    'hour': admit_hour.head(10),
    'dow': admit_dow.head(10),
    'weekend': is_weekend.head(10),
    'night': is_night.head(10),
    'LOS': train_df['LOS'].head(10)
})
print(sample_df.to_string())

# Conclusion
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

max_corr = max(abs(v) for v in correlations.values())
best_feat = max(correlations.items(), key=lambda x: abs(x[1]))

if max_corr < 0.05:
    print(f"\n⚠ All correlations are very low (max: {max_corr:.4f})")
    print("  Time features may not be very predictive.")
elif max_corr < 0.1:
    print(f"\n? Weak correlations (max: {max_corr:.4f})")
    print(f"  Best feature: {best_feat[0]} ({best_feat[1]:+.4f})")
    print("  May provide small improvement.")
else:
    print(f"\n✓ Found useful correlation!")
    print(f"  Best feature: {best_feat[0]} ({best_feat[1]:+.4f})")

# Check if weekend/night show meaningful LOS differences
weekend_diff = abs(train_df[is_weekend == 1]['LOS'].mean() - train_df[is_weekend == 0]['LOS'].mean())
night_diff = abs(train_df[is_night == 1]['LOS'].mean() - train_df[is_night == 0]['LOS'].mean())

print(f"\nLOS differences:")
print(f"  Weekend vs Weekday: {weekend_diff:.3f} days")
print(f"  Night vs Day: {night_diff:.3f} days")

if weekend_diff > 0.1 or night_diff > 0.1:
    print("\n  These differences might be meaningful for the model.")
else:
    print("\n  Very small differences - may not help much.")

print("\n" + "=" * 70)
