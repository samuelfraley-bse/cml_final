"""
Generate Both Submission Formats
Creates both probability and binary versions to be safe
"""

import pandas as pd
from pathlib import Path
import numpy as np

print("="*60)
print("DUAL FORMAT SUBMISSION GENERATOR")
print("="*60)

# Paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# Load test data to get IDs
print("\n[1/2] Loading test data...")
test_df = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

# Find ID column
id_col = None
for col in test_df.columns:
    if col.lower() == 'icustay_id':
        id_col = col
        break

icustay_ids = test_df[id_col]
print(f"  Found {len(icustay_ids)} test IDs")

# Process each submission file
print("\n[2/2] Creating both formats for each submission...")
submission_files = list(SUBMISSIONS_DIR.glob("submission_*.csv"))

if not submission_files:
    print("  ERROR: No submission files found!")
    print("  Run generate_kaggle_submissions.py first")
    exit(1)

for filepath in submission_files:
    print(f"\n  Processing: {filepath.name}")
    
    # Load submission
    sub_df = pd.read_csv(filepath)
    
    # Get probabilities (whether or not it has ID column)
    if 'HOSPITAL_EXPIRE_FLAG' in sub_df.columns:
        probabilities = sub_df['HOSPITAL_EXPIRE_FLAG'].values
    else:
        print(f"    ERROR: No HOSPITAL_EXPIRE_FLAG column found!")
        continue
    
    # Check if IDs are present
    has_ids = 'icustay_id' in sub_df.columns or 'ICUSTAY_ID' in sub_df.columns
    
    # Version 1: Probabilities with IDs
    prob_df = pd.DataFrame({
        'icustay_id': icustay_ids,
        'HOSPITAL_EXPIRE_FLAG': probabilities
    })
    
    prob_path = SUBMISSIONS_DIR / f"PROB_{filepath.stem}.csv"
    prob_df.to_csv(prob_path, index=False)
    print(f"    ✓ Probability version: {prob_path.name}")
    print(f"      Range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    
    # Version 2: Binary (0/1) with IDs
    # Convert probabilities to binary using 0.5 threshold
    binary = (probabilities >= 0.5).astype(int)
    
    binary_df = pd.DataFrame({
        'icustay_id': icustay_ids,
        'HOSPITAL_EXPIRE_FLAG': binary
    })
    
    binary_path = SUBMISSIONS_DIR / f"BINARY_{filepath.stem}.csv"
    binary_df.to_csv(binary_path, index=False)
    print(f"    ✓ Binary version: {binary_path.name}")
    print(f"      Positive rate: {binary.mean():.3f} (matches sample if random)")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nCreated TWO versions of each submission:")
print("\n1. PROB_*.csv - Probabilities (0.0 to 1.0)")
print("   → PDF says: 'will maximize your scores'")
print("   → Use for ROC-AUC or Log Loss metrics")
print("   → RECOMMENDED based on instructions")

print("\n2. BINARY_*.csv - Binary predictions (0 or 1)")
print("   → Matches the sample submission format")
print("   → Use if they actually want binary")
print("   → Fallback option")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("\n📤 Upload PROB_submission_rf.csv FIRST")
print("\nWhy?")
print("  - PDF explicitly says probabilities 'maximize scores'")
print("  - They mention .predict_proba() specifically")
print("  - Sample is likely just dummy data")

print("\n🔄 If that gets rejected or scores poorly:")
print("  → Try BINARY_submission_rf.csv instead")

print("\nYou have both versions ready - start with probabilities!")
print("="*60)