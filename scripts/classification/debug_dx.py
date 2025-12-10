"""
Diagnosis Features Debugging Script

This script investigates why diagnosis features are broken:
- diagnosis_count: NaN
- num_organ_systems: NaN
- has_respiratory, has_cardiac, has_infection: 0.000 importance
- primary_diagnosis_chapter: 0 correlation

Run from project root:
    python scripts/analysis/debug_diagnosis.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "notebooks" / "HEF"))

print("="*80)
print("DIAGNOSIS FEATURES DEBUGGING")
print("="*80)

# ============================================================================
# STEP 1: Load Raw Data
# ============================================================================

print("\n[Step 1] Loading raw data...")

DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")

print(f"✓ Train: {train_raw.shape}")
print(f"✓ Test: {test_raw.shape}")

# Find ID columns
print("\n[Step 2] Checking ID columns...")
print(f"Train columns: {train_raw.columns.tolist()[:10]}...")

hadm_col_train = None
for col in train_raw.columns:
    if 'HADM' in col.upper():
        hadm_col_train = col
        print(f"✓ Found HADM column in train: '{hadm_col_train}'")
        break

hadm_col_test = None
for col in test_raw.columns:
    if 'HADM' in col.upper():
        hadm_col_test = col
        print(f"✓ Found HADM column in test: '{hadm_col_test}'")
        break

if hadm_col_train:
    print(f"\nTrain HADM_ID stats:")
    print(f"  Unique values: {train_raw[hadm_col_train].nunique()}")
    print(f"  Missing values: {train_raw[hadm_col_train].isnull().sum()}")
    print(f"  Sample values: {train_raw[hadm_col_train].head(10).tolist()}")

if hadm_col_test:
    print(f"\nTest HADM_ID stats:")
    print(f"  Unique values: {test_raw[hadm_col_test].nunique()}")
    print(f"  Missing values: {test_raw[hadm_col_test].isnull().sum()}")
    print(f"  Sample values: {test_raw[hadm_col_test].head(10).tolist()}")

# ============================================================================
# STEP 2: Load Diagnosis Data
# ============================================================================

print("\n" + "="*80)
print("[Step 3] Loading MIMIC_diagnoses.csv...")
print("="*80)

diag_path = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF" / "extra_data" / "MIMIC_diagnoses.csv"
print(f"Path: {diag_path}")
print(f"Exists: {diag_path.exists()}")

if not diag_path.exists():
    print("❌ ERROR: MIMIC_diagnoses.csv not found!")
    sys.exit(1)

diagnosis_df = pd.read_csv(diag_path)
print(f"✓ Loaded: {diagnosis_df.shape}")
print(f"\nDiagnosis columns: {diagnosis_df.columns.tolist()}")

# Standardize column names
diagnosis_df.columns = diagnosis_df.columns.str.upper()
print(f"\nAfter uppercase: {diagnosis_df.columns.tolist()}")

# Check for HADM_ID
hadm_col_diag = None
for col in diagnosis_df.columns:
    if 'HADM' in col.upper():
        hadm_col_diag = col
        print(f"✓ Found HADM column in diagnosis: '{hadm_col_diag}'")
        break

if not hadm_col_diag:
    print("❌ ERROR: No HADM column found in diagnosis data!")
    sys.exit(1)

# Rename to lowercase
if hadm_col_diag != 'hadm_id':
    diagnosis_df = diagnosis_df.rename(columns={hadm_col_diag: 'hadm_id'})
    print(f"✓ Renamed '{hadm_col_diag}' → 'hadm_id'")

print(f"\nDiagnosis data stats:")
print(f"  Total records: {len(diagnosis_df)}")
print(f"  Unique hadm_id: {diagnosis_df['hadm_id'].nunique()}")
print(f"  Missing hadm_id: {diagnosis_df['hadm_id'].isnull().sum()}")
print(f"\nFirst 10 rows:")
print(diagnosis_df.head(10))

# ============================================================================
# STEP 3: Check Overlap Between Train and Diagnosis Data
# ============================================================================

print("\n" + "="*80)
print("[Step 4] Checking HADM_ID overlap...")
print("="*80)

if hadm_col_train:
    train_hadm_set = set(train_raw[hadm_col_train].dropna())
    diag_hadm_set = set(diagnosis_df['hadm_id'].dropna())
    
    overlap = train_hadm_set & diag_hadm_set
    
    print(f"\nTrain HADM_IDs: {len(train_hadm_set)}")
    print(f"Diagnosis HADM_IDs: {len(diag_hadm_set)}")
    print(f"Overlap: {len(overlap)} ({len(overlap)/len(train_hadm_set)*100:.1f}% of train)")
    
    if len(overlap) == 0:
        print("\n❌ CRITICAL: NO OVERLAP BETWEEN TRAIN AND DIAGNOSIS DATA!")
        print("   This explains why diagnosis features are useless!")
        print("\n   Possible causes:")
        print("   1. HADM_ID column name mismatch")
        print("   2. HADM_ID format mismatch (int vs string)")
        print("   3. Wrong diagnosis file")
        
        # Check data types
        print(f"\n   Train HADM_ID dtype: {train_raw[hadm_col_train].dtype}")
        print(f"   Diagnosis hadm_id dtype: {diagnosis_df['hadm_id'].dtype}")
        
        # Check sample values
        print(f"\n   Train HADM_ID samples: {list(train_raw[hadm_col_train].head(5))}")
        print(f"   Diagnosis hadm_id samples: {list(diagnosis_df['hadm_id'].head(5))}")
        
    else:
        print(f"\n✓ Good overlap found!")
        
        # Check a specific example
        sample_hadm = list(overlap)[0]
        print(f"\n   Example HADM_ID: {sample_hadm}")
        
        train_row = train_raw[train_raw[hadm_col_train] == sample_hadm].iloc[0]
        diag_rows = diagnosis_df[diagnosis_df['hadm_id'] == sample_hadm]
        
        print(f"   Train row for this HADM:")
        print(f"     ICUSTAY_ID: {train_row.get('ICUSTAY_ID', 'N/A')}")
        print(f"     SUBJECT_ID: {train_row.get('SUBJECT_ID', 'N/A')}")
        
        print(f"   Diagnosis rows for this HADM: {len(diag_rows)}")
        print(diag_rows[['SUBJECT_ID', 'hadm_id', 'SEQ_NUM', 'ICD9_CODE']].head())

# Check test overlap too
if hadm_col_test:
    test_hadm_set = set(test_raw[hadm_col_test].dropna())
    overlap_test = test_hadm_set & diag_hadm_set
    
    print(f"\nTest HADM_IDs: {len(test_hadm_set)}")
    print(f"Overlap with diagnosis: {len(overlap_test)} ({len(overlap_test)/len(test_hadm_set)*100:.1f}% of test)")

# ============================================================================
# STEP 4: Test Feature Creation
# ============================================================================

print("\n" + "="*80)
print("[Step 5] Testing feature creation manually...")
print("="*80)

if hadm_col_train:
    # Create a small test dataframe
    test_df = train_raw[[hadm_col_train]].head(100).copy()
    
    print(f"\nTest dataframe: {test_df.shape}")
    print(f"Unique HADM_IDs: {test_df[hadm_col_train].nunique()}")
    
    # Try to create diagnosis_count
    dx_counts = diagnosis_df.groupby('hadm_id').size().reset_index(name='diagnosis_count')
    print(f"\nDiagnosis counts created: {dx_counts.shape}")
    print(f"Sample:")
    print(dx_counts.head())
    
    # Try merge
    test_merged = test_df.merge(dx_counts, left_on=hadm_col_train, right_on='hadm_id', how='left')
    
    print(f"\nAfter merge:")
    print(f"  Shape: {test_merged.shape}")
    print(f"  diagnosis_count missing: {test_merged['diagnosis_count'].isnull().sum()}")
    print(f"  diagnosis_count mean: {test_merged['diagnosis_count'].mean()}")
    print(f"\nSample:")
    print(test_merged[[hadm_col_train, 'diagnosis_count']].head(10))

# ============================================================================
# STEP 5: Check After split_features_target
# ============================================================================

print("\n" + "="*80)
print("[Step 6] Checking what happens after split_features_target...")
print("="*80)

from hef_prep import split_features_target, ID_COLS

leak_cols = ["DEATHTIME", "DISCHTIME", "DOD", "DIAGNOSIS", "ICD9_diagnosis"]

X_train_raw, y_train, X_test_raw = split_features_target(
    train_df=train_raw,
    test_df=test_raw,
    task="class",
    leak_cols=leak_cols,
    id_cols=ID_COLS
)

print(f"\nAfter split_features_target:")
print(f"  X_train columns: {X_train_raw.columns.tolist()[:20]}...")
print(f"  X_test columns: {X_test_raw.columns.tolist()[:20]}...")

# Check if HADM_ID still exists
hadm_in_X_train = None
for col in X_train_raw.columns:
    if 'HADM' in col.upper():
        hadm_in_X_train = col
        print(f"\n✓ HADM column still in X_train: '{hadm_in_X_train}'")
        break

if not hadm_in_X_train:
    print("\n❌ CRITICAL: HADM_ID was removed by split_features_target!")
    print("   This means diagnosis features can't be merged!")
    print("\n   Solution: Keep HADM_ID through preprocessing, remove it later")

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSIS DEBUGGING SUMMARY")
print("="*80)

print("\n🔍 Key Findings:")

if hadm_col_train and hadm_col_diag:
    if len(overlap) == 0:
        print("\n❌ PROBLEM 1: No HADM_ID overlap between train and diagnosis data")
        print("   → Diagnosis features can't be merged!")
        
    if not hadm_in_X_train:
        print("\n❌ PROBLEM 2: HADM_ID removed by split_features_target")
        print("   → Can't merge diagnosis data after split!")
        
    print("\n💡 SOLUTIONS:")
    print("   1. DON'T drop HADM_ID in split_features_target")
    print("   2. Merge diagnosis BEFORE split_features_target")
    print("   3. Or: Keep HADM_ID through preprocessing, drop at the end")

print("\n" + "="*80)