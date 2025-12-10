"""
ICU History Features

Add features based on patient's previous ICU stays.
These are NOT leakage - they're known at admission time.

Key insight: Each row is a unique ICU stay, but same subject_id 
can appear multiple times. We can count PREVIOUS stays chronologically.
"""

import pandas as pd
import numpy as np


def add_icu_history_features(train_df: pd.DataFrame, 
                              test_df: pd.DataFrame,
                              subject_col: str = 'subject_id',
                              admit_col: str = 'ADMITTIME',
                              los_col: str = 'LOS') -> tuple:
    """
    Add ICU history features based on previous stays.
    
    For each ICU stay, we count how many PREVIOUS stays this patient had,
    using admission time to determine chronological order.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with subject_id, ADMITTIME, etc.
    test_df : pd.DataFrame
        Test data
    subject_col : str
        Column with patient identifier (default: 'subject_id')
    admit_col : str
        Column with admission timestamp (default: 'ADMITTIME')
    los_col : str
        Column with length of stay (default: 'LOS')
        Only used if available (might be leakage in train)
    
    Returns
    -------
    train_df, test_df : tuple of DataFrames
        DataFrames with ICU history features added
    """
    
    print("\n" + "="*80)
    print("ADDING ICU HISTORY FEATURES")
    print("="*80)
    
    # Combine train and test to get full patient history
    # Mark which dataset each row came from
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    train_copy['_dataset'] = 'train'
    test_copy['_dataset'] = 'test'
    
    # Need to ensure subject_col and admit_col exist
    if subject_col not in train_copy.columns:
        print(f"⚠️  {subject_col} not found in data")
        # Try case-insensitive match
        for col in train_copy.columns:
            if col.lower() == subject_col.lower():
                subject_col = col
                print(f"   Found {col} instead")
                break
    
    if admit_col not in train_copy.columns:
        print(f"⚠️  {admit_col} not found in data")
        for col in train_copy.columns:
            if col.lower() == admit_col.lower():
                admit_col = col
                print(f"   Found {col} instead")
                break
    
    # Combine datasets
    combined = pd.concat([train_copy, test_copy], axis=0, ignore_index=True)
    
    print(f"\n📊 Dataset Stats:")
    print(f"  Total ICU stays: {len(combined)}")
    print(f"  Unique patients: {combined[subject_col].nunique()}")
    print(f"  Avg stays per patient: {len(combined) / combined[subject_col].nunique():.2f}")
    
    # Check for patients with multiple stays
    stay_counts = combined[subject_col].value_counts()
    multi_stay_patients = (stay_counts > 1).sum()
    print(f"  Patients with multiple stays: {multi_stay_patients} ({multi_stay_patients/combined[subject_col].nunique()*100:.1f}%)")
    print(f"  Max stays for one patient: {stay_counts.max()}")
    
    # Convert admission time to datetime
    print(f"\n⏰ Processing admission times...")
    combined[admit_col] = pd.to_datetime(combined[admit_col], errors='coerce')
    
    # Sort by patient and admission time
    combined = combined.sort_values([subject_col, admit_col])
    
    print(f"  ✓ Sorted {len(combined)} stays chronologically")
    
    # ========================================================================
    # FEATURE 1: Number of previous ICU stays
    # ========================================================================
    print(f"\n[1/8] Calculating num_previous_icu_stays...")
    
    # For each patient, number their stays (0, 1, 2, ...)
    combined['_stay_number'] = combined.groupby(subject_col).cumcount()
    combined['num_previous_icu_stays'] = combined['_stay_number']
    
    print(f"  Distribution:")
    print(f"    First stay (0 previous):  {(combined['num_previous_icu_stays'] == 0).sum()} ({(combined['num_previous_icu_stays'] == 0).sum()/len(combined)*100:.1f}%)")
    print(f"    1 previous stay:          {(combined['num_previous_icu_stays'] == 1).sum()} ({(combined['num_previous_icu_stays'] == 1).sum()/len(combined)*100:.1f}%)")
    print(f"    2 previous stays:         {(combined['num_previous_icu_stays'] == 2).sum()} ({(combined['num_previous_icu_stays'] == 2).sum()/len(combined)*100:.1f}%)")
    print(f"    3+ previous stays:        {(combined['num_previous_icu_stays'] >= 3).sum()} ({(combined['num_previous_icu_stays'] >= 3).sum()/len(combined)*100:.1f}%)")
    
    # ========================================================================
    # FEATURE 2: Binary flags
    # ========================================================================
    print(f"\n[2/8] Creating binary flags...")
    
    combined['is_first_icu_stay'] = (combined['num_previous_icu_stays'] == 0).astype(int)
    combined['is_readmission'] = (combined['num_previous_icu_stays'] > 0).astype(int)
    combined['is_frequent_flyer'] = (combined['num_previous_icu_stays'] >= 3).astype(int)  # 3+ previous
    
    print(f"  is_first_icu_stay:  {combined['is_first_icu_stay'].sum()} patients")
    print(f"  is_readmission:     {combined['is_readmission'].sum()} patients")
    print(f"  is_frequent_flyer:  {combined['is_frequent_flyer'].sum()} patients")
    
    # ========================================================================
    # FEATURE 3: Days since last ICU stay
    # ========================================================================
    print(f"\n[3/8] Calculating days_since_last_icu...")
    
    # Get previous admission time for each patient
    combined['_prev_admit'] = combined.groupby(subject_col)[admit_col].shift(1)
    
    # Calculate days between this admission and previous
    combined['days_since_last_icu'] = (combined[admit_col] - combined['_prev_admit']).dt.days
    
    # Fill NaN (first stay) with a large number or special value
    combined['days_since_last_icu'] = combined['days_since_last_icu'].fillna(9999)
    
    # Get distribution (excluding first stays)
    readmissions = combined[combined['is_readmission'] == 1]['days_since_last_icu']
    if len(readmissions) > 0:
        print(f"  Readmissions only (excluding first stays):")
        print(f"    Median days since last ICU: {readmissions.median():.0f}")
        print(f"    Mean days since last ICU:   {readmissions.mean():.0f}")
        print(f"    Min days since last ICU:    {readmissions.min():.0f}")
    
    # ========================================================================
    # FEATURE 4: Readmission within timeframes
    # ========================================================================
    print(f"\n[4/8] Creating readmission timeframe flags...")
    
    combined['readmission_30d'] = ((combined['days_since_last_icu'] <= 30) & 
                                   (combined['days_since_last_icu'] < 9999)).astype(int)
    combined['readmission_90d'] = ((combined['days_since_last_icu'] <= 90) & 
                                   (combined['days_since_last_icu'] < 9999)).astype(int)
    combined['readmission_180d'] = ((combined['days_since_last_icu'] <= 180) & 
                                    (combined['days_since_last_icu'] < 9999)).astype(int)
    
    print(f"  Readmissions within 30 days:  {combined['readmission_30d'].sum()} ({combined['readmission_30d'].sum()/len(combined)*100:.2f}%)")
    print(f"  Readmissions within 90 days:  {combined['readmission_90d'].sum()} ({combined['readmission_90d'].sum()/len(combined)*100:.2f}%)")
    print(f"  Readmissions within 180 days: {combined['readmission_180d'].sum()} ({combined['readmission_180d'].sum()/len(combined)*100:.2f}%)")
    
    # ========================================================================
    # FEATURE 5: Log-transformed features (for skewed distributions)
    # ========================================================================
    print(f"\n[5/8] Creating log-transformed features...")
    
    combined['num_previous_icu_stays_log'] = np.log1p(combined['num_previous_icu_stays'])
    combined['days_since_last_icu_log'] = np.log1p(combined['days_since_last_icu'].clip(0, 9998))
    
    print(f"  ✓ Log-transformed stay counts and days")
    
    # ========================================================================
    # FEATURE 6: Historical LOS features (if available in train)
    # ========================================================================
    print(f"\n[6/8] Calculating historical LOS features...")
    
    if los_col in combined.columns:
        # Only use LOS from PREVIOUS stays (to avoid leakage)
        # Shift LOS by patient group
        combined['_prev_los'] = combined.groupby(subject_col)[los_col].shift(1)
        
        # Calculate cumulative stats from previous stays
        # For each stay, get mean/max/sum of ALL previous stays
        combined['avg_previous_los'] = combined.groupby(subject_col)['_prev_los'].expanding().mean().reset_index(level=0, drop=True)
        combined['max_previous_los'] = combined.groupby(subject_col)['_prev_los'].expanding().max().reset_index(level=0, drop=True)
        combined['total_previous_icu_days'] = combined.groupby(subject_col)['_prev_los'].expanding().sum().reset_index(level=0, drop=True)
        
        # Fill NaN (first stay) with 0
        combined['avg_previous_los'] = combined['avg_previous_los'].fillna(0)
        combined['max_previous_los'] = combined['max_previous_los'].fillna(0)
        combined['total_previous_icu_days'] = combined['total_previous_icu_days'].fillna(0)
        
        print(f"  ✓ Created avg_previous_los, max_previous_los, total_previous_icu_days")
    else:
        print(f"  ⚠️  {los_col} not available - skipping LOS history features")
        combined['avg_previous_los'] = 0
        combined['max_previous_los'] = 0
        combined['total_previous_icu_days'] = 0
    
    # ========================================================================
    # FEATURE 7: Interaction features
    # ========================================================================
    print(f"\n[7/8] Creating interaction features...")
    
    # Previous stays × recent readmission (high risk!)
    combined['frequent_recent_readmit'] = (
        (combined['num_previous_icu_stays'] >= 2) & 
        (combined['readmission_30d'] == 1)
    ).astype(int)
    
    print(f"  frequent_recent_readmit: {combined['frequent_recent_readmit'].sum()} patients")
    
    # ========================================================================
    # FEATURE 8: Clean up and split back to train/test
    # ========================================================================
    print(f"\n[8/8] Finalizing...")
    
    # Drop temporary columns (but keep _dataset until we split)
    temp_cols = [col for col in combined.columns if col.startswith('_') and col != '_dataset']
    combined = combined.drop(columns=temp_cols)
    
    # Split back to train and test
    train_with_history = combined[combined['_dataset'] == 'train'].drop(columns=['_dataset']).reset_index(drop=True)
    test_with_history = combined[combined['_dataset'] == 'test'].drop(columns=['_dataset']).reset_index(drop=True)
    
    # Verify shapes match
    assert len(train_with_history) == len(train_df), "Train length mismatch!"
    assert len(test_with_history) == len(test_df), "Test length mismatch!"
    
    print(f"\n✓ ICU history features added successfully!")
    
    # Summary of new features
    new_features = [
        'num_previous_icu_stays',
        'is_first_icu_stay',
        'is_readmission',
        'is_frequent_flyer',
        'days_since_last_icu',
        'readmission_30d',
        'readmission_90d',
        'readmission_180d',
        'num_previous_icu_stays_log',
        'days_since_last_icu_log',
        'avg_previous_los',
        'max_previous_los',
        'total_previous_icu_days',
        'frequent_recent_readmit'
    ]
    
    print(f"\n📋 New Features ({len(new_features)}):")
    for i, feat in enumerate(new_features, 1):
        if feat in train_with_history.columns:
            print(f"  {i:2d}. {feat}")
    
    print(f"\n" + "="*80)
    
    return train_with_history, test_with_history


if __name__ == "__main__":
    # Test the function
    from pathlib import Path
    
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset HEF"
    
    print("Testing ICU history feature extraction...")
    
    train_raw = pd.read_csv(DATA_DIR / "mimic_train_HEF.csv")
    test_raw = pd.read_csv(DATA_DIR / "mimic_test_HEF.csv")
    
    print(f"\nOriginal shapes:")
    print(f"  Train: {train_raw.shape}")
    print(f"  Test: {test_raw.shape}")
    
    train_with_hist, test_with_hist = add_icu_history_features(train_raw, test_raw)
    
    print(f"\nNew shapes:")
    print(f"  Train: {train_with_hist.shape}")
    print(f"  Test: {test_with_hist.shape}")
    
    print(f"\nNew features added: {train_with_hist.shape[1] - train_raw.shape[1]}")