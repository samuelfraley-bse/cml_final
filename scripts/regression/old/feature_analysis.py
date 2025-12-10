"""
LOS Regression - Feature Analysis Script (V2)
Now with:
- Diff column dropped
- Diagnosis-based features
- Better feature engineering

Run from scripts/regression/:
    python feature_analysis.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "regression":
    BASE_DIR = SCRIPT_DIR.parents[1]
    sys.path.append(str(BASE_DIR / "src"))
    sys.path.append(str(SCRIPT_DIR))
else:
    BASE_DIR = Path.cwd()
    sys.path.append(str(BASE_DIR / "src"))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 70)
print("LOS REGRESSION - FEATURE ANALYSIS V2")
print("=" * 70)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate patient age from DOB and ADMITTIME.

    From metadata: Diff is "Days to add to any datetime to become realistic"
    Since Diff applies to both DOB and ADMITTIME for the same patient,
    it cancels out when calculating age.
    """
    df = df.copy()

    if 'DOB' in df.columns and 'ADMITTIME' in df.columns:
        # Convert to datetime
        dob = pd.to_datetime(df['DOB'], errors='coerce')
        admit = pd.to_datetime(df['ADMITTIME'], errors='coerce')

        # Calculate age in years
        age_days = (admit - dob).dt.days
        df['age_years'] = age_days / 365.25

        # MIMIC-III shifts dates for patients >89 years old to protect privacy
        # These patients get shifted to ~300 years old
        # Cap at 90 to handle this
        df['age_years'] = df['age_years'].clip(upper=90)

        # Age groups (categorical)
        df['age_group'] = pd.cut(
            df['age_years'],
            bins=[0, 45, 65, 75, 90],
            labels=['young', 'middle', 'senior', 'elderly']
        )

    return df


def add_diagnosis_features(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add features based on DIAGNOSIS and ICD9_diagnosis columns.

    For training: pass train_df=None (will use df itself)
    For test: pass train_df to use same categories learned from training
    """
    df = df.copy()

    # --- ICD9 Code Features ---
    if 'ICD9_diagnosis' in df.columns:
        # High-LOS ICD9 codes (avg LOS > 5 days based on exploration)
        high_los_icd9 = [
            '51884',  # acute resp failure - 8.48 days
            '5770',   # acute pancreatitis - 8.30 days
            '430',    # subarachnoid hemorrhage - 7.76 days
            '44101',  # thoracic aortic aneurysm - 6.83 days
            '0380',   # strep septicemia - 6.70 days
            '51881',  # acute resp failure - 6.66 days
            '0389',   # septicemia - 5.12 days
            '431',    # intracerebral hemorrhage - 4.28 days
        ]

        # Low-LOS ICD9 codes (avg LOS < 2.5 days)
        low_los_icd9 = [
            '9951',   # angioedema - 1.49 days
            '4373',   # transient cerebral ischemia - 1.58 days
            '29181',  # alcohol withdrawal - 1.96 days
        ]

        # Original codes we were using
        other_icd9 = ['41401', '41071', '4241', '486', '5070']

        for code in high_los_icd9 + low_los_icd9 + other_icd9:
            df[f'icd9_{code}'] = (df['ICD9_diagnosis'] == code).astype(int)

        # Group by first 3 digits (disease category)
        df['icd9_category'] = df['ICD9_diagnosis'].astype(str).str[:3]

        # High-LOS ICD9 categories (first 3 digits)
        high_los_cats = ['430', '577', '518', '441', '038']
        for cat in high_los_cats:
            df[f'icd9_cat_{cat}'] = (df['icd9_category'] == cat).astype(int)

    # --- DIAGNOSIS Text Features ---
    if 'DIAGNOSIS' in df.columns:
        diag_upper = df['DIAGNOSIS'].fillna('').str.upper()

        # HIGH-LOS keywords (avg LOS > 4.5 days from exploration)
        df['diag_pancreatitis'] = diag_upper.str.contains('PANCREATITIS').astype(int)  # 8.12 days!
        df['diag_shock'] = diag_upper.str.contains('SHOCK').astype(int)  # 7.85 days!
        df['diag_arrest'] = diag_upper.str.contains('ARREST').astype(int)  # 5.37 days
        df['diag_transplant'] = diag_upper.str.contains('TRANSPLANT').astype(int)  # 5.53 days
        df['diag_failure'] = diag_upper.str.contains('FAILURE').astype(int)  # 4.53 days

        # Original high-LOS keywords
        df['diag_sepsis'] = diag_upper.str.contains('SEPSIS').astype(int)
        df['diag_pneumonia'] = diag_upper.str.contains('PNEUMONIA').astype(int)
        df['diag_hemorrhage'] = diag_upper.str.contains('HEMORRHAGE').astype(int)  # separate from BLEED
        df['diag_respiratory'] = diag_upper.str.contains('RESPIRATORY').astype(int)

        # LOW-LOS keywords (to help model identify shorter stays)
        df['diag_overdose'] = diag_upper.str.contains('OVERDOSE').astype(int)  # 2.56 days
        df['diag_diabetic'] = diag_upper.str.contains('DIABETIC|DKA').astype(int)  # 2.38 days
        df['diag_gi_bleed'] = diag_upper.str.contains('GI BLEED|UPPER GI|LOWER GI').astype(int)  # 2.64 days

        # Other useful keywords
        df['diag_cardiac'] = diag_upper.str.contains('HEART|CARDIAC|CORONARY|MI |STEMI|NSTEMI').astype(int)
        df['diag_neurological'] = diag_upper.str.contains('STROKE|SEIZURE|NEUROLOG|MENTAL').astype(int)
        df['diag_trauma'] = diag_upper.str.contains('TRAUMA|FRACTURE|INJURY|MVA|FALL').astype(int)
        df['diag_cancer'] = diag_upper.str.contains('CANCER|TUMOR|MALIGNAN|LYMPHOMA|LEUKEMIA').astype(int)
        df['diag_liver'] = diag_upper.str.contains('LIVER|HEPAT|CIRRHOSIS').astype(int)

        # Count of conditions (complexity proxy)
        condition_cols = [c for c in df.columns if c.startswith('diag_')]
        df['diag_count'] = df[condition_cols].sum(axis=1)

    # --- Interaction: SICU + EMERGENCY (highest LOS combination) ---
    if 'FIRST_CAREUNIT' in df.columns and 'ADMISSION_TYPE' in df.columns:
        df['sicu_emergency'] = ((df['FIRST_CAREUNIT'] == 'SICU') &
                                (df['ADMISSION_TYPE'] == 'EMERGENCY')).astype(int)

    return df


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering (improved version of los_prep.add_engineered_features)
    """
    df = df.copy()
    eps = 1e-3

    # --- Vital Sign Ranges (instability indicators) ---
    if 'HeartRate_Max' in df.columns and 'HeartRate_Min' in df.columns:
        df["HR_range"] = df["HeartRate_Max"] - df["HeartRate_Min"]
    if 'SysBP_Max' in df.columns and 'SysBP_Min' in df.columns:
        df["SysBP_range"] = df["SysBP_Max"] - df["SysBP_Min"]
    if 'MeanBP_Max' in df.columns and 'MeanBP_Min' in df.columns:
        df["MeanBP_range"] = df["MeanBP_Max"] - df["MeanBP_Min"]
    if 'RespRate_Max' in df.columns and 'RespRate_Min' in df.columns:
        df["RespRate_range"] = df["RespRate_Max"] - df["RespRate_Min"]
    if 'SpO2_Max' in df.columns and 'SpO2_Min' in df.columns:
        df["SpO2_range"] = df["SpO2_Max"] - df["SpO2_Min"]
    if 'TempC_Max' in df.columns and 'TempC_Min' in df.columns:
        df["Temp_range"] = df["TempC_Max"] - df["TempC_Min"]
    if 'Glucose_Max' in df.columns and 'Glucose_Min' in df.columns:
        df["Glucose_range"] = df["Glucose_Max"] - df["Glucose_Min"]

    # --- Critical Threshold Flags ---
    if 'HeartRate_Max' in df.columns:
        df["tachy_flag"] = (df["HeartRate_Max"] >= 120).astype(int)
        df["severe_tachy_flag"] = (df["HeartRate_Max"] >= 150).astype(int)
    if 'HeartRate_Min' in df.columns:
        df["brady_flag"] = (df["HeartRate_Min"] <= 50).astype(int)
    if 'SysBP_Min' in df.columns:
        df["hypotension_flag"] = (df["SysBP_Min"] < 90).astype(int)
    if 'MeanBP_Min' in df.columns:
        df["low_map_flag"] = (df["MeanBP_Min"] < 65).astype(int)
    if 'SpO2_Min' in df.columns:
        df["hypoxia_flag"] = (df["SpO2_Min"] < 92).astype(int)
        df["severe_hypoxia_flag"] = (df["SpO2_Min"] < 88).astype(int)
    if 'TempC_Max' in df.columns:
        df["fever_flag"] = (df["TempC_Max"] >= 38.3).astype(int)
    if 'TempC_Min' in df.columns:
        df["hypothermia_flag"] = (df["TempC_Min"] < 36).astype(int)
    if 'RespRate_Max' in df.columns:
        df["tachypnea_flag"] = (df["RespRate_Max"] >= 30).astype(int)
    if 'Glucose_Max' in df.columns:
        df["hyperglycemia_flag"] = (df["Glucose_Max"] >= 180).astype(int)
    if 'Glucose_Min' in df.columns:
        df["hypoglycemia_flag"] = (df["Glucose_Min"] <= 70).astype(int)

    # --- Derived Clinical Indices ---
    # Shock index (HR/SBP) - indicator of severity
    if 'HeartRate_Mean' in df.columns and 'SysBP_Mean' in df.columns:
        df["shock_index"] = df["HeartRate_Mean"] / (df["SysBP_Mean"].clip(lower=eps))

    # Pulse pressure
    if 'SysBP_Mean' in df.columns and 'DiasBP_Mean' in df.columns:
        df["pulse_pressure"] = df["SysBP_Mean"] - df["DiasBP_Mean"]

    # Mean arterial pressure estimate (if not already have MeanBP)
    if 'SysBP_Mean' in df.columns and 'DiasBP_Mean' in df.columns:
        df["map_calc"] = df["DiasBP_Mean"] + (df["SysBP_Mean"] - df["DiasBP_Mean"]) / 3

    # --- Instability Score (count of abnormal flags) ---
    flag_cols = [c for c in df.columns if c.endswith('_flag')]
    if flag_cols:
        df['instability_score'] = df[flag_cols].sum(axis=1)

    return df


def clean_min_bp_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean implausible blood pressure values"""
    lower_bounds = {
        "SysBP_Min": 40.0,
        "DiasBP_Min": 10.0,
        "MeanBP_Min": 30.0,
    }

    df = df.copy()
    for col, low in lower_bounds.items():
        if col not in df.columns:
            continue
        mask = (df[col] < low) & df[col].notna()
        if mask.sum() > 0:
            df.loc[mask, col] = np.nan
    return df


# =============================================================================
# SECTION 1: Load and Prepare Data
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: DATA LOADING")
print("=" * 70)

# Load raw data
DATA_DIR = BASE_DIR / "data" / "raw" / "MIMIC III dataset LOS"
train_df = pd.read_csv(DATA_DIR / "mimic_train_LOS.csv")
test_df = pd.read_csv(DATA_DIR / "mimic_test_LOS.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Target
y_full = train_df['LOS'].copy()

print(f"\nTarget (LOS) stats:")
print(f"  Mean: {y_full.mean():.2f}, Median: {y_full.median():.2f}")
print(f"  Min: {y_full.min():.2f}, Max: {y_full.max():.2f}")

# Age analysis removed - correlation with LOS was only 0.0117 (not useful)

# =============================================================================
# SECTION 2: Baseline Model (Current Approach)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: BASELINE MODEL (Current Approach)")
print("=" * 70)

# Columns to drop (leakage + useless)
drop_cols = [
    # IDs
    'icustay_id', 'subject_id', 'hadm_id',
    # Target and related
    'LOS', 'HOSPITAL_EXPIRE_FLAG',
    # Leakage (time-based)
    'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOB', 'DOD',
    # Currently dropping diagnosis (will add back as features)
    'ICD9_diagnosis', 'DIAGNOSIS',
    # Useless column
    'Diff',
]

# Prepare baseline features
X_baseline = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_baseline = clean_min_bp_outliers(X_baseline)

print(f"\nBaseline features: {X_baseline.shape[1]}")
print(f"Dropped 'Diff' column: ✓")

# Split
X_train_base, X_valid_base, y_train, y_valid = train_test_split(
    X_baseline, y_full, test_size=0.2, random_state=42
)

# Preprocessing
num_cols = X_train_base.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train_base.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}")

preprocessor_base = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols),
])

# Train baseline GB
print("\nTraining baseline Gradient Boosting...")
gb_base = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42,
)

pipe_base = Pipeline([
    ('preprocess', preprocessor_base),
    ('regressor', gb_base),
])

pipe_base.fit(X_train_base, y_train)
y_pred_base = np.maximum(pipe_base.predict(X_valid_base), 0)

baseline_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_base))
baseline_mae = mean_absolute_error(y_valid, y_pred_base)
baseline_r2 = r2_score(y_valid, y_pred_base)

print(f"\nBaseline Results (Diff dropped, no diagnosis features):")
print(f"  RMSE: {baseline_rmse:.4f}")
print(f"  MAE:  {baseline_mae:.4f}")
print(f"  R²:   {baseline_r2:.4f}")

# =============================================================================
# SECTION 3: With Diagnosis Features (Expanded)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: WITH DIAGNOSIS FEATURES (EXPANDED)")
print("=" * 70)

# Keep diagnosis columns for feature extraction
drop_cols_diag = [
    'icustay_id', 'subject_id', 'hadm_id',
    'LOS', 'HOSPITAL_EXPIRE_FLAG',
    'DISCHTIME', 'DEATHTIME', 'DOD', 'DOB', 'ADMITTIME',
    'Diff',
]

X_diag = train_df.drop(columns=[c for c in drop_cols_diag if c in train_df.columns])
X_diag = clean_min_bp_outliers(X_diag)
X_diag = add_diagnosis_features(X_diag)

# Now drop the original text columns (we've extracted features)
X_diag = X_diag.drop(columns=['ICD9_diagnosis', 'DIAGNOSIS', 'icd9_category'], errors='ignore')

print(f"\nWith diagnosis features: {X_diag.shape[1]} features")
print(f"Added: {X_diag.shape[1] - X_baseline.shape[1]} new features")

# Split (same random state for fair comparison)
X_train_diag, X_valid_diag, y_train_diag, y_valid_diag = train_test_split(
    X_diag, y_full, test_size=0.2, random_state=42
)

# New preprocessor
num_cols_diag = X_train_diag.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_diag = X_train_diag.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor_diag = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), num_cols_diag),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols_diag),
])

# Train
print("\nTraining GB with diagnosis features...")
gb_diag = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42,
)

pipe_diag = Pipeline([
    ('preprocess', preprocessor_diag),
    ('regressor', gb_diag),
])

pipe_diag.fit(X_train_diag, y_train_diag)
y_pred_diag = np.maximum(pipe_diag.predict(X_valid_diag), 0)

diag_rmse = np.sqrt(mean_squared_error(y_valid_diag, y_pred_diag))
diag_mae = mean_absolute_error(y_valid_diag, y_pred_diag)
diag_r2 = r2_score(y_valid_diag, y_pred_diag)

print(f"\nResults with diagnosis features:")
print(f"  RMSE: {diag_rmse:.4f} (Δ {baseline_rmse - diag_rmse:+.4f})")
print(f"  MAE:  {diag_mae:.4f}")
print(f"  R²:   {diag_r2:.4f}")

if diag_rmse < baseline_rmse:
    print("\n  ✓ Diagnosis features IMPROVED the score!")
else:
    print("\n  ✗ Diagnosis features did not help")

# =============================================================================
# SECTION 4: With Enhanced Feature Engineering
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: WITH ENHANCED FEATURE ENGINEERING")
print("=" * 70)

# Start from diagnosis features and add more
X_enhanced = train_df.drop(columns=[c for c in drop_cols_diag if c in train_df.columns])
X_enhanced = clean_min_bp_outliers(X_enhanced)
X_enhanced = add_diagnosis_features(X_enhanced)
X_enhanced = add_enhanced_features(X_enhanced)

# Drop original text columns
X_enhanced = X_enhanced.drop(columns=['ICD9_diagnosis', 'DIAGNOSIS', 'icd9_category'], errors='ignore')

print(f"\nWith enhanced features: {X_enhanced.shape[1]} features")
print(f"Added: {X_enhanced.shape[1] - X_diag.shape[1]} more features (clinical indices, flags)")

# Split
X_train_enh, X_valid_enh, y_train_enh, y_valid_enh = train_test_split(
    X_enhanced, y_full, test_size=0.2, random_state=42
)

# New preprocessor
num_cols_enh = X_train_enh.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_enh = X_train_enh.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor_enh = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), num_cols_enh),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols_enh),
])

# Train
print("\nTraining GB with enhanced features...")
gb_enh = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42,
)

pipe_enh = Pipeline([
    ('preprocess', preprocessor_enh),
    ('regressor', gb_enh),
])

pipe_enh.fit(X_train_enh, y_train_enh)
y_pred_enh = np.maximum(pipe_enh.predict(X_valid_enh), 0)

enh_rmse = np.sqrt(mean_squared_error(y_valid_enh, y_pred_enh))
enh_mae = mean_absolute_error(y_valid_enh, y_pred_enh)
enh_r2 = r2_score(y_valid_enh, y_pred_enh)

print(f"\nResults with enhanced features:")
print(f"  RMSE: {enh_rmse:.4f} (Δ {baseline_rmse - enh_rmse:+.4f})")
print(f"  MAE:  {enh_mae:.4f}")
print(f"  R²:   {enh_r2:.4f}")

if enh_rmse < baseline_rmse:
    print("\n  ✓ Enhanced features IMPROVED the score!")
else:
    print("\n  ✗ Enhanced features did not help")

# =============================================================================
# SECTION 5: Feature Importance Analysis
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: FEATURE IMPORTANCE (Enhanced Model)")
print("=" * 70)

# Get feature names
preprocessor_fitted = pipe_enh.named_steps['preprocess']
feature_names = num_cols_enh.copy()

if cat_cols_enh:
    ohe = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(cat_cols_enh).tolist()
    feature_names.extend(cat_feature_names)

importances = pipe_enh.named_steps['regressor'].feature_importances_

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 25 Most Important Features:")
print("-" * 55)
for i, (_, row) in enumerate(importance_df.head(25).iterrows()):
    bar = "█" * int(row['importance'] * 100)
    print(f"  {row['feature']:40} {row['importance']:.4f} {bar}")

# Show new features in top 25
new_feature_prefixes = ['diag_', 'icd9_', 'HR_range', 'SysBP_range', 'shock_', 'pulse_',
                        'instability', 'tachy', 'brady', 'hypo', 'fever', 'Temp_range',
                        'Glucose_range', 'tachypnea', 'hyperglycemia', 'severe', 'low_map']
new_in_top25 = [f for f in importance_df.head(25)['feature']
                if any(f.startswith(p) or p in f for p in new_feature_prefixes)]

if new_in_top25:
    print(f"\nNew engineered features in top 25: {len(new_in_top25)}")
    for f in new_in_top25:
        print(f"  - {f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)

results = [
    ("Baseline (Diff dropped)", baseline_rmse, baseline_mae, baseline_r2, X_baseline.shape[1]),
    ("+ Diagnosis features", diag_rmse, diag_mae, diag_r2, X_diag.shape[1]),
    ("+ Enhanced FE (all)", enh_rmse, enh_mae, enh_r2, X_enhanced.shape[1]),
]

# Sort by RMSE
results.sort(key=lambda x: x[1])

print("\nRanked by Validation RMSE (lower = better):")
print("-" * 70)
for i, (name, rmse, mae, r2, n_feat) in enumerate(results, 1):
    improvement = baseline_rmse - rmse
    pct = improvement / baseline_rmse * 100
    marker = "🏆" if i == 1 else "  "
    print(f"{marker} {i}. {name:30} RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.3f}  ({n_feat} feat)")

best_name, best_rmse, best_mae, best_r2, _ = results[0]

print(f"\n✓ Best approach: {best_name}")
print(f"  Improvement over baseline: {baseline_rmse - best_rmse:.4f} ({(baseline_rmse - best_rmse)/baseline_rmse*100:.1f}%)")

# =============================================================================
# NEXT STEPS
# =============================================================================
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

print("""
Based on the results:

1. If diagnosis features helped:
   - Use them in your final model
   - Consider adding more ICD9 code groups

2. If enhanced features helped:
   - Update los_prep.py with these new features
   - Use this preprocessing for XGBoost/NN

3. To create a submission with these features:
   - I can create an updated generate_los_submissions.py
   - Just let me know which feature set worked best

4. Still to try:
   - XGBoost (often better than GB for this)
   - Neural Network (benchmark method)
   - Hyperparameter tuning
""")

print("=" * 70)
print("ANALYSIS COMPLETE - Run and share output!")
print("=" * 70)
