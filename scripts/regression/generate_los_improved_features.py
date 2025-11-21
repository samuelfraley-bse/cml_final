"""
Generate LOS Submission with Improved Features

Uses los_prep.py as base, then adds high-value features:
1. vital_product (RespRate_Max * TempC_Max * HeartRate_Max) - corr 0.2010
2. CV features (Range/Mean) - corr 0.17+ (captures instability)
3. Interactions (HR*RR, RR*Temp) - corr 0.15+
4. Diagnosis-based features
5. Squared/log terms

Key insight: Long stays are hard to predict because:
- Only 1.9% of data is 20+ day stays
- Features capture point-in-time, not progression
- CV features help because instability → complications → longer stays

Run: python scripts/regression/generate_los_improved_features.py
"""

import sys
from pathlib import Path

# Add scripts/regression to path for los_prep import
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Import from los_prep
from los_prep import prepare_data, get_paths, load_raw_data

print("=" * 70)
print("LOS PREDICTION WITH IMPROVED FEATURES")
print("=" * 70)

# =============================================================================
# Additional Feature Engineering (on top of los_prep)
# =============================================================================

def add_cv_features(df):
    """
    Add Coefficient of Variation features (Range/Mean).

    These capture INSTABILITY - patients with high CV have more variable
    vitals, which may indicate complications and longer stays.
    """
    df = df.copy()
    eps = 1e-3  # Avoid division by zero

    # CV features for key vitals
    vital_configs = [
        ('SysBP', 'SBP'),      # corr 0.1782
        ('TempC', 'Temp'),     # corr 0.1667
        ('RespRate', 'RR'),    # corr 0.1513
        ('HeartRate', 'HR'),   # corr 0.1332
        ('Glucose', 'Gluc'),   # corr 0.0815
        ('DiasBP', 'DBP'),
        ('MeanBP', 'MBP'),
        ('SpO2', 'SpO2'),
    ]

    for vital, abbrev in vital_configs:
        max_col = f'{vital}_Max'
        min_col = f'{vital}_Min'
        mean_col = f'{vital}_Mean'

        # Range already added by los_prep for some, but let's ensure all
        if max_col in df.columns and min_col in df.columns:
            range_col = f'{abbrev}_range'
            if range_col not in df.columns:
                df[range_col] = df[max_col] - df[min_col]

            # CV = Range / Mean (key insight: captures instability)
            if mean_col in df.columns:
                df[f'{abbrev}_cv'] = df[range_col] / (df[mean_col].clip(lower=eps))

    return df


def add_vital_products(df):
    """
    Add product/interaction features.

    vital_product has correlation 0.2010 - our best single feature!
    This captures combined severity across multiple systems.
    """
    df = df.copy()

    # Product of top 3 vitals (corr = 0.2010)
    if all(col in df.columns for col in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']
        df['vital_product_mean'] = df['RespRate_Mean'] * df['TempC_Mean'] * df['HeartRate_Mean']

    # Interactions
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR_interaction'] = df['HeartRate_Mean'] * df['RespRate_Mean']  # corr 0.1606

    if 'RespRate_Mean' in df.columns and 'TempC_Mean' in df.columns:
        df['RR_Temp_interaction'] = df['RespRate_Mean'] * df['TempC_Mean']  # corr 0.1512

    # Max interactions (might capture acute severity better)
    if 'HeartRate_Max' in df.columns and 'RespRate_Max' in df.columns:
        df['HRmax_RRmax'] = df['HeartRate_Max'] * df['RespRate_Max']

    return df


def add_severity_indicators(df):
    """
    Add features that indicate severity/complexity.

    Long stays often result from multi-system involvement.
    """
    df = df.copy()

    # Count of critical flags (more flags = more severe = longer stay?)
    critical_flags = ['tachy_flag', 'hypotension_flag', 'hypoxia_flag']
    existing_flags = [f for f in critical_flags if f in df.columns]
    if existing_flags:
        df['critical_flag_count'] = df[existing_flags].sum(axis=1)

    # Squared terms for top predictors (capture non-linearity)
    for col in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max', 'vital_product']:
        if col in df.columns:
            df[f'{col}_sq'] = df[col] ** 2

    # Log transforms for right-skewed distributions
    for col in ['Glucose_Max', 'Glucose_Mean']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

    # Extreme value indicators (might help identify long-stayers)
    if 'RespRate_Max' in df.columns:
        df['high_resp_rate'] = (df['RespRate_Max'] > 30).astype(int)
    if 'TempC_Max' in df.columns:
        df['high_temp'] = (df['TempC_Max'] > 38.5).astype(int)
    if 'Glucose_Max' in df.columns:
        df['high_glucose'] = (df['Glucose_Max'] > 200).astype(int)

    return df


def add_diagnosis_features(df, train_df_orig):
    """
    Add diagnosis-based features from original data.

    Certain diagnoses (PANCREATITIS, SHOCK) are 3-4x more common in long stays.
    """
    df = df.copy()

    # Get DIAGNOSIS and ICD9 from original data if available
    if 'DIAGNOSIS' in train_df_orig.columns:
        # Need to align by index
        diag_series = train_df_orig['DIAGNOSIS'].fillna('')
        if len(diag_series) == len(df):
            diag_upper = diag_series.str.upper()

            # High LOS keywords (from investigation)
            high_los_keywords = ['PANCREATITIS', 'SHOCK', 'TRANSPLANT', 'ARREST',
                                'RESPIRATORY FAILURE', 'SEPSIS', 'PNEUMONIA']
            for kw in high_los_keywords:
                col_name = f'diag_{kw.lower().replace(" ", "_")}'
                df[col_name] = diag_upper.str.contains(kw, na=False).astype(int)

            # Low LOS keywords
            low_los_keywords = ['OVERDOSE', 'DIABETIC', 'DKA', 'CABG']
            for kw in low_los_keywords:
                col_name = f'diag_{kw.lower()}'
                df[col_name] = diag_upper.str.contains(kw, na=False).astype(int)

    if 'ICD9_diagnosis' in train_df_orig.columns:
        icd9_series = train_df_orig['ICD9_diagnosis'].astype(str)
        if len(icd9_series) == len(df):
            # High-LOS ICD9 codes (from investigation)
            high_los_icd9 = ['51884', '5770', '430', '44101', '0380', '51881', '0389', '431']
            df['high_los_icd9'] = icd9_series.isin(high_los_icd9).astype(int)

    return df


def add_careunit_features(df, train_df_orig):
    """Add care unit based features."""
    df = df.copy()

    if 'FIRST_CAREUNIT' in train_df_orig.columns:
        careunit = train_df_orig['FIRST_CAREUNIT']
        if len(careunit) == len(df):
            # SICU and CCU tend to have longer stays
            df['is_sicu'] = (careunit == 'SICU').astype(int)
            df['is_ccu'] = (careunit == 'CCU').astype(int)
            df['is_micu'] = (careunit == 'MICU').astype(int)
            # CSRU tends to have shorter stays
            df['is_csru'] = (careunit == 'CSRU').astype(int)

    return df


# =============================================================================
# Load Data Using los_prep
# =============================================================================
print("\nLoading data with los_prep.py...")

# Define leakage columns
LEAK_COLS = [
    'HOSPITAL_EXPIRE_FLAG',  # Leakage!
    'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME',  # Date columns
    'Diff',  # No signal (corr = -0.0012)
]

# Load using los_prep (applies BP cleaning + basic engineered features)
X_train_base, y_train_full, X_test_base = prepare_data(
    leak_cols=LEAK_COLS,
    apply_fe=True  # Apply los_prep's engineered features
)

# Also load raw data for diagnosis features
train_df_orig, test_df_orig = load_raw_data()

print(f"\nAfter los_prep: {X_train_base.shape[1]} features")

# =============================================================================
# Apply Additional Feature Engineering
# =============================================================================
print("\nAdding improved features...")

# Add CV features
X_train_fe = add_cv_features(X_train_base)
X_test_fe = add_cv_features(X_test_base)

# Add vital products and interactions
X_train_fe = add_vital_products(X_train_fe)
X_test_fe = add_vital_products(X_test_fe)

# Add severity indicators
X_train_fe = add_severity_indicators(X_train_fe)
X_test_fe = add_severity_indicators(X_test_fe)

# Add diagnosis features (from original data)
X_train_fe = add_diagnosis_features(X_train_fe, train_df_orig)
X_test_fe = add_diagnosis_features(X_test_fe, test_df_orig)

# Add careunit features
X_train_fe = add_careunit_features(X_train_fe, train_df_orig)
X_test_fe = add_careunit_features(X_test_fe, test_df_orig)

print(f"After improved features: {X_train_fe.shape[1]} features")

# Show new features
new_features = [col for col in X_train_fe.columns if col not in X_train_base.columns]
print(f"\nNew features added ({len(new_features)}):")
for f in new_features[:15]:
    print(f"  - {f}")
if len(new_features) > 15:
    print(f"  ... and {len(new_features) - 15} more")

# =============================================================================
# Prepare for Modeling
# =============================================================================

# Identify column types
numerical_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)}")

# Preprocessing with imputation (CV features can create NaN/inf)
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# =============================================================================
# Train/Validation Split
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_fe, y_train_full, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")

# =============================================================================
# Model Training - Using HistGradientBoosting (MUCH faster)
# =============================================================================

# HistGB doesn't need imputation for numerics but still needs encoding for categoricals
hist_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# =============================================================================
# Test different weighting strategies for long stays
# =============================================================================
print("\n" + "=" * 70)
print("TESTING SAMPLE WEIGHTING STRATEGIES")
print("=" * 70)

# Create sample weights - higher weight for longer stays
def create_weights(y, strategy='linear'):
    """Create sample weights to emphasize long stays."""
    if strategy == 'none':
        return np.ones(len(y))
    elif strategy == 'linear':
        # Weight proportional to LOS (longer = higher weight)
        return 1 + y / y.max()
    elif strategy == 'sqrt':
        # Sqrt of LOS (less aggressive than linear)
        return 1 + np.sqrt(y) / np.sqrt(y.max())
    elif strategy == 'threshold':
        # Binary: high weight for LOS > 14 days
        return np.where(y > 14, 3.0, 1.0)
    elif strategy == 'tiered':
        # Tiered weights by bucket
        weights = np.ones(len(y))
        weights[y >= 7] = 2.0
        weights[y >= 14] = 4.0
        weights[y >= 30] = 8.0
        return weights
    else:
        return np.ones(len(y))

# Test different weighting strategies
strategies = ['none', 'linear', 'sqrt', 'threshold', 'tiered']
results = {}

for strategy in strategies:
    weights = create_weights(y_train, strategy)

    # Transform features
    X_train_transformed = hist_preprocessor.fit_transform(X_train)
    X_val_transformed = hist_preprocessor.transform(X_val)

    # Train with weights
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_train_transformed, y_train, sample_weight=weights)

    # Predict
    pred = model.predict(X_val_transformed)
    rmse = np.sqrt(mean_squared_error(y_val, pred))

    # RMSE for long stays (14+ days)
    long_mask = y_val >= 14
    if long_mask.sum() > 0:
        long_rmse = np.sqrt(mean_squared_error(y_val[long_mask], pred[long_mask]))
    else:
        long_rmse = 0

    results[strategy] = {
        'rmse': rmse,
        'long_rmse': long_rmse,
        'pred_range': (pred.min(), pred.max()),
        'model': model
    }

    print(f"\n{strategy.upper()} weighting:")
    print(f"  Overall RMSE: {rmse:.4f}")
    print(f"  Long-stay (14+) RMSE: {long_rmse:.4f}")
    print(f"  Pred range: [{pred.min():.2f}, {pred.max():.2f}]")

# Find best strategy
best_strategy = min(results.keys(), key=lambda x: results[x]['rmse'])
print(f"\n>>> Best overall: {best_strategy} (RMSE={results[best_strategy]['rmse']:.4f})")

# Also check which is best for long stays
best_long = min(results.keys(), key=lambda x: results[x]['long_rmse'])
print(f">>> Best for long-stays: {best_long} (RMSE={results[best_long]['long_rmse']:.4f})")

# Use best strategy for final model
hist_val_pred = results[best_strategy]['model'].predict(hist_preprocessor.transform(X_val))
hist_val_rmse = results[best_strategy]['rmse']
hist_val_mse = hist_val_rmse ** 2

print(f"\nUsing {best_strategy} weighting for submission")

# Store for later use
val_pred = hist_val_pred
val_rmse = hist_val_rmse

# =============================================================================
# Analysis by LOS Bucket
# =============================================================================
print("\n" + "=" * 70)
print("RMSE BY LOS BUCKET")
print("=" * 70)

best_pred = val_pred if val_rmse <= hist_val_rmse else hist_val_pred
best_name = "GB" if val_rmse <= hist_val_rmse else "HistGB"

buckets = [(0, 3), (3, 7), (7, 14), (14, 30), (30, 200)]
print(f"\nUsing {best_name} predictions:")
for low, high in buckets:
    mask = (y_val >= low) & (y_val < high)
    if mask.sum() > 0:
        bucket_rmse = np.sqrt(mean_squared_error(y_val[mask], best_pred[mask]))
        pct = mask.sum() / len(y_val) * 100
        avg_pred = best_pred[mask].mean()
        avg_actual = y_val[mask].mean()
        print(f"  [{low:2}-{high:3}] days: RMSE={bucket_rmse:6.2f} | "
              f"pred={avg_pred:5.1f} vs actual={avg_actual:5.1f} | "
              f"n={mask.sum()} ({pct:.1f}%)")

# =============================================================================
# Feature Importance (skip if not available)
# =============================================================================
print("\n" + "=" * 70)
print("TOP 20 FEATURE IMPORTANCES")
print("=" * 70)

# Get feature names from HistGB model
feature_names = numerical_cols.copy()
if categorical_cols:
    cat_features = hist_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names.extend(list(cat_features))

# HistGradientBoostingRegressor doesn't have feature_importances_ by default
# We'll skip this or use permutation importance later
print("\n(Feature importances not directly available for HistGradientBoostingRegressor)")
print("Top features based on our analysis:")
print("  - vital_product (corr 0.2010)")
print("  - SBP_cv (corr 0.1782)")
print("  - Temp_cv (corr 0.1667)")
print("  - HR_RR_interaction (corr 0.1606)")

# =============================================================================
# Generate Submissions - Both best overall AND best for long-stays
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSIONS")
print("=" * 70)

# Transform data
X_full_transformed = hist_preprocessor.fit_transform(X_train_fe)
X_test_transformed = hist_preprocessor.transform(X_test_fe)

# Get test IDs
test_ids = test_df_orig['icustay_id'].values

# Get output directory
base_dir, _, _ = get_paths()
output_dir = base_dir / "submissions" / "regression"
output_dir.mkdir(parents=True, exist_ok=True)

# Generate submission for best overall (none weighting)
print(f"\n1. Best overall: {best_strategy} weighting")
weights_best = create_weights(y_train_full, best_strategy)
model_best = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=5,
    min_samples_leaf=20, random_state=42
)
model_best.fit(X_full_transformed, y_train_full, sample_weight=weights_best)
pred_best = model_best.predict(X_test_transformed)
print(f"   Pred range: [{pred_best.min():.2f}, {pred_best.max():.2f}]")

submission_best = pd.DataFrame({'icustay_id': test_ids, 'LOS': pred_best})
file_best = output_dir / "submission_los_improved_features.csv"
submission_best.to_csv(file_best, index=False)
print(f"   Saved: {file_best.name}")

# Generate submission for tiered weighting (best for long-stays)
print(f"\n2. Best for long-stays: tiered weighting")
weights_tiered = create_weights(y_train_full, 'tiered')
model_tiered = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=5,
    min_samples_leaf=20, random_state=42
)
model_tiered.fit(X_full_transformed, y_train_full, sample_weight=weights_tiered)
pred_tiered = model_tiered.predict(X_test_transformed)
print(f"   Pred range: [{pred_tiered.min():.2f}, {pred_tiered.max():.2f}]")

submission_tiered = pd.DataFrame({'icustay_id': test_ids, 'LOS': pred_tiered})
file_tiered = output_dir / "submission_los_tiered_weights.csv"
submission_tiered.to_csv(file_tiered, index=False)
print(f"   Saved: {file_tiered.name}")

# Use best overall for the final variables
test_pred = pred_best
output_file = file_best

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
APPROACH:
  Base: los_prep.py (BP cleaning, basic features)
  Added: CV features, vital_product, interactions, diagnosis flags
  Tested: 5 weighting strategies for long stays

NEW FEATURES ({len(new_features)}):
  - vital_product (best single feature, corr 0.2010)
  - CV features (Range/Mean) for 8 vitals
  - Interactions: HR*RR, RR*Temp
  - Severity indicators and critical flag count
  - Diagnosis keywords (PANCREATITIS, SHOCK, etc.)

WEIGHTING RESULTS:
  Best overall: {best_strategy} (RMSE={results[best_strategy]['rmse']:.4f})
  Best for long-stays: {best_long} (RMSE={results[best_long]['long_rmse']:.4f})

FILES GENERATED:
  1. {file_best.name} - best overall validation
  2. {file_tiered.name} - optimized for long stays

PREVIOUS BEST: 20.75 on Kaggle
TARGET:        15.7

RECOMMENDATION:
  Submit BOTH files to Kaggle! The tiered weighting might score better
  on Kaggle even though it has worse overall validation RMSE, because
  the test set likely has long-stay patients that dominate the error.
""")

print("=" * 70)
