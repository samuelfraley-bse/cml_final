"""
Analyze Residuals - Where is the model going wrong?

Compare predictions to actual values to find systematic errors.

Run: python scripts/regression/analyze_residuals.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from los_prep import prepare_data, get_paths, load_raw_data

print("=" * 70)
print("RESIDUAL ANALYSIS")
print("=" * 70)

# =============================================================================
# Load and prepare data (same as 20.0 submission)
# =============================================================================

def add_cv_features(df):
    df = df.copy()
    eps = 1e-3
    vitals = [('SysBP', 'SBP'), ('TempC', 'Temp'), ('RespRate', 'RR'),
              ('HeartRate', 'HR'), ('Glucose', 'Gluc'), ('DiasBP', 'DBP'),
              ('MeanBP', 'MBP'), ('SpO2', 'SpO2')]
    for vital, abbrev in vitals:
        max_col, min_col, mean_col = f'{vital}_Max', f'{vital}_Min', f'{vital}_Mean'
        if max_col in df.columns and min_col in df.columns:
            range_col = f'{abbrev}_range'
            if range_col not in df.columns:
                df[range_col] = df[max_col] - df[min_col]
            if mean_col in df.columns:
                df[f'{abbrev}_cv'] = df[range_col] / df[mean_col].clip(lower=eps)
    return df

def add_vital_products(df):
    df = df.copy()
    if all(c in df.columns for c in ['RespRate_Max', 'TempC_Max', 'HeartRate_Max']):
        df['vital_product'] = df['RespRate_Max'] * df['TempC_Max'] * df['HeartRate_Max']
    if 'HeartRate_Mean' in df.columns and 'RespRate_Mean' in df.columns:
        df['HR_RR_interaction'] = df['HeartRate_Mean'] * df['RespRate_Mean']
    return df

print("\nLoading data...")

LEAK_COLS = ['HOSPITAL_EXPIRE_FLAG', 'DOB', 'DOD', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'Diff']
X_train_base, y_train_full, X_test_base = prepare_data(leak_cols=LEAK_COLS, apply_fe=True)
train_df_orig, test_df_orig = load_raw_data()

# Add features
X_train_fe = add_cv_features(X_train_base)
X_train_fe = add_vital_products(X_train_fe)

# Prepare preprocessor
numerical_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Use same split as before
X_train, X_val, y_train, y_val = train_test_split(
    X_train_fe, y_train_full, test_size=0.2, random_state=42
)

# Also get original data for analysis
_, val_orig, _, _ = train_test_split(
    train_df_orig, y_train_full, test_size=0.2, random_state=42
)

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)

# Train model
model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=5,
    min_samples_leaf=20, random_state=42
)
model.fit(X_train_t, y_train)
pred = model.predict(X_val_t)

print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_val, pred)):.4f}")

# =============================================================================
# RESIDUAL ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS")
print("=" * 70)

residuals = y_val - pred
abs_residuals = np.abs(residuals)
squared_residuals = residuals ** 2

# Create analysis dataframe
analysis_df = pd.DataFrame({
    'actual': y_val.values,
    'predicted': pred,
    'residual': residuals.values,
    'abs_residual': abs_residuals.values,
    'squared_residual': squared_residuals.values,
})

# Add original columns for analysis
for col in ['DIAGNOSIS', 'ICD9_diagnosis', 'FIRST_CAREUNIT', 'ADMISSION_TYPE', 'GENDER', 'ETHNICITY']:
    if col in val_orig.columns:
        analysis_df[col] = val_orig[col].values

# Add some features
for col in ['HeartRate_Max', 'RespRate_Max', 'TempC_Max', 'Glucose_Max', 'SpO2_Min']:
    if col in X_val.columns:
        analysis_df[col] = X_val[col].values

# =============================================================================
# 1. ERROR BY LOS BUCKET
# =============================================================================
print("\n1. ERROR BY LOS BUCKET")
print("-" * 50)

buckets = [(0, 3), (3, 7), (7, 14), (14, 30), (30, 200)]
for low, high in buckets:
    mask = (analysis_df['actual'] >= low) & (analysis_df['actual'] < high)
    if mask.sum() > 0:
        subset = analysis_df[mask]
        rmse = np.sqrt(subset['squared_residual'].mean())
        mae = subset['abs_residual'].mean()
        mean_residual = subset['residual'].mean()
        n = len(subset)
        pct = n / len(analysis_df) * 100
        contribution = subset['squared_residual'].sum() / analysis_df['squared_residual'].sum() * 100

        print(f"\n[{low:2}-{high:3}] days (n={n}, {pct:.1f}% of data)")
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        print(f"  Mean residual: {mean_residual:+.2f} (negative = under-predicting)")
        print(f"  Contribution to total MSE: {contribution:.1f}%")
        print(f"  Avg pred: {subset['predicted'].mean():.1f} vs actual: {subset['actual'].mean():.1f}")

# =============================================================================
# 2. WORST PREDICTIONS
# =============================================================================
print("\n\n2. TOP 20 WORST PREDICTIONS")
print("-" * 50)

worst = analysis_df.nlargest(20, 'squared_residual')
print("\nActual | Predicted | Error | Diagnosis")
print("-" * 70)
for _, row in worst.iterrows():
    diag = row.get('DIAGNOSIS', 'N/A')
    if pd.isna(diag):
        diag = 'N/A'
    else:
        diag = str(diag)[:40]
    print(f"{row['actual']:6.1f} | {row['predicted']:9.1f} | {row['residual']:+6.1f} | {diag}")

# =============================================================================
# 3. ERROR BY CARE UNIT
# =============================================================================
print("\n\n3. ERROR BY CARE UNIT")
print("-" * 50)

if 'FIRST_CAREUNIT' in analysis_df.columns:
    for cu in analysis_df['FIRST_CAREUNIT'].unique():
        mask = analysis_df['FIRST_CAREUNIT'] == cu
        subset = analysis_df[mask]
        rmse = np.sqrt(subset['squared_residual'].mean())
        mean_res = subset['residual'].mean()
        n = len(subset)
        print(f"{cu:6}: RMSE={rmse:.2f}, Mean residual={mean_res:+.2f}, n={n}")

# =============================================================================
# 4. ERROR BY ADMISSION TYPE
# =============================================================================
print("\n\n4. ERROR BY ADMISSION TYPE")
print("-" * 50)

if 'ADMISSION_TYPE' in analysis_df.columns:
    for at in analysis_df['ADMISSION_TYPE'].unique():
        mask = analysis_df['ADMISSION_TYPE'] == at
        subset = analysis_df[mask]
        rmse = np.sqrt(subset['squared_residual'].mean())
        mean_res = subset['residual'].mean()
        n = len(subset)
        print(f"{at:12}: RMSE={rmse:.2f}, Mean residual={mean_res:+.2f}, n={n}")

# =============================================================================
# 5. DIRECTION OF ERROR
# =============================================================================
print("\n\n5. DIRECTION OF ERRORS")
print("-" * 50)

under_pred = (residuals > 0).sum()
over_pred = (residuals < 0).sum()
exact = (residuals == 0).sum()

print(f"Under-predictions (actual > pred): {under_pred} ({under_pred/len(residuals)*100:.1f}%)")
print(f"Over-predictions (actual < pred):  {over_pred} ({over_pred/len(residuals)*100:.1f}%)")

# By magnitude
large_under = (residuals > 10).sum()
large_over = (residuals < -10).sum()
print(f"\nLarge under-predictions (error > 10 days): {large_under}")
print(f"Large over-predictions (error < -10 days): {large_over}")

# =============================================================================
# 6. DIAGNOSIS PATTERNS IN WORST PREDICTIONS
# =============================================================================
print("\n\n6. COMMON DIAGNOSES IN WORST 10% PREDICTIONS")
print("-" * 50)

worst_10pct = analysis_df.nlargest(int(len(analysis_df) * 0.1), 'squared_residual')

if 'DIAGNOSIS' in worst_10pct.columns:
    # Extract keywords
    keywords = ['SEPSIS', 'PNEUMONIA', 'RESPIRATORY', 'CARDIAC', 'TRANSPLANT',
                'STROKE', 'PANCREATITIS', 'SHOCK', 'TRAUMA', 'FAILURE']

    print("\nKeyword frequency in worst 10% vs overall:")
    for kw in keywords:
        worst_pct = worst_10pct['DIAGNOSIS'].fillna('').str.upper().str.contains(kw).mean() * 100
        overall_pct = analysis_df['DIAGNOSIS'].fillna('').str.upper().str.contains(kw).mean() * 100
        if worst_pct > 0 or overall_pct > 0:
            ratio = worst_pct / overall_pct if overall_pct > 0 else 0
            print(f"  {kw:15}: worst={worst_pct:.1f}%, overall={overall_pct:.1f}%, ratio={ratio:.1f}x")

# =============================================================================
# 7. MSE DECOMPOSITION
# =============================================================================
print("\n\n7. MSE DECOMPOSITION BY LOS BUCKET")
print("-" * 50)

total_mse = analysis_df['squared_residual'].mean()
print(f"\nTotal MSE: {total_mse:.2f}")
print("\nBreakdown by bucket:")

for low, high in buckets:
    mask = (analysis_df['actual'] >= low) & (analysis_df['actual'] < high)
    if mask.sum() > 0:
        subset = analysis_df[mask]
        bucket_mse = subset['squared_residual'].mean()
        weight = len(subset) / len(analysis_df)
        contribution = bucket_mse * weight
        pct_contribution = contribution / total_mse * 100

        print(f"[{low:2}-{high:3}] days: MSE={bucket_mse:6.1f} × weight={weight:.3f} = {contribution:5.2f} ({pct_contribution:.1f}%)")

# =============================================================================
# 8. FEATURE VALUES IN WORST PREDICTIONS
# =============================================================================
print("\n\n8. FEATURE VALUES: WORST 10% vs OVERALL")
print("-" * 50)

features_to_check = ['HeartRate_Max', 'RespRate_Max', 'TempC_Max', 'Glucose_Max', 'SpO2_Min']

for feat in features_to_check:
    if feat in analysis_df.columns:
        worst_mean = worst_10pct[feat].mean()
        overall_mean = analysis_df[feat].mean()
        print(f"{feat:15}: worst={worst_mean:.1f}, overall={overall_mean:.1f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Calculate key stats
total_mse = analysis_df['squared_residual'].sum()
long_stay_mask = analysis_df['actual'] >= 14
long_stay_mse = analysis_df[long_stay_mask]['squared_residual'].sum()
long_stay_pct = long_stay_mse / total_mse * 100

print(f"""
KEY FINDINGS:
  Total validation MSE: {analysis_df['squared_residual'].mean():.2f}

  Patients with LOS >= 14 days:
    - {long_stay_mask.sum()} patients ({long_stay_mask.sum()/len(analysis_df)*100:.1f}% of data)
    - Contribute {long_stay_pct:.1f}% of total MSE

  The problem is clear: {long_stay_pct:.0f}% of our error comes from {long_stay_mask.sum()/len(analysis_df)*100:.0f}%
  of patients. These are the long-stay patients we're under-predicting.
""")

print("=" * 70)
