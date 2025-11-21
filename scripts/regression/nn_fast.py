"""
FAST Neural Network for LOS
Uses good default parameters instead of extensive search
Should run in 5-10 minutes instead of hours
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "regression":
    BASE_DIR = SCRIPT_DIR.parents[1]
    sys.path.append(str(BASE_DIR / "src"))
    sys.path.append(str(SCRIPT_DIR))
else:
    BASE_DIR = Path.cwd()
    sys.path.append(str(BASE_DIR / "src"))

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("FAST NEURAL NETWORK FOR LOS")
print("Target: Beat 15.7 benchmark | Runtime: 5-10 minutes")
print("="*70)

from los_prep import prepare_data, load_raw_data

OUTPUT_DIR = BASE_DIR / "submissions" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data WITH feature engineering
print("\n[1/5] Loading data...")
X, y, X_test = prepare_data(
    leak_cols=[
        "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS",
        "DOB", "DEATHTIME", "DISCHTIME", "DOD",
        "LOS", "HOSPITAL_EXPIRE_FLAG"
    ],
    apply_fe=True,
)

# Split
print("\n[2/5] Creating split...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
print("\n[3/5] Preprocessing...")
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols),
])

# Train 3 different NN architectures (quick comparison)
print("\n[4/5] Training 3 neural network architectures...")
print("  (This is much faster than hyperparameter search)")

architectures = {
    'small': (100, 50),           # 2 layers: 100 → 50
    'medium': (200, 100, 50),     # 3 layers: 200 → 100 → 50
    'large': (300, 150, 75),      # 3 layers: 300 → 150 → 75
}

best_model = None
best_rmse = float('inf')
best_arch_name = None

for arch_name, hidden_layers in architectures.items():
    print(f"\n  Training {arch_name.upper()} architecture: {hidden_layers}")
    
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=0.001,              # L2 regularization
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,             # Enough for convergence
        early_stopping=True,      # Stop if no improvement
        validation_fraction=0.1,
        n_iter_no_change=20,      # Patience
        random_state=42,
        verbose=False,
    )
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', mlp),
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Validate
    y_valid_pred = np.maximum(pipeline.predict(X_valid), 0)
    rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    
    print(f"    Validation RMSE: {rmse:.4f}")
    
    # Track best
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = pipeline
        best_arch_name = arch_name

print(f"\n  🏆 Best architecture: {best_arch_name.upper()}")
print(f"     RMSE: {best_rmse:.4f}")

# Final validation metrics
print("\n[5/5] Final validation...")
y_valid_pred = np.maximum(best_model.predict(X_valid), 0)
rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
mae = mean_absolute_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)

print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")

print(f"\n  vs Your GB:      21.3")
print(f"  vs NN Benchmark: 15.7")

# Predict on test
test_pred = np.maximum(best_model.predict(X_test), 0)

print(f"\nTest predictions:")
print(f"  Min:    {test_pred.min():.2f} days")
print(f"  Max:    {test_pred.max():.2f} days")
print(f"  Mean:   {test_pred.mean():.2f} days")
print(f"  Median: {np.median(test_pred):.2f} days")

# Get IDs
try:
    _, test_df = load_raw_data()
    id_col = [c for c in test_df.columns if c.lower() == 'icustay_id'][0]
    icustay_ids = test_df[id_col]
except:
    icustay_ids = range(len(test_pred))

submission = pd.DataFrame({
    'icustay_id': icustay_ids,
    'LOS': test_pred
})

output_file = OUTPUT_DIR / "submission_los_nn_fast.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ FAST NN SUBMISSION CREATED!")
print("="*70)
print(f"\nFile: {output_file}")
print(f"Architecture: {best_arch_name.upper()} - {architectures[best_arch_name]}")
print(f"Validation RMSE: {rmse:.4f}")

if rmse < 21.3:
    improvement = 21.3 - rmse
    print(f"\n🎉 IMPROVED over GB! -{improvement:.2f} points")
    
    if rmse < 15.7:
        print(f"🏆 BEAT NN BENCHMARK! ({rmse:.4f} < 15.7)")
        print("   This should score very well!")
    elif rmse < 18:
        print(f"📈 Close to benchmark! Gap: {rmse - 15.7:.2f} points")
        print("   Definitely worth submitting")
    else:
        print(f"📊 Better than GB, gap to benchmark: {rmse - 15.7:.2f} points")
        print("   Still worth trying on leaderboard")
else:
    print(f"\n⚠️  Similar to GB ({rmse:.4f} vs 21.3)")
    print("   But NN can perform differently on test - try it!")

print("\n💡 If this doesn't beat 15.7, try:")
print("   1. xgboost_los_submission.py (often best)")
print("   2. Deeper NN with more training (original nn_los_submission.py)")
print("="*70)