"""
Neural Network for LOS Regression
Target: Beat 15.7 benchmark (the "nn demanding" score)

Using sklearn's MLPRegressor - a simple feed-forward neural network
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

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("NEURAL NETWORK FOR LOS - BEAT 15.7 BENCHMARK")
print("="*70)
print("\nThe 'nn demanding' benchmark = Neural Network at 15.7")
print("Let's try to match or beat it!")

from los_prep import prepare_data, load_raw_data

OUTPUT_DIR = BASE_DIR / "submissions" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data WITH feature engineering
print("\n[1/6] Loading data with feature engineering...")
X, y, X_test = prepare_data(
    leak_cols=[
        "ADMITTIME", "ICD9_diagnosis", "DIAGNOSIS",
        "DOB", "DEATHTIME", "DISCHTIME", "DOD",
        "LOS", "HOSPITAL_EXPIRE_FLAG"
    ],
    apply_fe=True,
)

print(f"  LOS stats: mean={y.mean():.2f}, std={y.std():.2f}, median={y.median():.2f}")

# Split
print("\n[2/6] Creating split...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing (CRITICAL for neural networks - need scaled features!)
print("\n[3/6] Preprocessing (scaling is critical for NN)...")
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),  # Very important for NN!
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols),
])

# Neural Network with hyperparameter search
print("\n[4/6] Training neural network with hyperparameter search...")
print("  This will take 10-15 minutes...")

mlp = MLPRegressor(
    random_state=42,
    early_stopping=True,  # Stop if validation loss doesn't improve
    validation_fraction=0.1,
    max_iter=1000,
)

# Parameter grid
param_distributions = {
    'regressor__hidden_layer_sizes': [
        (100,), (200,), (100, 50), (200, 100), (300, 150),
        (100, 100), (200, 100, 50), (150, 100, 50),
    ],
    'regressor__activation': ['relu', 'tanh'],
    'regressor__alpha': [0.0001, 0.001, 0.01],  # L2 regularization
    'regressor__learning_rate': ['constant', 'adaptive'],
    'regressor__learning_rate_init': [0.001, 0.01],
    'regressor__batch_size': [32, 64, 128],
}

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', mlp),
])

# Randomized search (faster than grid)
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=30,  # Try 30 different combinations
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

random_search.fit(X_train, y_train)

print(f"\n  Best CV score: {-random_search.best_score_:.4f} (RMSE)")
print(f"  Best params:")
for param, value in random_search.best_params_.items():
    print(f"    {param}: {value}")

# Validate
print("\n[5/6] Validating...")
best_model = random_search.best_estimator_
y_valid_pred = np.maximum(best_model.predict(X_valid), 0)

rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
mae = mean_absolute_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)

print(f"  Validation RMSE: {rmse:.4f}")
print(f"  Validation MAE:  {mae:.4f}")
print(f"  Validation R²:   {r2:.4f}")

print(f"\n  vs Your GB:      21.3")
print(f"  vs NN Benchmark: 15.7")

# Predict
print("\n[6/6] Generating predictions...")
test_pred = np.maximum(best_model.predict(X_test), 0)

print(f"\nPrediction stats:")
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

output_file = OUTPUT_DIR / "submission_los_nn.csv"
submission.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✓ NEURAL NETWORK SUBMISSION CREATED!")
print("="*70)
print(f"\nFile: {output_file}")
print(f"Validation RMSE: {rmse:.4f}")

if rmse < 21.3:
    improvement = 21.3 - rmse
    print(f"\n🎉 IMPROVED over GB! -{improvement:.2f} points")
    
    if rmse < 15.7:
        print(f"🏆 BEAT NN BENCHMARK! ({rmse:.4f} < 15.7)")
        print("   This should score well on leaderboard!")
    elif rmse < 18:
        print(f"📈 Close to benchmark! Gap: {rmse - 15.7:.2f} points")
        print("   Worth submitting to see test performance")
    else:
        print(f"📊 Better than GB but not at benchmark yet")
        print(f"   Gap: {rmse - 15.7:.2f} points")
else:
    print(f"\n⚠️  Similar to GB ({rmse:.4f} vs 21.3)")
    print("   NN might still perform better on test set - try it!")

print("\nNote: Validation != Test performance")
print("Neural networks can sometimes perform better on test!")
print("="*70)