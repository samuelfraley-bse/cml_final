# LOS Regression Scripts

Scripts for predicting ICU **Length of Stay** (continuous regression task).

## Files

1. **los_prep.py** - Data loading and preprocessing module
2. **check_setup_los.py** - Verify setup before running
3. **quickstart_los.py** - Quick single GB submission (~2 min)
4. **generate_los_submissions.py** - Multiple models (RF, GB, ET, Ensemble) (~5 min)

## Key Differences from Classification

| Aspect | Classification (HEF) | Regression (LOS) |
|--------|---------------------|------------------|
| **Target** | HOSPITAL_EXPIRE_FLAG (0/1) | LOS (continuous, days) |
| **Models** | `*Classifier` | `*Regressor` |
| **Predictions** | `.predict_proba()[:, 1]` | `.predict()` |
| **Metrics** | ROC-AUC, Accuracy | RMSE, MAE, R² |
| **Output** | Probabilities (0-1) | Days (positive real numbers) |

## Quick Start

### 1. Check Setup
```bash
python check_setup_los.py
```

### 2. Generate First Submission
```bash
python quickstart_los.py
```

This creates: `submissions/regression/submission_los_gb.csv`

### 3. Generate Multiple Submissions (Optional)
```bash
python generate_los_submissions.py
```

Creates:
- `submission_los_rf.csv` - Random Forest
- `submission_los_gb.csv` - Gradient Boosting
- `submission_los_et.csv` - Extra Trees
- `submission_los_ensemble.csv` - Weighted average

## Important Notes

### ⚠️ De-standardization
The PDF warns:
> "Regression task: remember to make sure that your predictions are **de-standardized**."

Our scripts handle this correctly:
- We use `StandardScaler()` on **features** only (not target)
- Predictions come out in the original scale (days)
- No de-standardization needed!

### Clipping Negative Predictions
```python
predictions = np.maximum(predictions, 0)
```
LOS can't be negative, so we clip at 0.

### Expected LOS Range
Training data shows:
- Mean: ~5-6 days
- Median: ~3-4 days
- Range: 0-50+ days

If your predictions look weird (e.g., all < 1 or all > 100), something's wrong!

## Reusing Classification Code

Yes! Almost everything is reusable:
- ✅ Data loading logic
- ✅ Feature engineering
- ✅ Preprocessing pipeline
- ✅ Cross-validation approach
- ❌ Change: Classifier → Regressor
- ❌ Change: `.predict_proba()` → `.predict()`
- ❌ Change: ROC-AUC → RMSE/MAE/R²

## Example Output

```
Validation metrics:
  RMSE: 4.23 days  ← Root Mean Squared Error
  MAE:  2.87 days  ← Mean Absolute Error  
  R²:   0.65       ← R-squared (0-1, higher better)
```

## Tips for Better Scores

1. **Start simple** - GB without FE is a solid baseline
2. **Try feature engineering** - Medical features might help
3. **Ensemble** - Average RF + GB usually improves
4. **Hyperparameter tuning** - Grid search around baseline
5. **Check predictions** - Make sure they're reasonable!

## Troubleshooting

### "Cannot import los_prep"
Make sure `los_prep.py` is in:
- `scripts/regression/` folder, OR
- `src/` folder

### "LOS data directory not found"
Check that you have:
```
data/raw/MIMIC III dataset LOS/
  ├── mimic_train_LOS.csv
  └── mimic_test_LOS.csv
```

### Predictions seem wrong
Check:
1. Are they in the right range? (0-50 days, mean ~5)
2. Did you clip negatives?
3. Did you accidentally standardize the target?

## Next Steps

1. Run `check_setup_los.py` to verify everything
2. Run `quickstart_los.py` for first submission
3. Upload to Kaggle and see your score
4. Iterate based on leaderboard feedback!

Good luck! 🚀
