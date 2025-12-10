# Classification Scripts (HEF - Hospital Expire Flag)

Scripts for predicting patient mortality (binary classification).

## Files:
- `quickstart_submission.py` - Quick single RF submission
- `generate_kaggle_submission.py` - Generate multiple submissions (RF, GB, ET, Ensemble)
- `finetune_gb.py` - Fine-tune GB hyperparameters
- `xgboost_sub.py` - XGBoost model
- `stacking_sub.py` - Stacking ensemble
- `both_formats.py` - Generate probability + binary versions
- `check_setup.py` - Verify setup before running

## Best Result:
- Gradient Boosting: 0.727226 AUC (Kaggle leaderboard)

## Quick Start:
```bash
python quickstart_submission.py
```
