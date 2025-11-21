# CML Final Project - Progress Tracker

**Student:** [Your Name]  
**Project:** MIMIC-III ICU Predictions  
**Tasks:** Hospital Expire Flag (Classification) + Length of Stay (Regression)

---

## 📊 Current Best Scores

### Classification (HEF - Hospital Expire Flag)
| Model | Public Score | Date | Notes |
|-------|-------------|------|-------|
| **Gradient Boosting** | **0.727226** | 2024-11-19 | 🏆 Best classification score |
| Ensemble | 0.721090 | 2024-11-19 | Weighted average of models |
| XGBoost | 0.715164 | 2024-11-19 | Underperformed vs GB |
| Extra Trees | 0.688217 | 2024-11-19 | Lowest score |

**Metric:** ROC-AUC (higher is better)  
**Target:** Beat baseline, maximize AUC

### Regression (LOS - Length of Stay)
| Model | Public Score | Date | Notes |
|-------|-------------|------|-------|
| **GB Enhanced v2** | **20.751931** | 2024-11-19 | 🏆 Best - with diagnosis features |
| Ensemble Weighted | ~20.8 | 2024-11-19 | GB+RF+ET ensemble |
| Ensemble Average | ~20.8 | 2024-11-19 | Simple average |
| GB Diagnosis Only | 21.021833 | 2024-11-19 | Diagnosis features only |
| XGBoost | ~21.0 | 2024-11-19 | Slight regression |
| Gradient Boosting | 21.309904 | 2024-11-19 | Original baseline |
| Log-transform ensemble | TBD | - | Worse on validation |

**Metric:** Unknown (professor won't tell us - likely RMSE or MSE)
**Target:** Beat 15.7 (NN "demanding" benchmark)
**Gap:** Currently ~5 points behind benchmark

---

## 📅 Development Log

### 2024-11-19 - Initial Setup & Classification

**Completed:**
- ✅ Project organization (scripts/classification vs regression folders)
- ✅ Data exploration and preprocessing pipeline
- ✅ Feature engineering (medical features: shock index, pulse pressure, etc.)
- ✅ Classification baseline models (RF, GB, ET, Ensemble)
- ✅ Discovered GB performs best (0.727 AUC)
- ✅ Fixed submission format (probabilities vs binary, added icustay_id)
- ✅ Tried XGBoost and Stacking - didn't improve over GB

**Key Learnings:**
- Gradient Boosting consistently outperforms other tree methods
- Feature engineering didn't help classification (kept it simple)
- Ensemble didn't beat single GB model
- XGBoost underperformed (0.715 vs 0.727)

**Files Created:**
- `scripts/classification/quickstart_submission.py`
- `scripts/classification/generate_kaggle_submissions.py`
- `scripts/classification/finetune_gb.py`
- `scripts/classification/xgboost_sub.py`
- `scripts/classification/stacking_sub.py`

### 2024-11-19 - Regression (LOS) Initial Attempts

**Completed:**
- ✅ Adapted classification pipeline for regression
- ✅ Created `los_prep.py` module (similar to classification prep)
- ✅ Generated initial submissions (RF, GB, ET, Ensemble)
- ✅ GB achieved 21.3 RMSE (best so far)
- ✅ Identified benchmark: NN at 15.7 RMSE

**Files Created:**
- `scripts/regression/los_prep.py`
- `scripts/regression/quickstart_los.py`
- `scripts/regression/generate_los_submissions.py`
- `scripts/regression/nn_los_fast.py`
- `scripts/regression/xgb_los_fast.py`

### 2024-11-19 - Feature Engineering Deep Dive

**Key Discovery: Validation vs Kaggle Gap**
- Local validation RMSE: ~4.7
- Kaggle score: ~20.8
- This is suspicious! 4.7² ≈ 22, close to 20.8 → **Kaggle may be using MSE, not RMSE**

**Feature Analysis Completed:**

1. **Dropped Useless Features:**
   - `Diff` column: correlation with LOS = -0.0012 (zero signal)
   - Age: correlation = 0.0117 (not useful despite mean age 63.7)
   - ADMITTIME features (hour, weekend, night): max correlation 0.0205

2. **Found High-Value Diagnosis Features:**
   - ICD9 codes with high LOS: 51884, 5770, 430, 44101, 0380, 51881, 0389, 431
   - Diagnosis keywords: PANCREATITIS (8.12 days!), SHOCK (7.85 days), TRANSPLANT, ARREST
   - Low-LOS keywords: OVERDOSE (2.56), DIABETIC/DKA (2.38)

3. **Categorical Variable Power (eta-squared):**
   - FIRST_CAREUNIT: η²=0.0186 (most predictive)
   - ADMISSION_TYPE: η²=0.0105
   - INSURANCE: η²=0.0074

4. **Added Interaction Feature:**
   - `sicu_emergency`: SICU + EMERGENCY admission = 4.35 avg LOS

**Results:**
- Enhanced features improved Kaggle from 21.31 → 20.75 (2.6% improvement)
- Validation RMSE improved 4.7551 → 4.6793 (1.6%)

**Extreme Value Problem Identified:**
- Training LOS ranges 0.06 to 101.74 days
- Model predictions cap at ~17-19 days
- For LOS 20-50 days (1.9% of data): RMSE = 23.67!
- These outliers dominate the error

**Approaches Tested:**
- ✅ Ensemble (GB+RF+ET): ~20.8 on Kaggle (no improvement)
- ❌ Log-transform: Worse on validation (4.67 → 4.80)
- All models scoring similarly suggests the issue isn't the algorithm

**Files Created:**
- `scripts/regression/feature_analysis.py` - Main analysis script with all features
- `scripts/regression/test_age_calculation.py` - Verify age calculation
- `scripts/regression/test_admittime_features.py` - Test time features
- `scripts/regression/explore_all_features.py` - Comprehensive feature exploration
- `scripts/regression/generate_los_submissions_v2.py` - Submissions with new features
- `scripts/regression/generate_los_ensemble.py` - GB+RF+ET ensemble
- `scripts/regression/diagnose_kaggle_gap.py` - Investigate val vs Kaggle gap
- `scripts/regression/generate_los_logtransform.py` - Log-transform approach

---

## 🎯 Strategy & Insights

### Classification Strategy
1. **Stick with Gradient Boosting** - Clear winner at 0.727
2. **Simple is better** - Feature engineering didn't help
3. **Fine-tuning options exist** but current score is competitive

### Regression Strategy
1. **Current best:** 20.75 (improved from 21.3)
2. **Gap:** Still ~5 points from 15.7 benchmark
3. **Key Insight:** Validation RMSE² ≈ Kaggle score → might be MSE metric

**What Worked:**
- Diagnosis features (ICD9 codes + text keywords): +2.6% improvement
- Dropping useless features (Diff, age, time features)

**What Didn't Work:**
- Ensembles: All score ~20.8 (same as single models)
- Log-transform: Made things worse
- XGBoost: No improvement over GB

**Remaining Ideas to Try:**
- [ ] Neural Network (since benchmark says "NN demanding")
- [ ] Two-stage model (classify long-stay vs short-stay, then predict)
- [ ] Quantile regression (predict median instead of mean)
- [ ] More aggressive outlier handling

### Key Technical Decisions

**Data Preprocessing:**
- Median imputation for numerical features
- Most frequent imputation for categorical
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- BP outlier cleaning (values below physiological minimums)

**Feature Engineering:**
- Vital sign ranges (instability indicators)
- Critical threshold flags (tachycardia, hypotension, hypoxia)
- Shock index (HR/SBP ratio)
- Pulse pressure (SBP - DBP)
- **Note:** Helped classification validation but hurt test score - use cautiously

**Model Selection:**
- Classification: Gradient Boosting is clear winner
- Regression: Testing multiple approaches (GB baseline, then NN/XGB)

---

## 📁 Project Structure

```
cml_final/
├── scripts/
│   ├── classification/          # HEF (mortality) prediction
│   │   ├── quickstart_submission.py
│   │   ├── generate_kaggle_submissions.py
│   │   ├── finetune_gb.py
│   │   ├── xgboost_sub.py
│   │   ├── stacking_sub.py
│   │   ├── both_formats.py
│   │   ├── check_setup.py
│   │   └── README.md
│   └── regression/              # LOS (length of stay) prediction
│       ├── los_prep.py
│       ├── quickstart_los.py
│       ├── generate_los_submissions.py
│       ├── nn_los_fast.py
│       ├── xgb_los_fast.py
│       ├── check_setup_los.py
│       └── README.md
├── submissions/
│   ├── classification/          # HEF submission CSVs
│   └── regression/              # LOS submission CSVs
├── data/
│   ├── raw/
│   │   ├── MIMIC III dataset HEF/
│   │   └── MIMIC III dataset LOS/
│   └── output/
├── notebooks/                   # Jupyter notebooks for EDA
└── src/                         # Shared utilities
```

---

## 🔧 Environment Setup

**Python Packages:**
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- jupyter

**Installation:**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

---

## 📝 Notes & Observations

### What Worked
- ✅ Gradient Boosting consistently strong
- ✅ Clean data preprocessing pipeline
- ✅ Proper train/validation split with stratification
- ✅ BP outlier cleaning improved results

### What Didn't Work
- ❌ Feature engineering hurt classification test scores
- ❌ XGBoost underperformed GB for classification
- ❌ Ensemble/stacking didn't beat single GB
- ❌ Extra Trees significantly worse than RF/GB
- ❌ Age features for LOS (correlation only 0.0117)
- ❌ Time features for LOS (weekend, night - max correlation 0.02)
- ❌ Diff column (correlation -0.0012)
- ❌ Log-transform for regression (made validation worse)
- ❌ Ensembles for regression (all score same ~20.8)

### Open Questions
- Is Kaggle using MSE instead of RMSE? (4.7² ≈ 22 ≈ Kaggle 20.8)
- Can neural networks close the 5 point gap?
- Would a two-stage model help with extreme values?
- Are we fundamentally limited by the feature set?

---

## 🎯 Next Session TODO

**Immediate:**
1. [ ] Try Neural Network approach (benchmark hint: "NN demanding")
2. [ ] Consider two-stage model for extreme LOS values
3. [ ] Maybe accept ~20.8 is close to our limit with these features

**For Final Submission:**
- [ ] Document best approaches for both tasks
- [ ] Clean up code and add comments
- [ ] Prepare presentation/report
- [ ] Archive best submissions

**Note:** Professor won't reveal scoring metric, so we can't confirm MSE vs RMSE theory

---

## 📊 Leaderboard Submissions History

### Classification (HEF)
| Submission File | Score | Rank | Date | Notes |
|----------------|-------|------|------|-------|
| PROB_submission_gb.csv | 0.727226 | - | 2024-11-19 | Best so far |
| PROB_submission_ensemble.csv | 0.721090 | - | 2024-11-19 | |
| PROB_submission_xgboost.csv | 0.715164 | - | 2024-11-19 | Underperformed |
| PROB_submission_et.csv | 0.688217 | - | 2024-11-19 | Worst |

### Regression (LOS)
| Submission File | Score | Rank | Date | Notes |
|----------------|-------|------|------|-------|
| submission_los_gb_enhanced_v2.csv | 20.751931 | - | 2024-11-19 | 🏆 Best - diagnosis features |
| submission_los_ensemble_weighted.csv | ~20.8 | - | 2024-11-19 | GB+RF+ET ensemble |
| submission_los_gb_diagnosis_v2.csv | 21.021833 | - | 2024-11-19 | Diagnosis only |
| submission_los_xgboost.csv | ~21.0 | - | 2024-11-19 | No improvement |
| submission_los_gb.csv | 21.309904 | - | 2024-11-19 | Original baseline |
| submission_los_ensemble.csv | 21.457578 | - | 2024-11-19 | |
| submission_los_rf.csv | 21.782790 | - | 2024-11-19 | |
| submission_los_et.csv | 22.123555 | - | 2024-11-19 | |

---

## 💡 Tips for Future Me

1. **Always validate locally first** - Don't waste submissions on untested code
2. **Document parameter changes** - Easy to forget what you tried
3. **Keep old submissions** - Sometimes reverting is the right move
4. **The benchmark is a hint** - "nn demanding" = try neural networks!
5. **Simplicity often wins** - Feature engineering didn't always help
6. **Check your submission format** - icustay_id + probabilities/values
7. **Test correlations before using features** - Age (0.01) and time (0.02) were useless
8. **Extreme values dominate RMSE** - 1.9% of data (LOS 20-50 days) has RMSE 23.67!
9. **Validation/Kaggle gap can indicate metric mismatch** - 4.7² ≈ 22 ≈ Kaggle 20.8
10. **Ensembles don't always help** - If models are similar, ensemble won't improve

---

## 📚 Resources & References

- [MIMIC-III Documentation](https://mimic.mit.edu/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- Course materials and lectures
- Project PDF requirements

---

**Last Updated:** 2024-11-19
**Status:** Classification complete (0.727 AUC), Regression improved (20.75 → target 15.7)
