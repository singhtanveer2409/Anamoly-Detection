# Sensor Anomaly Detection — Predictive Modeling

Predicts whether a sensor reading is an anomaly (`target = 1`) from five engineered sensor channels (`X1`–`X5`) and a `Date`, on a highly imbalanced (~0.86% positive) industrial-telemetry-style dataset. Built as a Kaggle competition submission for [`ds-kaggle-mit-wpu`](https://www.kaggle.com/code/tanveersingh2409/final-submission).

🏆 **6th place on the leaderboard** — Public score: 0.827778 · Private score: 0.812500

## Overview

- **Task:** Binary classification — anomaly vs. normal sensor reading
- **Data:** 1,639,424 training rows / 409,856 test rows, 5 feature columns + `Date`
- **Class balance:** ~0.86% positive (anomaly) — accuracy is not a meaningful metric here
- **Evaluation metric:** F1 (binary), with precision/recall, ROC-AUC and PR-AUC also tracked
- **Final model:** Tuned LightGBM, F1-optimal decision threshold (0.91), validation F1 = **0.8059**

## Key Findings

- **`X3` and `X4` were exponentiated log-counts.** Their raw values ranged up to ~1e16–1e34, but `log(X3)` and `log(X4)` collapsed cleanly to small integers (0–88 and 0–80). Reversing the exponentiation lifted their correlation with the target from 0.02/0.05 to 0.33/0.38 and made `logX3`/`logX4` the single most important feature-engineering decision — confirmed later by feature-importance rankings on every tree model.
- **The anomaly rate is non-stationary**, drifting from ~2.4% (Dec 2020) to ~0.2% (Dec 2024), a >10x change. A `TimeSeriesSplit` backtest confirmed the model's F1/PR-AUC hold up across this drift rather than exploiting a stationary base rate.
- **Outliers in `X1`/`X2` are signal, not noise** — they concentrate where anomalies occur, so they were kept (not clipped) for the tree-based models.
- **Threshold tuning matters under severe imbalance.** The default 0.5 cutoff is arbitrary at 0.86% prevalence; sweeping the threshold and optimizing for F1 gave a meaningfully better operating point (0.91 for the final model) than the default.
- **Residual analysis:** false negatives cluster at lower `logX3`/`logX4` values — borderline anomalies near the "normal" boundary are the hardest to catch, suggesting a 6th sensor channel or a rate-of-change feature could help further.

## Notebook Structure

1. **Setup & Data Loading** — loads `train.parquet` / `test.parquet` / `sample_submission.parquet` (from a Kaggle path if present, else the local directory)
2. **Exploratory Data Analysis** — schema, missing values, duplicates, class balance, feature distributions, correlation analysis, time-based drift, outlier checks
3–4. **Data Cleaning & Feature Engineering** — reverses the `X3`/`X4` exponentiation, extracts calendar features (year, month, day-of-week, day-of-year, cyclical encodings) from `Date`
5. **Train/Validation Strategy** — stratified 80/20 hold-out for headline metrics + `TimeSeriesSplit` for out-of-time robustness
6–7. **Modeling** — classical baselines (Logistic Regression, Decision Tree, KNN, SVM) and advanced models (Random Forest, XGBoost, LightGBM, MLP neural network), all compared on the same validation fold
8. **Hyperparameter Tuning** — `RandomizedSearchCV` (3-fold stratified, scored on F1) applied to XGBoost and LightGBM; threshold sweep for the best model
9. **Model Evaluation & Comparison** — accuracy, precision, recall, F1 (binary + macro), ROC-AUC, PR-AUC for every model; ROC and precision-recall curves; feature importances
10. **Robustness Checks** — `TimeSeriesSplit` backtest and false-positive/false-negative residual analysis
11. **Final Model Selection & Test-Set Prediction** — best tuned candidate (by validation F1 at its optimal threshold) retrained on the full training set
12. **Submission File** — writes `submission.csv` / `submission.parquet` matching the sample submission schema

## Results

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Single interpretable tree |
| K-Nearest Neighbors | Distance-based; needs the `log(X3/X4)` fix + `RobustScaler` to work at all |
| SVM (RBF) | Trained on a subsample (quadratic training cost) |
| Random Forest | Tuned on a subsample for speed |
| XGBoost (tuned) | Full training data, `scale_pos_weight` set to imbalance ratio; best threshold F1 = 0.7460 |
| LightGBM (tuned) | Full training data, `is_unbalance=True`; best threshold F1 = **0.8059** — final model |
| MLP (Neural Network) | scikit-learn `MLPClassifier` on scaled subsample (TensorFlow/PyTorch unavailable in the execution environment) |

**Final model — LightGBM (tuned), threshold = 0.91, on the held-out validation set:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal (0) | 0.9989 | 0.9952 | 0.9970 |
| Anomaly (1) | 0.6095 | 0.8689 | 0.7165 |

Overall accuracy: 0.9941 (not the metric optimized for, given the class imbalance).

Predicted test-set anomaly rate: **0.8127%**, close to the training anomaly rate of 0.8563%, indicating the model is not drastically over- or under-predicting positives.

## Competition Result

| Metric | Score |
|---|---|
| Public leaderboard score | 0.827778 |
| Private leaderboard score | 0.812500 |
| Final leaderboard rank | **6th** |

Notebook: [Final_Submission — Version 1 on Kaggle](https://www.kaggle.com/code/tanveersingh2409/final-submission?scriptVersionId=349279530)

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost
```

## Data

The notebook expects `train.parquet`, `test.parquet`, and `sample_submission.parquet`. It looks first in a Kaggle-style path (`/kaggle/input/competitions/ds-kaggle-mit-wpu`), then falls back to the current directory. Place these three files alongside the notebook if running locally outside Kaggle.

## Usage

1. Install the requirements above.
2. Place `train.parquet`, `test.parquet`, and `sample_submission.parquet` in the working directory (or run on Kaggle with the competition dataset attached).
3. Run `final-submission.ipynb` top to bottom.
4. Predictions are written to `submission.csv` and `submission.parquet`, matching the schema of `sample_submission.parquet` (`ID`, `target` as string `'0'`/`'1'`).

## Reproducibility

`RANDOM_STATE = 42` is set globally (`np.random.seed`) and passed to all models and splits.
