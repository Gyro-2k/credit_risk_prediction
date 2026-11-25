# Credit Risk Prediction

Predicting loan default risk (good/bad credit) from applicant financial and demographic data using the German Credit dataset. The final model is deployed as a FastAPI service in Docker.

## What I Did

- Used the German Credit dataset (1,000 applicants, 300 bad risk and 700 good risk)
- Cleaned the data: dropped the unnamed index column, standardised column names, and filled missing `savings_account` and `checking_account` values with a `none` category
- Created a binary target `default` (1 = bad risk, 0 = good risk) from the original `risk` column, then dropped `risk`
- Ran EDA on credit amount and duration against default, a correlation heatmap, and default rate by loan purpose
- Split the data 75/25 with stratification to preserve the good/bad ratio
- Built a scikit-learn pipeline with `ColumnTransformer`: one-hot encoding for `sex`, `housing` and `purpose`, and ordinal encoding for `savings_account` and `checking_account`
- Handled class imbalance using `class_weight="balanced"` for Logistic Regression and Random Forest, and `scale_pos_weight` for XGBoost
- Trained and compared three models: Logistic Regression, Random Forest and XGBoost
- Selected XGBoost (learning rate 0.03, 600 trees, max depth 6) for the best balance between precision and recall on the bad risk class
- Saved the final pipeline with joblib and served it through a FastAPI service in Docker

## Results

Evaluated on the 250-row test set. Precision, recall and F1 are for the bad risk (default) class.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.64 | 0.43 | 0.63 | 0.51 |
| Random Forest | 0.76 | 0.67 | 0.40 | 0.50 |
| XGBoost (selected) | 0.73 | 0.56 | 0.53 | 0.54 |

**Takeaway:** Random Forest had the highest accuracy but missed 60% of actual defaulters. Logistic Regression caught the most defaulters but flagged too many good applicants. XGBoost gave the best balance and the highest F1 on the default class, which matters here because approving a risky applicant costs more than rejecting a safe one.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, seaborn, matplotlib, FastAPI, Docker
