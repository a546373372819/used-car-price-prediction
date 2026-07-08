# Regression part - Used car sale price prediction

This folder contains only Dušan's regression part of the project. The goal is to predict `sale_price` for used cars and save the tables/figures that will later be used in the project report.

## Files

- `make_subset.py` - creates the real project subset from the full Kaggle CSV.
- `preprocessing.py` - shared dataset cleanup and feature creation for regression and clustering.
- `regression.py` - runs EDA, regression-specific preprocessing, model training, evaluation and saves report evidence.
- `requirements_regression.txt` - Python packages needed locally.

## 1. Create the project subset

The specification says that the project should use a subset of several thousand rows and at least 15 relevant attributes. Run:

```bash
python make_subset.py Used_Car_Price_Prediction.csv used_cars_subset.csv --rows 5000
```

This creates:

```text
used_cars_subset.csv
used_cars_subset.metadata.json
```

The subset script removes invalid rows where `sale_price <= 0`, keeps a representative sample stratified by price quantiles, and excludes the following price-derived leakage columns by default:

```text
broker_quote, original_price, emi_starts_from, booking_down_pymnt
```

These columns are excluded because they are almost direct price estimates/derivatives in this dataset. Keeping them would make the regression unrealistically easy and harder to defend.

## 2. Run regression

```bash
python regression.py used_cars_subset.csv --target sale_price --output outputs/regression
```

For a quick first check:

```bash
python regression.py used_cars_subset.csv --target sale_price --output outputs/regression_fast --fast
```

## Output structure

The regression script creates:

```text
outputs/regression/
  figures/   # plots for the report
  tables/    # CSV/JSON evidence and metric tables
  models/    # fitted final model
```

Important figures:

- `01_sale_price_distribution.png`
- `02_log_sale_price_distribution.png`
- `03_missing_values.png`
- `04_correlation_heatmap.png`
- `05_numeric_relation_*.png`
- `06_outliers_boxplot_*.png`
- `07_categorical_target_mean_*.png`
- `08_model_comparison_rmse.png`
- `09_actual_vs_predicted.png`
- `10_residual_distribution.png`
- `11_residuals_vs_predicted.png`
- `12_feature_importance_transformed_features.png`, when the selected model exposes feature importance or coefficients
- `13_permutation_importance_original_features.png`

Important tables:

- `basic_statistics.csv`
- `missing_values.csv`
- `model_results.csv`
- `test_predictions.csv`
- `split_and_features_info.json`
- `permutation_importance_original_features.csv`

## Implemented models

The script compares:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Voting Regressor
- Stacking Regressor, unless `--fast` is used

Evaluation metrics:

- MAE
- RMSE
- R2 score

## Preprocessing summary

Shared preprocessing is in `preprocessing.py` so the clustering part can reuse
the same cleaned dataset representation. For clustering, call
`preprocess_raw_dataframe(..., require_positive_target=False)` if `sale_price`
should be kept only for cluster interpretation and not used as an input feature.

The shared module performs dataset-specific preparation for the uploaded CSV:

- cleans column names,
- removes invalid targets,
- removes price leakage columns by default,
- extracts date features from `ad_created_on`,
- creates `car_age`, `km_per_year`, `log_kms_run`, `make_model`, `make_body_type`, `fuel_transmission`, `brand`,

The regression script then performs regression-specific train-only steps:

- imputes missing numeric values with the median,
- clips numeric outliers using train-only 1% and 99% quantiles,
- scales numeric attributes,
- imputes categorical values with the most frequent value,
- applies one-hot encoding with unknown-category handling,
- applies `log1p` transformation to the target during model training.

All preprocessing steps that learn parameters are inside sklearn pipelines and are fitted only on the training split.
