"""
Regression part of the project: Used car sale price prediction.

Run example:
    python used_car_regression.py used_cars.csv --target sale_price --output outputs/regression

The script performs:
- basic EDA and saves figures/tables for the report,
- preprocessing with train-only fitted transformations,
- train/test split,
- training and comparison of several regression models,
- evaluation with MAE, RMSE, R2 and percentage-based error metrics,
- error analysis plots,
- feature importance and permutation importance for the selected model.
"""

import argparse
import json
import math
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance

from preprocessing import preprocess_raw_dataframe

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path_obj)
    if suffix == ".json":
        return pd.read_json(path_obj)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV, XLSX/XLS or JSON.")


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clips numerical columns to quantiles learned only from the training split."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_arr, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X_arr, self.upper, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target mean encoding for high-cardinality categorical features."""

    def __init__(self, smoothing=12.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        X_df = self._to_dataframe(X)
        y_arr = np.asarray(y, dtype=float)
        self.columns_ = X_df.columns.tolist()
        self.global_mean_ = float(np.nanmean(y_arr))
        self.maps_ = {}

        for col in self.columns_:
            work = pd.DataFrame({"category": self._clean(X_df[col]), "target": y_arr})
            stats = work.groupby("category")["target"].agg(["mean", "count"])
            weight = stats["count"] / (stats["count"] + self.smoothing)
            encoded = (weight * stats["mean"]) + ((1.0 - weight) * self.global_mean_)
            self.maps_[col] = encoded.to_dict()
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        encoded_cols = []
        for col in self.columns_:
            values = self._clean(X_df[col]) if col in X_df.columns else pd.Series(["unknown"] * len(X_df))
            encoded_cols.append(values.map(self.maps_[col]).fillna(self.global_mean_).to_numpy())
        return np.column_stack(encoded_cols) if encoded_cols else np.empty((len(X_df), 0))

    def get_feature_names_out(self, input_features=None):
        columns = input_features if input_features is not None else getattr(self, "columns_", [])
        return np.array([f"{col}_target_mean" for col in columns], dtype=object)

    @staticmethod
    def _to_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    @staticmethod
    def _clean(series: pd.Series) -> pd.Series:
        return (
            series
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
        )


def make_one_hot_encoder():
    """Compatible with newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            min_frequency=3,
            max_categories=120,
        )
    except TypeError:
        # Older scikit-learn versions do not support sparse_output/min_frequency/max_categories.
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def remove_target_outliers(df: pd.DataFrame, target_col: str, lower_q=0.05, upper_q=0.95) -> pd.DataFrame:
    lower = df[target_col].quantile(lower_q)
    upper = df[target_col].quantile(upper_q)
    return df[(df[target_col] >= lower) & (df[target_col] <= upper)].reset_index(drop=True)


def infer_feature_columns(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Avoid completely empty/all-constant columns.
    usable_numeric = [c for c in numeric_cols if X[c].nunique(dropna=True) > 1]
    usable_categorical = [c for c in categorical_cols if X[c].nunique(dropna=True) > 1]
    return usable_numeric, usable_categorical


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )
    categorical_target_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("target_mean", TargetMeanEncoder(smoothing=15.0)),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
        transformers.append(("cat_target", categorical_target_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def wrap_target_regressor(regressor):
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_models(numeric_cols, categorical_cols, fast: bool):
    def pipe(model):
        return Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )

    rf_estimators = 80 if fast else 260
    gb_estimators = 80 if fast else 260
    stack_cv = 3

    models = {
        "Linear Regression": wrap_target_regressor(pipe(LinearRegression())),
        "Ridge Regression": wrap_target_regressor(pipe(Ridge(alpha=10.0, random_state=RANDOM_STATE))),
        "Lasso Regression": wrap_target_regressor(pipe(Lasso(alpha=0.0005, max_iter=10000, random_state=RANDOM_STATE))),
        "Decision Tree": wrap_target_regressor(
            pipe(DecisionTreeRegressor(max_depth=18, min_samples_leaf=5, random_state=RANDOM_STATE))
        ),
        "Random Forest": wrap_target_regressor(
            pipe(
                RandomForestRegressor(
                    n_estimators=rf_estimators,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            )
        ),
        "Gradient Boosting": wrap_target_regressor(
            pipe(
                GradientBoostingRegressor(
                    n_estimators=gb_estimators,
                    learning_rate=0.04,
                    max_depth=3,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                )
            )
        ),
        "Histogram Gradient Boosting": wrap_target_regressor(
            pipe(
                HistGradientBoostingRegressor(
                    max_iter=260 if not fast else 80,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    random_state=RANDOM_STATE,
                )
            )
        ),
        "Voting Regressor": wrap_target_regressor(
            pipe(
                VotingRegressor(
                    estimators=[
                        ("ridge", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
                        (
                            "rf",
                            RandomForestRegressor(
                                n_estimators=max(80, rf_estimators // 2),
                                min_samples_leaf=2,
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        ),
                        (
                            "gbr",
                            GradientBoostingRegressor(
                                n_estimators=max(100, gb_estimators // 2),
                                learning_rate=0.04,
                                max_depth=3,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ],
                    weights=[1, 2, 2],
                    n_jobs=1,
                )
            )
        ),
        "Extra Trees": wrap_target_regressor(
            pipe(
                ExtraTreesRegressor(
                    n_estimators=420 if not fast else 160,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            )
        ),
    }

    if not fast:
        models["Stacking Regressor"] = wrap_target_regressor(
            pipe(
                StackingRegressor(
                    estimators=[
                        ("ridge", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
                        (
                            "rf",
                            RandomForestRegressor(
                                n_estimators=180,
                                min_samples_leaf=2,
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        ),
                        (
                            "gbr",
                            GradientBoostingRegressor(
                                n_estimators=180,
                                learning_rate=0.04,
                                max_depth=3,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                        (
                            "etr",
                            ExtraTreesRegressor(
                                n_estimators=220,
                                min_samples_leaf=2,
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        ),
                    ],
                    final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
                    cv=stack_cv,
                    n_jobs=1,
                    passthrough=False,
                )
            )
        )
    return models


def regression_metrics(y_true, y_pred):
    """
    Returns both absolute and relative regression metrics.

    MAE/RMSE are kept in the original price scale, because they show the real
    money error. Percentage metrics are added only to make the error easier to
    interpret in the report.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 0.0)

    abs_errors = np.abs(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Percentage errors are computed only where the real price is positive.
    valid_percentage = y_true > 0
    percentage_errors = np.full_like(y_true, np.nan, dtype=float)
    percentage_errors[valid_percentage] = (
        abs_errors[valid_percentage] / y_true[valid_percentage]
    ) * 100.0

    mape = float(np.nanmean(percentage_errors))
    mdape = float(np.nanmedian(percentage_errors))

    target_mean = float(np.nanmean(y_true))
    target_median = float(np.nanmedian(y_true))
    nmae_mean = float((mae / target_mean) * 100.0) if target_mean > 0 else np.nan
    nmae_median = float((mae / target_median) * 100.0) if target_median > 0 else np.nan

    # RMSLE is useful for prices because it evaluates relative/log-scale error
    # and is less dominated by very expensive cars.
    y_true_nonnegative = np.maximum(y_true, 0.0)
    rmsle = math.sqrt(mean_squared_error(np.log1p(y_true_nonnegative), np.log1p(y_pred)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "MdAPE_percent": mdape,
        "NMAE_mean_percent": nmae_mean,
        "NMAE_median_percent": nmae_median,
        "RMSLE": rmsle,
    }


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def shorten_labels(labels, max_len=28):
    return [str(x)[:max_len] + ("..." if len(str(x)) > max_len else "") for x in labels]


def save_eda_outputs(df: pd.DataFrame, target_col: str, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    # Tables for report.
    df.dtypes.astype(str).rename("dtype").to_csv(tables_dir / "column_types.csv")
    df.describe(include="all").transpose().to_csv(tables_dir / "basic_statistics.csv")
    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("missing_count", ascending=False)
    missing.to_csv(tables_dir / "missing_values.csv")

    plt = setup_matplotlib()

    # 1) Target distribution.
    fig = plt.figure(figsize=(8, 5))
    plt.hist(df[target_col].dropna(), bins=40)
    plt.title("Distribution of sale prices")
    plt.xlabel(target_col)
    plt.ylabel("Number of cars")
    plt.tight_layout()
    fig.savefig(figures_dir / "01_sale_price_distribution.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(df[target_col].dropna()), bins=40)
    plt.title("Distribution of log-transformed sale prices")
    plt.xlabel(f"log1p({target_col})")
    plt.ylabel("Number of cars")
    plt.tight_layout()
    fig.savefig(figures_dir / "02_log_sale_price_distribution.png", dpi=160)
    plt.close(fig)

    # 2) Missing values.
    top_missing = missing[missing["missing_count"] > 0].head(25)
    if len(top_missing) > 0:
        fig = plt.figure(figsize=(10, 5))
        plt.bar(shorten_labels(top_missing.index), top_missing["missing_count"].values)
        plt.title("Missing values by column")
        plt.xlabel("Column")
        plt.ylabel("Missing count")
        plt.xticks(rotation=70, ha="right")
        plt.tight_layout()
        fig.savefig(figures_dir / "03_missing_values.png", dpi=160)
        plt.close(fig)

    # 3) Correlation heatmap for numeric columns.
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns and numeric_df.shape[1] >= 2:
        candidate_cols = numeric_df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False).head(12).index.tolist()
        corr = numeric_df[candidate_cols].corr(numeric_only=True)
        fig = plt.figure(figsize=(9, 7))
        plt.imshow(corr.values, aspect="auto")
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(corr.columns)), shorten_labels(corr.columns, 16), rotation=70, ha="right")
        plt.yticks(range(len(corr.index)), shorten_labels(corr.index, 16))
        plt.title("Correlation heatmap for top numeric features")
        plt.tight_layout()
        fig.savefig(figures_dir / "04_correlation_heatmap.png", dpi=160)
        plt.close(fig)

        # 4) Scatter plots: numeric features vs target.
        top_numeric = [c for c in candidate_cols if c != target_col][:5]
        for idx, col in enumerate(top_numeric, start=1):
            fig = plt.figure(figsize=(7, 5))
            sample = df[[col, target_col]].dropna()
            if len(sample) > 3000:
                sample = sample.sample(3000, random_state=RANDOM_STATE)
            plt.scatter(sample[col], sample[target_col], alpha=0.45, s=12)
            plt.title(f"{col} vs sale price")
            plt.xlabel(col)
            plt.ylabel(target_col)
            plt.tight_layout()
            fig.savefig(figures_dir / f"05_numeric_relation_{idx}_{col}.png", dpi=160)
            plt.close(fig)

    # 5) Boxplots for outlier overview.
    numeric_cols = [c for c in numeric_df.columns if c != target_col]
    for idx, col in enumerate(numeric_cols[:6], start=1):
        values = numeric_df[col].dropna()
        if len(values) == 0:
            continue
        fig = plt.figure(figsize=(7, 4))
        plt.boxplot(values, vert=False)
        plt.title(f"Outlier overview: {col}")
        plt.xlabel(col)
        plt.tight_layout()
        fig.savefig(figures_dir / f"06_outliers_boxplot_{idx}_{col}.png", dpi=160)
        plt.close(fig)

    # 6) Categorical relations with target.
    categorical_cols = [c for c in df.columns if c != target_col and df[c].dtype == object]
    for idx, col in enumerate(categorical_cols[:4], start=1):
        grouped = df.groupby(col)[target_col].agg(["mean", "count"]).sort_values("count", ascending=False).head(15)
        if len(grouped) == 0:
            continue
        grouped = grouped.sort_values("mean", ascending=False)
        grouped.to_csv(tables_dir / f"target_by_category_{col}.csv")
        fig = plt.figure(figsize=(10, 5))
        plt.bar(shorten_labels(grouped.index), grouped["mean"].values)
        plt.title(f"Average sale price by {col}")
        plt.xlabel(col)
        plt.ylabel(f"Average {target_col}")
        plt.xticks(rotation=70, ha="right")
        plt.tight_layout()
        fig.savefig(figures_dir / f"07_categorical_target_mean_{idx}_{col}.png", dpi=160)
        plt.close(fig)


def evaluate_models(models, X_train, X_test, y_train, y_test):
    fitted_models = {}
    rows = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start
        pred = model.predict(X_test)
        metrics = regression_metrics(y_test, pred)
        rows.append(
            {
                "model": name,
                **metrics,
                "fit_time_seconds": fit_time,
            }
        )
        fitted_models[name] = model
    results = pd.DataFrame(rows).sort_values(
    ["MAPE_percent", "MdAPE_percent", "RMSE"], ascending=True).reset_index(drop=True)
    return results, fitted_models


def get_fitted_pipeline(model):
    if isinstance(model, TransformedTargetRegressor):
        return model.regressor_
    return model


def get_preprocess_and_estimator(best_model):
    pipeline = get_fitted_pipeline(best_model)
    if isinstance(pipeline, Pipeline):
        return pipeline.named_steps.get("preprocess"), pipeline.named_steps.get("model")
    return None, pipeline


def save_model_analysis(best_model, best_name: str, X_test, y_test, results_df: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    models_dir = output_dir / "models"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)
    ensure_dir(models_dir)

    plt = setup_matplotlib()

    pred = np.maximum(best_model.predict(X_test), 0.0)
    residuals = y_test - pred

    # Model comparison by absolute error.
    fig = plt.figure(figsize=(10, 5))
    cmp = results_df.sort_values("RMSE", ascending=False)
    plt.barh(cmp["model"], cmp["RMSE"])
    plt.title("Model comparison by RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("Model")
    plt.tight_layout()
    fig.savefig(figures_dir / "08_model_comparison_rmse.png", dpi=160)
    plt.close(fig)

    # Model comparison by relative/percentage error.
    if "MAPE_percent" in results_df.columns:
        fig = plt.figure(figsize=(10, 5))
        cmp_pct = results_df.sort_values("MAPE_percent", ascending=False)
        plt.barh(cmp_pct["model"], cmp_pct["MAPE_percent"])
        plt.title("Model comparison by MAPE")
        plt.xlabel("MAPE (%)")
        plt.ylabel("Model")
        plt.tight_layout()
        fig.savefig(figures_dir / "08b_model_comparison_mape.png", dpi=160)
        plt.close(fig)

    # Actual vs predicted.
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(y_test, pred, alpha=0.5, s=14)
    min_value = float(min(np.min(y_test), np.min(pred)))
    max_value = float(max(np.max(y_test), np.max(pred)))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.title(f"Actual vs predicted prices - {best_name}")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.tight_layout()
    fig.savefig(figures_dir / "09_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Residual distribution.
    fig = plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=40)
    plt.title(f"Residual distribution - {best_name}")
    plt.xlabel("Actual - predicted")
    plt.ylabel("Number of cars")
    plt.tight_layout()
    fig.savefig(figures_dir / "10_residual_distribution.png", dpi=160)
    plt.close(fig)

    # Residuals vs predicted.
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(pred, residuals, alpha=0.5, s=14)
    plt.axhline(0, linestyle="--")
    plt.title(f"Residuals vs predicted - {best_name}")
    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    plt.tight_layout()
    fig.savefig(figures_dir / "11_residuals_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Save predictions for later manual checks/report tables.
    abs_error = np.abs(residuals)
    abs_percentage_error = np.where(
        np.asarray(y_test, dtype=float) > 0,
        (abs_error / np.asarray(y_test, dtype=float)) * 100.0,
        np.nan,
    )

    # Save predictions together with original test features.
    y_test_arr = np.asarray(y_test, dtype=float)
    residuals_arr = y_test_arr - pred
    abs_error = np.abs(residuals_arr)

    abs_percentage_error = np.where(
        y_test_arr > 0,
        (abs_error / y_test_arr) * 100.0,
        np.nan,
    )

    pred_df = X_test.reset_index(drop=True).copy()

    pred_df.insert(0, "actual", y_test_arr)
    pred_df.insert(1, "predicted", pred)
    pred_df.insert(2, "residual", residuals_arr)
    pred_df.insert(3, "absolute_error", abs_error)
    pred_df.insert(4, "absolute_percentage_error", abs_percentage_error)

    pred_df.to_csv(tables_dir / "test_predictions_with_features.csv", index=False)

    worst_by_percentage = pred_df.sort_values(       "absolute_percentage_error",
        ascending=False
    ).head(50)

    worst_by_absolute = pred_df.sort_values(
        "absolute_error",
        ascending=False
    ).head(50)

    worst_by_percentage.to_csv(
        tables_dir / "worst_50_by_percentage_error.csv",
        index=False
    )

    worst_by_absolute.to_csv(
        tables_dir / "worst_50_by_absolute_error.csv",
        index=False
    )

    best_metrics = regression_metrics(y_test, pred)
    error_summary = {
        "best_model": best_name,
        "target_mean_test": float(np.mean(y_test)),
        "target_median_test": float(np.median(y_test)),
        **{k: float(v) for k, v in best_metrics.items()},
    }
    save_json(tables_dir / "best_model_error_summary.json", error_summary)

    preprocess, estimator = get_preprocess_and_estimator(best_model)

    # Internal feature importance / coefficients on transformed feature space.
    importance_df = None
    if preprocess is not None:
        try:
            feature_names = preprocess.get_feature_names_out()
        except Exception:
            feature_names = np.array([f"feature_{i}" for i in range(preprocess.transform(X_test.head(1)).shape[1])])

        if hasattr(estimator, "feature_importances_"):
            importance = estimator.feature_importances_
            importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_).ravel()
            importance_df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef), "coefficient": coef})

    if importance_df is not None and len(importance_df) > 0:
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df.to_csv(tables_dir / "model_feature_importance_transformed_features.csv", index=False)
        top_imp = importance_df.head(25).sort_values("importance", ascending=True)
        fig = plt.figure(figsize=(10, 7))
        plt.barh(shorten_labels(top_imp["feature"], 45), top_imp["importance"])
        plt.title(f"Top transformed feature importance - {best_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        fig.savefig(figures_dir / "12_feature_importance_transformed_features.png", dpi=160)
        plt.close(fig)

    # Permutation importance on original columns: more useful in the report.
    try:
        perm = permutation_importance(
            best_model,
            X_test,
            y_test,
            n_repeats=3,
            random_state=RANDOM_STATE,
            scoring="r2",
            n_jobs=1,
        )
        perm_df = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(tables_dir / "permutation_importance_original_features.csv", index=False)

        top_perm = perm_df.head(20).sort_values("importance_mean", ascending=True)
        fig = plt.figure(figsize=(10, 7))
        plt.barh(shorten_labels(top_perm["feature"], 40), top_perm["importance_mean"])
        plt.title(f"Permutation importance on original features - {best_name}")
        plt.xlabel("Mean decrease in R2")
        plt.ylabel("Feature")
        plt.tight_layout()
        fig.savefig(figures_dir / "13_permutation_importance_original_features.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        with open(tables_dir / "permutation_importance_error.txt", "w", encoding="utf-8") as f:
            f.write(str(exc))

    # Store fitted model for later use/demo.
    with open(models_dir / "best_regression_model.pkl", "wb") as f:
        pickle.dump(best_model, f)


def main():
    parser = argparse.ArgumentParser(description="Used car sale price regression project")
    parser.add_argument("data_path", help="Path to the used-car dataset file: CSV, XLSX/XLS or JSON")
    parser.add_argument("--target", default="sale_price", help="Target column name. Default: sale_price")
    parser.add_argument("--output", default="outputs/regression", help="Output directory for figures, tables and model")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split size. Default: 0.20")
    parser.add_argument("--sample", type=int, default=0, help="Optional random sample size. 0 means use all rows.")
    parser.add_argument("--keep-target-outliers", action="store_true", help="Do not remove extreme target outliers.")
    parser.add_argument("--fast", action="store_true", help="Faster run: fewer trees and no stacking regressor.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    raw_df = load_dataset(args.data_path)
    sample_size = args.sample if args.sample and args.sample > 0 else None
    df, target_col = preprocess_raw_dataframe(raw_df, args.target, sample_size=sample_size)

    if not args.keep_target_outliers:
        df = remove_target_outliers(df, target_col)

    numeric_cols, categorical_cols = infer_feature_columns(df, target_col)
    selected_columns = numeric_cols + categorical_cols + [target_col]
    df_model = df[selected_columns].copy()

    save_eda_outputs(df_model, target_col, output_dir)

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col].astype(float)

    price_bins = pd.qcut(y, q=10, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=price_bins,
    )

    split_info = {
        "rows_after_cleaning": int(len(df_model)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_column": target_col,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "test_size": args.test_size,
        "random_state": RANDOM_STATE,
    }
    save_json(tables_dir / "split_and_features_info.json", split_info)

    models = build_models(numeric_cols, categorical_cols, fast=args.fast)
    results_df, fitted_models = evaluate_models(models, X_train, X_test, y_train, y_test)
    results_df.to_csv(tables_dir / "model_results.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    save_model_analysis(best_model, best_name, X_test, y_test, results_df, output_dir)

    # Console output is intentionally compact; all report evidence is saved in output_dir.
    best_row = results_df.iloc[0]
    print("Best model:", best_name)
    print(f"MAE: {best_row['MAE']:.4f}")
    print(f"RMSE: {best_row['RMSE']:.4f}")
    print(f"R2: {best_row['R2']:.4f}")
    print(f"MAPE_percent: {best_row['MAPE_percent']:.4f}")
    print(f"MdAPE_percent: {best_row['MdAPE_percent']:.4f}")
    print(f"NMAE_mean_percent: {best_row['NMAE_mean_percent']:.4f}")
    print(f"NMAE_median_percent: {best_row['NMAE_median_percent']:.4f}")
    print(f"RMSLE: {best_row['RMSLE']:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
