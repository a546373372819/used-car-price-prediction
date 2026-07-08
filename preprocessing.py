"""
Shared preprocessing for the used-car project.

Both regression and clustering use the same raw dataset, so dataset-level
cleaning and feature creation live here. Task-specific supervised steps, such
as target outlier filtering and sklearn transformations fitted on a train split,
belong in the task script that needs them.
"""

import re
from datetime import datetime

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET_CANDIDATES = [
    "sale_price",
    "selling_price",
    "resale_price",
    "price",
    "listed_price",
    "current_price",
]

# These columns are strongly price-derived in this dataset and would cause data leakage.
PRICE_LEAKAGE_COLS = [
    "broker_quote",
    "original_price",
    "emi_starts_from",
    "booking_down_pymnt",
]

# Relevant attributes for the project. Shared preprocessing keeps these columns
# when they exist in the input CSV.
PROJECT_COLUMNS = [
    "car_name",
    "yr_mfr",
    "fuel_type",
    "kms_run",
    "sale_price",
    "city",
    "times_viewed",
    "body_type",
    "transmission",
    "variant",
    "assured_buy",
    "registered_city",
    "registered_state",
    "is_hot",
    "rto",
    "source",
    "make",
    "model",
    "car_availability",
    "total_owners",
    "car_rating",
    "ad_created_on",
    "fitness_certificate",
    "reserved",
    "warranty_avail",
]

DROP_NAME_HINTS = [
    "url",
    "link",
    "image",
    "photo",
    "thumbnail",
    "description",
    "seller_comment",
    "owner_comment",
]

ID_NAME_HINTS = [
    "id",
    "vin",
    "registration_number",
    "license_plate",
    "chassis",
]

NUMERIC_NAME_HINTS = [
    "year",
    "age",
    "km",
    "mileage",
    "odometer",
    "engine",
    "power",
    "torque",
    "seat",
    "owner",
    "rating",
    "distance",
    "cc",
    "bhp",
    "rpm",
    "price",
    "cost",
    "warranty",
    "manufacturing",
]

BOOL_TRUE = {"true", "yes", "y", "da", "1", "available", "present"}
BOOL_FALSE = {"false", "no", "n", "ne", "0", "not available", "absent"}


def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed_column"


def make_unique_columns(columns):
    seen = {}
    result = []
    for col in columns:
        base = clean_column_name(col)
        if base not in seen:
            seen[base] = 0
            result.append(base)
        else:
            seen[base] += 1
            result.append(f"{base}_{seen[base]}")
    return result


def extract_first_number(value):
    """Extract the first numeric value from strings like '1,498 CC' or '5.2 Lakh'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null", "--", "-"}:
        return np.nan

    multiplier = 1.0
    if "crore" in text or "cr" in text:
        multiplier = 10_000_000.0
    elif "lakh" in text or "lac" in text:
        multiplier = 100_000.0
    elif re.search(r"\bk\b", text) and "km" not in text:
        multiplier = 1_000.0

    text = text.replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return np.nan
    try:
        return float(match.group(0)) * multiplier
    except ValueError:
        return np.nan


def to_bool_numeric(series: pd.Series):
    lowered = series.astype(str).str.strip().str.lower()
    unique_values = set(lowered.dropna().unique())
    allowed = BOOL_TRUE | BOOL_FALSE | {"nan", "none", "null", ""}
    if unique_values and unique_values.issubset(allowed):
        return lowered.map(lambda x: 1.0 if x in BOOL_TRUE else (0.0 if x in BOOL_FALSE else np.nan))
    return None


def select_target_column(columns, requested_target: str | None) -> str:
    if requested_target:
        requested_clean = clean_column_name(requested_target)
        if requested_clean in columns:
            return requested_clean
        raise ValueError(
            f"Target column '{requested_target}' was not found after cleaning column names. "
            f"Available columns: {list(columns)}"
        )

    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Target column was not found automatically. Pass it explicitly with --target. "
        f"Tried: {TARGET_CANDIDATES}. Available columns: {list(columns)}"
    )


def clean_category_values(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
    )


def preprocess_raw_dataframe(
    raw_df: pd.DataFrame,
    requested_target: str | None = "sale_price",
    sample_size: int | None = None,
    require_positive_target: bool = True,
):
    """
    Project-level data cleaning and feature creation.

    Shared steps:
    - clean column names,
    - keep the relevant project columns,
    - remove price-leakage columns,
    - optionally validate a positive target column,
    - remove exact duplicate rows,
    - create common date, age, mileage and categorical interaction features.

    For clustering, pass require_positive_target=False if the target should only
    be retained for later interpretation rather than supervised training.
    """
    df = raw_df.copy()

    df.columns = make_unique_columns(df.columns)
    df = df.dropna(axis=1, how="all")

    target_col = select_target_column(df.columns, requested_target) if requested_target else None

    selected_cols = [c for c in PROJECT_COLUMNS if c in df.columns]
    if selected_cols:
        if target_col and target_col not in selected_cols:
            selected_cols.append(target_col)
        df = df[selected_cols].copy()

    leakage_cols = [c for c in PRICE_LEAKAGE_COLS if c in df.columns and c != target_col]
    df = df.drop(columns=leakage_cols, errors="ignore")

    if target_col:
        target_col = select_target_column(df.columns, requested_target)
        df[target_col] = df[target_col].map(extract_first_number)
        if require_positive_target:
            df = df.dropna(subset=[target_col])
            df = df[df[target_col] > 0]

    df = df.drop_duplicates().reset_index(drop=True)

    if sample_size is not None and sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    if "ad_created_on" in df.columns:
        raw_dates = df["ad_created_on"].astype(str).str.strip()
        try:
            ad_dates = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
        except TypeError:
            ad_dates = pd.to_datetime(raw_dates, errors="coerce")
        df["ad_year"] = ad_dates.dt.year
        df["ad_month"] = ad_dates.dt.month

    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if any(hint in col for hint in DROP_NAME_HINTS):
            drop_cols.append(col)
        elif any(col == hint or col.endswith("_" + hint) or col.startswith(hint + "_") for hint in ID_NAME_HINTS):
            drop_cols.append(col)
        elif df[col].nunique(dropna=True) >= max(50, int(0.98 * len(df))):
            drop_cols.append(col)
    df = df.drop(columns=sorted(set(drop_cols)), errors="ignore")

    name_candidates = [c for c in ["car_name", "name", "full_name", "title"] if c in df.columns]
    if name_candidates and "brand" not in df.columns:
        name_col = name_candidates[0]
        df["brand"] = df[name_col].astype(str).str.strip().str.split().str[0].str.lower()

    if "make" in df.columns and "model" in df.columns:
        df["make_model"] = clean_category_values(df["make"]) + "_" + clean_category_values(df["model"])

    if "make" in df.columns and "body_type" in df.columns:
        df["make_body_type"] = clean_category_values(df["make"]) + "_" + clean_category_values(df["body_type"])

    if "fuel_type" in df.columns and "transmission" in df.columns:
        df["fuel_transmission"] = clean_category_values(df["fuel_type"]) + "_" + clean_category_values(df["transmission"])

    for col in list(df.columns):
        if col == target_col:
            continue
        if df[col].dtype == object:
            bool_numeric = to_bool_numeric(df[col])
            if bool_numeric is not None:
                df[col] = bool_numeric

    for col in list(df.columns):
        if col == target_col:
            continue
        if df[col].dtype == object and any(hint in col for hint in NUMERIC_NAME_HINTS):
            parsed = df[col].map(extract_first_number)
            if parsed.notna().mean() >= 0.60:
                df[col] = parsed

    current_year = datetime.now().year
    reference_year = current_year
    if "ad_year" in df.columns:
        ad_years = pd.to_numeric(df["ad_year"], errors="coerce")
        plausible_years = ad_years[ad_years.between(2000, current_year + 1)]
        if len(plausible_years) > 0:
            reference_year = int(plausible_years.median())

    if "yr_mfr" in df.columns:
        yr = pd.to_numeric(df["yr_mfr"], errors="coerce")
        valid_year = yr.between(1980, reference_year + 1)
        if valid_year.mean() >= 0.5:
            df["car_age"] = np.where(valid_year, reference_year - yr, np.nan)

    year_cols = [
        c for c in df.columns
        if c != target_col
        and c not in {"ad_year"}
        and (c == "year" or c.endswith("_year") or "model_year" in c)
    ]

    for year_col in year_cols:
        year_num = pd.to_numeric(df[year_col], errors="coerce")
        valid_year = year_num.between(1980, reference_year + 1)
        if valid_year.mean() >= 0.5:
            df[f"{year_col}_age"] = np.where(valid_year, reference_year - year_num, np.nan)

    if "kms_run" in df.columns and "car_age" in df.columns:
        km = pd.to_numeric(df["kms_run"], errors="coerce")
        age = pd.to_numeric(df["car_age"], errors="coerce")
        df["km_per_year"] = km / (age.clip(lower=0) + 1.0)
        df["log_kms_run"] = np.log1p(km.clip(lower=0))
        df["log_km_per_year"] = np.log1p(df["km_per_year"].clip(lower=0))
        df["age_km_interaction"] = age.clip(lower=0) * np.log1p(km.clip(lower=0))

    if "car_age" in df.columns:
        age = pd.to_numeric(df["car_age"], errors="coerce")
        df["car_age_squared"] = age ** 2
        df["is_newer_car"] = (age <= 3).astype(float)
        df["is_old_car"] = (age >= 10).astype(float)

    if "kms_run" in df.columns:
        kms = pd.to_numeric(df["kms_run"], errors="coerce")
        median_kms = kms.median()
        df["high_mileage"] = (kms > median_kms).astype(float)

    if "times_viewed" in df.columns:
        views = pd.to_numeric(df["times_viewed"], errors="coerce")
        df["log_times_viewed"] = np.log1p(views.clip(lower=0))

    if "total_owners" in df.columns and "car_age" in df.columns:
        owners = pd.to_numeric(df["total_owners"], errors="coerce")
        age = pd.to_numeric(df["car_age"], errors="coerce")
        df["owners_per_year"] = owners / (age.clip(lower=0) + 1.0)

    engine_cols = [c for c in df.columns if c != target_col and ("engine" in c or c.endswith("cc"))]
    power_cols = [c for c in df.columns if c != target_col and ("power" in c or "bhp" in c)]
    if engine_cols and power_cols:
        engine = pd.to_numeric(df[engine_cols[0]], errors="coerce")
        power = pd.to_numeric(df[power_cols[0]], errors="coerce")
        df["power_per_engine"] = power / engine.replace(0, np.nan)

    return df.reset_index(drop=True), target_col
