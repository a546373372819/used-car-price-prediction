

import argparse
import json
from pathlib import Path

import pandas as pd

RANDOM_STATE = 42
DEFAULT_ROWS = 5000
TARGET_COL = "sale_price"


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def stratified_price_sample(df: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    if n_rows <= 0 or len(df) <= n_rows:
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if TARGET_COL not in df.columns:
        # If the raw file does not have sale_price for some reason, fall back to
        # simple random sampling. The regression script will validate the target later.
        return df.sample(n=n_rows, random_state=random_state).reset_index(drop=True)

    work = df.copy()
    prices = pd.to_numeric(work[TARGET_COL], errors="coerce")

    try:
        valid_prices = prices.dropna()
        if valid_prices.nunique() < 2:
            raise ValueError("Not enough unique valid prices for qcut.")
        work["_price_bin_for_sampling"] = pd.qcut(
            prices,
            q=min(10, valid_prices.nunique()),
            duplicates="drop",
        )
    except ValueError:
        work["_price_bin_for_sampling"] = 0

    parts = []
    grouped = work.groupby("_price_bin_for_sampling", observed=False, group_keys=False, dropna=False)
    for i, (_, group) in enumerate(grouped):
        group_share = len(group) / len(work)
        take = max(1, int(round(n_rows * group_share)))
        take = min(take, len(group))
        parts.append(group.sample(n=take, random_state=random_state + i))

    sampled = pd.concat(parts, axis=0)

    if len(sampled) > n_rows:
        sampled = sampled.sample(n=n_rows, random_state=random_state)
    elif len(sampled) < n_rows:
        remaining = work.drop(index=sampled.index, errors="ignore")
        missing = min(n_rows - len(sampled), len(remaining))
        if missing > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=missing, random_state=random_state)],
                axis=0,
            )

    sampled = sampled.drop(columns=["_price_bin_for_sampling"], errors="ignore")
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def create_subset(input_path: str, output_path: str, n_rows: int, random_state: int):
    df = load_csv(input_path)
    original_rows, original_cols = df.shape

    sampled = stratified_price_sample(df, n_rows=n_rows, random_state=random_state)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)

    metadata = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_shape": [int(original_rows), int(original_cols)],
        "subset_shape": [int(sampled.shape[0]), int(sampled.shape[1])],
        "target_column_used_for_sampling": TARGET_COL if TARGET_COL in df.columns else None,
        "random_state": int(random_state),
        "sampling": "stratified by sale_price quantile bins when sale_price exists, otherwise random sampling",
        "cleaning_done_here": False,
        "cleaning_note": (
            "Column cleaning, relevant-column selection, leakage-column removal, "
            "invalid sale_price removal and duplicate removal are handled in preprocessing.py."
        ),
        "kept_columns": sampled.columns.tolist(),
    }
    meta_path = out_path.with_suffix(".metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return sampled, metadata, meta_path


def main():
    parser = argparse.ArgumentParser(description="Create row subset for used-car regression.")
    parser.add_argument("input_csv", help="Path to the full Used_Car_Price_Prediction.csv file")
    parser.add_argument("output_csv", nargs="?", default="used_cars_subset.csv", help="Output subset CSV path")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of rows in the subset. Default: 5000")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed. Default: 42")
    args = parser.parse_args()

    sampled, metadata, meta_path = create_subset(
        input_path=args.input_csv,
        output_path=args.output_csv,
        n_rows=args.rows,
        random_state=args.random_state,
    )

    print(f"Saved subset: {args.output_csv}")
    print(f"Rows x columns: {sampled.shape[0]} x {sampled.shape[1]}")
    print(f"Metadata: {meta_path}")
    print("Cleaning/preprocessing moved to preprocessing.py")


if __name__ == "__main__":
    main()
