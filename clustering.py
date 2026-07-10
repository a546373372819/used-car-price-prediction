
import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from preprocessing import preprocess_raw_dataframe

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# columns for clustering
PREFERRED_NUMERIC = [
    "car_age",
    "kms_run",
    "km_per_year",
    "total_owners",
    "times_viewed",
]

# long right tail
SKEWED_NUMERIC = ["kms_run", "km_per_year", "times_viewed"]

# categorical columns, skipping high-cardinality stuff
PREFERRED_CATEGORICAL = [
    "fuel_type",
    "transmission",
    "body_type",
    "car_rating",
    "make",
]

MAX_CATEGORY_CARDINALITY = 25


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


def make_one_hot_encoder():
    """one-hot encoder """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            min_frequency=15,
            max_categories=MAX_CATEGORY_CARDINALITY,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def select_cluster_features(df: pd.DataFrame, target_col: str):
    """pick the numeric + categorical columns to cluster on"""
    numeric_cols = [
        c for c in PREFERRED_NUMERIC
        if c in df.columns
        and c != target_col
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
    ]
    if len(numeric_cols) < 2:
        auto_numeric = df.drop(columns=[target_col], errors="ignore").select_dtypes(
            include=[np.number]
        ).columns.tolist()
        numeric_cols = [c for c in auto_numeric if df[c].nunique(dropna=True) > 1]

    categorical_cols = [
        c for c in PREFERRED_CATEGORICAL
        if c in df.columns
        and c != target_col
        and 1 < df[c].nunique(dropna=True) <= MAX_CATEGORY_CARDINALITY
    ]
    if not categorical_cols:
        for c in df.columns:
            if c == target_col or c in numeric_cols:
                continue
            if df[c].dtype == object and 1 < df[c].nunique(dropna=True) <= MAX_CATEGORY_CARDINALITY:
                categorical_cols.append(c)

    return numeric_cols, categorical_cols


def build_feature_pipeline(numeric_cols, categorical_cols):
    """skewed numerics get log1p'd, everything numeric is
    scaled, categoricals are one-hot'd. Scaling matters because clustering uses
    distances - without it kms_run would dominate everything."""
    skewed = [c for c in numeric_cols if c in SKEWED_NUMERIC]
    plain = [c for c in numeric_cols if c not in SKEWED_NUMERIC]

    log_numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    plain_numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if skewed:
        transformers.append(("num_log", log_numeric, skewed))
    if plain:
        transformers.append(("num", plain_numeric, plain))
    if categorical_cols:
        transformers.append(("cat", categorical, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def clip_numeric_outliers(df: pd.DataFrame, numeric_cols, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """clip numerics at the 1%/99% quantiles so outliers don't pull the clusters"""
    df = df.copy()
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        low, high = values.quantile(lower_q), values.quantile(upper_q)
        df[col] = values.clip(low, high)
    return df


def cluster_scores(X, labels):
    """the three metrics. DBSCAN noise is -1, so I drop it before scoring"""
    mask = labels != -1
    unique_labels = set(labels[mask])
    n_clusters = len(unique_labels)
    n_noise = int((labels == -1).sum())

    result = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    # need at least 2 clusters for these to mean anything
    if n_clusters >= 2 and mask.sum() > n_clusters:
        X_valid = X[mask]
        labels_valid = labels[mask]
        result["silhouette"] = float(silhouette_score(X_valid, labels_valid))
        result["davies_bouldin"] = float(davies_bouldin_score(X_valid, labels_valid))
        result["calinski_harabasz"] = float(calinski_harabasz_score(X_valid, labels_valid))
    return result


def choose_kmeans_k(X, k_min, k_max, output_dir: Path):
    """run kmeans for a range of k, save the elbow + silhouette plots, and return
    the k with the best silhouette"""
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        scores = cluster_scores(X, labels)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": scores["silhouette"],
                "davies_bouldin": scores["davies_bouldin"],
                "calinski_harabasz": scores["calinski_harabasz"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(tables_dir / "kmeans_k_selection.csv", index=False)

    plt = setup_matplotlib()

    # elbow plot (inertia vs k)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["k"], metrics_df["inertia"], marker="o")
    plt.title("Elbow method: inertia vs number of clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.xticks(metrics_df["k"])
    plt.tight_layout()
    fig.savefig(figures_dir / "01_kmeans_elbow.png", dpi=160)
    plt.close(fig)

    # silhouette vs k
    fig = plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="tab:green")
    plt.title("Silhouette score vs number of clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.xticks(metrics_df["k"])
    plt.tight_layout()
    fig.savefig(figures_dir / "02_kmeans_silhouette.png", dpi=160)
    plt.close(fig)

    best_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    return best_k, metrics_df


def plot_dendrogram(X, output_dir: Path):
    """dendrogram for the hierarchical clustering. truncated, since the full tree
    with ~5000 points is just a black blob"""
    from scipy.cluster.hierarchy import dendrogram, linkage

    figures_dir = output_dir / "figures"
    ensure_dir(figures_dir)

    # Ward linkage on the PCA space
    linkage_matrix = linkage(X, method="ward")

    plt = setup_matplotlib()
    fig = plt.figure(figsize=(11, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90.0,
        leaf_font_size=9.0,
        show_contracted=True,
    )
    plt.title("Hierarchical clustering dendrogram (Ward linkage, last 30 merges)")
    plt.xlabel("Sample index or (cluster size)")
    plt.ylabel("Merge distance")
    plt.tight_layout()
    fig.savefig(figures_dir / "03_dendrogram.png", dpi=160)
    plt.close(fig)


def estimate_dbscan_eps(X, min_samples: int, output_dir: Path):
    """guess an eps for DBSCAN from the k-distance graph: distance to each points
    min_samples-th neighbour, sorted, and the knee is roughly where density drops"""
    figures_dir = output_dir / "figures"
    ensure_dir(figures_dir)

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    plt = setup_matplotlib()
    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(k_distances)), k_distances)
    plt.title(f"DBSCAN k-distance graph (k={min_samples})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {min_samples}-th nearest neighbour")
    plt.tight_layout()
    fig.savefig(figures_dir / "04_dbscan_k_distance.png", dpi=160)
    plt.close(fig)

    # knee = biggest jump in the top half of the curve
    upper = k_distances[int(0.5 * len(k_distances)):]
    knee_index = int(np.argmax(np.diff(upper)))
    eps_candidate = float(upper[knee_index])
    if not np.isfinite(eps_candidate) or eps_candidate <= 0:
        eps_candidate = float(np.median(k_distances))
    return eps_candidate


def run_dbscan(X, min_samples: int, output_dir: Path):
    """one eps guess is not good, so try a few around it and keep the best silhouette"""
    tables_dir = output_dir / "tables"
    ensure_dir(tables_dir)

    base_eps = estimate_dbscan_eps(X, min_samples, output_dir)
    # 0.6x .. 1.5x of the guess
    eps_grid = sorted({round(base_eps * f, 4) for f in (0.6, 0.8, 1.0, 1.25, 1.5)})

    rows = []
    labelled = {}
    for eps in eps_grid:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        scores = cluster_scores(X, labels)
        rows.append({"eps": eps, "min_samples": min_samples, **scores})
        labelled[eps] = labels

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(tables_dir / "dbscan_eps_selection.csv", index=False)

    # keep only usable results: >=2 clusters and <half the points as noise
    n_samples = len(X)
    usable = grid_df[
        (grid_df["n_clusters"] >= 2)
        & (grid_df["n_noise"] <= 0.5 * n_samples)
        & grid_df["silhouette"].notna()
    ]
    if len(usable) == 0:
        return None, None, base_eps, grid_df

    best_row = usable.sort_values("silhouette", ascending=False).iloc[0]
    best_eps = float(best_row["eps"])
    return labelled[best_eps], best_eps, base_eps, grid_df


def plot_pca_scatter(X_2d, labels, title, filename, output_dir: Path):
    figures_dir = output_dir / "figures"
    ensure_dir(figures_dir)

    plt = setup_matplotlib()
    fig = plt.figure(figsize=(9, 7))
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        mask = labels == lab
        name = "noise" if lab == -1 else f"cluster {lab}"
        marker = "x" if lab == -1 else "o"
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=14, alpha=0.55, label=name, marker=marker)
    plt.title(title)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend(loc="best", fontsize=8, markerscale=1.2)
    plt.tight_layout()
    fig.savefig(figures_dir / filename, dpi=160)
    plt.close(fig)


def build_cluster_profiles(
    df_features: pd.DataFrame,
    numeric_cols,
    categorical_cols,
    target_col: str,
    target_values: pd.Series,
    labels,
    output_dir: Path,
):
    """numeric averages per cluster,
    typical categories, and the avg/median price per cluster.
 price bar chart and the profile heatmap too"""
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    profile_df = df_features.copy()
    profile_df["cluster"] = labels
    profile_df[target_col] = np.asarray(target_values, dtype=float)

    # don't include DBSCAN noise points in the profiles
    profile_df = profile_df[profile_df["cluster"] != -1].copy()

    # average of each numeric column per cluster
    numeric_profile = profile_df.groupby("cluster")[numeric_cols].mean().round(2)
    numeric_profile.insert(0, "size", profile_df.groupby("cluster").size())
    numeric_profile[f"{target_col}_mean"] = (
        profile_df.groupby("cluster")[target_col].mean().round(2)
    )
    numeric_profile[f"{target_col}_median"] = (
        profile_df.groupby("cluster")[target_col].median().round(2)
    )
    numeric_profile.to_csv(tables_dir / "cluster_profiles_numeric.csv")


    if categorical_cols:
        def top_value(series: pd.Series):
            counts = series.astype(str).value_counts()
            return counts.index[0] if len(counts) else "unknown"

        cat_profile = profile_df.groupby("cluster")[categorical_cols].agg(top_value)
        cat_profile.to_csv(tables_dir / "cluster_profiles_categorical.csv")

        global_share = {
            col: profile_df[col].astype(str).value_counts(normalize=True)
            for col in categorical_cols
        }
        distinctive_rows = {}
        for cluster_id, group in profile_df.groupby("cluster"):
            row = {}
            for col in categorical_cols:
                local = group[col].astype(str).value_counts(normalize=True)
                # skip missing placeholders and anything under 10% of the cluster
                local = local[~local.index.isin(["nan", "none", "unknown", ""])]
                local = local[local >= 0.10]
                if len(local) == 0:
                    row[col] = top_value(group[col])
                    continue
                lift = (local / global_share[col].reindex(local.index)).fillna(0.0)
                best = lift.idxmax()
                row[col] = f"{best} ({local[best] * 100:.0f}%, x{lift[best]:.2f})"
            distinctive_rows[cluster_id] = row
        distinctive_df = pd.DataFrame(distinctive_rows).T
        distinctive_df.index.name = "cluster"
        distinctive_df.to_csv(tables_dir / "cluster_profiles_categorical_distinctive.csv")
    else:
        cat_profile = pd.DataFrame()

    # price per cluster
    price_profile = profile_df.groupby("cluster")[target_col].agg(
        ["count", "mean", "median", "min", "max"]
    ).round(2)
    price_profile.to_csv(tables_dir / "cluster_price_profile.csv")

    plt = setup_matplotlib()

    # bar chart of average price per cluster
    fig = plt.figure(figsize=(9, 5))
    ordered = price_profile.sort_values("mean")
    plt.bar([f"cluster {i}" for i in ordered.index], ordered["mean"].values)
    plt.title("Average sale price per cluster")
    plt.xlabel("Cluster")
    plt.ylabel(f"Average {target_col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(figures_dir / "06_cluster_avg_price.png", dpi=160)
    plt.close(fig)

    # profile heatmap. z-score each column so they share a colour scale
    heat = numeric_profile[numeric_cols].copy()
    normed = (heat - heat.mean(axis=0)) / heat.std(axis=0).replace(0, np.nan)
    normed = normed.fillna(0.0)
    fig = plt.figure(figsize=(1.6 * max(len(numeric_cols), 4), 0.9 * len(normed) + 2))
    plt.imshow(normed.values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    plt.colorbar(label="Standardised cluster mean")
    plt.xticks(range(len(numeric_cols)), shorten_labels(numeric_cols, 16), rotation=45, ha="right")
    plt.yticks(range(len(normed)), [f"cluster {i}" for i in normed.index])
    plt.title("Cluster numeric profiles (standardised means)")
    plt.tight_layout()
    fig.savefig(figures_dir / "07_cluster_profile_heatmap.png", dpi=160)
    plt.close(fig)

    return numeric_profile, cat_profile, price_profile


def main():
    #obrisati
    parser = argparse.ArgumentParser(description="Used car market segmentation (clustering)")
    parser.add_argument("data_path", help="Path to the used-car dataset file: CSV, XLSX/XLS or JSON")
    parser.add_argument("--target", default="sale_price", help="Target column, used only for interpretation. Default: sale_price")
    parser.add_argument("--output", default="outputs/clustering", help="Output directory for figures, tables and model")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum number of clusters to try for K-Means")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum number of clusters to try for K-Means")
    parser.add_argument("--pca-variance", type=float, default=0.90, help="Retained variance ratio for PCA used before clustering")
    parser.add_argument("--dbscan-min-samples", type=int, default=10, help="min_samples for DBSCAN")
    parser.add_argument("--sample", type=int, default=0, help="Optional random sample size. 0 means use all rows.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    models_dir = output_dir / "models"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)
    ensure_dir(models_dir)

    # shared cleaning
    raw_df = load_dataset(args.data_path)
    sample_size = args.sample if args.sample and args.sample > 0 else None
    df, target_col = preprocess_raw_dataframe(
        raw_df,
        requested_target=args.target,
        sample_size=sample_size,
        require_positive_target=True,
    )

    # pick features + clip outliers
    numeric_cols, categorical_cols = select_cluster_features(df, target_col)
    if len(numeric_cols) + len(categorical_cols) < 2:
        raise ValueError("Not enough usable attributes were found for clustering.")

    df = clip_numeric_outliers(df, numeric_cols)

    feature_cols = numeric_cols + categorical_cols
    df_features = df[feature_cols].copy()
    target_values = df[target_col].astype(float)

    # encode + scale, then PCA
    preprocessor = build_feature_pipeline(numeric_cols, categorical_cols)
    X_encoded = preprocessor.fit_transform(df_features)

    # ~90% variance PCA
    pca = PCA(n_components=args.pca_variance, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_encoded)

    # 2D for the plots
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_encoded)

    setup_info = {
        "rows_used": int(len(df_features)),
        "target_column": target_col,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "skewed_log_features": [c for c in numeric_cols if c in SKEWED_NUMERIC],
        "encoded_dimensions": int(X_encoded.shape[1]),
        "pca_components_used": int(X_pca.shape[1]),
        "pca_variance_retained": float(np.sum(pca.explained_variance_ratio_)),
        "pca_2d_variance_retained": float(np.sum(pca_2d.explained_variance_ratio_)),
        "random_state": RANDOM_STATE,
    }
    save_json(tables_dir / "clustering_setup.json", setup_info)

    # K-Means: find best k, then fit
    best_k, kmeans_metrics = choose_kmeans_k(X_pca, args.k_min, args.k_max, output_dir)
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_pca)

    # hierarchical + dendrogram, same k as K-Means
    plot_dendrogram(X_pca, output_dir)
    agglo = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    agglo_labels = agglo.fit_predict(X_pca)

    # DBSCAN
    dbscan_labels, dbscan_eps, dbscan_base_eps, _ = run_dbscan(
        X_pca, args.dbscan_min_samples, output_dir
    )

    # score all three and compare
    comparison_rows = []
    algo_labels = {
        "KMeans": kmeans_labels,
        "Agglomerative": agglo_labels,
    }
    if dbscan_labels is not None:
        algo_labels["DBSCAN"] = dbscan_labels

    for name, labels in algo_labels.items():
        scores = cluster_scores(X_pca, labels)
        row = {"algorithm": name, **scores}
        if name == "KMeans":
            row["k"] = best_k
        if name == "DBSCAN":
            row["eps"] = dbscan_eps
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(tables_dir / "algorithm_comparison.csv", index=False)

    # one scatter plot per algorithm
    plot_pca_scatter(X_2d, kmeans_labels, f"K-Means clusters (k={best_k}) in PCA space", "05a_pca_kmeans.png", output_dir)
    plot_pca_scatter(X_2d, agglo_labels, f"Agglomerative clusters (k={best_k}) in PCA space", "05b_pca_agglomerative.png", output_dir)
    if dbscan_labels is not None:
        plot_pca_scatter(X_2d, dbscan_labels, f"DBSCAN clusters (eps={dbscan_eps:.3f}) in PCA space", "05c_pca_dbscan.png", output_dir)

    # winner = best silhouette (need >=2 clusters, and skip DBSCAN if it turned
    # too much into noise)
    valid_comp = comparison_df[
        (comparison_df["n_clusters"] >= 2) & comparison_df["silhouette"].notna()
    ].copy()
    valid_comp = valid_comp[valid_comp["n_noise"] <= 0.3 * len(X_pca)]
    if len(valid_comp) == 0:
        best_algo = "KMeans"
    else:
        best_algo = valid_comp.sort_values("silhouette", ascending=False).iloc[0]["algorithm"]
    final_labels = algo_labels[best_algo]

    # same plot under a fixed name for the report
    plot_pca_scatter(
        X_2d, final_labels,
        f"Final clustering ({best_algo}) in PCA space",
        "05_pca_final_clusters.png", output_dir,
    )

    # describe the clusters
    numeric_profile, cat_profile, price_profile = build_cluster_profiles(
        df_features, numeric_cols, categorical_cols, target_col, target_values, final_labels, output_dir
    )

    final_scores = cluster_scores(X_pca, final_labels)
    summary = {
        "selected_algorithm": best_algo,
        "kmeans_best_k": best_k,
        "dbscan_eps": dbscan_eps,
        "dbscan_base_eps_estimate": dbscan_base_eps,
        "final_metrics": final_scores,
        "n_features_numeric": len(numeric_cols),
        "n_features_categorical": len(categorical_cols),
    }
    save_json(tables_dir / "clustering_summary.json", summary)

    # save the fitted stuff + labels for later
    with open(models_dir / "clustering_model.pkl", "wb") as f:
        pickle.dump(
            {
                "preprocessor": preprocessor,
                "pca": pca,
                "pca_2d": pca_2d,
                "kmeans": kmeans,
                "selected_algorithm": best_algo,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
            },
            f,
        )

    labels_out = df_features.copy()
    labels_out[target_col] = np.asarray(target_values, dtype=float)
    labels_out["cluster"] = final_labels
    labels_out.to_csv(tables_dir / "clustered_rows.csv", index=False)

    # quick print, the rest is in output_dir
    print("Clustering finished.")
    print(f"Selected algorithm: {best_algo}")
    print(f"K-Means best k (Elbow + Silhouette): {best_k}")
    print(f"Final clusters: {final_scores['n_clusters']} (noise points: {final_scores['n_noise']})")
    print(f"Silhouette: {final_scores['silhouette']:.4f}")
    print(f"Davies-Bouldin: {final_scores['davies_bouldin']:.4f}")
    print(f"Calinski-Harabasz: {final_scores['calinski_harabasz']:.1f}")
    print("Algorithm comparison:")
    print(comparison_df.to_string(index=False))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
