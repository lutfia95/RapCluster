"""Exploratory analysis of a dataset under the selected preprocessing."""
import numpy as np
from sklearn.decomposition import PCA

from preprocessing import numeric_matrix

MAX_SAMPLE_ROWS = 20000       # PCA, correlation and distances use a random subset of larger datasets
MAX_PCA_COMPONENTS = 20
MAX_CORRELATION_COLUMNS = 150
N_OUTLIERS = 25
MISSING_BINS = [(0, 0, "none"), (0, 0.1, "≤ 10%"), (0.1, 0.25, "10–25%"), (0.25, 0.5, "25–50%"),
                (0.5, 0.75, "50–75%"), (0.75, 1.0, "75–100%")]


def rounded(values, digits=4):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def column_boxes(X):
    """Per-column box-plot statistics (whiskers at the most extreme values within 1.5 IQR)."""
    q1, median, q3 = np.percentile(X, [25, 50, 75], axis=0)
    iqr = q3 - q1
    lower = np.where(X >= q1 - 1.5 * iqr, X, np.inf).min(axis=0)
    upper = np.where(X <= q3 + 1.5 * iqr, X, -np.inf).max(axis=0)
    return {"q1": rounded(q1), "median": rounded(median), "q3": rounded(q3),
            "lower": rounded(lower), "upper": rounded(upper), "mean": rounded(X.mean(axis=0))}


def missing_overview(df, intensity_start_index):
    columns, raw = numeric_matrix(df, intensity_start_index)
    missing = np.isnan(raw)
    fraction = missing.mean(axis=1)
    bins = []
    for low, high, label in MISSING_BINS:
        in_bin = fraction == 0 if high == 0 else (fraction > low) & (fraction <= high)
        bins.append({"label": label, "rows": int(in_bin.sum())})
    return {
        "total_cells": int(missing.size),
        "missing_cells": int(missing.sum()),
        "columns": [str(c) for c in columns],
        "per_column": missing.sum(axis=0).astype(int).tolist(),
        "rows_by_missing_fraction": bins,
    }


def explore(df, intensity_start_index, result):
    X, transformed = result["data"], result["transformed"]
    columns = result["columns"]
    rng = np.random.default_rng(0)
    sample = np.arange(X.shape[0])
    if X.shape[0] > MAX_SAMPLE_ROWS:
        sample = np.sort(rng.choice(X.shape[0], MAX_SAMPLE_ROWS, replace=False))
    Xs = X[sample]

    n_components = min(MAX_PCA_COMPONENTS, Xs.shape[1], Xs.shape[0])
    ratio = PCA(n_components=n_components).fit(Xs).explained_variance_ratio_

    corr_idx = np.arange(len(columns))
    if len(columns) > MAX_CORRELATION_COLUMNS:
        corr_idx = np.unique(np.linspace(0, len(columns) - 1, MAX_CORRELATION_COLUMNS).astype(int))
    with np.errstate(all="ignore"):
        corr = np.corrcoef(Xs[:, corr_idx], rowvar=False) if len(corr_idx) > 1 else np.ones((1, 1))
    corr = np.where(np.isfinite(corr), corr, 0.0)

    # Rows far from the overall median profile, as robust z-scores.
    distance = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    mad = 1.4826 * np.median(np.abs(distance - np.median(distance)))
    z = (distance - np.median(distance)) / mad if mad > 0 else np.zeros_like(distance)
    top = np.argsort(-z, kind="stable")[:N_OUTLIERS]
    counts, edges = np.histogram(distance, bins=30)

    return {
        "summary": result["summary"],
        "options": result["options"],
        "sampled_rows": int(sample.size),
        "missing": missing_overview(df, intensity_start_index),
        "pca": {"explained_variance_ratio": rounded(ratio, 5),
                "cumulative": rounded(np.cumsum(ratio), 5), "n_rows": int(sample.size)},
        "distributions": {"columns": columns, "before_scaling": column_boxes(transformed), "after_scaling": column_boxes(X)},
        "correlation": {"columns": [columns[i] for i in corr_idx], "values": rounded(corr, 3),
                        "subset": bool(len(corr_idx) < len(columns))},
        "outliers": {
            "rows": [{"name": result["names"][i], "row": result["row_numbers"][i],
                      "distance": round(float(distance[i]), 4), "robust_z": round(float(z[i]), 2)} for i in top],
            "histogram": {"counts": counts.tolist(), "edges": rounded(edges)},
        },
    }
