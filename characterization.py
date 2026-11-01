"""Cluster characterization, computed on all rows of the preprocessed data that was clustered.

All statistics are descriptive. The clusters were built to differ, so test p-values are not valid
inference; effect sizes (Cliff's delta, epsilon squared) are the primary output.
"""
import math

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import false_discovery_control, kruskal, rankdata

MEDOID_SAMPLE = 1000        # medoids of larger clusters are estimated on a random subset of members
N_REPRESENTATIVES = 5
TOP_FEATURES = 15
MAX_OUTLIERS = 100
MAX_NOISE_NAMES = 500
MAX_DETAIL_CLUSTERS = 100   # profile bands and the full effect-size matrix are kept for the largest clusters

TEST_NOTE = ("Descriptive only: the clusters were formed to differ, so these p-values are not valid inference "
             "and are expected to be tiny. Use the effect sizes to judge how strongly a feature differs.")


def _round(values, digits=4):
    """Rounded nested lists with NaN/inf replaced by None (valid JSON)."""
    array = np.round(np.asarray(values, dtype=float), digits)
    if array.ndim > 1:
        return [_round(row, digits) for row in array]
    return [float(v) if math.isfinite(v) else None for v in array]


def _number(value, digits=4):
    return round(float(value), digits) if value is not None and math.isfinite(value) else None


def cluster_order(labels):
    """Real clusters in numeric order, noise last."""
    return sorted(set(labels.tolist()), key=lambda c: (c == -1, c))


def medoid_index(X, idx, rng):
    sample = idx if len(idx) <= MEDOID_SAMPLE else np.sort(rng.choice(idx, MEDOID_SAMPLE, replace=False))
    return int(sample[np.argmin(cdist(X[sample], X[sample]).sum(axis=1))])


def feature_tests(X, labels, columns):
    """Kruskal–Wallis per feature across the real clusters, with epsilon squared and BH-adjusted p-values."""
    real = [c for c in cluster_order(labels) if c != -1]
    groups = [np.flatnonzero(labels == c) for c in real]
    if len(groups) < 2:
        return None
    n = int(sum(len(g) for g in groups))
    h_values, p_values = [], []
    for j in range(X.shape[1]):
        try:
            h, p = kruskal(*[X[g, j] for g in groups])
        except ValueError:  # all values identical
            h, p = np.nan, np.nan
        h_values.append(h)
        p_values.append(p)
    h_values, p_values = np.array(h_values, dtype=float), np.array(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)
    valid = np.isfinite(p_values)
    if valid.any():
        q_values[valid] = false_discovery_control(p_values[valid], method="bh")
    epsilon_squared = h_values / (n - 1)
    rows = [{"feature": str(columns[j]), "H": _number(h_values[j], 3), "epsilon_squared": _number(epsilon_squared[j]),
             "p": float(p_values[j]) if valid[j] else None, "q": float(q_values[j]) if valid[j] else None}
            for j in range(X.shape[1])]
    rows.sort(key=lambda row: -(row["epsilon_squared"] if row["epsilon_squared"] is not None else -1))
    return {"test": "Kruskal–Wallis", "correction": "Benjamini–Hochberg", "effect_size": "epsilon squared (H / (n − 1))",
            "n_groups": len(groups), "n": n, "noise_excluded": bool(np.any(labels == -1)), "note": TEST_NOTE, "rows": rows}


def characterize(X, labels, names, columns):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = X.shape[0]
    rng = np.random.default_rng(0)
    clusters = cluster_order(labels)
    ranks = rankdata(X, axis=0)  # average ranks per feature; Cliff's delta = 2·AUC − 1

    sizes = {c: int(np.sum(labels == c)) for c in clusters}
    detail_clusters = set(sorted(clusters, key=lambda c: -sizes[c])[:MAX_DETAIL_CLUSTERS])
    profiles = {key: [] for key in ("lower", "q1", "median", "q3", "upper", "mean")}
    profile_clusters, delta_clusters, delta_rows = [], [], []
    entries, outliers = [], []

    for c in clusters:
        idx = np.flatnonzero(labels == c)
        rows = X[idx]
        q1, median, q3 = np.percentile(rows, [25, 50, 75], axis=0)
        medoid = medoid_index(X, idx, rng)
        closest = idx[np.argsort(np.linalg.norm(rows - X[medoid], axis=1), kind="stable")[:N_REPRESENTATIVES]]
        entry = {"cluster": int(c), "size": len(idx), "fraction": len(idx) / n,
                 "medoid": names[medoid], "representatives": [names[i] for i in closest], "top_features": []}

        if c in detail_clusters:
            iqr = q3 - q1
            profile_clusters.append(int(c))
            profiles["q1"].append(q1)
            profiles["median"].append(median)
            profiles["q3"].append(q3)
            profiles["mean"].append(rows.mean(axis=0))
            # Box-plot whiskers: most extreme values within 1.5 IQR of the quartiles.
            profiles["lower"].append(np.where(rows >= q1 - 1.5 * iqr, rows, np.inf).min(axis=0))
            profiles["upper"].append(np.where(rows <= q3 + 1.5 * iqr, rows, -np.inf).max(axis=0))

        if c != -1 and len(idx) < n:
            n_in, n_out = len(idx), n - len(idx)
            u = ranks[idx].sum(axis=0) - n_in * (n_in + 1) / 2
            delta = 2 * u / (n_in * n_out) - 1
            if c in detail_clusters:
                delta_clusters.append(int(c))
                delta_rows.append(delta)
            top = np.argsort(-np.abs(delta), kind="stable")[:TOP_FEATURES]
            rest_median = np.median(X[labels != c][:, top], axis=0)
            entry["top_features"] = [
                {"feature": str(columns[j]), "cliffs_delta": _number(delta[j]),
                 "median_in": _number(median[j]), "median_rest": _number(rest_median[k])}
                for k, j in enumerate(top)
            ]

        if c != -1 and len(idx) >= 5:
            # Distance to the cluster's median profile as a robust z-score within the cluster.
            distance = np.linalg.norm(rows - median, axis=1)
            mad = 1.4826 * np.median(np.abs(distance - np.median(distance)))
            z = (distance - np.median(distance)) / mad if mad > 0 else np.zeros_like(distance)
            outliers.extend(zip(idx.tolist(), distance.tolist(), z.tolist()))
        entries.append(entry)

    outliers.sort(key=lambda item: -item[2])
    noise_idx = np.flatnonzero(labels == -1)
    return {
        "space": "preprocessed data used for clustering (all columns), all rows",
        "columns": [str(column) for column in columns],
        "clusters": entries,
        "profiles": {"clusters": profile_clusters, **{key: _round(values) for key, values in profiles.items()},
                     "truncated": len(profile_clusters) < len(clusters)},
        "cliffs_delta": {"clusters": delta_clusters, "values": _round(delta_rows),
                         "description": "Cliff's delta of each cluster vs. all other rows (−1 … 1; |δ| ≥ 0.474 is often called large)"},
        "feature_tests": feature_tests(X, labels, columns),
        "outliers": [{"name": names[i], "cluster": int(labels[i]), "distance": _number(d), "robust_z": _number(z, 2)}
                     for i, d, z in outliers[:MAX_OUTLIERS]],
        "noise": {"count": int(noise_idx.size), "names": [names[i] for i in noise_idx[:MAX_NOISE_NAMES]],
                  "truncated": bool(noise_idx.size > MAX_NOISE_NAMES)},
        "settings": {"medoid_sample": MEDOID_SAMPLE, "representatives": N_REPRESENTATIVES,
                     "top_features": TOP_FEATURES, "max_detail_clusters": MAX_DETAIL_CLUSTERS},
    }
