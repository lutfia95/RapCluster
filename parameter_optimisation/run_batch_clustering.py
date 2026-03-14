from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from hdbscan import HDBSCAN
from sklearn import cluster, mixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import umap


LOGGER = logging.getLogger("rapcluster.batch")


DEFAULT_SCORING_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": {
        "silhouette": 0.20,
        "calinski_harabasz": 0.10,
        "davies_bouldin": 0.10,
        "stability_ari": 0.30,
        "assigned_fraction": 0.15,
        "singleton_penalty": 0.05,
        "cluster_count_penalty": 0.05,
        "runtime_penalty": 0.05,
    },
    "stability_first": {
        "silhouette": 0.15,
        "calinski_harabasz": 0.05,
        "davies_bouldin": 0.05,
        "stability_ari": 0.45,
        "assigned_fraction": 0.15,
        "singleton_penalty": 0.05,
        "cluster_count_penalty": 0.05,
        "runtime_penalty": 0.05,
    },
    "quality_first": {
        "silhouette": 0.30,
        "calinski_harabasz": 0.15,
        "davies_bouldin": 0.15,
        "stability_ari": 0.20,
        "assigned_fraction": 0.10,
        "singleton_penalty": 0.04,
        "cluster_count_penalty": 0.03,
        "runtime_penalty": 0.03,
    },
    "coverage_first": {
        "silhouette": 0.15,
        "calinski_harabasz": 0.05,
        "davies_bouldin": 0.10,
        "stability_ari": 0.25,
        "assigned_fraction": 0.25,
        "singleton_penalty": 0.08,
        "cluster_count_penalty": 0.07,
        "runtime_penalty": 0.05,
    },
    "fast": {
        "silhouette": 0.15,
        "calinski_harabasz": 0.05,
        "davies_bouldin": 0.10,
        "stability_ari": 0.20,
        "assigned_fraction": 0.15,
        "singleton_penalty": 0.05,
        "cluster_count_penalty": 0.05,
        "runtime_penalty": 0.25,
    },
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "MiniBatchKMeans": {"n_clusters": 8, "random_state": 42, "init": "k-means++"},
    "KMeans": {"n_clusters": 8, "random_state": 42, "init": "k-means++", "algorithm": "lloyd"},
    "AffinityPropagation": {"damping": 0.5, "preference": -50, "affinity": "euclidean"},
    "MeanShift": {"bandwidth": 1.0, "cluster_all": False},
    "SpectralClustering": {"n_clusters": 8, "affinity": "nearest_neighbors", "n_neighbors": 10, "assign_labels": "kmeans", "random_state": 42},
    "AgglomerativeClustering": {"n_clusters": 8, "linkage": "ward", "metric": "euclidean"},
    "DBSCAN": {"eps": 0.5, "min_samples": 5, "metric": "euclidean", "leaf_size": 30, "algorithm": "auto"},
    "HDBSCAN": {"min_cluster_size": 5, "min_samples": 5, "cluster_selection_epsilon": 0.0, "max_cluster_size": 40, "metric": "euclidean", "leaf_size": 20, "cluster_selection_method": "eom"},
    "OPTICS": {"min_samples": 5, "xi": 0.05, "min_cluster_size": 0.05, "metric": "minkowski", "predecessor_correction": True, "algorithm": "auto"},
    "BIRCH": {"n_clusters": 8, "threshold": 0.5, "branching_factor": 50, "compute_labels": True},
    "GaussianMixture": {"n_components": 8, "covariance_type": "full", "random_state": 42, "init_params": "kmeans"},
}


@dataclass
class RepeatResult:
    labels: np.ndarray
    runtime_seconds: float
    metrics: dict[str, Any]
    reduction_method: str
    reduction_params: dict[str, Any]
    clustering_params: dict[str, Any]
    seed: int


class ConfigError(RuntimeError):
    pass


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RapCluster-compatible batch clustering with config-driven hyperparameter search.",
    )
    parser.add_argument(
        "config",
        help="Path to config.yml",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    LOGGER.info("Loading config from %s", path)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError("Top-level config must be a mapping.")
    LOGGER.info("Config loaded successfully")
    return data


def read_table(file_path: str, file_type: str) -> pd.DataFrame:
    LOGGER.info("Reading input table %s as %s", file_path, file_type)
    file_type_norm = file_type.lower()
    if file_type_norm == "csv":
        return pd.read_csv(file_path)
    if file_type_norm == "tsv":
        return pd.read_csv(file_path, sep="\t")
    if file_type_norm == "xlsx":
        return pd.read_excel(file_path)
    raise ConfigError(f"Unsupported file_type: {file_type}")


def load_data(config: Mapping[str, Any]) -> tuple[np.ndarray, list[str], np.ndarray, pd.DataFrame]:
    input_cfg = as_mapping(config.get("input"), "input")
    prep_cfg = as_mapping(config.get("preprocessing", {}), "preprocessing")

    file_path = str(input_cfg["file"])
    file_type = str(input_cfg.get("file_type", infer_file_type(file_path)))
    name_column = str(input_cfg.get("name_column", "name"))
    intensity_start_index = int(input_cfg["intensity_start_index"])
    LOGGER.info(
        "Preparing data from %s (name_column=%s, intensity_start_index=%d)",
        file_path,
        name_column,
        intensity_start_index,
    )

    df = read_table(file_path, file_type)
    LOGGER.info("Input table shape: rows=%d, columns=%d", df.shape[0], df.shape[1])
    if intensity_start_index < 0 or intensity_start_index >= len(df.columns):
        raise ConfigError(
            f"intensity_start_index={intensity_start_index} is outside the available column range 0..{len(df.columns) - 1}.",
        )

    intensity_cols = df.columns[intensity_start_index:]
    for col in intensity_cols:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)

    data = df[intensity_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    missing_strategy = str(prep_cfg.get("missing_value_strategy", "zero")).lower()
    LOGGER.info("Applying missing value strategy: %s", missing_strategy)
    if missing_strategy == "zero":
        data[np.isnan(data)] = 0.0
        valid_mask = np.ones(data.shape[0], dtype=bool)
    elif missing_strategy == "drop_row":
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
    else:
        raise ConfigError("preprocessing.missing_value_strategy must be one of: zero, drop_row")

    filtered_df = df.loc[valid_mask].copy()

    if bool(prep_cfg.get("log_transform", True)):
        LOGGER.info("Applying log2(x + 1) transform")
        data = np.log2(data + 1.0)

    if bool(prep_cfg.get("drop_all_zero_rows", True)):
        before_drop = data.shape[0]
        nonzero_mask = ~(np.all(data == 0.0, axis=1))
        data = data[nonzero_mask]
        filtered_df = filtered_df.loc[nonzero_mask].copy()
        LOGGER.info("Dropped %d all-zero rows", before_drop - data.shape[0])

    if name_column in filtered_df.columns:
        names = filtered_df[name_column].astype(str).fillna("").replace("", np.nan).fillna("NA").tolist()
    else:
        warnings.warn(f"Name column '{name_column}' not found. Using default names.")
        names = [f"Node_{idx + 1}" for idx in range(data.shape[0])]

    scale = bool(prep_cfg.get("scale", True))
    scale_axis = str(prep_cfg.get("scale_axis", "row")).lower()
    if scale:
        LOGGER.info("Scaling data with StandardScaler on %s axis", scale_axis)
        scaler = StandardScaler()
        if scale_axis == "row":
            data_scaled = scaler.fit_transform(data.T).T
        elif scale_axis == "column":
            data_scaled = scaler.fit_transform(data)
        else:
            raise ConfigError("preprocessing.scale_axis must be either 'row' or 'column'.")
    else:
        data_scaled = data.copy()

    data_log10_transformed = np.log10(np.maximum(data, 0.0) + 1.0)
    LOGGER.info("Finished preprocessing: samples=%d, features=%d", data_scaled.shape[0], data_scaled.shape[1])
    return data_scaled, names, data_log10_transformed, filtered_df


def infer_file_type(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix in {".xls", ".xlsx"}:
        return "xlsx"
    raise ConfigError(f"Could not infer file type from extension: {suffix}")


def as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping.")
    return dict(value)


def normalize_nullable(value: Any) -> Any:
    if isinstance(value, str) and value.lower() in {"null", "none"}:
        return None
    return value


def build_parameter_sets(algorithm_name: str, algorithm_cfg: Mapping[str, Any], search_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    enabled = bool(algorithm_cfg.get("enabled", False))
    if not enabled:
        LOGGER.info("Skipping disabled algorithm: %s", algorithm_name)
        return []

    strategy = str(search_cfg.get("strategy", "grid")).lower()
    defaults = dict(DEFAULT_PARAMS.get(algorithm_name, {}))

    if strategy == "manual_list":
        manual_list = algorithm_cfg.get("param_list", [])
        if not isinstance(manual_list, Sequence):
            raise ConfigError(f"algorithms.{algorithm_name}.param_list must be a list.")
        result: list[dict[str, Any]] = []
        for item in manual_list:
            if not isinstance(item, Mapping):
                raise ConfigError(f"Every param_list entry for {algorithm_name} must be a mapping.")
            merged = defaults | {k: normalize_nullable(v) for k, v in item.items()}
            result.append(clean_algorithm_params(algorithm_name, merged))
        LOGGER.info("Built %d manual parameter sets for %s", len(result), algorithm_name)
        return result

    if strategy != "grid":
        raise ConfigError("search.strategy must be 'grid' or 'manual_list'.")

    param_grid = algorithm_cfg.get("param_grid", {})
    if not isinstance(param_grid, Mapping):
        raise ConfigError(f"algorithms.{algorithm_name}.param_grid must be a mapping.")

    if not param_grid:
        return [clean_algorithm_params(algorithm_name, defaults)]

    keys = list(param_grid.keys())
    values_product: list[list[Any]] = []
    for key in keys:
        raw_values = param_grid[key]
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
            raise ConfigError(f"algorithms.{algorithm_name}.param_grid.{key} must be a list.")
        values_product.append([normalize_nullable(value) for value in raw_values])

    result = []
    for combo in itertools.product(*values_product):
        merged = defaults | dict(zip(keys, combo, strict=True))
        result.append(clean_algorithm_params(algorithm_name, merged))
    LOGGER.info("Built %d grid parameter sets for %s", len(result), algorithm_name)
    return result


def clean_algorithm_params(algorithm_name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = {key: normalize_nullable(value) for key, value in params.items()}

    if algorithm_name == "AgglomerativeClustering" and cleaned.get("linkage") == "ward":
        cleaned["metric"] = "euclidean"

    if algorithm_name == "SpectralClustering" and cleaned.get("affinity") != "nearest_neighbors":
        cleaned.pop("n_neighbors", None)

    if algorithm_name == "MeanShift" and cleaned.get("bandwidth") is None:
        cleaned.pop("bandwidth", None)

    if algorithm_name == "HDBSCAN" and cleaned.get("max_cluster_size") is None:
        cleaned.pop("max_cluster_size", None)

    if algorithm_name == "BIRCH" and cleaned.get("n_clusters") is None:
        cleaned["n_clusters"] = None

    return cleaned


def prepare_reduction_base(config: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    reduction_cfg = as_mapping(config.get("dimensionality_reduction", {}), "dimensionality_reduction")
    enabled = bool(reduction_cfg.get("enabled", False))
    if not enabled:
        LOGGER.info("Dimensionality reduction disabled")
        return False, "None", {}
    method = str(reduction_cfg.get("method", "None"))
    params = as_mapping(reduction_cfg.get("params", {}), "dimensionality_reduction.params")
    LOGGER.info("Dimensionality reduction enabled: method=%s params=%s", method, dict(params))
    return True, method, {key: normalize_nullable(value) for key, value in params.items()}


def apply_dimensionality_reduction(X: np.ndarray, method: str, params: Mapping[str, Any]) -> np.ndarray:
    LOGGER.info("Applying dimensionality reduction: method=%s", method)
    if method == "None":
        return X
    if method == "UMAP":
        reducer = umap.UMAP(**params)
        return reducer.fit_transform(X)
    if method == "TSNE":
        reducer = TSNE(**params)
        return reducer.fit_transform(X)
    if method == "PCA":
        reducer = PCA(**params)
        return reducer.fit_transform(X)
    raise ConfigError(f"Unsupported dimensionality reduction method: {method}")


def build_connectivity(X: np.ndarray) -> Any:
    if X.shape[0] < 3:
        return None
    n_neighbors = min(X.shape[0] - 1, 10)
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
    return 0.5 * (graph + graph.T)


def instantiate_algorithm(algorithm_name: str, params: Mapping[str, Any], connectivity: Any) -> Any:
    config = dict(params)

    if algorithm_name == "MiniBatchKMeans":
        return cluster.MiniBatchKMeans(**config)
    if algorithm_name == "KMeans":
        return cluster.KMeans(**config)
    if algorithm_name == "AffinityPropagation":
        if config.get("affinity") == "precomputed":
            config["affinity"] = "euclidean"
        config.pop("random_state", None)
        return cluster.AffinityPropagation(**config)
    if algorithm_name == "MeanShift":
        config = {key: value for key, value in config.items() if value is not None and key != "n_jobs"}
        return cluster.MeanShift(**config)
    if algorithm_name == "SpectralClustering":
        return cluster.SpectralClustering(**config)
    if algorithm_name == "AgglomerativeClustering":
        if config.get("linkage") == "ward":
            config["metric"] = "euclidean"
        if connectivity is not None:
            return cluster.AgglomerativeClustering(connectivity=connectivity, **config)
        return cluster.AgglomerativeClustering(**config)
    if algorithm_name == "DBSCAN":
        return cluster.DBSCAN(**config)
    if algorithm_name == "HDBSCAN":
        return HDBSCAN(**config)
    if algorithm_name == "OPTICS":
        return cluster.OPTICS(**config)
    if algorithm_name == "BIRCH":
        return cluster.Birch(**config)
    if algorithm_name == "GaussianMixture":
        if config.get("init_params") == "‘k-means++":
            config["init_params"] = "k-means++"
        if config.get("init_params") == "‘random_from_data’":
            config["init_params"] = "random_from_data"
        return mixture.GaussianMixture(**config)
    raise ConfigError(f"Unsupported algorithm: {algorithm_name}")


def fit_predict_labels(estimator: Any, X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(estimator, "fit_predict"):
            labels = estimator.fit_predict(X)
        else:
            estimator.fit(X)
            labels = getattr(estimator, "labels_", estimator.predict(X))
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise RuntimeError("Estimator did not produce a one-dimensional label vector.")
    return labels_arr.astype(int, copy=False)


def evaluate_single_run(
    labels: np.ndarray,
    X_eval: np.ndarray,
    small_cluster_threshold: int,
    ignore_noise_for_internal_scores: bool,
) -> dict[str, Any]:
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise RuntimeError("Labels must be one-dimensional.")

    n_total = int(labels_arr.size)
    noise_mask = labels_arr == -1
    non_noise_mask = ~noise_mask
    n_noise = int(noise_mask.sum())
    assigned_fraction = float(non_noise_mask.mean()) if n_total else 0.0
    noise_fraction = 1.0 - assigned_fraction

    assigned_labels = labels_arr[non_noise_mask]
    cluster_sizes_series = pd.Series(assigned_labels).value_counts(sort=False)
    cluster_sizes = cluster_sizes_series.to_dict()
    n_clusters = int(cluster_sizes_series.shape[0])
    n_singletons = int((cluster_sizes_series == 1).sum())
    singleton_fraction = float(n_singletons / n_clusters) if n_clusters else 1.0
    small_cluster_fraction = 0.0
    if n_clusters:
        small_clusters = cluster_sizes_series[cluster_sizes_series <= small_cluster_threshold]
        small_cluster_fraction = float(small_clusters.sum() / max(int(cluster_sizes_series.sum()), 1))

    silhouette = -1.0
    calinski_harabasz = 0.0
    davies_bouldin = -1.0

    if ignore_noise_for_internal_scores and n_noise > 0:
        X_score = X_eval[non_noise_mask]
        labels_score = assigned_labels
    else:
        X_score = X_eval
        labels_score = labels_arr

    unique_score_labels = np.unique(labels_score)
    if X_score.shape[0] >= 2 and unique_score_labels.size >= 2:
        try:
            silhouette = float(silhouette_score(X_score, labels_score))
            calinski_harabasz = float(calinski_harabasz_score(X_score, labels_score))
            davies_bouldin = float(davies_bouldin_score(X_score, labels_score))
        except Exception:
            silhouette = -1.0
            calinski_harabasz = 0.0
            davies_bouldin = -1.0

    return {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski_harabasz,
        "davies_bouldin_score": davies_bouldin,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": noise_fraction,
        "assigned_fraction": assigned_fraction,
        "n_singletons": n_singletons,
        "singleton_fraction": singleton_fraction,
        "small_cluster_fraction": small_cluster_fraction,
        "cluster_sizes_json": json.dumps(cluster_sizes, sort_keys=True),
    }


def cluster_count_penalty(n_clusters: int, n_samples: int) -> float:
    if n_samples <= 1:
        return 1.0
    if n_clusters <= 1:
        return 1.0
    upper_soft_limit = max(2.0, math.sqrt(float(n_samples)))
    if n_clusters <= upper_soft_limit:
        return 0.0
    penalty = (float(n_clusters) - upper_soft_limit) / upper_soft_limit
    return float(min(1.0, max(0.0, penalty)))


def inject_seed(params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    seeded = dict(params)
    if "random_state" in seeded:
        seeded["random_state"] = seed
    return seeded


def summarize_repeats(
    algorithm_name: str,
    param_index: int,
    repeat_results: Sequence[RepeatResult],
    n_samples: int,
) -> dict[str, Any]:
    LOGGER.info(
        "Summarizing %d repeat(s) for %s param_set=%d",
        len(repeat_results),
        algorithm_name,
        param_index,
    )
    pairwise_ari_values: list[float] = []
    if len(repeat_results) >= 2:
        for left_idx in range(len(repeat_results)):
            for right_idx in range(left_idx + 1, len(repeat_results)):
                left = repeat_results[left_idx].labels
                right = repeat_results[right_idx].labels
                pairwise_ari_values.append(float(adjusted_rand_score(left, right)))

    summary: dict[str, Any] = {
        "algorithm": algorithm_name,
        "param_set_index": param_index,
        "n_repeats": len(repeat_results),
        "reduction_method": repeat_results[0].reduction_method,
        "reduction_params_json": json.dumps(repeat_results[0].reduction_params, sort_keys=True),
        "params_json": json.dumps(repeat_results[0].clustering_params, sort_keys=True),
        "seed_list_json": json.dumps([item.seed for item in repeat_results]),
    }

    metric_keys = [
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "assigned_fraction",
        "n_singletons",
        "singleton_fraction",
        "small_cluster_fraction",
        "runtime_seconds",
    ]

    for key in metric_keys:
        values = [
            float(item.runtime_seconds) if key == "runtime_seconds" else float(item.metrics[key])
            for item in repeat_results
        ]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values, ddof=0))

    summary["cluster_sizes_json"] = repeat_results[0].metrics["cluster_sizes_json"]
    summary["stability_ari_mean"] = float(np.mean(pairwise_ari_values)) if pairwise_ari_values else 1.0
    summary["stability_ari_std"] = float(np.std(pairwise_ari_values, ddof=0)) if pairwise_ari_values else 0.0
    summary["cluster_count_penalty"] = cluster_count_penalty(
        int(round(summary["n_clusters_mean"])),
        n_samples,
    )
    return summary


def normalize_series(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-12):
        return pd.Series(np.full(numeric.shape[0], 0.5), index=numeric.index, dtype=float)
    normalized = (numeric - min_value) / (max_value - min_value)
    if higher_is_better:
        return normalized.astype(float)
    return (1.0 - normalized).astype(float)


def apply_overall_scoring(results_df: pd.DataFrame, config: Mapping[str, Any], n_samples: int) -> pd.DataFrame:
    scoring_cfg = as_mapping(config.get("scoring", {}), "scoring")
    mode = str(scoring_cfg.get("mode", "balanced"))
    LOGGER.info("Scoring %d result rows using mode=%s", results_df.shape[0], mode)
    weights = dict(DEFAULT_SCORING_WEIGHTS.get(mode, DEFAULT_SCORING_WEIGHTS["balanced"]))
    weights.update(as_mapping(scoring_cfg.get("weights", {}), "scoring.weights"))

    output = results_df.copy()
    output["silhouette_norm"] = normalize_series(output["silhouette_score_mean"], higher_is_better=True)
    output["calinski_harabasz_norm"] = normalize_series(output["calinski_harabasz_score_mean"], higher_is_better=True)
    db_input = output["davies_bouldin_score_mean"].replace(-1.0, np.nan).fillna(output["davies_bouldin_score_mean"].max())
    output["davies_bouldin_norm"] = normalize_series(db_input, higher_is_better=False)
    output["stability_ari_norm"] = normalize_series(output["stability_ari_mean"], higher_is_better=True)
    output["assigned_fraction_norm"] = normalize_series(output["assigned_fraction_mean"], higher_is_better=True)
    output["singleton_penalty_norm"] = normalize_series(output["singleton_fraction_mean"], higher_is_better=False)
    output["cluster_count_penalty_norm"] = normalize_series(output["cluster_count_penalty"], higher_is_better=False)
    output["runtime_penalty_norm"] = normalize_series(output["runtime_seconds_mean"], higher_is_better=False)

    overall = (
        weights["silhouette"] * output["silhouette_norm"]
        + weights["calinski_harabasz"] * output["calinski_harabasz_norm"]
        + weights["davies_bouldin"] * output["davies_bouldin_norm"]
        + weights["stability_ari"] * output["stability_ari_norm"]
        + weights["assigned_fraction"] * output["assigned_fraction_norm"]
        + weights["singleton_penalty"] * output["singleton_penalty_norm"]
        + weights["cluster_count_penalty"] * output["cluster_count_penalty_norm"]
        + weights["runtime_penalty"] * output["runtime_penalty_norm"]
    )

    penalties_cfg = as_mapping(scoring_cfg.get("penalties", {}), "scoring.penalties")
    penalty_factor = np.ones(output.shape[0], dtype=float)

    if bool(penalties_cfg.get("one_cluster", True)):
        penalty_factor *= np.where(output["n_clusters_mean"] <= 1.0, 0.25, 1.0)
    if bool(penalties_cfg.get("too_many_noise_points", True)):
        penalty_factor *= np.where(output["assigned_fraction_mean"] < 0.5, 0.70, 1.0)
    if bool(penalties_cfg.get("too_many_singletons", True)):
        penalty_factor *= np.where(output["singleton_fraction_mean"] > 0.5, 0.80, 1.0)
    if bool(penalties_cfg.get("degenerate_solution", True)):
        degenerate = (output["silhouette_score_mean"] < 0.0) & (output["n_clusters_mean"] <= 1.0)
        penalty_factor *= np.where(degenerate, 0.10, 1.0)

    output["overall_score"] = overall * penalty_factor
    output = output.sort_values(["overall_score", "stability_ari_mean", "silhouette_score_mean"], ascending=[False, False, False]).reset_index(drop=True)
    output["rank_global"] = np.arange(1, output.shape[0] + 1)
    output["rank_within_algorithm"] = output.groupby("algorithm").cumcount() + 1
    output["n_samples"] = n_samples
    output["scoring_mode"] = mode
    return output


def run_single_repeat(
    X_original: np.ndarray,
    evaluation_cfg: Mapping[str, Any],
    reduction_enabled: bool,
    reduction_method: str,
    reduction_base_params: Mapping[str, Any],
    algorithm_name: str,
    algorithm_params: Mapping[str, Any],
    seed: int,
) -> RepeatResult:
    LOGGER.info(
        "Running repeat: algorithm=%s seed=%d reduction=%s",
        algorithm_name,
        seed,
        reduction_method if reduction_enabled else "None",
    )
    reduction_params = inject_seed(reduction_base_params, seed)
    X_reduced = apply_dimensionality_reduction(X_original, reduction_method, reduction_params) if reduction_enabled else X_original

    clustering_space = str(evaluation_cfg.get("clustering_space", "reduced")).lower()
    evaluation_space = str(evaluation_cfg.get("evaluation_space", "original")).lower()
    ignore_noise_for_internal_scores = bool(evaluation_cfg.get("ignore_noise_for_internal_scores", True))
    small_cluster_threshold = int(evaluation_cfg.get("small_cluster_threshold", 2))

    if clustering_space == "original":
        X_cluster = X_original
    elif clustering_space == "reduced":
        X_cluster = X_reduced
    else:
        raise ConfigError("evaluation.clustering_space must be either 'original' or 'reduced'.")

    if evaluation_space == "original":
        X_eval = X_original
    elif evaluation_space == "reduced":
        X_eval = X_reduced
    else:
        raise ConfigError("evaluation.evaluation_space must be either 'original' or 'reduced'.")

    clustering_params = inject_seed(algorithm_params, seed)
    connectivity = build_connectivity(X_cluster) if algorithm_name == "AgglomerativeClustering" else None
    estimator = instantiate_algorithm(algorithm_name, clustering_params, connectivity)

    start = time.time()
    labels = fit_predict_labels(estimator, X_cluster)
    runtime_seconds = time.time() - start
    metrics = evaluate_single_run(
        labels=labels,
        X_eval=X_eval,
        small_cluster_threshold=small_cluster_threshold,
        ignore_noise_for_internal_scores=ignore_noise_for_internal_scores,
    )
    LOGGER.info(
        "Completed repeat: algorithm=%s seed=%d runtime=%.3fs clusters=%d assigned_fraction=%.3f silhouette=%.3f",
        algorithm_name,
        seed,
        runtime_seconds,
        metrics["n_clusters"],
        metrics["assigned_fraction"],
        metrics["silhouette_score"],
    )
    return RepeatResult(
        labels=labels,
        runtime_seconds=runtime_seconds,
        metrics=metrics,
        reduction_method=reduction_method if reduction_enabled else "None",
        reduction_params=dict(reduction_params),
        clustering_params=dict(clustering_params),
        seed=seed,
    )


def save_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing table: %s", path)
    df.to_csv(path, sep="\t", index=False)


def save_best_labels_table(
    path: Path,
    names: Sequence[str],
    raw_df: pd.DataFrame,
    best_repeat: RepeatResult,
    input_cfg: Mapping[str, Any],
) -> None:
    LOGGER.info("Writing best labels table: %s", path)
    output_df = raw_df.copy().reset_index(drop=True)
    output_df.insert(0, "_rapcluster_name", list(names))
    output_df["_rapcluster_cluster"] = best_repeat.labels
    output_df.to_csv(path, sep="\t", index=False)


def dump_yaml(path: Path, data: Mapping[str, Any]) -> None:
    LOGGER.info("Writing YAML: %s", path)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def main() -> int:
    configure_logging()
    args = parse_args()
    LOGGER.info("Starting RapCluster batch clustering")
    config = load_yaml(args.config)

    X_original, names, _data_log10_transformed, raw_df = load_data(config)
    n_samples = int(X_original.shape[0])
    if n_samples < 2:
        raise ConfigError("Need at least two rows after preprocessing to run clustering.")
    LOGGER.info("Data ready for clustering: n_samples=%d", n_samples)

    evaluation_cfg = as_mapping(config.get("evaluation", {}), "evaluation")
    search_cfg = as_mapping(config.get("search", {}), "search")
    output_cfg = as_mapping(config.get("output", {}), "output")
    algorithms_cfg = as_mapping(config.get("algorithms", {}), "algorithms")
    input_cfg = as_mapping(config.get("input", {}), "input")

    repeats = int(evaluation_cfg.get("repeats", 5))
    random_seeds = evaluation_cfg.get("random_seeds", [11, 22, 33, 44, 55])
    if not isinstance(random_seeds, Sequence) or isinstance(random_seeds, (str, bytes)):
        raise ConfigError("evaluation.random_seeds must be a list.")
    seeds = [int(seed) for seed in random_seeds[:repeats]]
    if len(seeds) < repeats:
        raise ConfigError("evaluation.random_seeds must contain at least as many entries as evaluation.repeats.")

    reduction_enabled, reduction_method, reduction_base_params = prepare_reduction_base(config)
    max_combinations = int(search_cfg.get("max_combinations_per_algorithm", 200))
    LOGGER.info("Search setup: repeats=%d max_combinations_per_algorithm=%d", repeats, max_combinations)

    results: list[dict[str, Any]] = []
    best_repeat_lookup: dict[tuple[str, int], RepeatResult] = {}

    for algorithm_name, algorithm_cfg_any in algorithms_cfg.items():
        LOGGER.info("Processing algorithm: %s", algorithm_name)
        algorithm_cfg = as_mapping(algorithm_cfg_any, f"algorithms.{algorithm_name}")
        parameter_sets = build_parameter_sets(algorithm_name, algorithm_cfg, search_cfg)
        if not parameter_sets:
            continue
        if len(parameter_sets) > max_combinations:
            warnings.warn(
                f"Algorithm {algorithm_name} generated {len(parameter_sets)} parameter combinations; truncating to {max_combinations}.",
            )
            LOGGER.warning(
                "Algorithm %s generated %d parameter combinations; truncating to %d",
                algorithm_name,
                len(parameter_sets),
                max_combinations,
            )
            parameter_sets = parameter_sets[:max_combinations]

        for param_index, param_set in enumerate(parameter_sets, start=1):
            LOGGER.info(
                "Evaluating %s param_set=%d/%d params=%s",
                algorithm_name,
                param_index,
                len(parameter_sets),
                json.dumps(param_set, sort_keys=True),
            )
            repeat_results: list[RepeatResult] = []
            for seed in seeds:
                try:
                    repeat_result = run_single_repeat(
                        X_original=X_original,
                        evaluation_cfg=evaluation_cfg,
                        reduction_enabled=reduction_enabled,
                        reduction_method=reduction_method,
                        reduction_base_params=reduction_base_params,
                        algorithm_name=algorithm_name,
                        algorithm_params=param_set,
                        seed=seed,
                    )
                    repeat_results.append(repeat_result)
                except Exception as exc:
                    warnings.warn(
                        f"Skipping repeat for {algorithm_name} param_set={param_index} seed={seed}: {exc}",
                    )
                    LOGGER.warning(
                        "Skipping repeat: algorithm=%s param_set=%d seed=%d error=%s",
                        algorithm_name,
                        param_index,
                        seed,
                        exc,
                    )

            if not repeat_results:
                LOGGER.warning("No successful repeats for %s param_set=%d", algorithm_name, param_index)
                continue

            summary = summarize_repeats(
                algorithm_name=algorithm_name,
                param_index=param_index,
                repeat_results=repeat_results,
                n_samples=n_samples,
            )
            results.append(summary)
            LOGGER.info(
                "Summary complete: algorithm=%s param_set=%d overall_inputs ready, stability_ari_mean=%.3f",
                algorithm_name,
                param_index,
                summary["stability_ari_mean"],
            )

            best_repeat = max(
                repeat_results,
                key=lambda item: (
                    item.metrics["assigned_fraction"],
                    item.metrics["silhouette_score"],
                    -item.runtime_seconds,
                ),
            )
            best_repeat_lookup[(algorithm_name, param_index)] = best_repeat

    if not results:
        raise RuntimeError("No successful clustering runs were completed.")

    results_df = pd.DataFrame(results)
    ranked_df = apply_overall_scoring(results_df, config=config, n_samples=n_samples)
    LOGGER.info("Ranking complete. Total successful parameter summaries: %d", ranked_df.shape[0])

    output_dir = Path(str(output_cfg.get("output_dir", "rapcluster_results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory ready: %s", output_dir)

    save_table(output_dir / "all_runs.tsv", ranked_df)

    best_per_algorithm = ranked_df.sort_values(["algorithm", "overall_score"], ascending=[True, False]).groupby("algorithm", as_index=False).head(1)
    save_table(output_dir / "best_per_algorithm.tsv", best_per_algorithm)

    top_k = int(search_cfg.get("top_k_to_report", 10))
    save_table(output_dir / "top_global.tsv", ranked_df.head(top_k))
    save_table(output_dir / "best_overall.tsv", ranked_df.head(1))

    best_row = ranked_df.iloc[0]
    best_algorithm = str(best_row["algorithm"])
    best_param_index = int(best_row["param_set_index"])
    best_repeat = best_repeat_lookup[(best_algorithm, best_param_index)]
    LOGGER.info(
        "Best overall result: algorithm=%s param_set=%d overall_score=%.4f",
        best_algorithm,
        best_param_index,
        float(best_row["overall_score"]),
    )

    best_config = {
        "input": dict(input_cfg),
        "preprocessing": as_mapping(config.get("preprocessing", {}), "preprocessing"),
        "evaluation": dict(evaluation_cfg),
        "dimensionality_reduction": {
            "enabled": reduction_enabled,
            "method": reduction_method,
            "params": best_repeat.reduction_params,
        },
        "algorithm": {
            "name": best_algorithm,
            "params": best_repeat.clustering_params,
        },
        "summary": {
            "overall_score": float(best_row["overall_score"]),
            "rank_global": int(best_row["rank_global"]),
            "silhouette_score_mean": float(best_row["silhouette_score_mean"]),
            "calinski_harabasz_score_mean": float(best_row["calinski_harabasz_score_mean"]),
            "davies_bouldin_score_mean": float(best_row["davies_bouldin_score_mean"]),
            "stability_ari_mean": float(best_row["stability_ari_mean"]),
            "assigned_fraction_mean": float(best_row["assigned_fraction_mean"]),
            "runtime_seconds_mean": float(best_row["runtime_seconds_mean"]),
        },
    }
    dump_yaml(output_dir / "best_config.yml", best_config)

    if bool(output_cfg.get("save_labels_table", True)):
        save_best_labels_table(
            output_dir / "best_overall_labels.tsv",
            names=names,
            raw_df=raw_df,
            best_repeat=best_repeat,
            input_cfg=input_cfg,
        )

    LOGGER.info("Finished successfully")
    print(f"Finished. Results written to: {output_dir}")
    print(f"Best overall: {best_algorithm} (param_set_index={best_param_index}, overall_score={best_row['overall_score']:.4f})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
