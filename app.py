import io
import os
import re
import sys
import html
import time
import zipfile
import platform
from importlib import metadata
import uuid
import shutil
import hashlib
import tempfile
import warnings
import glob
import fcntl
import contextlib
import gzip
import signal
import logging
import resource
import importlib
import threading
import multiprocessing
import multiprocessing.forkserver
from datetime import datetime, timezone
import umap
import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
try:
    from sklearn.utils._param_validation import InvalidParameterError
except ImportError:  # older scikit-learn
    class InvalidParameterError(ValueError):
        pass
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, mixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from matplotlib import colormaps
from flask import Flask, Response, request, jsonify, send_file
from flask_cors import CORS

import preprocessing as pp
from characterization import TEST_NOTE, characterize
from exploration import explore as explore_dataset
import stability as stab
import sweep as sw


app = Flask(__name__)
CORS(app) # Enable CORS for all routes


# Session storage lives on disk so that every gunicorn worker sees the same runs.
DATA_ROOT = os.environ.get("RAPCLUSTER_DATA_DIR", os.path.join(tempfile.gettempdir(), "rapcluster_sessions"))
SESSION_TTL_SECONDS = float(os.environ.get("RAPCLUSTER_SESSION_TTL_HOURS", 12)) * 3600
CLEANUP_INTERVAL_SECONDS = 600
MAX_RECORDED_WARNINGS = 20
ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_last_cleanup = 0.0

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DBCV_MAX_SAMPLES = 2000
STABILITY_RESAMPLES = 10
STABILITY_SEEDS = 5
STABILITY_FRACTION = 0.8
STABILITY_LIMITS = {"resamples": (3, 30), "seeds": (0, 10)}
SWEEP_TIME_FRACTION = 0.85  # no new sweep trial starts after this share of the job time limit

EXAMPLE_DATASETS = [
    {
        "id": "protein_profiles",
        "title": "Protein profiles (3000 × 120)",
        "description": "Fractionation-style intensity profiles; the original RapCluster example.",
        "path": "example_data.tsv",
        "name_column": "Nodes",
        "intensity_start_index": 1,
        "preprocessing": {"transform": "log2p1", "scaling": "row_zscore"},
    },
    {
        "id": "time_course",
        "title": "Time-course profiles (1200 × 24)",
        "description": "Six expression patterns (rising, falling, peaks, oscillating, dip) with varying abundance. Class column = true pattern.",
        "path": "examples/time_course_profiles.tsv",
        "name_column": "Name",
        "intensity_start_index": 2,
        "preprocessing": {"transform": "log2p1", "scaling": "row_zscore"},
    },
    {
        "id": "gaussian_blobs",
        "title": "Gaussian blobs (600 × 10)",
        "description": "Five well-separated groups; a good first test for KMeans or GaussianMixture. Class column = true group.",
        "path": "examples/gaussian_blobs.tsv",
        "name_column": "Name",
        "intensity_start_index": 2,
        "preprocessing": {"transform": "none", "scaling": "column_zscore"},
    },
    {
        "id": "density_noise",
        "title": "Uneven density with noise (700 × 8)",
        "description": "Three clusters of different spread plus scattered noise points; try DBSCAN, HDBSCAN or OPTICS. Class column = truth.",
        "path": "examples/density_with_noise.tsv",
        "name_column": "Name",
        "intensity_start_index": 2,
        "preprocessing": {"transform": "none", "scaling": "column_zscore"},
    },
    {
        "id": "iris",
        "title": "Iris flowers (150 × 4)",
        "description": "The classic small dataset with three species. Class column = species.",
        "path": "examples/iris.tsv",
        "name_column": "Name",
        "intensity_start_index": 2,
        "preprocessing": {"transform": "none", "scaling": "column_zscore"},
    },
]

# Capacity guidance per algorithm: (comfortable up to n, workable up to n) observations.
# Above the second value an algorithm is not recommended on the hosted server.
ALGORITHM_CAPACITY = {
    "MiniBatchKMeans": (200_000, 1_000_000, "Linear in the number of samples; suited to large data."),
    "BIRCH": (200_000, 1_000_000, "Single pass over the data with a compact tree; suited to large data."),
    "KMeans": (100_000, 500_000, "Roughly linear per iteration."),
    "GaussianMixture": (50_000, 200_000, "Linear in samples, but full covariances grow quickly with the number of features."),
    "BisectingKMeans": (100_000, 500_000, "Repeated 2-means splits; roughly linear in the number of samples."),
    "BayesianGaussianMixture": (20_000, 100_000, "Variational inference over all components; slower than GaussianMixture."),
    "DBSCAN": (20_000, 100_000, "Neighborhood queries; memory can grow sharply with a large eps."),
    "HDBSCAN": (20_000, 100_000, "Builds a minimum spanning tree; slows down in high dimensions."),
    "AgglomerativeClustering": (10_000, 30_000, "Hierarchical merging; memory and time grow faster than linear."),
    "OPTICS": (5_000, 30_000, "Reachability ordering is slow in practice."),
    "MeanShift": (5_000, 20_000, "Repeated neighborhood searches for every seed."),
    "SpectralClustering": (5_000, 20_000, "Eigen-decomposition of an n × n affinity graph."),
    "AffinityPropagation": (2_000, 5_000, "Quadratic time and memory in the number of samples."),
}
SPECTRAL_RBF_CAPACITY = (2_000, 8_000, "The rbf affinity builds a dense n × n matrix.")
REDUCTION_CAPACITY = {
    "PCA": (500_000, 2_000_000, "Fast linear projection."),
    "UMAP": (50_000, 200_000, "Approximate nearest-neighbor graph; scales well."),
    "TSNE": (5_000, 20_000, "Slow for many samples; consider PCA or UMAP for large data."),
}
# Silhouette uses all pairwise distances between the clustered samples.
METRIC_CAPACITY = (10_000, 40_000, "The silhouette score compares all pairs of samples.")
LARGE_RESPONSE_MB = 50
LEVEL_ORDER = ["recommended", "may_be_slow", "not_recommended"]

# Job execution and result-size limits; all can be overridden with environment variables.
MAX_UPLOAD_MB = int(os.environ.get("RAPCLUSTER_MAX_UPLOAD_MB", 50))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
JOB_TIMEOUT_SECONDS = int(os.environ.get("RAPCLUSTER_JOB_TIMEOUT_SECONDS", 600))
JOB_MEMORY_MB = int(os.environ.get("RAPCLUSTER_JOB_MEMORY_MB", 4096))
MAX_CONCURRENT_JOBS = int(os.environ.get("RAPCLUSTER_MAX_CONCURRENT_JOBS", 3))
MAX_PLOT_POINTS = int(os.environ.get("RAPCLUSTER_MAX_PLOT_POINTS", 5000))
SILHOUETTE_SAMPLE_SIZE = 5000
MAX_PROFILE_REQUEST = 500
MAX_CHANGED_SAMPLES = 20000
GZIP_MIN_BYTES = 1024

# Flask rejects larger requests itself (the extra megabyte covers the multipart envelope).
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024 * 1024
app.logger.setLevel(logging.INFO)


DEFAULT_PARAMS = {
    "MiniBatchKMeans": {"n_clusters": 8, "random_state": 42, "init": "k-means++"},
    "KMeans": {"n_clusters": 8, "random_state": 42, "init": "k-means++", "algorithm": "lloyd"},
    "AffinityPropagation": {"damping": 0.5, "preference": -50, "affinity": "euclidean"},
    "MeanShift": {"bandwidth": 1.0, "cluster_all": False},
    "SpectralClustering": {"n_clusters": 8, "affinity": "nearest_neighbors", "n_neighbors": 10, "assign_labels": "kmeans", "random_state": 42},
    "AgglomerativeClustering": {"n_clusters": 8, "linkage": "ward", "metric": "euclidean"}, # Ward needs euclidean
    "DBSCAN": {"eps": 0.5, "min_samples": 5, "metric": "euclidean", "leaf_size": 30, "algorithm": "auto"},
    "HDBSCAN": {"min_cluster_size": 5, "min_samples": 5, "cluster_selection_epsilon": 0.0, "max_cluster_size": 40, "metric": "euclidean", "leaf_size": 20, "cluster_selection_method": "eom"},
    "OPTICS": {"min_samples": 5, "xi": 0.05, "min_cluster_size": 0.05, "metric": "minkowski", "predecessor_correction": True, "algorithm": "auto"},
    "BIRCH": {"n_clusters": 8, "threshold": 0.5, "branching_factor": 50, "compute_labels": True},
    "GaussianMixture": {"n_components": 8, "covariance_type": "full", "random_state": 42, "init_params": "kmeans"},
    "BayesianGaussianMixture": {"n_components": 10, "weight_concentration_prior_type": "dirichlet_process",
                                "weight_concentration_prior": None, "covariance_type": "full", "init_params": "kmeans",
                                "max_iter": 500, "random_state": 42},
    "BisectingKMeans": {"n_clusters": 8, "bisecting_strategy": "biggest_inertia", "init": "k-means++", "random_state": 42},
}


ALGORITHM_PARAMS = {
    "MiniBatchKMeans": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1},
        {"name": "init", "type": "select", "default": "k-means++", "options": ["k-means++", "random"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "KMeans": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1},
        {"name": "init", "type": "select", "default": "k-means++", "options": ["k-means++", "random"]},
        {"name": "algorithm", "type": "select", "default": "lloyd", "options": ["lloyd", "elkan"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "AffinityPropagation": [
        {"name": "damping", "type": "float", "default": 0.5, "min": 0.5, "max": 1.0, "step": 0.01},
        {"name": "preference", "type": "float", "default": -50, "min": -1000, "max": 0, "step": 1},
        {"name": "affinity", "type": "select", "default": "euclidean", "options": ["euclidean"]}, # 'precomputed' requires special input, keep simple for now
    ],
    "MeanShift": [
        {"name": "bandwidth", "type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "nullable": True},
        {"name": "cluster_all", "type": "boolean", "default": True},
    ],
    "SpectralClustering": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1},
        {"name": "affinity", "type": "select", "default": "nearest_neighbors", "options": ["nearest_neighbors", "rbf"]}, # rbf requires gamma
        {"name": "n_neighbors", "type": "int", "default": 10, "min": 1, "max": 50, "step": 1}, # Only for nearest_neighbors affinity
        {"name": "assign_labels", "type": "select", "default": "kmeans", "options": ["kmeans", "discretize"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "AgglomerativeClustering": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1},
        {"name": "linkage", "type": "select", "default": "ward", "options": ["ward", "complete", "average", "single"]},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "l1", "l2", "manhattan", "cosine"]}, # Ward only works with euclidean
    ],
    "DBSCAN": [
        {"name": "eps", "type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1},
        {"name": "min_samples", "type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "manhattan", "chebyshev", "minkowski"]},
        {"name": "leaf_size", "type": "int", "default": 30, "min": 10, "max": 100, "step": 10},
        {"name": "algorithm", "type": "select", "default": "auto", "options": ["auto", "ball_tree", "kd_tree", "brute"]},
    ],
    "HDBSCAN": [
        {"name": "min_cluster_size", "type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
        {"name": "min_samples", "type": "int", "default": 15, "min": 1, "max": 20, "step": 1, "nullable": True},
        {"name": "cluster_selection_epsilon", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "max_cluster_size", "type": "int", "default": 20, "min": 10, "max": 100, "step": 10, "nullable": True},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "manhattan", "l1", "l2", "chebyshev"]},
        {"name": "leaf_size", "type": "int", "default": 20, "min": 10, "max": 100, "step": 10},
        {"name": "cluster_selection_method", "type": "select", "default": "eom", "options": ["eom", "leaf"]},
    ],
    "OPTICS": [
        {"name": "min_samples", "type": "int", "default": 5, "min": 2, "max": 50, "step": 1},
        {"name": "xi", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "min_cluster_size", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}, # Can also be int
        {"name": "metric", "type": "select", "default": "minkowski", "options": ["minkowski", "euclidean", "manhattan", "chebyshev"]},
        {"name": "predecessor_correction", "type": "boolean", "default": True},
        {"name": "algorithm", "type": "select", "default": "auto", "options": ["auto", "ball_tree", "kd_tree", "brute"]},
    ],
    "BIRCH": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1, "nullable": True}, # Can be None
        {"name": "threshold", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "branching_factor", "type": "int", "default": 50, "min": 20, "max": 200, "step": 10},
        {"name": "compute_labels", "type": "boolean", "default": True},
    ],
    "GaussianMixture": [
        {"name": "n_components", "type": "int", "default": 8, "min": 1, "max": 50, "step": 1},
        {"name": "covariance_type", "type": "select", "default": "full", "options": ["full", "tied", "diag", "spherical"]},
        {"name": "init_params", "type": "select", "default": "kmeans", "options": ["kmeans", "random", "k-means++", "random_from_data"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "BayesianGaussianMixture": [
        {"name": "n_components", "type": "int", "default": 10, "min": 2, "max": 50, "step": 1},
        {"name": "weight_concentration_prior_type", "type": "select", "default": "dirichlet_process",
         "options": ["dirichlet_process", "dirichlet_distribution"]},
        {"name": "weight_concentration_prior", "type": "float", "default": None, "min": 0.0001, "max": 1000, "step": 0.01, "nullable": True},
        {"name": "covariance_type", "type": "select", "default": "full", "options": ["full", "tied", "diag", "spherical"]},
        {"name": "init_params", "type": "select", "default": "kmeans", "options": ["kmeans", "random", "k-means++", "random_from_data"]},
        {"name": "max_iter", "type": "int", "default": 500, "min": 50, "max": 5000, "step": 50},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "BisectingKMeans": [
        {"name": "n_clusters", "type": "int", "default": 8, "min": 2, "max": 50, "step": 1},
        {"name": "bisecting_strategy", "type": "select", "default": "biggest_inertia", "options": ["biggest_inertia", "largest_cluster"]},
        {"name": "init", "type": "select", "default": "k-means++", "options": ["k-means++", "random"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
}

REDUCTION_PARAMS = {
    "UMAP": [
        {"name": "n_components", "type": "int", "default": 2, "min": 2, "max": 10, "step": 1},
        {"name": "n_neighbors", "type": "int", "default": 15, "min": 2, "max": 200, "step": 1},
        {"name": "min_dist", "type": "float", "default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "manhattan", "cosine"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "TSNE": [
        {"name": "n_components", "type": "int", "default": 2, "min": 2, "max": 3, "step": 1},
        {"name": "perplexity", "type": "float", "default": 30.0, "min": 5.0, "max": 100.0, "step": 1.0},
        {"name": "learning_rate", "type": "float", "default": 200.0, "min": 10.0, "max": 1000.0, "step": 10.0},
        {"name": "n_iter", "type": "int", "default": 1000, "min": 250, "max": 5000, "step": 50},
        {"name": "metric", "type": "select", "default": "euclidean", "options": ["euclidean", "manhattan", "cosine"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
    "PCA": [
        {"name": "n_components", "type": "int", "default": 2, "min": 2, "max": 10, "step": 1},
        {"name": "svd_solver", "type": "select", "default": "auto", "options": ["auto", "full", "arpack", "randomized"]},
        {"name": "random_state", "type": "int", "default": 42, "min": 1, "max": 999},
    ],
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_valid_id(value):
    return isinstance(value, str) and bool(ID_RE.match(value))


def get_session_dir(session_id, create=False):
    """Returns the storage directory of a session; ids are validated to prevent path traversal."""
    if not is_valid_id(session_id):
        raise ValueError("Invalid session id")
    path = os.path.join(DATA_ROOT, session_id)
    if create:
        os.makedirs(path, exist_ok=True)
    if os.path.isdir(path):
        os.utime(path, None)  # mark the session as active for the expiry cleanup
    return path


def cleanup_expired_sessions(force=False):
    """Deletes sessions that were idle longer than the TTL (fallback when the page-close signal is lost)."""
    global _last_cleanup
    now = time.time()
    if not force and now - _last_cleanup < CLEANUP_INTERVAL_SECONDS:
        return
    _last_cleanup = now
    if not os.path.isdir(DATA_ROOT):
        return
    for name in os.listdir(DATA_ROOT):
        path = os.path.join(DATA_ROOT, name)
        try:
            if os.path.isdir(path) and now - os.path.getmtime(path) > SESSION_TTL_SECONDS:
                kill_session_jobs(path)
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def write_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp_path, path)


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def detect_file_type(filename):
    extension = os.path.splitext(filename)[1].lower()
    if extension == '.csv':
        return 'csv'
    if extension == '.tsv':
        return 'tsv'
    if extension in ['.xls', '.xlsx']:
        return 'xlsx'
    return None


def dataset_paths(session_dir, dataset_id):
    dataset_dir = os.path.join(session_dir, "datasets", dataset_id)
    return dataset_dir, os.path.join(dataset_dir, "meta.json")


def store_dataset(session_dir, file):
    """Stores an uploaded file once per session; identical re-uploads reuse the existing dataset."""
    filename = os.path.basename(file.filename)
    content = file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadTooLarge(filename)
    sha256 = hashlib.sha256(content).hexdigest()

    datasets_dir = os.path.join(session_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    for existing_id in os.listdir(datasets_dir):
        meta = read_json(dataset_paths(session_dir, existing_id)[1])
        if meta and meta["sha256"] == sha256 and meta["filename"] == filename:
            return meta

    dataset_id = uuid.uuid4().hex
    dataset_dir, meta_path = dataset_paths(session_dir, dataset_id)
    os.makedirs(dataset_dir)
    data_file = f"data{os.path.splitext(filename)[1].lower()}"
    with open(os.path.join(dataset_dir, data_file), "wb") as fh:
        fh.write(content)

    meta = {
        "dataset_id": dataset_id,
        "filename": filename,
        "file_type": detect_file_type(filename),
        "data_file": data_file,
        "sha256": sha256,
        "size_bytes": len(content),
        "uploaded_at": now_iso(),
    }
    write_json(meta_path, meta)
    return meta


def load_dataset_meta(session_dir, dataset_id):
    if not is_valid_id(dataset_id):
        return None
    return read_json(dataset_paths(session_dir, dataset_id)[1])


def run_dir_path(session_dir, run_id):
    return os.path.join(session_dir, "runs", run_id)


def list_run_records(session_dir):
    runs_dir = os.path.join(session_dir, "runs")
    if not os.path.isdir(runs_dir):
        return []
    records = [read_json(os.path.join(runs_dir, run_id, "run.json")) for run_id in os.listdir(runs_dir)]
    return sorted((r for r in records if r), key=lambda r: r.get("number", 0))


def format_warnings(caught):
    """Keeps user-relevant warnings (no library deprecation noise), de-duplicated and capped."""
    messages = []
    for w in caught:
        category = w.category.__name__
        if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning)):
            continue
        message = f"{category}: {w.message}"
        if message not in messages:
            messages.append(message)
    return messages[:MAX_RECORDED_WARNINGS]


def read_table(file_content, file_type, sheet_name=None):
    try:
        if file_type == 'csv':
            return pd.read_csv(file_content)
        if file_type == 'tsv':
            return pd.read_csv(file_content, sep='\t')
        if file_type == 'xlsx':
            return pd.read_excel(file_content, sheet_name=sheet_name if sheet_name else 0)
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
    raise ValueError(f"Unsupported file type: {file_type}")


def apply_dimensionality_reduction(X, method, params):
    """Applies the selected dimensionality reduction; the clustering uses all returned dimensions."""
    if method == "UMAP":
        reducer = umap.UMAP(**params)
        X_reduced = reducer.fit_transform(X)
    elif method == "TSNE":
        reducer = TSNE(**params)
        X_reduced = reducer.fit_transform(X)
    elif method == "PCA":
        reducer = PCA(**params)
        X_reduced = reducer.fit_transform(X)
    else:
        X_reduced = X
    return np.asarray(X_reduced, dtype=np.float64)


def display_coordinates(space, method):
    """2D coordinates for the plots: the first two reduced dimensions, or a PCA projection of the data when
    clustering ran on all features. Returns (coordinates, description)."""
    dims = space.shape[1]
    if method != "None" and dims >= 2:
        note = (f"the two {method} dimensions" if dims == 2
                else f"the first two of the {dims} {method} dimensions used for clustering")
        return space[:, :2], note
    if dims == 2:
        return space, "the two data columns"
    if dims == 1:
        return np.column_stack([space[:, 0], np.zeros(len(space))]), "the single data column (Y = 0)"
    projection = PCA(n_components=2, random_state=0).fit_transform(space)
    return projection, f"a PCA projection for display only; clustering used all {dims} features"


def model_details(algo, X, labels):
    """Algorithm-specific extras: soft membership per row, model criteria and per-cluster information."""
    details = {"membership": None, "metrics": {}, "model": {}}
    if isinstance(algo, HDBSCAN):
        details["membership"] = np.asarray(algo.probabilities_, dtype=float)
        persistence = getattr(algo, "cluster_persistence_", None)
        if persistence is not None and len(persistence):
            details["model"]["cluster_persistence"] = {str(i): float(p) for i, p in enumerate(persistence)}
        details["model"]["membership_meaning"] = "HDBSCAN membership probability (strength of the row's cluster membership; 0 for noise)."
    elif isinstance(algo, (mixture.GaussianMixture, mixture.BayesianGaussianMixture)):
        details["membership"] = algo.predict_proba(X).max(axis=1)
        details["model"]["membership_meaning"] = "Posterior probability of the assigned mixture component."
        details["model"]["converged"] = bool(algo.converged_)
        details["model"]["n_iter"] = int(algo.n_iter_)
        if isinstance(algo, mixture.BayesianGaussianMixture):
            components = np.unique(algo.predict(X))  # the relabelled clusters 0..k-1 follow this order
            details["model"]["component_weights"] = {str(i): float(algo.weights_[c]) for i, c in enumerate(components)}
            details["model"]["n_components_max"] = int(algo.n_components)
            details["model"]["n_components_used"] = int(components.size)
            details["model"]["n_components_weight_above_1pct"] = int(np.sum(algo.weights_ > 0.01))
        else:
            details["metrics"]["bic"] = float(algo.bic(X))
            details["metrics"]["aic"] = float(algo.aic(X))
    return details


def run_clustering(X, algo_name, config, connectivity=None, details=None):
    """Runs the selected clustering algorithm. Returns (labels, runtime, error_message, error_code).
    When a dict is passed as details it is filled with model_details()."""
    try:
        if algo_name == "AgglomerativeClustering":
            # Ensure 'ward' linkage uses 'euclidean' metric
            if config.get("linkage") == "ward" and config.get("metric") != "euclidean":
                warnings.warn("Ward linkage requires 'euclidean' metric. Overriding metric to 'euclidean'.")
                config["metric"] = "euclidean"
            algo = cluster.AgglomerativeClustering(connectivity=connectivity, **config)
        elif algo_name == "MiniBatchKMeans":
            algo = cluster.MiniBatchKMeans(**config)
        elif algo_name == "KMeans":
            algo = cluster.KMeans(**config)
        elif algo_name == "AffinityPropagation":

            if config.get("affinity") == "precomputed":
                warnings.warn("AffinityPropagation with 'precomputed' affinity requires a distance matrix input. Proceeding with 'euclidean'.")
                config["affinity"] = "euclidean" # Fallback if precomputed is not given
            algo = cluster.AffinityPropagation(**{k: v for k, v in config.items() if k not in ['random_state']})
        elif algo_name == "MeanShift":
            # MeanShift does not take 'n_jobs' directly, relies on underlying KDTree if used
            algo = cluster.MeanShift(**{k: v for k, v in config.items() if v is not None and k != 'n_jobs'}) # Remove n_jobs if it's there
        elif algo_name == "SpectralClustering":
            # SpectralClustering can use n_jobs with 'nearest_neighbors' affinity
            algo = cluster.SpectralClustering(**config)
        elif algo_name == "DBSCAN":
            algo = cluster.DBSCAN(**config)
        elif algo_name == "HDBSCAN":
            # HDBSCAN has n_jobs
            algo = HDBSCAN(**config)
        elif algo_name == "OPTICS":
            algo = cluster.OPTICS(**config)
        elif algo_name == "BIRCH":
            algo = cluster.Birch(**config)
        elif algo_name == "GaussianMixture":
            algo = mixture.GaussianMixture(**config)
        elif algo_name == "BayesianGaussianMixture":
            algo = mixture.BayesianGaussianMixture(**config)
        elif algo_name == "BisectingKMeans":
            algo = cluster.BisectingKMeans(**config)
        else:
            return None, None, f"Unknown algorithm: {algo_name}", "invalid_parameters"

        # Warnings (e.g. convergence) are recorded by the caller and stored with the run.
        t0 = time.time()
        if hasattr(algo, 'fit_predict'):
            labels = algo.fit_predict(X)
        else:
            algo.fit(X)
            labels = getattr(algo, 'labels_', algo.predict(X))
        runtime = time.time() - t0

        if labels is None:
            raise ValueError("Clustering did not produce labels.")
        labels = np.asarray(labels)
        if algo_name == "BayesianGaussianMixture":
            # Components without rows are dropped; the clusters are numbered 0..k-1 in component order.
            labels = np.unique(labels, return_inverse=True)[1].reshape(labels.shape)
        if details is not None:
            details.update(model_details(algo, X, labels))

    except MemoryError:
        return None, None, "The algorithm ran out of memory.", "memory_exceeded"
    except Exception as e:
        code = "invalid_parameters" if isinstance(e, (InvalidParameterError, TypeError)) else "clustering_failed"
        return None, None, str(e), code

    return np.asarray(labels), runtime, None, None


def silhouette(X, labels):
    """Silhouette on a fixed-seed subsample for large data (the full score compares all pairs of samples)."""
    if len(labels) > SILHOUETTE_SAMPLE_SIZE:
        try:
            return silhouette_score(X, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=0)
        except ValueError:
            pass  # the subsample happened to contain a single cluster
    return silhouette_score(X, labels)


def evaluate(labels, X):
    """Calculates clustering evaluation metrics."""
    if labels is None or len(set(labels)) <= 1:
        return -1, 0, -1, -1

    unique_labels = set(labels)
    if -1 in unique_labels:
        non_noise_mask = (labels != -1)
        if np.sum(non_noise_mask) < 2:
            return -1, 0, -1, -1

        filtered_labels = labels[non_noise_mask]
        filtered_X = X[non_noise_mask]

        if len(set(filtered_labels)) <= 1:
            return -1, 0, -1, -1

        sil_score = silhouette(filtered_X, filtered_labels)
        ch_score = calinski_harabasz_score(filtered_X, filtered_labels)
        db_score = davies_bouldin_score(filtered_X, filtered_labels)
        n_clusters = len(set(filtered_labels))
    else:
        sil_score = silhouette(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        n_clusters = len(unique_labels)

    return sil_score, n_clusters, ch_score, db_score


def parse_param_json(raw, label):
    """Parses a parameter JSON object; the string 'None' becomes a real None."""
    if not raw:
        return {}
    try:
        params = json.loads(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {label} parameters JSON: {e}")
    if not isinstance(params, dict):
        raise ValueError(f"Invalid {label} parameters JSON: expected an object")
    return {k: (None if isinstance(v, str) and v.lower() == 'none' else v) for k, v in params.items()}


def parse_preprocessing(raw):
    """Preprocessing options JSON, validated and merged with the defaults ('none' stays a string here)."""
    try:
        options = json.loads(raw) if raw else {}
    except ValueError as e:
        raise ValueError(f"Invalid preprocessing options JSON: {e}")
    if not isinstance(options, dict):
        raise ValueError("Invalid preprocessing options JSON: expected an object")
    return pp.validate_options(options)


def cluster_colors_for(labels):
    cmap = colormaps.get_cmap("tab20")
    colors_map = {}
    color_idx = 0
    for label in sorted(set(labels.tolist())):
        if label == -1:
            colors_map[label] = "#808080"
        else:
            colors_map[label] = mcolors.rgb2hex(cmap(color_idx % 20))
            color_idx += 1
    return colors_map


def stratified_sample(labels, max_points, seed=0):
    """Indices of at most ~max_points samples, proportional per cluster but keeping every cluster visible."""
    n = len(labels)
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    clusters, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    minimum = max(1, min(20, max_points // (2 * len(clusters))))
    quota = np.minimum(counts, np.maximum(minimum, np.floor(counts * max_points / n).astype(int)))
    chosen = [rng.choice(np.flatnonzero(inverse == i), q, replace=False) for i, q in enumerate(quota)]
    return np.sort(np.concatenate(chosen))


def cluster_profile_summary(labels, profiles):
    """Per-cluster size, mean and median of the row-scaled profiles, computed on all samples."""
    clusters = sorted(set(labels.tolist()))
    means, medians, sizes = [], [], []
    for cluster_id in clusters:
        rows = profiles[labels == cluster_id]
        sizes.append(int(rows.shape[0]))
        means.append(rows.mean(axis=0))
        medians.append(np.median(rows, axis=0))
    return {
        "clusters": clusters,
        "sizes": sizes,
        "mean": np.round(np.array(means, dtype=float), 4).tolist(),
        "median": np.round(np.array(medians, dtype=float), 4).tolist(),
    }


def build_result_payload(metrics, names, X, labels, profiles, intensity_columns, row_values=None):
    """Interactive result: a stratified sample for plots plus full-data cluster summaries.
    row_values: optional {key: per-row values} added to each plotted row (e.g. membership)."""
    colors_map = cluster_colors_for(labels)
    sample = stratified_sample(labels, MAX_PLOT_POINTS)
    rounded_profiles = np.round(profiles[sample].astype(float), 4)
    row_values = {key: values for key, values in (row_values or {}).items() if values is not None}
    cluster_data = [{
        "name": names[i],
        "x": round(float(X[i, 0]), 5),
        "y": round(float(X[i, 1]), 5),
        "cluster": int(labels[i]),
        "color": colors_map[int(labels[i])],
        "profile_values": rounded_profiles[j].tolist(),
        **{key: round(float(values[i]), 4) for key, values in row_values.items()},
    } for j, i in enumerate(sample)]
    return {
        "status": "success",
        "metrics": metrics,
        "cluster_data": cluster_data,
        "cluster_colors": colors_map,
        "intensity_columns": intensity_columns,
        "cluster_profiles": cluster_profile_summary(labels, profiles),
        "sampling": {
            "total": int(len(labels)),
            "shown": int(len(sample)),
            "sampled": bool(len(sample) < len(labels)),
            "max_points": MAX_PLOT_POINTS,
            "method": "stratified by cluster (fixed seed)",
        },
    }


def format_cell(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_labels_tsv(path, names, labels, X, row_numbers, extra=None):
    """Cluster assignments with the original identifier and the 1-based data row in the uploaded file.
    extra: optional {column name: per-row values} appended as further columns (e.g. membership, stability)."""
    extra = {name: np.asarray(values).tolist() for name, values in (extra or {}).items() if values is not None}
    order = sorted(range(len(names)), key=lambda i: (labels[i], names[i].lower()))
    with open(path, "w") as fh:
        fh.write("\t".join(["Name", "OriginalRow", "Cluster", "X", "Y", *extra]) + "\n")
        for i in order:
            cells = "".join(f"\t{format_cell(values[i])}" for values in extra.values())
            fh.write(f"{names[i]}\t{row_numbers[i]}\t{labels[i]}\t{X[i, 0]:.6f}\t{X[i, 1]:.6f}{cells}\n")


def load_clustering_space(run_dir, arrays):
    """The data the run was clustered on; runs without space.npy were clustered on their 2D coordinates."""
    path = os.path.join(run_dir, "space.npy")
    return np.load(path) if os.path.isfile(path) else arrays["embedding"]


def load_run_arrays(run_dir):
    """Full (unsampled) per-sample data of a completed run."""
    with open(os.path.join(run_dir, "names.json")) as fh:
        names = json.load(fh)
    return {
        "names": names,
        "labels": np.load(os.path.join(run_dir, "labels.npy")),
        "embedding": np.load(os.path.join(run_dir, "embedding.npy")),
        "profiles": np.load(os.path.join(run_dir, "profiles.npy"), mmap_mode="r"),
        "intensities": np.load(os.path.join(run_dir, "intensities.npy"), mmap_mode="r"),
    }


# ---------- Background jobs ----------

class JobFailure(Exception):
    def __init__(self, code, detail=None):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


class UploadTooLarge(Exception):
    pass


def error_info(code, detail=None):
    """User-facing message and hint for a structured error code."""
    limit_time = (f"{JOB_TIMEOUT_SECONDS / 60:.3g} minutes" if JOB_TIMEOUT_SECONDS >= 60
                  else f"{JOB_TIMEOUT_SECONDS} seconds")
    limit_memory = f"{JOB_MEMORY_MB / 1024:.3g} GB" if JOB_MEMORY_MB >= 1024 else f"{JOB_MEMORY_MB} MB"
    catalog = {
        "invalid_parameters": ("The parameters were rejected.",
                               "Check the parameter named in the message and try another value."),
        "unsupported_data": ("The data could not be processed.",
                             "Check the file, the name column and the data start column index; "
                             "'Check data & settings' lists data problems."),
        "clustering_failed": ("The algorithm could not produce a result with these settings.",
                              "Try other parameters or another algorithm."),
        "timeout": (f"The run exceeded the time limit of {limit_time} and was stopped.",
                    "Use a faster algorithm (see the capacity guidance), reduce the dimensions with PCA or UMAP, "
                    "or use a smaller dataset."),
        "memory_exceeded": (f"The run exceeded the memory limit of {limit_memory} and was stopped.",
                            "Algorithms such as AffinityPropagation or SpectralClustering (rbf) need memory that grows "
                            "with the square of the number of rows; use MiniBatchKMeans, BIRCH or HDBSCAN instead."),
        "worker_crashed": ("The run stopped unexpectedly, most likely because the server ran out of memory.",
                           "Try a more scalable algorithm or a smaller dataset."),
        "cancelled": ("The run was cancelled.", ""),
        "server_busy": (f"The server is already running {MAX_CONCURRENT_JOBS} analyses.",
                        "Please try again in a moment."),
        "internal_error": ("An unexpected server error occurred.",
                           "Please report the run ID shown in Run History."),
    }
    message, hint = catalog.get(code, catalog["internal_error"])
    if detail and detail != message:
        message = f"{message} {detail}"
    return message, hint


def job_files(run_dir, kind):
    return {name: os.path.join(run_dir, f"{name}_{kind}.json") for name in ("job", "progress", "outcome")}


# Job processes are tracked through /proc on Linux (the server). Elsewhere (e.g. macOS, which has no /proc)
# psutil is used if installed; without it a basic fallback still keeps runs from being reported as crashed.
HAVE_PROC = os.path.exists("/proc/self/stat")
try:
    import psutil
except ImportError:
    psutil = None
# ru_maxrss is in kilobytes on Linux but in bytes on macOS.
MAXRSS_PER_MB = 1024 * 1024 if sys.platform == "darwin" else 1024


def own_peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MAXRSS_PER_MB


def process_stat(pid):
    """(state, start_time) of a process, or None if it does not exist."""
    if HAVE_PROC:
        try:
            with open(f"/proc/{pid}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            return fields[0], int(fields[19])
        except (OSError, IndexError, ValueError):
            return None
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            state = "Z" if process.status() == psutil.STATUS_ZOMBIE else "R"
            return state, int(round(process.create_time() * 100))
        except (psutil.Error, OSError):
            return None
    try:  # no start time available: no guard against a recycled pid
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except OSError:
        pass  # exists, but owned by another user
    return "R", None


def process_alive(job):
    stat = process_stat(job.get("pid")) if job and job.get("pid") else None
    # The start time guards against a recycled pid; zombies have already finished.
    return bool(stat) and stat[1] == job.get("start_ticks") and stat[0] != "Z"


def process_rss_mb(pid):
    """Current memory (resident set) of a process in MB; 0 if it cannot be read."""
    if HAVE_PROC:
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        return 0.0
    if psutil is not None:
        try:
            return psutil.Process(pid).memory_info().rss / (1024 * 1024)
        except (psutil.Error, OSError):
            return 0.0
    # Without psutil only the own process can be measured (its peak, which is enough for the limit check).
    return own_peak_rss_mb() if pid == os.getpid() else 0.0


def kill_job_process(job):
    if process_alive(job):
        try:
            os.kill(job["pid"], signal.SIGKILL)
        except OSError:
            pass


_job_context = None


def job_context():
    """Jobs are forked from a clean forkserver with the app preloaded: fast start, no inherited thread pools."""
    global _job_context
    if _job_context is None:
        context = multiprocessing.get_context("forkserver")
        context.set_forkserver_preload(["app"])
        _job_context = context
    return _job_context


def running_job_count():
    count = 0
    job_paths = (glob.glob(os.path.join(DATA_ROOT, "*", "runs", "*", "job_*.json"))
                 + glob.glob(os.path.join(DATA_ROOT, "*", "sweeps", "*", "job_*.json")))
    for job_path in job_paths:
        outcome_path = os.path.join(os.path.dirname(job_path), os.path.basename(job_path).replace("job_", "outcome_", 1))
        if not os.path.exists(outcome_path) and process_alive(read_json(job_path)):
            count += 1
    return count


def launch_job(run_dir, kind, spec):
    files = job_files(run_dir, kind)
    for stale in (files["progress"], files["outcome"]):
        if os.path.exists(stale):
            os.remove(stale)
    module = importlib.import_module("app") if __name__ == "__main__" else sys.modules[__name__]
    process = job_context().Process(target=module.run_job, args=(spec,), daemon=True)
    process.start()
    job = {
        "kind": kind,
        "pid": process.pid,
        "start_ticks": (process_stat(process.pid) or (None, None))[1],
        "started_at": time.time(),
        "timeout_seconds": JOB_TIMEOUT_SECONDS,
        "memory_limit_mb": JOB_MEMORY_MB,
    }
    write_json(files["job"], job)
    threading.Thread(target=process.join, daemon=True).start()  # releases the handle once the job ends
    app.logger.info("Started %s job for run %s (pid %s)", kind, os.path.basename(run_dir), process.pid)
    return job


def finalize_job(run_dir, kind, state, code=None, detail=None, record_updates=None):
    """Writes the job outcome; for clustering jobs the run record is completed as well. First writer wins."""
    files = job_files(run_dir, kind)
    if os.path.exists(files["outcome"]):
        return read_json(files["outcome"])
    message, hint = error_info(code, detail) if code else (None, None)
    outcome = {"state": state, "error_code": code, "error": message, "hint": hint, "finished_at": time.time()}

    if kind == "cluster":
        record = read_json(os.path.join(run_dir, "run.json")) or {}
        record.update(record_updates or {})
        if state != "success":
            job = read_json(files["job"]) or {}
            record.update({"status": "cancelled" if code == "cancelled" else "failed",
                           "error": message, "error_code": code, "error_hint": hint})
            if record.get("runtime_seconds", {}).get("total") is None and job.get("started_at"):
                record.setdefault("runtime_seconds", {})["total"] = time.time() - job["started_at"]
        write_json(os.path.join(run_dir, "run.json"), record)
    write_json(files["outcome"], outcome)
    return outcome


def refresh_job(run_dir, kind):
    """Current job state; also enforces the time and memory limits (backs up the in-job watchdog)."""
    files = job_files(run_dir, kind)
    outcome = read_json(files["outcome"])
    if outcome:
        return outcome
    job = read_json(files["job"])
    if job is None:
        return None
    elapsed = time.time() - job["started_at"]
    progress = read_json(files["progress"]) or {}

    if not process_alive(job):
        time.sleep(0.2)  # the job may be writing its outcome right now
        return read_json(files["outcome"]) or finalize_job(run_dir, kind, "failed", "worker_crashed")
    if elapsed > job["timeout_seconds"]:
        kill_job_process(job)
        return finalize_job(run_dir, kind, "failed", "timeout")
    memory_mb = process_rss_mb(job["pid"])
    if memory_mb > job["memory_limit_mb"]:
        kill_job_process(job)
        return finalize_job(run_dir, kind, "failed", "memory_exceeded")

    return {"state": "running", "stage": progress.get("stage", "starting"), "message": progress.get("message", ""),
            "elapsed_seconds": elapsed, "memory_mb": memory_mb}


def kill_session_jobs(session_dir):
    for folder in ("runs", "sweeps"):
        for job_path in glob.glob(os.path.join(session_dir, folder, "*", "job_*.json")):
            kill_job_process(read_json(job_path))


def start_watchdog(run_dir, kind, started):
    """In-job guard: stops the process when it exceeds the memory or time limit; also tracks peak memory."""
    peak = {"mb": 0.0}

    def watch():
        while True:
            rss = process_rss_mb(os.getpid())
            peak["mb"] = max(peak["mb"], rss)
            if rss > JOB_MEMORY_MB:
                finalize_job(run_dir, kind, "failed", "memory_exceeded")
                os._exit(3)
            if time.time() - started > JOB_TIMEOUT_SECONDS:
                finalize_job(run_dir, kind, "failed", "timeout")
                os._exit(4)
            time.sleep(0.25)

    threading.Thread(target=watch, daemon=True).start()
    return peak


def job_logger(run_dir, run_id):
    logger = logging.getLogger(f"rapcluster.job.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(f"%(asctime)s [run {run_id}] %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(os.path.join(run_dir, "job.log"))):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_job(spec):
    """Entry point of a job process."""
    run_dir, kind = spec["run_dir"], spec["kind"]
    started = time.time()
    peak = start_watchdog(run_dir, kind, started)
    logger = job_logger(run_dir, os.path.basename(run_dir))
    logger.info("%s job started (pid %s)", kind, os.getpid())
    try:
        if kind == "cluster":
            execute_cluster_job(spec, peak, logger)
        elif kind == "sweep":
            execute_sweep_job(spec, peak, logger, started)
        else:
            execute_stability_job(spec, logger)
        logger.info("%s job finished in %.1f s", kind, time.time() - started)
    except MemoryError:
        logger.error("Out of memory")
        finalize_job(run_dir, kind, "failed", "memory_exceeded")
    except Exception as e:
        logger.exception("Unexpected error")
        finalize_job(run_dir, kind, "failed", "internal_error", str(e))


def set_progress(run_dir, kind, stage, message):
    write_json(job_files(run_dir, kind)["progress"], {"stage": stage, "message": message, "updated_at": time.time()})


def preprocessing_key(dataset_meta, preprocessing):
    raw = json.dumps([dataset_meta["sha256"], preprocessing["name_column"], preprocessing["intensity_start_index"],
                      preprocessing["sheet_name"], preprocessing.get("options")], sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def cached_preprocessing(session_dir, dataset_meta, data_path, preprocessing):
    """Preprocessed data, reused across runs with the same dataset and preprocessing settings."""
    key = preprocessing_key(dataset_meta, preprocessing)
    cache_dir = os.path.join(session_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"prep_{key}.npz")
    hit = os.path.isfile(path)
    if hit:
        with np.load(path) as cached:
            data = {name: cached[name] for name in cached.files}
    else:
        df = read_table(data_path, dataset_meta["file_type"], preprocessing["sheet_name"])
        result = pp.preprocess(df, preprocessing["name_column"], preprocessing["intensity_start_index"],
                               preprocessing.get("options"))
        data = {"data": result["data"], "log10": result["log10"], "names": np.array(result["names"], dtype=str),
                "columns": np.array(result["columns"], dtype=str), "row_numbers": np.array(result["row_numbers"]),
                "name_column_found": np.array(result["name_column_found"]),
                "summary": np.array(json.dumps(result["summary"]))}
        tmp_path = f"{path}.tmp.npz"
        np.savez(tmp_path, **data)
        os.replace(tmp_path, path)
    if not bool(data["name_column_found"]):
        warnings.warn(f"Name column '{preprocessing['name_column']}' not found. Using default names.")
    data["summary"] = json.loads(str(data["summary"]))
    return data, key, hit


def cached_embedding(session_dir, prep_key, X, method, params):
    """Reduced data (all requested dimensions), reused when only the clustering settings change."""
    if method not in REDUCTION_PARAMS:
        return apply_dimensionality_reduction(X, method, params), False
    # "v2" marks cache files that hold all reduced dimensions.
    raw = json.dumps(["v2", prep_key, method, params], sort_keys=True, default=str)
    path = os.path.join(session_dir, "cache", f"emb_{hashlib.sha1(raw.encode()).hexdigest()[:16]}.npy")
    if os.path.isfile(path):
        return np.load(path), True
    embedding = apply_dimensionality_reduction(X, method, params)
    tmp_path = f"{path}.tmp.npy"
    np.save(tmp_path, embedding)
    os.replace(tmp_path, path)
    return embedding, False


def add_model_details(characterization, labels, membership, model):
    """Adds mean membership and (HDBSCAN) persistence to the per-cluster summaries."""
    persistence = model.get("cluster_persistence") or {}
    weights = model.get("component_weights") or {}
    for entry in characterization["clusters"]:
        key = str(entry["cluster"])
        if membership is not None:
            entry["mean_membership"] = round(float(membership[labels == entry["cluster"]].mean()), 4)
        if key in persistence:
            entry["persistence"] = round(persistence[key], 4)
        if key in weights:
            entry["component_weight"] = round(weights[key], 4)


def execute_cluster_job(spec, peak, logger):
    run_dir, session_dir = spec["run_dir"], spec["session_dir"]
    record = read_json(os.path.join(run_dir, "run.json"))
    preprocessing = record["preprocessing"]
    value_label = pp.value_label(preprocessing.get("options") or pp.DEFAULT_OPTIONS)
    characterization = None
    reduction_method, reduction_params = record["reduction"]["method"], dict(record["reduction"]["params"])
    algorithm, clustering_params = record["clustering"]["algorithm"], dict(record["clustering"]["params"])
    dataset_meta = load_dataset_meta(session_dir, record["dataset"]["dataset_id"])
    data_path = os.path.join(dataset_paths(session_dir, dataset_meta["dataset_id"])[0], dataset_meta["data_file"])

    stage_seconds, cache = {}, {}
    caught = []
    t_start = time.perf_counter()

    def timed_stage(name, message):
        set_progress(run_dir, "cluster", name, message)
        logger.info(message)
        stage_seconds[name] = time.perf_counter()
        return name

    def end_stage(name):
        stage_seconds[name] = time.perf_counter() - stage_seconds[name]

    def failure_updates():
        return {"warnings": format_warnings(caught), "stage_seconds": stage_seconds,
                "preprocessing_summary": record.get("preprocessing_summary"),
                "runtime_seconds": {"total": time.perf_counter() - t_start, "clustering": None},
                "peak_memory_mb": peak["mb"]}

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            stage = timed_stage("loading", "Reading and preprocessing the data")
            try:
                prep, prep_key, cache["preprocessing"] = cached_preprocessing(
                    session_dir, dataset_meta, data_path, preprocessing)
            except ValueError as e:
                raise JobFailure("unsupported_data", str(e))
            end_stage(stage)
            X0 = prep["data"]
            names = prep["names"].tolist()
            record["preprocessing_summary"] = prep["summary"]
            record["dataset"]["n_observations"] = int(X0.shape[0])
            record["dataset"]["n_features"] = int(X0.shape[1])
            if X0.shape[0] < 2:
                raise JobFailure("unsupported_data", "Fewer than two non-zero rows remain after preprocessing.")

            stage = timed_stage("reducing", f"Dimensionality reduction ({reduction_method})")
            try:
                X, cache["embedding"] = cached_embedding(session_dir, prep_key, X0, reduction_method, reduction_params)
                display, display_note = display_coordinates(X, reduction_method)
            except MemoryError:
                raise JobFailure("memory_exceeded")
            except (ValueError, TypeError) as e:
                raise JobFailure("invalid_parameters", f"{reduction_method}: {e}")
            end_stage(stage)

            stage = timed_stage("clustering", f"Clustering with {algorithm}")
            connectivity = None
            if algorithm == "AgglomerativeClustering":
                try:
                    connectivity = build_connectivity(X)
                except Exception as e:
                    warnings.warn(f"Could not compute connectivity graph for Agglomerative: {e}. Running without connectivity.")
            details = {}
            labels, runtime, error, error_code = run_clustering(X, algorithm, clustering_params, connectivity, details)
            if error:
                raise JobFailure(error_code, error)
            membership = details.get("membership")
            end_stage(stage)

            stage = timed_stage("evaluating", "Computing validation metrics")
            sil_score, n_clusters, ch_score, db_score = evaluate(labels, X)
            metrics = {
                "silhouette_score": float(sil_score),
                "silhouette_sample_size": min(SILHOUETTE_SAMPLE_SIZE, int(np.sum(labels != -1))),
                "n_clusters": int(n_clusters),
                "calinski_harabasz_score": float(ch_score),
                "davies_bouldin_score": float(db_score),
                "runtime_seconds": float(runtime),
                "dbcv": compute_dbcv(X, labels),
                **cluster_size_stats(labels),
                **details.get("metrics", {}),
            }
            if membership is not None:
                real = labels != -1
                metrics["mean_membership"] = float(membership[real].mean()) if real.any() else None
            end_stage(stage)

            stage = timed_stage("characterizing", "Characterizing the clusters")
            try:
                characterization = characterize(X0, labels, names, prep["columns"].tolist())
                characterization["value_label"] = value_label
                add_model_details(characterization, labels, membership, details.get("model", {}))
            except Exception as e:  # descriptive extras must not fail the run
                logger.warning("Cluster characterization failed: %s", e, exc_info=True)
                warnings.warn(f"Cluster characterization could not be computed: {e}")
            end_stage(stage)

        stage = timed_stage("saving", "Preparing the results")
        X = np.asarray(X, dtype=np.float64)
        display = np.asarray(display, dtype=np.float64)
        np.save(os.path.join(run_dir, "embedding.npy"), display)  # 2D plot coordinates
        if X is not display:
            np.save(os.path.join(run_dir, "space.npy"), X)  # the space the clustering ran in
        record["reduction"]["clustering_dims"] = int(X.shape[1])
        record["display"] = display_note
        np.save(os.path.join(run_dir, "labels.npy"), labels)
        np.save(os.path.join(run_dir, "profiles.npy"), X0.astype(np.float32))
        np.save(os.path.join(run_dir, "intensities.npy"), prep["log10"].astype(np.float32))
        np.save(os.path.join(run_dir, "row_numbers.npy"), prep["row_numbers"])
        if membership is not None:
            np.save(os.path.join(run_dir, "membership.npy"), membership.astype(np.float32))
        write_json(os.path.join(run_dir, "names.json"), names)
        write_labels_tsv(os.path.join(run_dir, "labels.tsv"), names, labels, display, prep["row_numbers"].tolist(),
                         {"Membership": membership})
        if characterization is not None:
            write_json(os.path.join(run_dir, "characterization.json"), characterization)
        result = build_result_payload(metrics, names, display, labels, X0, prep["columns"].tolist(),
                                      {"membership": membership})
        result["display"] = {"description": display_note, "clustering_dims": int(X.shape[1])}
        result["value_label"] = value_label
        end_stage(stage)

        record.update({
            "status": "success",
            "n_clusters": int(n_clusters),
            "n_noise": int(np.sum(labels == -1)),
            "metrics": metrics,
            "sampling": result["sampling"],
            "model": details.get("model") or None,
            "cache": cache,
            "stage_seconds": stage_seconds,
            "warnings": format_warnings(caught),
            "runtime_seconds": {"total": time.perf_counter() - t_start, "clustering": float(runtime)},
            "peak_memory_mb": max(peak["mb"], own_peak_rss_mb()),
        })
        result["run"] = record
        write_json(os.path.join(run_dir, "result.json"), result)
        finalize_job(run_dir, "cluster", "success", record_updates=record)

    except JobFailure as e:
        logger.warning("Run failed (%s): %s", e.code, e.detail)
        finalize_job(run_dir, "cluster", "failed", e.code, e.detail, record_updates=failure_updates())


def parse_stability_options(raw):
    """{resamples, seeds} from the request body, validated against STABILITY_LIMITS."""
    options = {"resamples": STABILITY_RESAMPLES, "seeds": STABILITY_SEEDS}
    for key, (low, high) in STABILITY_LIMITS.items():
        if raw.get(key) is None:
            continue
        try:
            value = int(raw[key])
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a whole number.")
        if value != raw[key] or not low <= value <= high:
            raise ValueError(f"'{key}' must be a whole number between {low} and {high}.")
        options[key] = value
    return options


def execute_stability_job(spec, logger):
    """Re-clusters random subsamples (and, for seeded algorithms, all rows with other seeds) of the run's
    embedding and compares every solution with the run's labels (see stability.py)."""
    run_dir = spec["run_dir"]
    options = spec.get("options") or {"resamples": STABILITY_RESAMPLES, "seeds": STABILITY_SEEDS}
    record = read_json(os.path.join(run_dir, "run.json"))
    arrays = load_run_arrays(run_dir)
    X, labels, names = load_clustering_space(run_dir, arrays), arrays["labels"], arrays["names"]
    algorithm, params = record["clustering"]["algorithm"], dict(record["clustering"]["params"])
    seeded = params.get("random_state") is not None
    n_seeds = options["seeds"] if seeded else 0
    rng = np.random.default_rng(0)
    n = len(labels)
    sample_size = max(2, int(round(STABILITY_FRACTION * n)))
    total = options["resamples"] + n_seeds

    plan = [("subsample", None)] * options["resamples"]
    plan += [("seed", params["random_state"] + k) for k in range(1, n_seeds + 1)]
    solutions = []
    t_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (kind, seed) in enumerate(plan):
            message = (f"Subsample {i + 1} of {options['resamples']}" if kind == "subsample"
                       else f"Seed {seed} ({i + 1 - options['resamples']} of {n_seeds})")
            set_progress(run_dir, "stability", kind, f"{message} — solution {i + 1} of {total}")
            idx = np.sort(rng.choice(n, sample_size, replace=False)) if kind == "subsample" else np.arange(n)
            X_sub = X[idx]
            config = dict(params, random_state=seed) if kind == "seed" else dict(params)
            connectivity = build_connectivity(X_sub) if algorithm == "AgglomerativeClustering" else None
            sub_labels, _, error, error_code = run_clustering(X_sub, algorithm, config, connectivity)
            if error:
                finalize_job(run_dir, "stability", "failed", error_code, f"Stability solution {i + 1} failed: {error}")
                return
            solutions.append({"kind": kind, "seed": seed, "idx": idx, "labels": np.asarray(sub_labels)})

    set_progress(run_dir, "stability", "analyzing", "Matching clusters and computing row stability")
    summary, details, per_row = stab.analyze(labels, solutions, names)
    runtime = time.perf_counter() - t_start
    settings = {"n_resamples": options["resamples"], "n_seeds": n_seeds, "seeded_algorithm": seeded,
                "subsample_fraction": STABILITY_FRACTION, "space": f"the {X.shape[1]}-dimensional data the run was clustered on"}
    details.update({"summary": summary, "settings": settings, "runtime_seconds": runtime, "method": STABILITY_METHOD})

    set_progress(run_dir, "stability", "saving", "Saving the stability results")
    np.save(os.path.join(run_dir, "row_stability.npy"), per_row["row_stability"].astype(np.float32))
    np.save(os.path.join(run_dir, "consensus.npy"), per_row["consensus"])
    np.save(os.path.join(run_dir, "consensus_agreement.npy"), per_row["agreement"].astype(np.float32))
    write_json(os.path.join(run_dir, "stability.json"), details)
    update_run_files_with_stability(run_dir, arrays, per_row, details["clusters"])

    record = read_json(os.path.join(run_dir, "run.json"))  # re-read: only the stability field changes
    record["stability"] = {
        **summary,
        "n_resamples": options["resamples"],
        "n_seeds": n_seeds,
        "subsample_fraction": STABILITY_FRACTION,
        "runtime_seconds": runtime,
        "method": STABILITY_METHOD,
    }
    write_json(os.path.join(run_dir, "run.json"), record)
    result = read_json(os.path.join(run_dir, "result.json"))
    if result is not None:
        result["run"] = record
        write_json(os.path.join(run_dir, "result.json"), result)
    finalize_job(run_dir, "stability", "success")


STABILITY_METHOD = ("Each solution re-clusters a random 80% subsample (or all rows with another random seed) of the "
                    "data the run was clustered on, with the run's settings. Clusters are matched one-to-one to the run's clusters by Jaccard overlap. Cluster "
                    "stability: mean best Jaccard per cluster (≥ 0.75 stable, 0.6–0.75 doubtful, < 0.6 unstable). Row "
                    "stability: share of the solutions containing a row in which it keeps its cluster. Mean ARI: "
                    "adjusted Rand index of each solution against the run's labels.")


def update_run_files_with_stability(run_dir, arrays, per_row, clusters):
    """Adds row stability and the consensus label to labels.tsv, the plotted rows and the characterization."""
    names, labels, X = arrays["names"], arrays["labels"], arrays["embedding"]
    row_numbers_path = os.path.join(run_dir, "row_numbers.npy")
    membership_path = os.path.join(run_dir, "membership.npy")
    membership = np.load(membership_path) if os.path.isfile(membership_path) else None
    if os.path.isfile(row_numbers_path):  # older runs without row_numbers.npy keep their labels.tsv
        tmp_path = os.path.join(run_dir, "labels.tsv.tmp")
        write_labels_tsv(tmp_path, names, labels, X, np.load(row_numbers_path).tolist(), {
            "Membership": membership, "RowStability": per_row["row_stability"],
            "ConsensusCluster": per_row["consensus"], "ConsensusAgreement": per_row["agreement"]})
        os.replace(tmp_path, os.path.join(run_dir, "labels.tsv"))

    result = read_json(os.path.join(run_dir, "result.json"))
    if result is not None:
        # The plotted rows are the same deterministic stratified sample that build_result_payload chose.
        sample = stratified_sample(labels, result["sampling"]["max_points"])
        rows = result["cluster_data"]
        if len(sample) == len(rows) and all(rows[j]["name"] == names[i] for j, i in enumerate(sample)):
            for j, i in enumerate(sample):
                value = per_row["row_stability"][i]
                rows[j]["stability"] = round(float(value), 3) if np.isfinite(value) else None
            write_json(os.path.join(run_dir, "result.json"), result)

    characterization = read_json(os.path.join(run_dir, "characterization.json"))
    if characterization is not None:
        by_cluster = {entry["cluster"]: entry for entry in clusters}
        for entry in characterization["clusters"]:
            match = by_cluster.get(entry["cluster"], {})
            if "mean_jaccard" in match:
                entry["stability_jaccard"] = round(match["mean_jaccard"], 4)
                entry["stability_verdict"] = match["verdict"]
            if match.get("mean_row_stability") is not None:
                entry["mean_row_stability"] = round(match["mean_row_stability"], 4)
        write_json(os.path.join(run_dir, "characterization.json"), characterization)


def screening_trial(X, algorithm, params):
    """Metrics of one sweep trial on the screening subsample (nothing is saved)."""
    connectivity = build_connectivity(X) if algorithm == "AgglomerativeClustering" else None
    details = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels, runtime, error, error_code = run_clustering(X, algorithm, params, connectivity, details)
    if error:
        return None, error, error_code
    sil, n_clusters, ch, db = evaluate(labels, X)
    valid = n_clusters >= 2
    return {
        "n_clusters": int(n_clusters),
        "silhouette_score": float(sil) if valid else None,
        "calinski_harabasz_score": float(ch) if valid else None,
        "davies_bouldin_score": float(db) if valid else None,
        "dbcv": compute_dbcv(X, labels) if valid else None,
        "noise_fraction": float(np.mean(labels == -1)),
        "runtime_seconds": float(runtime),
        **details.get("metrics", {}),
    }, None, None


SWEEP_TRIAL_METRICS = ("n_clusters", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", "dbcv",
                       "noise_fraction", "min_cluster_size", "bic", "aic", "mean_membership", "runtime_seconds")


def execute_sweep_job(spec, peak, logger, started):
    """Runs the planned trials; every confirmed trial becomes a normal run of the session."""
    sweep_dir, session_dir = spec["run_dir"], spec["session_dir"]
    sweep_path = os.path.join(sweep_dir, "sweep.json")
    sweep = read_json(sweep_path)
    settings, trials, screening = sweep["settings"], sweep["trials"], sweep["screening"]
    algorithm = settings["algorithm"]
    dataset_meta = load_dataset_meta(session_dir, sweep["dataset"]["dataset_id"])
    deadline = started + SWEEP_TIME_FRACTION * JOB_TIMEOUT_SECONDS

    def trial_params(trial):
        return {**settings["clustering_params"], **trial["values"]}

    def save():
        sweep["peak_memory_mb"] = peak["mb"]
        write_json(sweep_path, sweep)

    def out_of_time(trial):
        if time.time() < deadline:
            return False
        trial["state"] = "skipped"
        sweep["stopped_reason"] = (f"Stopped before the time limit of {JOB_TIMEOUT_SECONDS / 60:.3g} minutes; "
                                   "remaining trials were skipped. Use fewer trials or screening on a subsample.")
        return True

    if screening:
        set_progress(sweep_dir, "sweep", "preparing", "Preprocessing and embedding the data")
        template = new_run_record(sweep["session_id"], "template", None, dataset_meta, settings)
        data_path = os.path.join(dataset_paths(session_dir, dataset_meta["dataset_id"])[0], dataset_meta["data_file"])
        try:
            prep, prep_key, _ = cached_preprocessing(session_dir, dataset_meta, data_path, template["preprocessing"])
            X, _ = cached_embedding(session_dir, prep_key, prep["data"], settings["reduction_method"],
                                    dict(settings["reduction_params"]))
        except (ValueError, TypeError) as e:
            finalize_job(sweep_dir, "sweep", "failed", "unsupported_data", str(e))
            return
        n = X.shape[0]
        idx = np.sort(np.random.default_rng(0).choice(n, max(10, min(n, int(round(screening["fraction"] * n)))), replace=False))
        screening["n_rows"] = int(len(idx))
        for i, trial in enumerate(trials):
            if out_of_time(trial):
                continue
            set_progress(sweep_dir, "sweep", "screening", f"Screening trial {i + 1} of {len(trials)} on {len(idx)} rows")
            metrics, error, code = screening_trial(X[idx], algorithm, trial_params(trial))
            if error:
                message, _ = error_info(code, error)
                trial.update({"state": "failed", "phase": "screening", "error": message, "error_code": code})
            else:
                trial.update({"state": "screened", "screening_metrics": metrics})
            save()
        confirmed = sw.pick_confirmed(trials, screening)
        for i, trial in enumerate(trials):
            if trial["state"] == "screened" and i not in confirmed:
                trial["state"] = "not_confirmed"
        save()
    else:
        confirmed = list(range(len(trials)))

    for position, i in enumerate(confirmed):
        trial = trials[i]
        if out_of_time(trial):
            save()
            continue
        set_progress(sweep_dir, "sweep", "running",
                     f"Trial {trial['trial']} ({position + 1} of {len(confirmed)}): "
                     + ", ".join(f"{k}={v}" for k, v in trial["values"].items()))
        run_id = uuid.uuid4().hex
        trial_dir = os.path.join(sweep_dir, "trials", run_id)
        os.makedirs(trial_dir)
        record = new_run_record(sweep["session_id"], run_id, None, dataset_meta,
                                dict(settings, clustering_params=trial_params(trial)))
        record["sweep"] = {"sweep_id": sweep["sweep_id"], "trial": trial["trial"], "values": trial["values"]}
        write_json(os.path.join(trial_dir, "run.json"), record)
        trial.update({"state": "running", "run_id": run_id})
        save()
        try:
            execute_cluster_job({"run_dir": trial_dir, "session_dir": session_dir}, peak, logger)
        except Exception as e:  # a broken trial must not stop the sweep
            logger.exception("Sweep trial %s failed", trial["trial"])
            finalize_job(trial_dir, "cluster", "failed", "internal_error", str(e))
        record = read_json(os.path.join(trial_dir, "run.json"))
        if record.get("status") == "success":
            with session_lock(session_dir):  # the trial joins the session's runs with the next number
                record["number"] = next_run_number(session_dir)
                write_json(os.path.join(trial_dir, "run.json"), record)
                result = read_json(os.path.join(trial_dir, "result.json"))
                result["run"] = record
                write_json(os.path.join(trial_dir, "result.json"), result)
                os.makedirs(os.path.join(session_dir, "runs"), exist_ok=True)
                os.rename(trial_dir, run_dir_path(session_dir, run_id))
            trial.update({"state": "success", "run_number": record["number"],
                          "metrics": {k: record["metrics"].get(k) for k in SWEEP_TRIAL_METRICS}})
        else:
            trial.update({"state": "failed", "phase": "full data", "run_id": None, "error": record.get("error"),
                          "error_code": record.get("error_code")})
            shutil.rmtree(trial_dir, ignore_errors=True)
        save()

    sweep["runtime_seconds"] = time.time() - started
    save()
    finalize_job(sweep_dir, "sweep", "success")


def build_connectivity(X):
    connectivity = kneighbors_graph(X, n_neighbors=min(X.shape[0]-1, 10), include_self=False)
    return 0.5 * (connectivity + connectivity.T) # Symmetrize


def cluster_size_stats(labels):
    real = labels[labels != -1]
    sizes = np.unique(real, return_counts=True)[1] if real.size else np.array([], dtype=int)
    return {
        "min_cluster_size": int(sizes.min()) if sizes.size else None,
        "max_cluster_size": int(sizes.max()) if sizes.size else None,
        "cluster_size_imbalance": float(sizes.max() / sizes.min()) if sizes.size else None,
        "noise_fraction": float(np.mean(labels == -1)),
    }


def compute_dbcv(X, labels):
    """Density-based clustering validation (DBCV) on at most DBCV_MAX_SAMPLES points; None when undefined."""
    idx = np.arange(len(labels))
    if len(idx) > DBCV_MAX_SAMPLES:
        idx = np.sort(np.random.default_rng(0).choice(len(labels), DBCV_MAX_SAMPLES, replace=False))
    sub_labels = labels[idx]
    real = sub_labels != -1
    if len(np.unique(sub_labels[real])) < 2:
        return None
    # validity_index expects consecutive cluster ids 0..k-1 (noise stays -1).
    consecutive = np.full(sub_labels.shape, -1, dtype=np.intp)
    consecutive[real] = np.unique(sub_labels[real], return_inverse=True)[1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = validity_index(np.ascontiguousarray(X[idx], dtype=np.float64), consecutive)
    except Exception:
        return None
    return float(score) if np.isfinite(score) else None


def upload_too_large_response(filename=""):
    return jsonify({
        "error": f"{filename + ' is' if filename else 'The file is'} larger than {MAX_UPLOAD_MB} MB, "
                 f"the upload limit of the RapCluster server.",
        "code": "upload_too_large",
        "limit_mb": MAX_UPLOAD_MB,
        "limit_source": "RapCluster server",
    }), 413


_forkserver_warm_pid = None


@app.before_request
def warm_up_job_server():
    """Starts this worker's job forkserver in the background on its first request (e.g. the page load),
    so the first analysis does not wait for the app to be imported."""
    global _forkserver_warm_pid
    if _forkserver_warm_pid != os.getpid():
        _forkserver_warm_pid = os.getpid()
        job_context()
        threading.Thread(target=multiprocessing.forkserver.ensure_running, daemon=True).start()


@app.errorhandler(413)
def request_too_large(_error):
    return upload_too_large_response()


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error("Unhandled server error: %s", error, exc_info=True)
    message, hint = error_info("internal_error")
    return jsonify({"error": message, "hint": hint, "code": "internal_error"}), 500


@app.after_request
def compress_response(response):
    """gzip JSON responses (results compress about 3-5x)."""
    if (response.direct_passthrough or response.status_code < 200 or "Content-Encoding" in response.headers
            or not response.mimetype == "application/json"
            or "gzip" not in request.headers.get("Accept-Encoding", "")):
        return response
    data = response.get_data()
    if len(data) >= GZIP_MIN_BYTES:
        response.set_data(gzip.compress(data, compresslevel=5))
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"
    return response


@app.route('/api/limits', methods=['GET'])
def get_limits():
    return jsonify({
        "max_upload_mb": MAX_UPLOAD_MB,
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "job_memory_mb": JOB_MEMORY_MB,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "max_plot_points": MAX_PLOT_POINTS,
        "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
    })


@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Returns the list of available clustering algorithms and their parameters."""
    return jsonify({
        "clustering_algorithms": list(ALGORITHM_PARAMS.keys()),
        "dimensionality_reduction_methods": list(REDUCTION_PARAMS.keys()),
        "algorithm_params": ALGORITHM_PARAMS,
        "reduction_params": REDUCTION_PARAMS,
        "default_params": DEFAULT_PARAMS
    })


@app.route('/api/sessions/<session_id>/datasets', methods=['POST'])
def upload_dataset(session_id):
    """Stores a dataset for the session so it can be clustered repeatedly without re-uploading."""
    cleanup_expired_sessions()
    try:
        session_dir = get_session_dir(session_id, create=True)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({"error": "No file part in the request"}), 400
    if detect_file_type(file.filename) is None:
        return jsonify({"error": "Unsupported file format. Please upload CSV, TSV, or Excel."}), 400

    try:
        return jsonify(store_dataset(session_dir, file)), 201
    except UploadTooLarge as e:
        return upload_too_large_response(str(e))


@app.route('/api/sessions/<session_id>/runs', methods=['GET'])
def list_runs(session_id):
    try:
        session_dir = get_session_dir(session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"runs": list_run_records(session_dir)})


@app.route('/api/sessions/<session_id>/runs/<run_id>', methods=['GET'])
def get_run(session_id, run_id):
    """Returns the stored interactive result of a run (sampled points plus full-data summaries)."""
    try:
        _, run_dir, record = load_run_record(session_id, run_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except LookupError as e:
        return jsonify({"error": str(e), "code": "run_not_found"}), 404

    result = read_json(os.path.join(run_dir, "result.json"))
    if result is None:
        return jsonify({"error": record.get("error") or "This run has no results.", "code": "no_result", "run": record}), 409
    result["run"] = record  # always the latest record (e.g. after a stability analysis)
    return jsonify(result)


def job_status_response(run_dir, kind):
    status = refresh_job(run_dir, kind)
    if status is None:
        return None
    status = dict(status)
    status["run"] = read_json(os.path.join(run_dir, "run.json"))
    return status


@app.route('/api/sessions/<session_id>/runs/<run_id>/status', methods=['GET'])
def run_status(session_id, run_id):
    """Progress of a background job (kind=cluster or kind=stability)."""
    kind = request.args.get('kind', 'cluster')
    try:
        _, run_dir, _ = load_run_record(session_id, run_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except LookupError as e:
        return jsonify({"error": str(e), "code": "run_not_found"}), 404
    status = job_status_response(run_dir, kind) if kind in ("cluster", "stability") else None
    if status is None:
        return jsonify({"error": f"No {kind} job for this run.", "code": "job_not_found"}), 404
    return jsonify(status)


@app.route('/api/sessions/<session_id>/runs/<run_id>/cancel', methods=['POST'])
def cancel_run(session_id, run_id):
    kind = request.args.get('kind', 'cluster')
    try:
        _, run_dir, _ = load_run_record(session_id, run_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except LookupError as e:
        return jsonify({"error": str(e), "code": "run_not_found"}), 404
    files = job_files(run_dir, kind)
    job = read_json(files["job"])
    if job is None:
        return jsonify({"error": f"No {kind} job for this run.", "code": "job_not_found"}), 404
    if not os.path.exists(files["outcome"]):
        kill_job_process(job)
        finalize_job(run_dir, kind, "cancelled", "cancelled")
        app.logger.info("Cancelled %s job of run %s", kind, run_id)
    return jsonify(job_status_response(run_dir, kind))


@app.route('/api/sessions/<session_id>/runs/<run_id>/profiles', methods=['POST'])
def run_profiles(session_id, run_id):
    """log10 raw-intensity profiles of selected samples (kept out of the main result to keep it small)."""
    try:
        _, run_dir, _ = load_run_record(session_id, run_id)
        arrays = load_run_arrays(run_dir)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except (LookupError, OSError):
        return jsonify({"error": "Run data not found.", "code": "run_not_found"}), 404

    wanted = set(((request.get_json(silent=True) or {}).get("names") or [])[:MAX_PROFILE_REQUEST])
    profiles = [
        {"name": name, "cluster": int(arrays["labels"][i]),
         "intensities": np.round(np.asarray(arrays["intensities"][i], dtype=float), 5).tolist()}
        for i, name in enumerate(arrays["names"]) if name in wanted
    ]
    return jsonify({"profiles": profiles})


@app.route('/api/sessions/<session_id>/runs/<run_id>/characterization', methods=['GET'])
def run_characterization(session_id, run_id):
    """Cluster characterization computed on all rows at the end of the run."""
    try:
        _, run_dir, _ = load_run_record(session_id, run_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except LookupError as e:
        return jsonify({"error": str(e), "code": "run_not_found"}), 404
    data = read_json(os.path.join(run_dir, "characterization.json"))
    if data is None:
        return jsonify({"error": "No cluster characterization is available for this run.", "code": "no_result"}), 404
    return jsonify(data)


@app.route('/api/sessions/<session_id>/runs/<run_id>/members.txt', methods=['GET'])
def run_cluster_members(session_id, run_id):
    """Names of all members of one cluster, one per line (e.g. for enrichment tools)."""
    try:
        _, run_dir, record = load_run_record(session_id, run_id)
        arrays = load_run_arrays(run_dir)
        cluster_id = int(request.args.get('cluster', ''))
    except ValueError as e:
        return jsonify({"error": f"Invalid request: {e}", "code": "invalid_parameters"}), 400
    except (LookupError, OSError):
        return jsonify({"error": "Run data not found.", "code": "run_not_found"}), 404
    members = [name for name, label in zip(arrays["names"], arrays["labels"]) if label == cluster_id]
    if not members:
        return jsonify({"error": f"Cluster {cluster_id} has no members.", "code": "invalid_parameters"}), 404
    filename = f"run_{record['number']}_{'noise' if cluster_id == -1 else f'cluster_{cluster_id}'}_members.txt"
    return Response("\n".join(members) + "\n", mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route('/api/sessions/<session_id>/explore', methods=['POST'])
def explore_data(session_id):
    """Exploratory analysis of the session dataset under the selected preprocessing (no clustering)."""
    try:
        session_dir = get_session_dir(session_id)
        name_column = request.form.get('nameColumn')
        intensity_start_index = int(request.form.get('intensityStartIndex', 1))
        sheet_name = request.form.get('sheetName', '').strip()
        options = parse_preprocessing(request.form.get('preprocessing', '{}'))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input parameter: {e}", "code": "invalid_parameters"}), 400

    dataset_meta = load_dataset_meta(session_dir, request.form.get('datasetId', '').strip())
    if dataset_meta is None:
        return jsonify({"error": "Dataset not found for this session. Please upload the file again.",
                        "code": "dataset_not_found"}), 404
    data_path = os.path.join(dataset_paths(session_dir, dataset_meta["dataset_id"])[0], dataset_meta["data_file"])
    try:
        df = read_table(data_path, dataset_meta["file_type"], sheet_name if dataset_meta["file_type"] == 'xlsx' else None)
        result = pp.preprocess(df, name_column, intensity_start_index, options)
    except ValueError as e:
        return jsonify({"error": f"Preprocessing failed: {e}", "code": "unsupported_data"}), 400

    payload = explore_dataset(df, intensity_start_index, result)
    payload["value_label"] = pp.value_label(options)
    payload["dataset"] = {"filename": dataset_meta["filename"]}
    return jsonify(payload)


@app.route('/api/sessions/<session_id>/runs/<run_id>/labels.tsv', methods=['GET'])
def download_run_labels(session_id, run_id):
    try:
        session_dir = get_session_dir(session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not is_valid_id(run_id):
        return jsonify({"error": "Invalid run id"}), 400

    labels_path = os.path.join(run_dir_path(session_dir, run_id), "labels.tsv")
    if not os.path.isfile(labels_path):
        return jsonify({"error": "Labels not found for this run."}), 404
    record = read_json(os.path.join(run_dir_path(session_dir, run_id), "run.json")) or {}
    download_name = f"run_{record.get('number', run_id)}_{record.get('clustering', {}).get('algorithm', 'labels')}.tsv"
    return send_file(labels_path, mimetype="text/tab-separated-values", as_attachment=True, download_name=download_name)


def keyed_labels(names, labels):
    """Maps (name, occurrence) -> cluster so that duplicate identifiers still align between runs."""
    occurrences = {}
    keyed = {}
    for name, label in zip(names, labels.tolist()):
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        keyed[(name, occurrence)] = label
    return keyed


def compare_labelings(arrays_a, arrays_b):
    """Agreement between two clusterings: ARI/NMI, contingency, overlap-based cluster matching and changed samples."""
    labels_a = keyed_labels(arrays_a["names"], arrays_a["labels"])
    labels_b = keyed_labels(arrays_b["names"], arrays_b["labels"])
    common = [key for key in labels_a if key in labels_b]
    if len(common) < 2:
        raise ValueError("The two runs share fewer than two samples, so they cannot be compared.")

    a = np.array([labels_a[key] for key in common])
    b = np.array([labels_b[key] for key in common])
    a_clusters = np.unique(a)
    b_clusters = np.unique(b)
    counts = contingency_matrix(a, b)  # rows follow a_clusters, columns follow b_clusters
    sizes_a = counts.sum(axis=1)
    sizes_b = counts.sum(axis=0)

    # Cluster ids are arbitrary, so pair clusters by maximal Jaccard overlap (Hungarian algorithm).
    # Noise (-1) is never matched to a real cluster; it only corresponds to noise.
    rows_real = [i for i, c in enumerate(a_clusters) if c != -1]
    cols_real = [j for j, c in enumerate(b_clusters) if c != -1]
    matches = []
    if rows_real and cols_real:
        overlap = counts[np.ix_(rows_real, cols_real)]
        union = sizes_a[rows_real][:, None] + sizes_b[cols_real][None, :] - overlap
        jaccard = overlap / union
        for r, c in zip(*linear_sum_assignment(-jaccard)):
            if overlap[r, c] > 0:
                i, j = rows_real[r], cols_real[c]
                matches.append({
                    "cluster_a": int(a_clusters[i]),
                    "cluster_b": int(b_clusters[j]),
                    "size_a": int(sizes_a[i]),
                    "size_b": int(sizes_b[j]),
                    "overlap": int(overlap[r, c]),
                    "jaccard": float(jaccard[r, c]),
                })
    matches.sort(key=lambda m: m["cluster_a"])

    partner = {m["cluster_a"]: m["cluster_b"] for m in matches}
    if -1 in a_clusters and -1 in b_clusters:
        partner[-1] = -1
    matched_a = {m["cluster_a"] for m in matches}
    matched_b = {m["cluster_b"] for m in matches}

    changed = [
        {"name": key[0], "cluster_a": int(ca), "cluster_b": int(cb)}
        for key, ca, cb in zip(common, a, b)
        if partner.get(int(ca)) != int(cb)
    ]

    return {
        "n_common": len(common),
        "n_only_a": len(labels_a) - len(common),
        "n_only_b": len(labels_b) - len(common),
        "adjusted_rand_index": float(adjusted_rand_score(a, b)),
        "normalized_mutual_info": float(normalized_mutual_info_score(a, b)),
        "contingency": {
            "clusters_a": [int(c) for c in a_clusters],
            "clusters_b": [int(c) for c in b_clusters],
            "counts": counts.tolist(),
        },
        "matches": matches,
        "unmatched_a": [int(c) for c in a_clusters if c != -1 and int(c) not in matched_a],
        "unmatched_b": [int(c) for c in b_clusters if c != -1 and int(c) not in matched_b],
        "n_changed": len(changed),
        "changed_samples": changed[:MAX_CHANGED_SAMPLES],
        "changed_truncated": len(changed) > MAX_CHANGED_SAMPLES,
    }


@app.route('/api/sessions/<session_id>/compare', methods=['GET'])
def compare_runs(session_id):
    """Compares two completed runs of the session on all samples."""
    try:
        session_dir = get_session_dir(session_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400

    run_ids = [request.args.get('run_a', ''), request.args.get('run_b', '')]
    if not all(is_valid_id(run_id) for run_id in run_ids):
        return jsonify({"error": "Two valid run ids (run_a, run_b) are required.", "code": "invalid_parameters"}), 400

    try:
        arrays = [load_run_arrays(run_dir_path(session_dir, run_id)) for run_id in run_ids]
    except OSError:
        return jsonify({"error": "Both runs must exist and have completed successfully.", "code": "run_not_found"}), 404

    try:
        comparison = compare_labelings(*arrays)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "unsupported_data"}), 400

    comparison.update({"run_a": run_ids[0], "run_b": run_ids[1]})
    return jsonify(comparison)


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@app.route('/api/sessions/<session_id>/close', methods=['POST'])
def close_session(session_id):
    """Stops running jobs and deletes all datasets and runs of a session (called when the page is closed)."""
    try:
        session_dir = get_session_dir(session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    kill_session_jobs(session_dir)
    shutil.rmtree(session_dir, ignore_errors=True)
    return jsonify({"status": "deleted"})


def parse_run_settings(form):
    """Preprocessing, reduction and clustering settings of a run request (raises ValueError)."""
    settings = {
        "name_column": form.get('nameColumn'),
        "intensity_start_index": int(form.get('intensityStartIndex', 1)),
        "sheet_name": form.get('sheetName', '').strip(),
        "reduction_method": form.get('reductionMethod', 'None'),
        "reduction_params": parse_param_json(form.get('reductionParams', '{}'), "reduction"),
        "algorithm": form.get('clusteringAlgorithm'),
        "clustering_params": parse_param_json(form.get('clusteringParams', '{}'), "clustering"),
        "preprocessing_options": parse_preprocessing(form.get('preprocessing', '{}')),
    }
    if settings["algorithm"] not in ALGORITHM_PARAMS:
        raise ValueError(f"Unknown clustering algorithm: {settings['algorithm']}")
    if settings["reduction_method"] != 'None' and settings["reduction_method"] not in REDUCTION_PARAMS:
        raise ValueError(f"Unknown reduction method: {settings['reduction_method']}")
    return settings


def resolve_request_dataset(session_dir):
    """The dataset named by datasetId, or the uploaded file. Returns (dataset_meta, None) or (None, error response)."""
    dataset_id = request.form.get('datasetId', '').strip()
    if dataset_id:
        dataset_meta = load_dataset_meta(session_dir, dataset_id)
        if dataset_meta is None:
            return None, (jsonify({
                "error": "Dataset not found for this session. Please upload the file again.",
                "code": "dataset_not_found"
            }), 404)
        return dataset_meta, None
    file = request.files.get('file')
    if file is None or file.filename == '':
        return None, (jsonify({"error": "No file or dataset id in the request", "code": "invalid_parameters"}), 400)
    if detect_file_type(file.filename) is None:
        return None, (jsonify({"error": "Unsupported file format. Please upload CSV, TSV, or Excel.", "code": "unsupported_data"}), 400)
    try:
        return store_dataset(session_dir, file), None
    except UploadTooLarge as e:
        return None, upload_too_large_response(str(e))


@contextlib.contextmanager
def session_lock(session_dir):
    """Serializes run numbering within a session across gunicorn workers and job processes."""
    with open(os.path.join(session_dir, ".lock"), "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def next_run_number(session_dir):
    """Next run number of the session; call while holding session_lock."""
    counter_path = os.path.join(session_dir, "run_counter.json")
    last = max((read_json(counter_path) or {}).get("last", 0),
               max((r.get("number", 0) for r in list_run_records(session_dir)), default=0))
    write_json(counter_path, {"last": last + 1})
    return last + 1


def new_run_record(session_id, run_id, number, dataset_meta, settings):
    base_link = f"/sessions/{session_id}/runs/{run_id}"
    links = {
        "result": base_link,
        "status": f"{base_link}/status",
        "cancel": f"{base_link}/cancel",
        "labels_tsv": f"{base_link}/labels.tsv",
        "download_zip": f"{base_link}/download.zip",
        "report": f"{base_link}/report.html",
        "stability": f"{base_link}/stability",
        "profiles": f"{base_link}/profiles",
        "characterization": f"{base_link}/characterization",
        "members": f"{base_link}/members.txt",
    }
    reduction_params, clustering_params = settings["reduction_params"], settings["clustering_params"]
    sheet_name = settings["sheet_name"]
    return {
        "run_id": run_id,
        "session_id": session_id,
        "number": number,
        "created_at": now_iso(),
        "status": "running",
        "dataset": {
            "dataset_id": dataset_meta["dataset_id"],
            "filename": dataset_meta["filename"],
            "sha256": dataset_meta["sha256"],
            "size_bytes": dataset_meta["size_bytes"],
            "n_observations": None,
            "n_features": None,
        },
        "preprocessing": {
            "name_column": settings["name_column"],
            "intensity_start_index": settings["intensity_start_index"],
            "sheet_name": sheet_name if dataset_meta["file_type"] == 'xlsx' and sheet_name else None,
            **pp.describe(settings["preprocessing_options"]),
            "options": settings["preprocessing_options"],
        },
        "preprocessing_summary": None,
        "reduction": {"method": settings["reduction_method"], "params": dict(reduction_params)},
        "clustering": {"algorithm": settings["algorithm"], "params": dict(clustering_params)},
        "random_seed": {
            "reduction": reduction_params.get("random_state"),
            "clustering": clustering_params.get("random_state"),
        },
        "n_clusters": None,
        "n_noise": None,
        "metrics": None,
        "stability": None,
        "runtime_seconds": {"total": None, "clustering": None},
        "peak_memory_mb": None,
        "limits": {"timeout_seconds": JOB_TIMEOUT_SECONDS, "memory_mb": JOB_MEMORY_MB},
        "warnings": [],
        "error": None,
        "error_code": None,
        "error_hint": None,
        "links": links,
    }


@app.route('/api/cluster', methods=['POST'])
def cluster_data():
    """Starts a clustering job for a session dataset and returns immediately (HTTP 202); poll the status link."""
    cleanup_expired_sessions()
    try:
        session_id = request.form.get('sessionId') or uuid.uuid4().hex
        session_dir = get_session_dir(session_id, create=True)
        settings = parse_run_settings(request.form)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input parameter: {e}", "code": "invalid_parameters"}), 400

    dataset_meta, error_response = resolve_request_dataset(session_dir)
    if error_response:
        return error_response

    if running_job_count() >= MAX_CONCURRENT_JOBS:
        message, hint = error_info("server_busy")
        return jsonify({"error": message, "hint": hint, "code": "server_busy"}), 503

    run_id = uuid.uuid4().hex
    run_dir = run_dir_path(session_dir, run_id)
    os.makedirs(run_dir)
    with session_lock(session_dir):
        record = new_run_record(session_id, run_id, next_run_number(session_dir), dataset_meta, settings)
        write_json(os.path.join(run_dir, "run.json"), record)

    try:
        launch_job(run_dir, "cluster", {"kind": "cluster", "run_dir": run_dir, "session_dir": session_dir})
    except Exception as e:
        app.logger.error("Could not start job for run %s: %s", run_id, e, exc_info=True)
        outcome = finalize_job(run_dir, "cluster", "failed", "internal_error", f"Could not start the job: {e}")
        return jsonify({"error": outcome["error"], "code": "internal_error",
                        "run": read_json(os.path.join(run_dir, "run.json"))}), 500

    return jsonify({"run": record, "state": "running"}), 202


# ---------- Parameter sweep ----------

def sweep_dir_path(session_dir, sweep_id):
    return os.path.join(session_dir, "sweeps", sweep_id)


def load_sweep(session_id, sweep_id):
    """(sweep_dir, sweep) or raises LookupError/ValueError."""
    session_dir = get_session_dir(session_id)
    if not is_valid_id(sweep_id):
        raise ValueError("Invalid sweep id")
    sweep_dir = sweep_dir_path(session_dir, sweep_id)
    sweep = read_json(os.path.join(sweep_dir, "sweep.json"))
    if sweep is None:
        raise LookupError("Sweep not found. The session may have expired.")
    return sweep_dir, sweep


def sweep_response(sweep_dir, sweep):
    """The sweep with its job state; trials that never finished are shown as not run once the job has ended."""
    job = refresh_job(sweep_dir, "sweep") or {"state": "failed", "error": "The sweep job was not found."}
    sweep = read_json(os.path.join(sweep_dir, "sweep.json")) or sweep  # refresh_job may have just finished it
    if job["state"] != "running":
        for trial in sweep["trials"]:
            if trial["state"] in ("pending", "running"):
                trial["state"] = "cancelled" if job["state"] == "cancelled" else "not_run"
    return {**sweep, "job": job}


@app.route('/api/sessions/<session_id>/sweeps', methods=['POST'])
def start_sweep(session_id):
    """Starts a parameter sweep (form fields as for /api/cluster plus a 'sweep' JSON object)."""
    cleanup_expired_sessions()
    try:
        session_dir = get_session_dir(session_id, create=True)
        settings = parse_run_settings(request.form)
        config = json.loads(request.form.get('sweep') or '{}')
        if not isinstance(config, dict):
            raise ValueError("The sweep settings must be a JSON object.")
        trials, info = sw.plan_trials(settings["algorithm"], config.get("parameters"), config.get("max_trials", 20),
                                      ALGORITHM_PARAMS[settings["algorithm"]])
        screening = sw.validate_screening(config.get("screening"), len(trials))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid sweep settings: {e}", "code": "invalid_parameters"}), 400

    dataset_meta, error_response = resolve_request_dataset(session_dir)
    if error_response:
        return error_response
    if running_job_count() >= MAX_CONCURRENT_JOBS:
        message, hint = error_info("server_busy")
        return jsonify({"error": message, "hint": hint, "code": "server_busy"}), 503

    sweep_id = uuid.uuid4().hex
    sweep_dir = sweep_dir_path(session_dir, sweep_id)
    os.makedirs(sweep_dir)
    base_link = f"/sessions/{session_id}/sweeps/{sweep_id}"
    sweep = {
        "sweep_id": sweep_id,
        "session_id": session_id,
        "created_at": now_iso(),
        "dataset": {key: dataset_meta[key] for key in ("dataset_id", "filename", "sha256", "size_bytes")},
        "settings": settings,
        "algorithm": settings["algorithm"],
        **info,
        "screening": screening,
        "trials": [{"trial": i + 1, "values": values, "state": "pending"} for i, values in enumerate(trials)],
        "stopped_reason": None,
        "time_limit_seconds": JOB_TIMEOUT_SECONDS,
        "links": {"self": base_link, "cancel": f"{base_link}/cancel", "config": f"{base_link}/config.json"},
    }
    write_json(os.path.join(sweep_dir, "sweep.json"), sweep)
    try:
        launch_job(sweep_dir, "sweep", {"kind": "sweep", "run_dir": sweep_dir, "session_dir": session_dir})
    except Exception as e:
        app.logger.error("Could not start sweep %s: %s", sweep_id, e, exc_info=True)
        finalize_job(sweep_dir, "sweep", "failed", "internal_error", f"Could not start the job: {e}")
    return jsonify(sweep_response(sweep_dir, sweep)), 202


@app.route('/api/sessions/<session_id>/sweeps/<sweep_id>', methods=['GET'])
def get_sweep(session_id, sweep_id):
    try:
        sweep_dir, sweep = load_sweep(session_id, sweep_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except LookupError as e:
        return jsonify({"error": str(e), "code": "sweep_not_found"}), 404
    return jsonify(sweep_response(sweep_dir, sweep))


@app.route('/api/sessions/<session_id>/sweeps/<sweep_id>/cancel', methods=['POST'])
def cancel_sweep(session_id, sweep_id):
    try:
        sweep_dir, sweep = load_sweep(session_id, sweep_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except LookupError as e:
        return jsonify({"error": str(e), "code": "sweep_not_found"}), 404
    files = job_files(sweep_dir, "sweep")
    if not os.path.exists(files["outcome"]):
        kill_job_process(read_json(files["job"]))
        finalize_job(sweep_dir, "sweep", "cancelled", "cancelled")
    return jsonify(sweep_response(sweep_dir, sweep))


@app.route('/api/sessions/<session_id>/sweeps/<sweep_id>/config.json', methods=['GET'])
def download_sweep_config(session_id, sweep_id):
    """Everything needed to repeat the sweep: data hash, fixed settings, swept values and the trials."""
    try:
        sweep_dir, sweep = load_sweep(session_id, sweep_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except LookupError as e:
        return jsonify({"error": str(e), "code": "sweep_not_found"}), 404
    sweep = sweep_response(sweep_dir, sweep)
    settings = sweep["settings"]
    config = {
        "rapcluster_sweep": sweep["sweep_id"],
        "created_at": sweep["created_at"],
        "dataset": sweep["dataset"],
        "preprocessing": {"name_column": settings["name_column"], "intensity_start_index": settings["intensity_start_index"],
                          "sheet_name": settings["sheet_name"] or None, "options": settings["preprocessing_options"]},
        "reduction": {"method": settings["reduction_method"], "params": settings["reduction_params"]},
        "clustering": {"algorithm": settings["algorithm"], "fixed_params": settings["clustering_params"]},
        "sweep": {key: sweep[key] for key in ("parameters", "grid_size", "max_trials", "search", "random_seed")},
        "screening": sweep["screening"],
        "trials": [{key: trial.get(key) for key in ("trial", "values", "state", "run_number", "metrics",
                                                       "screening_metrics", "error_code", "error")}
                   for trial in sweep["trials"]],
        "stopped_reason": sweep["stopped_reason"],
        "note": ("Preprocessing and dimensionality reduction were identical for all trials. Each confirmed trial is a "
                 "regular run; its ZIP download contains reproduce.py and config.json for that run."),
    }
    response = Response(json.dumps(config, indent=2), mimetype="application/json")
    response.headers["Content-Disposition"] = f"attachment; filename=rapcluster_sweep_{sweep_id[:8]}.json"
    return response


def load_run_record(session_id, run_id):
    """Returns (session_dir, run_dir, record) or raises LookupError/ValueError."""
    session_dir = get_session_dir(session_id)
    if not is_valid_id(run_id):
        raise ValueError("Invalid run id")
    run_dir = run_dir_path(session_dir, run_id)
    record = read_json(os.path.join(run_dir, "run.json"))
    if record is None:
        raise LookupError("Run not found. The session may have expired.")
    return session_dir, run_dir, record


@app.route('/api/sessions/<session_id>/runs/<run_id>/stability', methods=['POST'])
def compute_run_stability(session_id, run_id):
    """Starts a background stability analysis of a completed run (poll status?kind=stability)."""
    try:
        _, run_dir, record = load_run_record(session_id, run_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except LookupError as e:
        return jsonify({"error": str(e), "code": "run_not_found"}), 404
    if record["status"] != "success":
        return jsonify({"error": "Stability can only be computed for successful runs.", "code": "invalid_parameters"}), 409

    current = refresh_job(run_dir, "stability")
    if current and current.get("state") == "running":
        return jsonify(job_status_response(run_dir, "stability")), 202
    if running_job_count() >= MAX_CONCURRENT_JOBS:
        message, hint = error_info("server_busy")
        return jsonify({"error": message, "hint": hint, "code": "server_busy"}), 503

    try:
        options = parse_stability_options(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400

    launch_job(run_dir, "stability", {"kind": "stability", "run_dir": run_dir, "options": options,
                                      "session_dir": os.path.dirname(os.path.dirname(run_dir))})
    return jsonify({"state": "running", "stage": "starting", "run": record, "options": options}), 202


@app.route('/api/sessions/<session_id>/runs/<run_id>/stability', methods=['GET'])
def get_run_stability(session_id, run_id):
    """Details of the last stability analysis (cluster Jaccard, distributions, least stable rows)."""
    try:
        _, run_dir, _ = load_run_record(session_id, run_id)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "invalid_parameters"}), 400
    except LookupError as e:
        return jsonify({"error": str(e), "code": "run_not_found"}), 404
    details = read_json(os.path.join(run_dir, "stability.json"))
    if details is None:
        return jsonify({"error": "No stability analysis has been computed for this run.", "code": "not_computed"}), 404
    return jsonify(details)


# ---------- Example datasets ----------

@app.route('/api/examples', methods=['GET'])
def list_examples():
    return jsonify({"examples": [
        {key: value for key, value in example.items() if key != "path"} | {"filename": os.path.basename(example["path"])}
        for example in EXAMPLE_DATASETS
        if os.path.isfile(os.path.join(APP_DIR, example["path"]))
    ]})


@app.route('/api/examples/<example_id>/file', methods=['GET'])
def download_example(example_id):
    example = next((e for e in EXAMPLE_DATASETS if e["id"] == example_id), None)
    path = os.path.join(APP_DIR, example["path"]) if example else None
    if path is None or not os.path.isfile(path):
        return jsonify({"error": "Example dataset not found."}), 404
    return send_file(path, mimetype="text/tab-separated-values", as_attachment=True,
                     download_name=os.path.basename(path))


# ---------- Download All ----------

def package_versions():
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ["numpy", "pandas", "scikit-learn", "scipy", "umap-learn", "hdbscan", "matplotlib", "Flask", "openpyxl"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def cluster_name(cluster_id):
    return "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"


def group_by_cluster(labels):
    """Sample indices per cluster; real clusters in numeric order, noise last."""
    clusters = sorted(set(labels.tolist()), key=lambda c: (c == -1, c))
    return {cluster_id: np.flatnonzero(labels == cluster_id) for cluster_id in clusters}


def figure_bytes(fig, fmt):
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt)
    return buffer.getvalue()


def scatter_figure(record, embedding, colors, groups):
    fig = Figure(figsize=(7, 6), dpi=150)
    ax = fig.subplots()
    many = len(embedding) > 20000  # rasterize huge point clouds so the SVG stays small
    for cluster_id, idx in groups.items():
        ax.scatter(embedding[idx, 0], embedding[idx, 1], s=3 if many else 8, linewidths=0, rasterized=many,
                   c=colors.get(str(cluster_id), "#000000"), label=f"{cluster_name(cluster_id)} (n={len(idx)})")
    reduction = record["reduction"]["method"]
    ax.set_title(f"Run #{record['number']}: {record['clustering']['algorithm']}"
                 + (f" on {reduction}" if reduction != "None" else ""))
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    if record.get("display"):
        ax.text(0.01, 0.01, f"Axes: {record['display']}", transform=ax.transAxes, fontsize=6, color="#666")
    if len(groups) <= 20:
        ax.legend(fontsize=7, markerscale=2, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def sizes_figure(colors, groups):
    fig = Figure(figsize=(7, 3.5), dpi=150)
    ax = fig.subplots()
    labels = ["Noise" if c == -1 else str(c) for c in groups]
    ax.bar(labels, [len(idx) for idx in groups.values()],
           color=[colors.get(str(c), "#000000") for c in groups])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Samples")
    ax.set_title("Cluster sizes")
    fig.tight_layout()
    return fig


def profile_tables(columns, profiles, groups):
    """Per-cluster mean (centroid) and median of the preprocessed profiles."""
    if len(columns) != profiles.shape[1]:
        columns = [f"Dimension {i + 1}" for i in range(profiles.shape[1])]
    header = "Cluster\tSize\t" + "\t".join(columns) + "\n"
    tables = {}
    for statistic, function in [("mean", np.mean), ("median", np.median)]:
        lines = [header]
        for cluster_id, idx in groups.items():
            values = function(np.asarray(profiles[idx], dtype=float), axis=0)
            lines.append(f"{cluster_id}\t{len(idx)}\t" + "\t".join(f"{v:.6f}" for v in values) + "\n")
        tables[statistic] = "".join(lines)
    return tables


def profile_band_figure(characterization, colors, max_panels=12):
    """Median profile with IQR band for the largest clusters (small multiples)."""
    profiles = characterization["profiles"]
    sizes = {entry["cluster"]: entry["size"] for entry in characterization["clusters"]}
    chosen = sorted(sorted(profiles["clusters"], key=lambda c: -sizes[c])[:max_panels], key=lambda c: (c == -1, c))
    columns = characterization["columns"]
    n_cols = min(3, len(chosen))
    n_rows = -(-len(chosen) // n_cols)
    fig = Figure(figsize=(3.3 * n_cols, 2.4 * n_rows + 0.4), dpi=150)
    axes = fig.subplots(n_rows, n_cols, squeeze=False, sharey=True)
    x = np.arange(len(columns))
    for ax, cluster_id in zip(axes.flat, chosen):
        i = profiles["clusters"].index(cluster_id)
        color = colors.get(str(cluster_id), "#000000")
        ax.fill_between(x, np.array(profiles["q1"][i], dtype=float), np.array(profiles["q3"][i], dtype=float),
                        color=color, alpha=0.3, linewidth=0)
        ax.plot(x, np.array(profiles["median"][i], dtype=float), color=color, linewidth=1.2)
        ax.set_title(f"{cluster_name(cluster_id)} (n={sizes[cluster_id]})", fontsize=8)
        ax.tick_params(labelsize=6)
        if len(columns) <= 12:
            ax.set_xticks(x, columns, rotation=90, fontsize=5)
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    title = f"Median profile with IQR band ({characterization.get('value_label', 'preprocessed values')})"
    if len(chosen) < len(sizes):
        title += f"; {len(chosen)} largest of {len(sizes)} groups"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def characterization_files(characterization):
    """TSV tables of the cluster characterization for the ZIP download."""
    files = {}
    optional = [(key, label) for key, label in (("mean_membership", "MeanMembership"), ("persistence", "Persistence"),
                                                 ("component_weight", "ComponentWeight"))
                if any(key in entry for entry in characterization["clusters"])]
    lines = ["Cluster\tSize\tFraction\tMedoid\tRepresentatives" + "".join(f"\t{label}" for _, label in optional) + "\n"]
    for entry in characterization["clusters"]:
        lines.append(f"{entry['cluster']}\t{entry['size']}\t{entry['fraction']:.6f}\t{entry['medoid']}\t"
                     f"{', '.join(entry['representatives'])}"
                     + "".join(f"\t{format_cell(entry.get(key))}" for key, _ in optional) + "\n")
    files["cluster_summary.tsv"] = "".join(lines)

    profiles, columns = characterization["profiles"], characterization["columns"]
    keys = ("lower", "q1", "median", "q3", "upper", "mean")
    lines = ["Cluster\tColumn\tLowerWhisker\tQ1\tMedian\tQ3\tUpperWhisker\tMean\n"]
    for i, cluster_id in enumerate(profiles["clusters"]):
        for j, column in enumerate(columns):
            lines.append(f"{cluster_id}\t{column}\t" + "\t".join(
                "" if profiles[key][i][j] is None else str(profiles[key][i][j]) for key in keys) + "\n")
    files["cluster_profile_bands.tsv"] = "".join(lines)

    lines = ["Cluster\tRank\tFeature\tCliffsDelta\tMedianInCluster\tMedianRest\n"]
    for entry in characterization["clusters"]:
        for rank, feature in enumerate(entry["top_features"], start=1):
            lines.append(f"{entry['cluster']}\t{rank}\t{feature['feature']}\t{feature['cliffs_delta']}\t"
                         f"{feature['median_in']}\t{feature['median_rest']}\n")
    files["distinguishing_features.tsv"] = "".join(lines)

    delta = characterization["cliffs_delta"]
    lines = ["Cluster\t" + "\t".join(columns) + "\n"]
    for cluster_id, row in zip(delta["clusters"], delta["values"]):
        lines.append(f"{cluster_id}\t" + "\t".join("" if v is None else str(v) for v in row) + "\n")
    files["cliffs_delta_matrix.tsv"] = "".join(lines)

    tests = characterization.get("feature_tests")
    if tests:
        lines = ["Feature\tH\tEpsilonSquared\tP\tQ_BH\n"]
        for row in tests["rows"]:
            lines.append("\t".join("" if row[k] is None else str(row[k])
                                    for k in ("feature", "H", "epsilon_squared", "p", "q")) + "\n")
        files["feature_tests.tsv"] = "".join(lines)

    lines = ["Name\tCluster\tDistanceToClusterMedian\tRobustZ\n"]
    for row in characterization["outliers"]:
        lines.append(f"{row['name']}\t{row['cluster']}\t{row['distance']}\t{row['robust_z']}\n")
    files["outliers.tsv"] = "".join(lines)
    return files


def preprocessing_rows(record):
    prep = record["preprocessing"]
    described = pp.describe(prep.get("options") or pp.DEFAULT_OPTIONS)
    summary = record.get("preprocessing_summary") or {}
    rows = [("Name column", prep["name_column"]), ("Data start column index", prep["intensity_start_index"]),
            ("Sheet", prep.get("sheet_name") or "–"), ("Missing values", described["missing_values"]),
            ("Transformation", described["transform"]), ("All-zero rows", described["zero_rows"]),
            ("Column filters", described["filters"]), ("Scaling", described["scaling"])]
    if summary:
        correlated = [f"{c['column']} (with {c['correlated_with']})" for c in summary["columns_removed_correlated"]]
        rows += [("Rows used", f"{summary['rows_used']} of {summary['rows_in']}"),
                 ("Rows removed (too many missing values)", summary["rows_removed_missing"]),
                 ("Rows removed (all zero)", summary["rows_removed_zero"]),
                 ("Missing cells imputed", summary["cells_imputed"]),
                 ("Columns used", f"{summary['columns_used']} of {summary['columns_in']}"),
                 ("Columns removed (low variance)", short_list(summary["columns_removed_low_variance"], 30) or "none"),
                 ("Columns removed (correlated)", short_list(correlated, 30) or "none")]
    return rows


def methods_text(record):
    def params_text(params):
        return ", ".join(f"{k}={'None' if v is None else v}" for k, v in params.items())
    prep = record["preprocessing"]
    described = pp.describe(prep.get("options") or pp.DEFAULT_OPTIONS)
    summary = record.get("preprocessing_summary") or {}
    text = (f"Data values were read from column index {prep['intensity_start_index']} onward. Missing values were "
            f"{described['missing_values']}, values were transformed as {described['transform']}, and all-zero rows "
            f"were {described['zero_rows']}. ")
    if described["filters"] != "none":
        text += f"Filters: {described['filters']}. "
    text += f"Scaling: {described['scaling']}. "
    if summary:
        text += (f"{summary['rows_used']} of {summary['rows_in']} rows and {summary['columns_used']} of "
                 f"{summary['columns_in']} columns were used. ")
    dims = record["reduction"].get("clustering_dims")
    if record["reduction"]["method"] != "None":
        text += (f"The data were reduced with {record['reduction']['method']} "
                 f"({params_text(record['reduction']['params'])})"
                 + (f" to {dims} dimensions, all of which were used for clustering. " if dims
                    else " and the first two dimensions were kept. "))
    elif dims:
        text += f"Clustering used all {dims} features without dimensionality reduction. "
    text += (f"Clustering used {record['clustering']['algorithm']} ({params_text(record['clustering']['params'])}). "
             f"Analysis was performed with RapCluster.")
    return text


def characterization_html(characterization, profiles_svg, max_rows=50):
    e = html.escape
    if characterization is None:
        return "<p>Not available for this run.</p>"
    rows = []
    for entry in characterization["clusters"][:max_rows]:
        features = ", ".join(f"{f['feature']} ({f['cliffs_delta']:+.2f})" for f in entry["top_features"][:3]
                             if f["cliffs_delta"] is not None) or "–"
        rows.append(f"<tr><td>{e(cluster_name(entry['cluster']))}</td><td>{entry['size']}</td>"
                    f"<td>{100 * entry['fraction']:.1f}%</td><td>{e(entry['medoid'])}</td>"
                    f"<td>{e(', '.join(entry['representatives'][:3]))}</td><td>{e(features)}</td></tr>")
    more = (f"<p>Showing {max_rows} of {len(characterization['clusters'])} groups; see cluster_summary.tsv.</p>"
            if len(characterization["clusters"]) > max_rows else "")
    tests = characterization.get("feature_tests")
    tests_html = ""
    if tests:
        top = "".join(f"<tr><td>{e(r['feature'])}</td><td>{r['epsilon_squared']}</td><td>{r['H']}</td></tr>"
                      for r in tests["rows"][:10])
        tests_html = (f"<h3>Features differing most between clusters (Kruskal–Wallis)</h3>"
                      f"<p><em>{e(TEST_NOTE)}</em></p>"
                      f"<table><tr><th>Feature</th><th>ε²</th><th>H</th></tr>{top}</table>"
                      f"<p>All features with p-values and Benjamini–Hochberg q-values: feature_tests.tsv.</p>")
    svg_start = profiles_svg.find("<svg") if profiles_svg else -1
    return f"""<p>Computed on all rows of the data that was clustered ({e(characterization.get('value_label', ''))}).
The medoid is the most central member (estimated on up to {characterization['settings']['medoid_sample']} members for large
clusters); representative members are those closest to the medoid. Cliff's δ compares each cluster with all other rows
(−1 … 1).</p>
<table><tr><th>Cluster</th><th>Size</th><th>Share</th><th>Medoid</th><th>Representative members</th>
<th>Most distinguishing features (Cliff's δ)</th></tr>{''.join(rows)}</table>{more}
<div class="figure">{profiles_svg[svg_start:] if svg_start >= 0 else ''}</div>
{tests_html}"""


def stability_tsv(details):
    lines = ["Cluster\tSize\tMeanJaccard\tStdJaccard\tMinJaccard\tVerdict\tMeanRowStability\n"]
    for entry in details["clusters"]:
        lines.append("\t".join(format_cell(entry.get(key)) for key in (
            "cluster", "size", "mean_jaccard", "std_jaccard", "min_jaccard", "verdict", "mean_row_stability")) + "\n")
    return "".join(lines)


def stability_rows(stability):
    """Report rows for a computed stability analysis."""
    if not stability:
        return [("Stability", "not computed")]
    rows = [("Stability (mean ARI vs. the run)", f"{stability['mean_ari']:.3f} ± {stability['std_ari']:.3f}"),
            ("Stability solutions", f"{stability['n_resamples']} subsamples of {100 * stability['subsample_fraction']:.0f}%"
                                    + (f" + {stability['n_seeds']} other seeds" if stability.get("n_seeds") else ""))]
    if stability.get("mean_row_stability") is not None:  # older analyses only have the mean ARI
        rows += [("Mean row stability", f"{stability['mean_row_stability']:.3f} "
                                        f"({100 * stability['share_stable_rows']:.1f}% of rows ≥ {stab.STABLE_ROW})"),
                 ("Clusters stable / doubtful / unstable",
                  f"{stability['n_stable_clusters']} / {stability['n_doubtful_clusters']} / {stability['n_unstable_clusters']}"),
                 ("Rows whose consensus cluster differs", stability["consensus_changed"])]
    return rows


def model_rows(record):
    """Algorithm-specific criteria for the report (mixture models, HDBSCAN)."""
    metrics, model = record["metrics"], record.get("model") or {}
    rows = []
    if metrics.get("bic") is not None:
        rows += [("BIC (lower is better)", f"{metrics['bic']:.1f}"), ("AIC (lower is better)", f"{metrics['aic']:.1f}")]
    if metrics.get("mean_membership") is not None:
        rows.append(("Mean membership of clustered rows", f"{metrics['mean_membership']:.3f}"))
    if "n_components_used" in model:
        rows.append(("Mixture components used", f"{model['n_components_used']} of at most {model['n_components_max']} "
                                                 f"({model['n_components_weight_above_1pct']} with weight > 1%)"))
    if "converged" in model:
        rows.append(("Converged", f"{'yes' if model['converged'] else 'no'} after {model['n_iter']} iterations"))
    return rows


def report_html(record, groups, scatter_svg, versions, characterization=None, profiles_svg=None):
    e = html.escape
    metrics = record["metrics"]
    def fmt(value, digits=3):
        return "–" if value is None else f"{value:.{digits}f}"

    def table(rows):
        return "<table>" + "".join(f"<tr><th>{e(str(k))}</th><td>{e(str(v))}</td></tr>" for k, v in rows) + "</table>"

    params_rows = lambda params: [(k, "None" if v is None else v) for k, v in params.items()] or [("–", "–")]
    total = sum(len(rows) for rows in groups.values())
    size_rows = "".join(
        f"<tr><td>{e(cluster_name(c))}</td><td>{len(rows)}</td><td>{100 * len(rows) / total:.1f}%</td></tr>"
        for c, rows in groups.items()
    )
    stability = record.get("stability")
    svg_start = scatter_svg.find("<svg")
    warnings_html = "".join(f"<li>{e(w)}</li>" for w in record["warnings"]) or "<li>None</li>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RapCluster Run #{record['number']} Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; color: #333; }}
h1 {{ color: #28a745; }} h2 {{ color: #28a745; border-bottom: 2px solid #eee; padding-bottom: 4px; margin-top: 28px; }}
table {{ border-collapse: collapse; margin: 8px 0; }} th, td {{ border: 1px solid #ddd; padding: 5px 9px; text-align: left; }}
th {{ background: #f3f5f7; }} .figure svg {{ max-width: 100%; height: auto; }} code {{ background: #f3f5f7; padding: 1px 4px; }}
</style></head><body>
<h1>RapCluster analysis report — Run #{record['number']}</h1>
<p>Created {e(record['created_at'])}. Status: <strong>{e(record['status'])}</strong>.</p>

<h2>Summary</h2>
{table([("Algorithm", record['clustering']['algorithm']), ("Dimensionality reduction", record['reduction']['method']),
        ("Clusters", record['n_clusters']), ("Noise points", f"{record['n_noise']} ({100 * record['n_noise'] / total:.1f}%)"),
        ("Silhouette (higher is better)", fmt(metrics['silhouette_score'])),
        ("Calinski–Harabasz (higher is better)", fmt(metrics['calinski_harabasz_score'])),
        ("Davies–Bouldin (lower is better)", fmt(metrics['davies_bouldin_score'])),
        ("DBCV (higher is better)", fmt(metrics.get('dbcv'))),
        *model_rows(record),
        *stability_rows(stability),
        ("Runtime", f"{fmt(record['runtime_seconds']['total'], 2)} s total, {fmt(record['runtime_seconds']['clustering'], 2)} s clustering"),
        ("Peak memory", f"{fmt(record['peak_memory_mb'], 1)} MB")])}

<h2>Scatter plot</h2>
<div class="figure">{scatter_svg[svg_start:] if svg_start >= 0 else ''}</div>

<h2>Cluster sizes</h2>
<table><tr><th>Cluster</th><th>Size</th><th>Share</th></tr>{size_rows}</table>

<h2>Cluster characterization</h2>
{characterization_html(characterization, profiles_svg)}

<h2>Dataset and preprocessing</h2>
{table([("File", record['dataset']['filename']), ("SHA-256", record['dataset']['sha256']),
        ("Rows × features used", f"{record['dataset']['n_observations']} × {record['dataset']['n_features']}"),
        *preprocessing_rows(record)])}

<h2>Parameters</h2>
<h3>Dimensionality reduction: {e(record['reduction']['method'])}</h3>
{table(params_rows(record['reduction']['params']))}
<h3>Clustering: {e(record['clustering']['algorithm'])}</h3>
{table(params_rows(record['clustering']['params']))}
<p>Random seeds — reduction: {e(str(record['random_seed']['reduction']))}, clustering: {e(str(record['random_seed']['clustering']))}.</p>

<h2>Methods text</h2>
<p>{e(methods_text(record))}</p>

<h2>Warnings</h2>
<ul>{warnings_html}</ul>

<h2>Reproducibility</h2>
<p>Run <code>python reproduce.py {e(record['dataset']['filename'])}</code> in the extracted folder (settings are read from <code>config.json</code>).</p>
{table(list(versions.items()))}
</body></html>
"""


ZIP_README = """RapCluster results — Run #{number}
Profile values: {value_label}

cluster_assignments.tsv     Cluster per sample with original identifier (Name) and 1-based data row (OriginalRow)
cluster_sizes.tsv           Number and share of samples per cluster
cluster_centroids.tsv       Per-cluster mean of the preprocessed profiles (centroids)
cluster_medians.tsv         Per-cluster median of the preprocessed profiles
cluster_summary.tsv         Size, share, medoid and representative members per cluster
cluster_profile_bands.tsv   Per-cluster quartiles, whiskers and mean for every column
distinguishing_features.tsv Top features per cluster by Cliff's delta (cluster vs. all other rows)
cliffs_delta_matrix.tsv     Cliff's delta of every cluster for every column
feature_tests.tsv           Kruskal–Wallis per feature with epsilon squared and BH q-values (descriptive only:
                            the clusters were formed to differ, so p-values are not valid inference)
outliers.tsv                Rows furthest from their cluster's median profile (robust z-score)
noise_points.tsv            Rows labelled as noise (-1), if any
cluster_stability.tsv       Per-cluster Jaccard stability and verdict (only after a stability analysis; row stability
                            and consensus cluster are then also columns of cluster_assignments.tsv)
cluster_members/            One text file per cluster with its member names (e.g. for enrichment tools)
metrics.json                Validation metrics, stability, runtime and memory
scatter.png / .svg          Scatter plot colored by cluster (the 2D view is described in the report)
cluster_sizes.png / .svg    Cluster-size bar chart
cluster_profiles.png / .svg Median profile with IQR band per cluster
config.json                 Preprocessing, dimensionality reduction and clustering settings incl. random seeds
run_record.json             Complete run record (status, warnings, timings, dataset hash, preprocessing summary)
software_versions.json      Python and package versions used on the server
warnings.txt                Warnings raised during the run
report.html                 Human-readable analysis report
reproduce.py                Standalone script: python reproduce.py <data file> (needs preprocessing.py next to it)
preprocessing.py            RapCluster's preprocessing steps, used by reproduce.py
"""


def load_finished_run(session_id, run_id, action):
    """(run_dir, record, result) of a successful run, or (None, None, error response)."""
    try:
        _, run_dir, record = load_run_record(session_id, run_id)
    except ValueError as e:
        return None, None, (jsonify({"error": str(e), "code": "invalid_parameters"}), 400)
    except LookupError as e:
        return None, None, (jsonify({"error": str(e), "code": "run_not_found"}), 404)
    result = read_json(os.path.join(run_dir, "result.json"))
    if result is None:
        return None, None, (jsonify({"error": f"Only successful runs can be {action}.", "code": "no_result"}), 409)
    return run_dir, record, result


def report_parts(run_dir, record, result):
    """Full-data figures and tables shared by report.html and the ZIP download."""
    arrays = load_run_arrays(run_dir)  # full data, not the plotting sample
    groups = group_by_cluster(arrays["labels"])
    colors = result["cluster_colors"]
    scatter = scatter_figure(record, arrays["embedding"], colors, groups)
    characterization = read_json(os.path.join(run_dir, "characterization.json"))
    profile_fig = profile_band_figure(characterization, colors) if characterization else None
    return {
        "arrays": arrays, "groups": groups, "colors": colors, "versions": package_versions(),
        "scatter": scatter, "scatter_svg": figure_bytes(scatter, "svg"),
        "characterization": characterization, "profile_fig": profile_fig,
        "profiles_svg": figure_bytes(profile_fig, "svg").decode("utf-8") if profile_fig else None,
    }


@app.route('/api/sessions/<session_id>/runs/<run_id>/report.html', methods=['GET'])
def view_run_report(session_id, run_id):
    """The run's HTML report (the same report.html as in the ZIP), shown in the browser."""
    run_dir, record, result = load_finished_run(session_id, run_id, "shown as a report")
    if run_dir is None:
        return result
    parts = report_parts(run_dir, record, result)
    page = report_html(record, parts["groups"], parts["scatter_svg"].decode("utf-8"), parts["versions"],
                       parts["characterization"], parts["profiles_svg"])
    return Response(page, mimetype="text/html")


@app.route('/api/sessions/<session_id>/runs/<run_id>/download.zip', methods=['GET'])
def download_run_zip(session_id, run_id):
    run_dir, record, result = load_finished_run(session_id, run_id, "downloaded")
    if run_dir is None:
        return result

    parts = report_parts(run_dir, record, result)
    arrays, groups, colors, versions = parts["arrays"], parts["groups"], parts["colors"], parts["versions"]
    total = len(arrays["labels"])
    scatter, scatter_svg = parts["scatter"], parts["scatter_svg"]
    sizes = sizes_figure(colors, groups)
    profiles = profile_tables(result.get("intensity_columns") or [], arrays["profiles"], groups)
    characterization, profile_fig, profiles_svg = parts["characterization"], parts["profile_fig"], parts["profiles_svg"]
    config = {
        "rapcluster_run": record["number"],
        "created_at": record["created_at"],
        "dataset": {key: record["dataset"][key] for key in ("filename", "sha256", "size_bytes")},
        "preprocessing": record["preprocessing"],
        "reduction": record["reduction"],
        "clustering": record["clustering"],
        "random_seed": record["random_seed"],
    }
    metrics = {
        **record["metrics"],
        "n_noise": record["n_noise"],
        "stability": record.get("stability"),
        "runtime_seconds": record["runtime_seconds"],
        "stage_seconds": record.get("stage_seconds"),
        "peak_memory_mb": record["peak_memory_mb"],
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", ZIP_README.format(
            number=record["number"], value_label=result.get("value_label", "preprocessed values")))
        archive.write(os.path.join(run_dir, "labels.tsv"), "cluster_assignments.tsv")
        archive.writestr("cluster_sizes.tsv", "Cluster\tSize\tFraction\n" + "".join(
            f"{c}\t{len(idx)}\t{len(idx) / total:.6f}\n" for c, idx in groups.items()))
        archive.writestr("cluster_centroids.tsv", profiles["mean"])
        archive.writestr("cluster_medians.tsv", profiles["median"])
        archive.writestr("metrics.json", json.dumps(metrics, indent=2))
        archive.writestr("scatter.png", figure_bytes(scatter, "png"))
        archive.writestr("scatter.svg", scatter_svg)
        archive.writestr("cluster_sizes.png", figure_bytes(sizes, "png"))
        archive.writestr("cluster_sizes.svg", figure_bytes(sizes, "svg"))
        archive.writestr("config.json", json.dumps(config, indent=2))
        archive.writestr("run_record.json", json.dumps(record, indent=2))
        archive.writestr("software_versions.json", json.dumps(versions, indent=2))
        archive.writestr("warnings.txt", "\n".join(record["warnings"]) or "No warnings.")
        archive.writestr("report.html", report_html(record, groups, scatter_svg.decode("utf-8"), versions,
                                                    characterization, profiles_svg))
        archive.write(os.path.join(APP_DIR, "rapcluster_reproduce.py"), "reproduce.py")
        archive.write(os.path.join(APP_DIR, "preprocessing.py"), "preprocessing.py")
        stability_details = read_json(os.path.join(run_dir, "stability.json"))
        if stability_details:
            archive.writestr("cluster_stability.tsv", stability_tsv(stability_details))
        if characterization:
            for filename, content in characterization_files(characterization).items():
                archive.writestr(filename, content)
            archive.writestr("cluster_profiles.png", figure_bytes(profile_fig, "png"))
            archive.writestr("cluster_profiles.svg", profiles_svg)
        names = arrays["names"]
        if -1 in groups:
            archive.writestr("noise_points.tsv", "Name\n" + "".join(f"{names[i]}\n" for i in groups[-1]))
        for cluster_id, idx in groups.items():
            member_file = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
            archive.writestr(f"cluster_members/{member_file}.txt", "".join(f"{names[i]}\n" for i in idx))
    buffer.seek(0)

    download_name = f"rapcluster_run_{record['number']}_{record['clustering']['algorithm']}.zip"
    return send_file(buffer, mimetype="application/zip", as_attachment=True, download_name=download_name)


# ---------- Preflight checks ----------

def capacity_level(n, limits):
    comfortable, workable = limits[0], limits[1]
    if n <= comfortable:
        return "recommended"
    return "may_be_slow" if n <= workable else "not_recommended"


def worse_level(level, steps=1):
    return LEVEL_ORDER[min(LEVEL_ORDER.index(level) + steps, len(LEVEL_ORDER) - 1)]


def algorithm_guidance(algorithm, params, n, n_dims):
    limits = ALGORITHM_CAPACITY.get(algorithm)
    if limits is None:
        return {"algorithm": algorithm, "level": "may_be_slow", "note": "No capacity information for this algorithm."}
    if algorithm == "SpectralClustering" and params.get("affinity") == "rbf":
        limits = SPECTRAL_RBF_CAPACITY
    level = capacity_level(n, limits)
    note = limits[2]
    if algorithm in ("GaussianMixture", "BayesianGaussianMixture") and params.get("covariance_type", "full") == "full" and n_dims > 50:
        level = worse_level(level)
        note += f" Clustering runs on {n_dims} features with full covariance, which is slower."
    if algorithm == "AffinityPropagation" or limits is SPECTRAL_RBF_CAPACITY:
        matrices = 3 if algorithm == "AffinityPropagation" else 1
        note += f" Estimated memory for {n} samples: {matrices * n * n * 8 / 1024 ** 2:,.0f} MB."
    return {
        "algorithm": algorithm,
        "level": level,
        "note": note,
        "comfortable_up_to": limits[0],
        "workable_up_to": limits[1],
    }


def short_list(items, limit=8):
    items = [str(item) for item in items]
    return ", ".join(items[:limit]) + (" …" if len(items) > limit else "")


def dataset_checks(df, intensity_start_index, name_column, size_bytes, options):
    intensity_cols, raw = pp.numeric_matrix(df, intensity_start_index)
    missing = df[intensity_cols].isna().to_numpy()
    non_numeric = np.isnan(raw) & ~missing
    n_rows, n_features = raw.shape
    described = pp.describe(options)
    row_scaling = options["scaling"].startswith("row_")

    try:
        result = pp.preprocess(df, name_column, intensity_start_index, options)
        prep_summary, prep_error = result["summary"], None
        n_used, n_cols_used = prep_summary["rows_used"], prep_summary["columns_used"]
    except ValueError as e:
        result, prep_summary, prep_error = None, None, str(e)
        n_used, n_cols_used = 0, n_features

    filled = np.where(np.isnan(raw), 0.0, raw)
    zero_rows = int(np.all(filled == 0.0, axis=1).sum())
    duplicate_rows = int(pd.DataFrame(filled).duplicated().sum())
    below_minus_one = int((filled <= -1).sum())
    log_transform = options["transform"] in ("log2p1", "log10p1")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        column_sd = np.nanstd(raw, axis=0)
    constant = [str(c) for c, sd in zip(intensity_cols, column_sd) if not sd > 0]
    # Rough size of the interactive JSON response (one rounded profile per plotted sample).
    estimated_output_mb = min(n_used, MAX_PLOT_POINTS) * (n_cols_used * 8 + 120) / 1024 ** 2

    checks = [
        {"check": "Observations", "value": f"{n_rows} rows ({n_used} used after preprocessing)",
         "level": "error" if n_used < 2 else "ok",
         "message": "At least two rows are needed after preprocessing." if n_used < 2 else ""},
        {"check": "Features", "value": f"{n_features} data columns ({n_cols_used} used)",
         "level": "warning" if n_cols_used < 2 else "ok",
         "message": "Check the data start column index and the column filters." if n_cols_used < 2 else ""},
    ]
    if prep_error:
        checks.append({"check": "Preprocessing", "value": "failed", "level": "error", "message": prep_error})
    checks += [
        {"check": "Missing values", "value": f"{int(missing.sum())} cells ({100 * missing.mean():.2f}%)",
         "level": "warning" if missing.any() else "ok",
         "message": f"Missing values are {described['missing_values']}." if missing.any() else ""},
        {"check": "Non-numeric values", "value": f"{int(non_numeric.sum())} cells",
         "level": "warning" if non_numeric.any() else "ok",
         "message": ("Text in data columns is treated as a missing value. Is the data start column index correct?"
                     if non_numeric.any() else "")},
        {"check": "Values ≤ −1", "value": f"{below_minus_one} cells",
         "level": "error" if below_minus_one and log_transform else ("info" if below_minus_one else "ok"),
         "message": (f"{described['transform']} is undefined for values ≤ −1; choose the signed log or no transformation."
                     if below_minus_one and log_transform else
                     "Negative values are fine with the selected transformation." if below_minus_one else "")},
        {"check": "Constant features", "value": f"{len(constant)} columns",
         "level": "info" if constant else "ok",
         "message": ((f"Removed before clustering: {short_list(constant)}" if options["remove_low_variance"] else
                      f"No variation in: {short_list(constant)}. Enable 'Remove low-variance columns' to drop them.")
                     if constant else "")},
        {"check": "All-zero rows", "value": f"{zero_rows} rows",
         "level": "info" if zero_rows else "ok",
         "message": ("" if not zero_rows else "These rows are removed before clustering." if options["remove_zero_rows"]
                     else "These rows are kept; with per-row scaling they have no defined profile shape." if row_scaling
                     else "These rows are kept.")},
    ]
    if prep_summary and prep_summary["rows_removed_missing"]:
        checks.append({"check": "Rows with many missing values", "value": f"{prep_summary['rows_removed_missing']} rows",
                       "level": "info", "message": f"Removed: more than {options['max_missing_fraction']:.0%} of their values are missing."})
    if prep_summary and prep_summary["columns_removed_correlated"]:
        removed = [c["column"] for c in prep_summary["columns_removed_correlated"]]
        checks.append({"check": "Correlated columns", "value": f"{len(removed)} columns", "level": "info",
                       "message": f"Removed by the correlation filter: {short_list(removed)}"})
    checks.append({"check": "Duplicate rows", "value": f"{duplicate_rows} rows",
                   "level": "warning" if duplicate_rows else "ok",
                   "message": "Identical profiles can distort density-based methods and metrics." if duplicate_rows else ""})

    # Does the scaling suit the shape of the data?
    if row_scaling and n_cols_used < 3:
        checks.append({"check": "Scaling", "value": described["scaling"], "level": "warning",
                       "message": "Per-row scaling with fewer than 3 columns removes almost all information. Use per-column scaling."})
    elif row_scaling and n_cols_used <= 10:
        checks.append({"check": "Scaling", "value": described["scaling"], "level": "info",
                       "message": "Per-row scaling compares the shape of each row. It suits ordered measurements of the same "
                                  "quantity (fractions, time points). If the columns are different measurements "
                                  "(e.g. sizes or concentrations of different things), use per-column scaling."})
    elif options["scaling"] == "none" and result is not None:
        sd = result["transformed"].std(axis=0)
        positive = sd[sd > 0]
        if positive.size > 1 and positive.max() / positive.min() > 100:
            checks.append({"check": "Scaling", "value": "none", "level": "warning",
                           "message": f"Column spreads differ by a factor of {positive.max() / positive.min():,.0f}; "
                                      "distance-based clustering will be dominated by the widest columns. "
                                      "Consider per-column scaling."})

    if name_column in df.columns:
        names = df[name_column].astype(str)
        duplicated_names = names[names.duplicated(keep=False)].unique().tolist()
        checks.append({
            "check": "Duplicate identifiers", "value": f"{len(duplicated_names)} identifiers",
            "level": "warning" if duplicated_names else "ok",
            "message": (f"Repeated in column '{name_column}': {short_list(duplicated_names, 5)}" if duplicated_names else ""),
        })
    else:
        checks.append({"check": "Identifier column", "value": f"'{name_column}' not found", "level": "warning",
                       "message": "Generic names (Node_1, Node_2, …) will be used."})

    checks += [
        {"check": "Upload size", "value": f"{size_bytes / 1024 ** 2:.2f} MB", "level": "ok", "message": ""},
        {"check": "Estimated result size", "value": f"{estimated_output_mb:.1f} MB",
         "level": "warning" if estimated_output_mb > LARGE_RESPONSE_MB else "ok",
         "message": "Large results can make the browser slow." if estimated_output_mb > LARGE_RESPONSE_MB else ""},
    ]

    summary = {"n_rows": int(n_rows), "n_rows_used": n_used, "n_features": int(n_cols_used),
               "n_features_in": int(n_features), "size_bytes": int(size_bytes),
               "estimated_output_mb": estimated_output_mb, "preprocessing": prep_summary}
    return summary, checks


@app.route('/api/sessions/<session_id>/preflight', methods=['POST'])
def preflight(session_id):
    """Inspects the dataset and the requested configuration before a run (no clustering is performed)."""
    try:
        session_dir = get_session_dir(session_id)
        name_column = request.form.get('nameColumn')
        intensity_start_index = int(request.form.get('intensityStartIndex', 1))
        sheet_name = request.form.get('sheetName', '').strip()
        reduction_method = request.form.get('reductionMethod', 'None')
        reduction_params = parse_param_json(request.form.get('reductionParams', '{}'), "reduction")
        clustering_algo = request.form.get('clusteringAlgorithm')
        clustering_params = parse_param_json(request.form.get('clusteringParams', '{}'), "clustering")
        preprocessing_options = parse_preprocessing(request.form.get('preprocessing', '{}'))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input parameter: {e}"}), 400

    dataset_meta = load_dataset_meta(session_dir, request.form.get('datasetId', '').strip())
    if dataset_meta is None:
        return jsonify({"error": "Dataset not found for this session. Please upload the file again.",
                        "code": "dataset_not_found"}), 404

    data_path = os.path.join(dataset_paths(session_dir, dataset_meta["dataset_id"])[0], dataset_meta["data_file"])
    try:
        df = read_table(data_path, dataset_meta["file_type"],
                        sheet_name if dataset_meta["file_type"] == 'xlsx' else None)
        summary, checks = dataset_checks(df, intensity_start_index, name_column, dataset_meta["size_bytes"],
                                         preprocessing_options)
    except ValueError as e:
        return jsonify({"error": f"Data processing error: {e}"}), 400

    n = summary["n_rows_used"]
    if reduction_method == 'None':
        n_dims = summary["n_features"]
    else:
        defaults = {p["name"]: p["default"] for p in REDUCTION_PARAMS.get(reduction_method, [])}
        n_dims = int(reduction_params.get("n_components") or defaults.get("n_components", 2))
    selected = algorithm_guidance(clustering_algo, clustering_params, n, n_dims)
    items = [{"component": "Clustering", "name": clustering_algo, "level": selected["level"],
              "message": selected["note"]}]

    if reduction_method in REDUCTION_CAPACITY:
        limits = REDUCTION_CAPACITY[reduction_method]
        items.append({"component": "Dimensionality reduction", "name": reduction_method,
                      "level": capacity_level(n, limits), "message": limits[2]})
    elif summary["n_features"] > 50:
        items.append({"component": "Dimensionality reduction", "name": "None", "level": "may_be_slow",
                      "message": f"Clustering directly on {summary['n_features']} features; distance-based methods "
                                 f"are slower and less reliable in high dimensions. Consider PCA or UMAP."})

    items.append({"component": "Validation metrics", "name": "Silhouette score",
                  "level": capacity_level(n, METRIC_CAPACITY), "message": METRIC_CAPACITY[2]})
    if summary["estimated_output_mb"] > LARGE_RESPONSE_MB:
        items.append({"component": "Result transfer", "name": f"{summary['estimated_output_mb']:.0f} MB",
                      "level": "may_be_slow", "message": "The interactive result is large and may load slowly."})

    overall = max((item["level"] for item in items), key=LEVEL_ORDER.index)
    guidance = [
        algorithm_guidance(name, {p["name"]: p["default"] for p in params}, n, n_dims)
        for name, params in ALGORITHM_PARAMS.items()
    ]

    return jsonify({
        "dataset": {"filename": dataset_meta["filename"], **summary},
        "effective_dimensions": n_dims,
        "checks": checks,
        "has_data_errors": any(check["level"] == "error" for check in checks),
        "compatibility": {"overall": overall, "items": items},
        "algorithm_guidance": guidance,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
