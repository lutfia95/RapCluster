#!/usr/bin/env python3
"""Reproduces a RapCluster run outside the web app.

Usage:
    python reproduce.py path/to/your_data_file [config.json]

Requires: numpy pandas scikit-learn umap-learn hdbscan openpyxl, and preprocessing.py next to this script.
The script mirrors RapCluster's pipeline (preprocessing -> dimensionality reduction ->
clustering -> evaluation) using the settings stored in config.json, writes
reproduced_assignments.tsv and prints the validation metrics.
"""
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn import cluster, mixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess  # noqa: E402  (shipped next to this script)


def read_table(path, sheet_name):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".csv":
        return pd.read_csv(path)
    if extension == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_excel(path, sheet_name=sheet_name if sheet_name else 0)


def tsne_kwargs(params):
    """RapCluster stores the iteration count as n_iter; scikit-learn renamed it to max_iter (1.5) and removed
    n_iter (1.7). Passes it under the name the installed version accepts."""
    params = dict(params)
    if "n_iter" in params and "max_iter" in inspect.signature(TSNE.__init__).parameters:
        params["max_iter"] = params.pop("n_iter")
    return params


def reduce(X, method, params, all_dimensions=True):
    """Clustering uses every reduced dimension (or all features without reduction). Configs without
    "clustering_dims" describe runs that used only the first two dimensions."""
    if method == "UMAP":
        import umap
        X = umap.UMAP(**params).fit_transform(X)
    elif method == "TSNE":
        X = TSNE(**tsne_kwargs(params)).fit_transform(X)
    elif method == "PCA":
        X = PCA(**params).fit_transform(X)
    return X if all_dimensions else X[:, :2]


def run_clustering(X, algorithm, params):
    params = dict(params)
    if algorithm == "AgglomerativeClustering":
        connectivity = kneighbors_graph(X, n_neighbors=min(X.shape[0] - 1, 10), include_self=False)
        connectivity = 0.5 * (connectivity + connectivity.T)
        if params.get("linkage") == "ward":
            params["metric"] = "euclidean"
        model = cluster.AgglomerativeClustering(connectivity=connectivity, **params)
    elif algorithm == "AffinityPropagation":
        params.pop("random_state", None)
        model = cluster.AffinityPropagation(**params)
    elif algorithm == "MeanShift":
        model = cluster.MeanShift(**{k: v for k, v in params.items() if v is not None and k != "n_jobs"})
    elif algorithm == "HDBSCAN":
        from hdbscan import HDBSCAN
        model = HDBSCAN(**params)
    elif algorithm == "BIRCH":
        model = cluster.Birch(**params)
    elif algorithm == "GaussianMixture":
        model = mixture.GaussianMixture(**params)
    elif algorithm == "BayesianGaussianMixture":
        # RapCluster numbers the used components 0..k-1 in component order.
        labels = mixture.BayesianGaussianMixture(**params).fit_predict(X)
        return np.unique(labels, return_inverse=True)[1].reshape(labels.shape)
    else:
        model = getattr(cluster, algorithm)(**params)  # e.g. KMeans, BisectingKMeans, DBSCAN
    return model.fit_predict(X)


def evaluate(X, labels):
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return {}
    X, labels = X[mask], labels[mask]
    return {
        "silhouette_score": silhouette_score(X, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X, labels),
        "davies_bouldin_score": davies_bouldin_score(X, labels),
        "n_clusters": len(set(labels)),
    }


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path) as fh:
        config = json.load(fh)

    df = read_table(data_path, config["preprocessing"].get("sheet_name"))
    prep = config["preprocessing"]
    # Without "options" in config.json the default preprocessing is used.
    result = preprocess(df, prep["name_column"], prep["intensity_start_index"], prep.get("options"))
    X, names = result["data"], result["names"]
    X = reduce(X, config["reduction"]["method"], config["reduction"]["params"],
               all_dimensions="clustering_dims" in config["reduction"])
    labels = run_clustering(X, config["clustering"]["algorithm"], config["clustering"]["params"])

    out_path = "reproduced_assignments.tsv"
    # X/Y: the first two dimensions of the clustering space (for data without reduction RapCluster's plots use a
    # PCA projection instead, so these coordinates can differ from cluster_assignments.tsv; the clusters do not).
    second = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
    pd.DataFrame({"Name": names, "Cluster": labels, "X": X[:, 0], "Y": second}).to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")
    for name, value in evaluate(X, labels).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
