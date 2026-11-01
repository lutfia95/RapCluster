"""Generates the bundled example datasets (deterministic; re-run to recreate the TSV files).

All values are positive intensities because RapCluster applies log2(x + 1) before clustering.
The "Class" column holds the known groups and is not used for clustering (data start index 2).
"""
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_blobs

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(2024)


def save(df, filename):
    df.to_csv(os.path.join(OUT_DIR, filename), sep="\t", index=False, float_format="%.6f")
    print(f"{filename}: {df.shape[0]} rows x {df.shape[1]} columns")


def gaussian_blobs():
    X, y = make_blobs(n_samples=600, centers=5, n_features=10, cluster_std=1.0, random_state=7)
    X = X - X.min() + 1.0
    df = pd.DataFrame(X, columns=[f"F{i + 1}" for i in range(X.shape[1])])
    df.insert(0, "Class", [f"Blob{label + 1}" for label in y])
    df.insert(0, "Name", [f"S{i + 1}" for i in range(len(df))])
    save(df, "gaussian_blobs.tsv")


def density_with_noise():
    sizes, spreads = [300, 200, 120], [0.4, 1.0, 2.0]
    centers = RNG.uniform(-8, 8, size=(3, 8))
    parts, labels = [], []
    for k, (size, spread) in enumerate(zip(sizes, spreads)):
        parts.append(centers[k] + RNG.normal(scale=spread, size=(size, 8)))
        labels += [f"Cluster{k + 1}"] * size
    n_noise = 80
    parts.append(RNG.uniform(-12, 12, size=(n_noise, 8)))
    labels += ["Noise"] * n_noise
    X = np.vstack(parts)
    X = X - X.min() + 1.0
    order = RNG.permutation(len(X))
    df = pd.DataFrame(X[order], columns=[f"F{i + 1}" for i in range(8)])
    df.insert(0, "Class", np.array(labels)[order])
    df.insert(0, "Name", [f"P{i + 1}" for i in range(len(df))])
    save(df, "density_with_noise.tsv")


def time_course_profiles():
    t = np.linspace(0, 1, 24)
    patterns = {
        "Rising": 1 + 9 * t,
        "Falling": 10 - 9 * t,
        "EarlyPeak": 1 + 9 * np.exp(-((t - 0.2) ** 2) / 0.01),
        "LatePeak": 1 + 9 * np.exp(-((t - 0.8) ** 2) / 0.01),
        "Oscillating": 5.5 + 4.5 * np.sin(2 * np.pi * 2 * t),
        "Transient dip": 10 - 8 * np.exp(-((t - 0.5) ** 2) / 0.02),
    }
    rows, labels = [], []
    for name, pattern in patterns.items():
        for _ in range(200):
            scale = RNG.lognormal(mean=3, sigma=1)  # genes differ in absolute abundance
            rows.append(scale * pattern * RNG.lognormal(sigma=0.15, size=t.size))
            labels.append(name)
    X = np.array(rows)
    order = RNG.permutation(len(X))
    df = pd.DataFrame(X[order], columns=[f"T{i + 1:02d}" for i in range(t.size)])
    df.insert(0, "Class", np.array(labels)[order])
    df.insert(0, "Name", [f"Gene{i + 1}" for i in range(len(df))])
    save(df, "time_course_profiles.tsv")


def iris():
    data = load_iris(as_frame=True)
    df = data.data.copy()
    df.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
    df.insert(0, "Class", [data.target_names[i] for i in data.target])
    df.insert(0, "Name", [f"Flower{i + 1}" for i in range(len(df))])
    save(df, "iris.tsv")


if __name__ == "__main__":
    gaussian_blobs()
    density_with_noise()
    time_course_profiles()
    iris()
