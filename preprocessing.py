"""RapCluster preprocessing, shared by the web app and the standalone reproduce.py.

Steps, in order:
  1. read the data columns as numbers (comma decimals accepted, text becomes missing)
  2. remove rows with too many missing values
  3. impute the remaining missing values
  4. transform (log, signed log or none)
  5. remove all-zero rows
  6. remove low-variance and highly correlated columns
  7. scale rows or columns

The defaults reproduce RapCluster's original fixed pipeline: missing -> 0, log2(x + 1),
all-zero rows removed, per-row z-score.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

DEFAULT_OPTIONS = {
    "max_missing_fraction": 1.0,
    "imputation": "zero",
    "transform": "log2p1",
    "remove_zero_rows": True,
    "remove_low_variance": False,
    "variance_threshold": 0.0,
    "correlation_filter": False,
    "correlation_threshold": 0.95,
    "scaling": "row_zscore",
}

IMPUTATION = {
    "zero": "replaced with 0",
    "min": "replaced with the column minimum",
    "half_min": "replaced with half of the column minimum",
    "median": "replaced with the column median",
}
TRANSFORM = {
    "log2p1": "log2(x + 1)",
    "log10p1": "log10(x + 1)",
    "signed_log2": "sign(x) · log2(|x| + 1)",
    "none": "none",
}
SCALING = {
    "row_zscore": "per-row z-score",
    "row_minmax": "per-row min–max scaling to [0, 1]",
    "column_zscore": "per-column z-score",
    "column_robust": "per-column robust scaling (median and IQR)",
    "column_minmax": "per-column min–max scaling to [0, 1]",
    "none": "none",
}
MAX_CORRELATION_COLUMNS = 5000


def validate_options(raw):
    """Options merged with the defaults; raises ValueError for unknown keys or invalid values."""
    raw = dict(raw or {})
    unknown = set(raw) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown preprocessing option(s): {', '.join(sorted(unknown))}")
    options = {**DEFAULT_OPTIONS, **raw}
    for key, choices in (("imputation", IMPUTATION), ("transform", TRANSFORM), ("scaling", SCALING)):
        if options[key] not in choices:
            raise ValueError(f"Invalid {key}: {options[key]!r}. Choose one of: {', '.join(choices)}")
    for key in ("remove_zero_rows", "remove_low_variance", "correlation_filter"):
        if not isinstance(options[key], bool):
            raise ValueError(f"{key} must be true or false")
    for key, low, high in (("max_missing_fraction", 0.0, 1.0), ("variance_threshold", 0.0, float("inf")),
                           ("correlation_threshold", 0.5, 1.0)):
        try:
            options[key] = float(options[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        if not low <= options[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
    return options


def describe(options):
    """Human-readable description of each step (stored in the run record and used in the report)."""
    filters = []
    if options["max_missing_fraction"] < 1.0:
        filters.append(f"rows with more than {options['max_missing_fraction']:.0%} missing values removed")
    if options["remove_low_variance"]:
        filters.append("constant columns removed" if options["variance_threshold"] == 0
                       else f"columns with standard deviation ≤ {options['variance_threshold']:g} removed")
    if options["correlation_filter"]:
        filters.append(f"columns correlated above |r| = {options['correlation_threshold']:g} with an earlier column removed")
    return {
        "missing_values": IMPUTATION[options["imputation"]],
        "transform": TRANSFORM[options["transform"]],
        "zero_rows": "removed" if options["remove_zero_rows"] else "kept",
        "filters": "; ".join(filters) or "none",
        "scaling": SCALING[options["scaling"]],
    }


def value_label(options):
    """Short label for the values that are clustered and plotted, e.g. 'log2(x + 1), per-row z-score'."""
    parts = [TRANSFORM[options["transform"]]] if options["transform"] != "none" else []
    if options["scaling"] != "none":
        parts.append(SCALING[options["scaling"]])
    return ", ".join(parts) or "values as uploaded"


def numeric_matrix(df, intensity_start_index):
    """Data columns as floats (comma decimals accepted); unparsable cells become NaN."""
    intensity_cols = df.columns[intensity_start_index:]
    if len(intensity_cols) == 0:
        raise ValueError("No data columns found after the data start column index.")
    as_text = df[intensity_cols].astype(str).apply(lambda col: col.str.replace(',', '.', regex=False))
    return intensity_cols, as_text.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)


def impute(raw, method):
    """Fills NaN per column; returns the filled matrix and the columns that had no observed value."""
    missing = np.isnan(raw)
    filled = raw.copy()
    if not missing.any():
        return filled, np.zeros(raw.shape[1], dtype=bool)
    all_missing = missing.all(axis=0)
    if method == "zero":
        fill = np.zeros(raw.shape[1])
    else:
        observed = np.where(missing, np.nan, raw)
        with np.errstate(all="ignore"):
            if method == "median":
                fill = np.array([np.median(col[~np.isnan(col)]) if not empty else 0.0
                                 for col, empty in zip(observed.T, all_missing)])
            else:
                fill = np.array([np.min(col[~np.isnan(col)]) if not empty else 0.0
                                 for col, empty in zip(observed.T, all_missing)])
                if method == "half_min":
                    fill = fill / 2.0
    filled[missing] = fill[np.nonzero(missing)[1]]
    return filled, all_missing


def transform(values, method):
    if method == "none":
        return values.copy()
    if method == "signed_log2":
        return np.sign(values) * np.log2(np.abs(values) + 1.0)
    if np.any(values <= -1):
        raise ValueError(f"{TRANSFORM[method]} is undefined for values ≤ −1. Choose the signed log or no "
                         "transformation for data with negative values.")
    return np.log2(values + 1.0) if method == "log2p1" else np.log10(values + 1.0)


def correlated_columns(X, threshold):
    """Greedy filter: a column is removed when |r| > threshold with an earlier kept column."""
    if X.shape[1] > MAX_CORRELATION_COLUMNS:
        raise ValueError(f"The correlation filter supports at most {MAX_CORRELATION_COLUMNS} columns.")
    with np.errstate(all="ignore"):
        corr = np.abs(np.corrcoef(X, rowvar=False))
    corr = np.nan_to_num(np.atleast_2d(corr))
    kept, removed = [], {}
    for j in range(X.shape[1]):
        partners = [k for k in kept if corr[j, k] > threshold]
        if partners:
            removed[j] = partners[0]
        else:
            kept.append(j)
    return removed


def scale(X, method):
    if method == "row_zscore":
        return StandardScaler().fit_transform(X.T).T
    if method == "row_minmax":
        low = X.min(axis=1, keepdims=True)
        span = X.max(axis=1, keepdims=True) - low
        span[span == 0] = 1.0
        return (X - low) / span
    if method == "column_zscore":
        return StandardScaler().fit_transform(X)
    if method == "column_robust":
        return RobustScaler().fit_transform(X)
    if method == "column_minmax":
        return MinMaxScaler().fit_transform(X)
    return X.copy()


def preprocess(df, name_column, intensity_start_index, options):
    """Runs all steps. Returns the clustered matrix plus everything needed for plots, reports and checks."""
    options = validate_options(options)
    intensity_cols, raw = numeric_matrix(df, intensity_start_index)
    columns = np.array(intensity_cols.astype(str))
    n_rows, n_cols = raw.shape
    missing = np.isnan(raw)

    rows = np.flatnonzero(missing.mean(axis=1) <= options["max_missing_fraction"])
    filled, all_missing = impute(raw[rows], options["imputation"])
    cells_imputed = int(missing[rows].sum())
    transformed = transform(filled, options["transform"])

    rows_removed_zero = 0
    if options["remove_zero_rows"]:
        nonzero = ~np.all(filled == 0.0, axis=1)
        rows_removed_zero = int((~nonzero).sum())
        rows, filled, transformed = rows[nonzero], filled[nonzero], transformed[nonzero]
    if rows.size < 2:
        raise ValueError("Fewer than two rows remain after preprocessing.")

    keep_cols = np.ones(n_cols, dtype=bool)
    low_variance = []
    if options["remove_low_variance"]:
        std = transformed.std(axis=0)
        low = std <= max(options["variance_threshold"], 1e-12)
        low_variance = columns[low].tolist()
        keep_cols &= ~low
    correlated = []
    if options["correlation_filter"] and keep_cols.sum() > 1:
        candidates = np.flatnonzero(keep_cols)
        for j, partner in correlated_columns(transformed[:, candidates], options["correlation_threshold"]).items():
            keep_cols[candidates[j]] = False
            correlated.append({"column": str(columns[candidates[j]]), "correlated_with": str(columns[candidates[partner]])})
    if not keep_cols.any():
        raise ValueError("All data columns were removed by the column filters.")

    # C order keeps the scaler's summation order, so the default settings give bit-identical results.
    transformed = np.ascontiguousarray(transformed[:, keep_cols])
    data = scale(transformed, options["scaling"])

    name_found = name_column in df.columns
    if name_found:
        names = df[name_column].astype(str).to_numpy()[rows].tolist()
    else:
        names = [f"Node_{i + 1}" for i in range(rows.size)]
    kept_filled = filled[:, keep_cols]

    summary = {
        "rows_in": int(n_rows),
        "rows_removed_missing": int(n_rows - (missing.mean(axis=1) <= options["max_missing_fraction"]).sum()),
        "rows_removed_zero": rows_removed_zero,
        "rows_used": int(rows.size),
        "columns_in": int(n_cols),
        "columns_removed_low_variance": low_variance,
        "columns_removed_correlated": correlated,
        "columns_used": int(keep_cols.sum()),
        "missing_cells": int(missing.sum()),
        "cells_imputed": cells_imputed,
        "columns_all_missing": columns[all_missing].tolist(),
    }
    return {
        "data": data,
        "transformed": transformed,
        # signed log10 of the imputed values, for raw-intensity line plots
        "log10": np.sign(kept_filled) * np.log10(np.abs(kept_filled) + 1.0),
        "names": names,
        "columns": columns[keep_cols].tolist(),
        "row_numbers": (rows + 1).tolist(),
        "name_column_found": name_found,
        "summary": summary,
        "options": options,
    }
