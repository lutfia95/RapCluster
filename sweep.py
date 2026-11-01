"""Parameter sweep planning: validates the swept parameters and chooses the trials.

One or two parameters of one algorithm are swept over explicit value lists. All combinations are tried
(grid search) when they fit into the trial budget; otherwise a fixed-seed random subset of the grid is used.
"""
import itertools

import numpy as np

MAX_SWEEP_PARAMETERS = 2
MAX_VALUES_PER_PARAMETER = 50
MAX_TRIALS = 30
SCREENING_METRICS = {  # name in run metrics, direction
    "silhouette": ("silhouette_score", "max"),
    "calinski_harabasz": ("calinski_harabasz_score", "max"),
    "davies_bouldin": ("davies_bouldin_score", "min"),
    "dbcv": ("dbcv", "max"),
    "bic": ("bic", "min"),
}


def coerce_value(definition, value):
    """A value of the parameter's type (raises ValueError), within the parameter's documented range."""
    name, kind = definition["name"], definition["type"]
    if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
        if definition.get("nullable"):
            return None
        raise ValueError(f"'{name}' cannot be None.")
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "false"):
            return str(value).lower() == "true"
        raise ValueError(f"'{name}' must be true or false, not {value!r}.")
    if kind == "select":
        if value not in definition["options"]:
            raise ValueError(f"'{name}' must be one of {', '.join(map(str, definition['options']))}, not {value!r}.")
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number, not {value!r}.")
    if not np.isfinite(number):
        raise ValueError(f"'{name}' must be a finite number.")
    if kind == "int":
        if number != int(number):
            raise ValueError(f"'{name}' must be a whole number, not {value!r}.")
        number = int(number)
    low, high = definition.get("min"), definition.get("max")
    if (low is not None and number < low) or (high is not None and number > high):
        raise ValueError(f"'{name}' must be between {low} and {high}; {value!r} is outside.")
    return number


def plan_trials(algorithm, parameters, max_trials, definitions):
    """parameters: [{"name", "values"}]; definitions: the algorithm's parameter list.
    Returns (trials, info) where trials are dicts of swept values and info describes the search."""
    if not isinstance(parameters, list) or not 1 <= len(parameters) <= MAX_SWEEP_PARAMETERS:
        raise ValueError(f"Choose one or two parameters to sweep (at most {MAX_SWEEP_PARAMETERS}).")
    by_name = {d["name"]: d for d in definitions}
    names, value_lists = [], []
    for item in parameters:
        name = item.get("name") if isinstance(item, dict) else None
        if name not in by_name:
            raise ValueError(f"{algorithm} has no parameter {name!r}.")
        if name in names:
            raise ValueError(f"Parameter {name!r} is listed twice.")
        if name == "random_state":
            raise ValueError("random_state is not a model parameter; use the stability analysis to check seed dependence.")
        raw = item.get("values")
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Give at least one value for {name!r}.")
        values = []
        for value in raw:
            coerced = coerce_value(by_name[name], value)
            if coerced not in values:
                values.append(coerced)
        if len(values) > MAX_VALUES_PER_PARAMETER:
            raise ValueError(f"At most {MAX_VALUES_PER_PARAMETER} values per parameter ({name!r} has {len(values)}).")
        names.append(name)
        value_lists.append(values)
    try:
        max_trials = int(max_trials)
    except (TypeError, ValueError):
        raise ValueError("The maximum number of trials must be a whole number.")
    if not 1 <= max_trials <= MAX_TRIALS:
        raise ValueError(f"The maximum number of trials must be between 1 and {MAX_TRIALS}.")

    grid = [dict(zip(names, combination)) for combination in itertools.product(*value_lists)]
    if len(grid) <= max_trials:
        trials, search = grid, "grid"
    else:
        chosen = np.sort(np.random.default_rng(0).choice(len(grid), max_trials, replace=False))
        trials, search = [grid[i] for i in chosen], "random"
    info = {"parameters": [{"name": n, "values": v} for n, v in zip(names, value_lists)],
            "grid_size": len(grid), "max_trials": max_trials, "search": search, "random_seed": 0 if search == "random" else None}
    return trials, info


def validate_screening(raw, n_trials):
    """Optional subsample screening: run every trial on a subsample, confirm the best top_k on all rows."""
    if not raw or not raw.get("enabled"):
        return None
    try:
        fraction = float(raw.get("fraction", 0.2))
        top_k = int(raw.get("top_k", 3))
    except (TypeError, ValueError):
        raise ValueError("Screening fraction and number of confirmed trials must be numbers.")
    metric = raw.get("metric", "silhouette")
    if not 0.05 <= fraction <= 0.9:
        raise ValueError("The screening subsample must be between 5% and 90% of the rows.")
    if not 1 <= top_k <= n_trials:
        raise ValueError(f"The number of confirmed trials must be between 1 and {n_trials}.")
    if metric not in SCREENING_METRICS:
        raise ValueError(f"Unknown screening metric {metric!r}.")
    return {"fraction": fraction, "top_k": top_k, "metric": metric, "direction": SCREENING_METRICS[metric][1]}


def pick_confirmed(trials, screening):
    """Indices of the top_k screened trials by the screening metric (trials without the metric come last)."""
    key, direction = SCREENING_METRICS[screening["metric"]]
    sign = -1 if direction == "max" else 1

    def score(index):
        value = (trials[index].get("screening_metrics") or {}).get(key)
        return (0, sign * value) if value is not None and np.isfinite(value) else (1, 0)

    candidates = [i for i, trial in enumerate(trials) if trial["state"] == "screened"]
    return sorted(sorted(candidates, key=score)[:screening["top_k"]])
