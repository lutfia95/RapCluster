"""Extended stability analysis.

The run's labels are the reference. Each repeated solution (a re-clustering of a random subsample,
or of all rows with a different random seed) is compared with the reference:
  * cluster stability: Jaccard overlap of each reference cluster with its best-matching cluster
    (Hennig's clusterboot criterion),
  * row stability: how often a row keeps its cluster after one-to-one matching of the clusters,
  * consensus: the most frequent matched label of each row over the reference and all solutions,
  * agreement between solutions: pairwise ARI/NMI on the rows they share.
Noise (-1) is not a cluster: a noise row is "kept" when it stays noise.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

UNMATCHED = -2          # solution cluster without a reference partner
STABLE_JACCARD = 0.75   # commonly read as a stable cluster
DOUBTFUL_JACCARD = 0.6  # below this a cluster is commonly read as dissolved
STABLE_ROW = 0.8
MAX_PAIRWISE = 1000
N_LEAST_STABLE = 50


def jaccard_matrix(reference, solution):
    """Jaccard overlap between the real clusters of two labelings of the same rows."""
    ref_clusters = np.unique(reference[reference != -1])
    sol_clusters = np.unique(solution[solution != -1])
    if ref_clusters.size == 0 or sol_clusters.size == 0:
        return np.zeros((ref_clusters.size, sol_clusters.size)), ref_clusters, sol_clusters
    ref_index = np.searchsorted(ref_clusters, reference)
    sol_index = np.searchsorted(sol_clusters, solution)
    both = (reference != -1) & (solution != -1)
    counts = np.zeros((ref_clusters.size, sol_clusters.size))
    np.add.at(counts, (ref_index[both], sol_index[both]), 1)
    ref_sizes = np.bincount(ref_index[reference != -1], minlength=ref_clusters.size)
    sol_sizes = np.bincount(sol_index[solution != -1], minlength=sol_clusters.size)
    union = ref_sizes[:, None] + sol_sizes[None, :] - counts
    return np.divide(counts, union, out=np.zeros_like(counts), where=union > 0), ref_clusters, sol_clusters


def match_to_reference(reference, solution):
    """Solution labels renamed to the reference clusters they overlap most (one-to-one)."""
    jaccard, ref_clusters, sol_clusters = jaccard_matrix(reference, solution)
    mapping = {}
    if jaccard.size:
        rows, cols = linear_sum_assignment(-jaccard)
        mapping = {int(sol_clusters[c]): int(ref_clusters[r]) for r, c in zip(rows, cols) if jaccard[r, c] > 0}
    lookup = np.vectorize(lambda label: -1 if label == -1 else mapping.get(int(label), UNMATCHED), otypes=[int])
    return (lookup(solution) if solution.size else solution.astype(int)), jaccard, ref_clusters


def pairwise_agreement(solutions, rng):
    """ARI/NMI between pairs of solutions on the rows both contain (at most MAX_PAIRWISE pairs)."""
    pairs = [(a, b) for a in range(len(solutions)) for b in range(a + 1, len(solutions))]
    if len(pairs) > MAX_PAIRWISE:
        pairs = [pairs[i] for i in rng.choice(len(pairs), MAX_PAIRWISE, replace=False)]
    ari, nmi = [], []
    for a, b in pairs:
        _, pos_a, pos_b = np.intersect1d(solutions[a]["idx"], solutions[b]["idx"], assume_unique=True, return_indices=True)
        if pos_a.size < 2:
            continue
        la, lb = solutions[a]["labels"][pos_a], solutions[b]["labels"][pos_b]
        ari.append(float(adjusted_rand_score(la, lb)))
        nmi.append(float(normalized_mutual_info_score(la, lb)))
    return ari, nmi


def verdict(jaccard):
    if jaccard >= STABLE_JACCARD:
        return "stable"
    return "doubtful" if jaccard >= DOUBTFUL_JACCARD else "unstable"


def analyze(reference, solutions, names):
    """reference: labels of all n rows; solutions: [{"kind", "idx", "labels"}] with idx into the rows."""
    reference = np.asarray(reference)
    n = reference.size
    rng = np.random.default_rng(0)
    ref_clusters = sorted(set(reference.tolist()), key=lambda c: (c == -1, c))
    columns = ref_clusters + ([-1] if -1 not in ref_clusters else []) + [UNMATCHED]
    column_of = {label: i for i, label in enumerate(columns)}

    kept = np.zeros(n)
    seen = np.zeros(n)
    votes = np.zeros((n, len(columns)), dtype=np.int32)
    votes[np.arange(n), [column_of[label] for label in reference]] += 1
    jaccards = {c: [] for c in ref_clusters if c != -1}
    ari_reference = []

    for solution in solutions:
        idx, labels = solution["idx"], np.asarray(solution["labels"])
        ref_sub = reference[idx]
        mapped, jaccard, present = match_to_reference(ref_sub, labels)
        for i, cluster in enumerate(present):
            jaccards[int(cluster)].append(float(jaccard[i].max()) if jaccard.shape[1] else 0.0)
        seen[idx] += 1
        kept[idx] += mapped == ref_sub
        votes[idx, [column_of[label] for label in mapped]] += 1
        ari_reference.append({"kind": solution["kind"], "ari": float(adjusted_rand_score(ref_sub, labels)),
                              "nmi": float(normalized_mutual_info_score(ref_sub, labels))})

    with np.errstate(invalid="ignore", divide="ignore"):
        row_stability = np.where(seen > 0, kept / seen, np.nan)
    consensus = np.array(columns)[votes.argmax(axis=1)]
    agreement = votes.max(axis=1) / votes.sum(axis=1)
    pairwise_ari, pairwise_nmi = pairwise_agreement(solutions, rng)

    sizes = {c: int(np.sum(reference == c)) for c in ref_clusters}
    clusters = []
    for cluster in ref_clusters:
        values = jaccards.get(cluster)
        members = reference == cluster
        entry = {"cluster": int(cluster), "size": sizes[cluster],
                 "mean_row_stability": float(np.nanmean(row_stability[members])) if members.any() else None}
        if values:
            mean = float(np.mean(values))
            entry.update({"mean_jaccard": mean, "std_jaccard": float(np.std(values)), "min_jaccard": float(np.min(values)),
                          "n_solutions": len(values), "verdict": verdict(mean)})
        clusters.append(entry)

    finite = row_stability[np.isfinite(row_stability)]
    counts, edges = np.histogram(finite, bins=20, range=(0, 1))
    order = np.argsort(np.where(np.isfinite(row_stability), row_stability, 2.0), kind="stable")[:N_LEAST_STABLE]
    real = [c for c in clusters if "mean_jaccard" in c]
    aris = [item["ari"] for item in ari_reference]
    summary = {
        "mean_ari": float(np.mean(aris)) if aris else None,
        "std_ari": float(np.std(aris)) if aris else None,
        "mean_row_stability": float(np.mean(finite)) if finite.size else None,
        "median_row_stability": float(np.median(finite)) if finite.size else None,
        "share_stable_rows": float(np.mean(finite >= STABLE_ROW)) if finite.size else None,
        "n_stable_clusters": sum(c["verdict"] == "stable" for c in real),
        "n_doubtful_clusters": sum(c["verdict"] == "doubtful" for c in real),
        "n_unstable_clusters": sum(c["verdict"] == "unstable" for c in real),
        "mean_cluster_jaccard": float(np.mean([c["mean_jaccard"] for c in real])) if real else None,
        "pairwise_ari_mean": float(np.mean(pairwise_ari)) if pairwise_ari else None,
        "consensus_changed": int(np.sum(consensus != reference)),
        "consensus_unassigned": int(np.sum(consensus == UNMATCHED)),
    }
    details = {
        "clusters": clusters,
        "thresholds": {"stable_jaccard": STABLE_JACCARD, "doubtful_jaccard": DOUBTFUL_JACCARD, "stable_row": STABLE_ROW},
        "distributions": {"vs_reference": ari_reference, "pairwise_ari": pairwise_ari, "pairwise_nmi": pairwise_nmi},
        "rows": {
            "histogram": {"counts": counts.tolist(), "edges": np.round(edges, 3).tolist()},
            "least_stable": [{"name": names[i], "cluster": int(reference[i]), "stability": float(row_stability[i]),
                              "times_included": int(seen[i]), "consensus": int(consensus[i]),
                              "agreement": round(float(agreement[i]), 3)}
                             for i in order if np.isfinite(row_stability[i])],
        },
    }
    return summary, details, {"row_stability": row_stability, "consensus": consensus, "agreement": agreement}
