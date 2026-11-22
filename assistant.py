"""Clustering Assistant: a local language model (Qwen2.5 7B Instruct via Ollama) that explains a run's results.

The model never computes anything itself: every number it may talk about is written into its context by
build_context() from what RapCluster already calculated (metrics, cluster sizes, characterization, stability,
preprocessing, settings, warnings and the other runs of the session). stream_chat() forwards the conversation
to the Ollama server and yields the answer piece by piece.
"""
import json
import os
import re
import urllib.error
import urllib.request

ASSISTANT_URL = os.environ.get("RAPCLUSTER_ASSISTANT_URL", "http://127.0.0.1:11434").rstrip("/")
ASSISTANT_MODEL = os.environ.get("RAPCLUSTER_ASSISTANT_MODEL", "qwen2.5:7b")
# Qwen2.5 7B reads up to 32K tokens; 8K covers the run context plus a conversation and keeps CPU memory low.
CONTEXT_TOKENS = int(os.environ.get("RAPCLUSTER_ASSISTANT_CONTEXT", 8192))
MAX_ANSWER_TOKENS = int(os.environ.get("RAPCLUSTER_ASSISTANT_MAX_TOKENS", 700))
MAX_HISTORY_MESSAGES = 8         # earlier turns are dropped so the context stays within CONTEXT_TOKENS
MAX_MENTIONED_RUNS = 2           # other runs named in the questions whose full results are added
# Repetition guard: small models can get stuck repeating their last paragraph. When the last REPEAT_WINDOW
# characters already occurred REPEAT_LIMIT times in the answer, the answer is cut after the first occurrence.
REPEAT_WINDOW = 80
REPEAT_LIMIT = 3
REPEAT_NOTE = "\n\n(The answer started to repeat itself and was stopped here. Ask again or rephrase the question.)"
MAX_QUESTION_CHARS = 2000
MAX_DETAIL_CLUSTERS = 25         # clusters described one by one; the rest are summarized
MAX_FEATURES_PER_CLUSTER = 3

SYSTEM_PROMPT = """You are the RapCluster Clustering Assistant. RapCluster is a web tool that clusters tabular data \
(rows = items such as proteins or genes, columns = measurements). You explain ONE clustering run to the user.

Rules:
- Use only the facts in the RUN RESULTS below. Never invent numbers, clusters, features or row names. If something \
is not in the results, say that RapCluster did not compute it here and, if useful, where in RapCluster it can be computed.
- Explain plainly for a biologist, not a statistician. Be concise: short paragraphs or bullet points, usually under \
200 words unless the user asks for detail.
- Interpret metrics with care: they are guides, not proof. Silhouette ranges -1..1 (above ~0.5 well separated, \
0.25-0.5 moderate, below ~0.25 weak or overlapping). Davies-Bouldin: lower is better, below ~1 is good. \
Calinski-Harabasz: higher is better, only comparable between runs on the same data. DBCV (-1..1, higher is better) \
suits density-based clusters and noise. Cluster -1 is noise (rows not assigned to any cluster). Cliff's delta \
(-1..1) compares a feature inside a cluster with all other rows: |delta| above ~0.47 is a large difference. \
Stability: Jaccard >= 0.75 stable, 0.6-0.75 doubtful, below 0.6 unstable; ARI near 1 means repeated runs agree.
- Never repeat a sentence, paragraph or json block you have already written in this answer. Say each thing once, \
then stop.
- You explain the run named at the top of RUN RESULTS. For other runs you only have the short summary lines, \
unless their full results are added below as FULL RESULTS OF RUN #N; if the user asks for details of another run \
you do not have, say so and suggest mentioning it by number (e.g. "run 5") or opening its Clustering Assistant.
- When you suggest different settings, explain why, and add ONE fenced ```json block with only the settings that \
change (at least one changed value, never empty "params"), in this layout (all keys optional):
{"preprocessing": {"options": {"scaling": "row_zscore"}}, "reduction": {"method": "UMAP", "params": {"n_neighbors": 30}}, \
"clustering": {"algorithm": "HDBSCAN", "params": {"min_cluster_size": 20}}}
Use only algorithm, method, parameter and option names that appear in the results or in AVAILABLE SETTINGS. \
The user can apply that block to the form with one click.
"""


class AssistantUnavailable(Exception):
    """The model server cannot be reached or does not have the model."""


def _fmt(value, digits=3):
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _q(value):
    """q-values: very small ones in scientific notation instead of 0.0000."""
    if value is None:
        return "not available"
    return f"{value:.2e}" if value < 0.001 else f"{value:.4f}"


def _cluster_label(cluster):
    return "Noise (-1)" if int(cluster) == -1 else f"Cluster {cluster}"


def _params(params):
    return ", ".join(f"{k}={'None' if v is None else v}" for k, v in (params or {}).items()) or "defaults"


def _metric_lines(record):
    metrics = record.get("metrics") or {}
    lines = [
        f"- Clusters found: {record.get('n_clusters')}",
        f"- Noise rows (cluster -1): {record.get('n_noise')}",
        f"- Silhouette (higher is better, -1..1): {_fmt(metrics.get('silhouette_score'))}",
        f"- Davies-Bouldin (lower is better): {_fmt(metrics.get('davies_bouldin_score'))}",
        f"- Calinski-Harabasz (higher is better): {_fmt(metrics.get('calinski_harabasz_score'), 1)}",
        f"- DBCV (higher is better, -1..1): {_fmt(metrics.get('dbcv'))}",
    ]
    for key, label in (("bic", "BIC (lower is better)"), ("aic", "AIC (lower is better)"),
                       ("mean_membership", "Mean membership probability")):
        if metrics.get(key) is not None:
            lines.append(f"- {label}: {_fmt(metrics[key], 3 if key == 'mean_membership' else 1)}")
    return lines


def _cluster_lines(characterization, stability_details, labels_sizes, max_clusters=MAX_DETAIL_CLUSTERS):
    stability_by_cluster = {c["cluster"]: c for c in (stability_details or {}).get("clusters", [])}
    if characterization:
        entries = characterization["clusters"]
    else:
        total = sum(labels_sizes.values()) or 1
        entries = [{"cluster": c, "size": n, "fraction": n / total} for c, n in labels_sizes.items()]
    lines = []
    for entry in entries[:max_clusters]:
        parts = [f"{entry['size']} rows ({100 * entry['fraction']:.1f}%)"]
        features = [f"{f['feature']} (delta {f['cliffs_delta']:+.2f}, median {f['median_in']} vs {f['median_rest']} in other rows)"
                    for f in entry.get("top_features", [])[:MAX_FEATURES_PER_CLUSTER] if f.get("cliffs_delta") is not None]
        if features:
            parts.append("most distinguishing features: " + "; ".join(features))
        if entry.get("medoid"):
            parts.append(f"most central member: {entry['medoid']}")
        if entry.get("representatives"):
            parts.append("representative members: " + ", ".join(entry["representatives"][:3]))
        for key, label in (("mean_membership", "mean membership"), ("persistence", "persistence"),
                           ("component_weight", "component weight")):
            if entry.get(key) is not None:
                parts.append(f"{label} {entry[key]:.3f}")
        stab = stability_by_cluster.get(entry["cluster"])
        if stab and stab.get("mean_jaccard") is not None:
            parts.append(f"stability Jaccard {stab['mean_jaccard']:.2f} ({stab['verdict']})")
        lines.append(f"- {_cluster_label(entry['cluster'])}: " + "; ".join(parts))
    if len(entries) > max_clusters:
        rest = entries[max_clusters:]
        lines.append(f"- ... and {len(rest)} smaller clusters with {sum(e['size'] for e in rest)} rows in total "
                     f"(listed in cluster_summary.tsv of the ZIP download).")
    return lines


def build_context(record, *, characterization=None, stability_details=None, labels_sizes=None,
                  preprocessing_rows=(), model_rows=(), stability_rows=(), methods_text="", other_runs=(),
                  available_settings=None, display_note=None, title=None, compact=False):
    """The run's results as plain text for the model (kept compact: CPU models read slowly).

    compact=True (other runs the user asks about) leaves out the outliers, the methods text and fewer clusters.
    """
    dataset = record.get("dataset") or {}
    sections = [
        title or f"RUN RESULTS - Run #{record.get('number')} (status: {record.get('status')})",
        "",
        "DATA",
        f"- File: {dataset.get('filename')}; rows x features clustered: {dataset.get('n_observations')} x {dataset.get('n_features')}",
        *[f"- {label}: {value}" for label, value in preprocessing_rows],
        "",
        "SETTINGS",
        f"- Dimensionality reduction: {record['reduction']['method']} ({_params(record['reduction'].get('params'))})",
        f"- Dimensions used for clustering: {record['reduction'].get('clustering_dims', 'not recorded')}",
        f"- Clustering algorithm: {record['clustering']['algorithm']} ({_params(record['clustering'].get('params'))})",
    ]
    if display_note:
        sections.append(f"- Scatter plot shows: {display_note}")
    sections += ["", "QUALITY METRICS", *_metric_lines(record), *[f"- {label}: {value}" for label, value in model_rows]]
    sections += ["", "STABILITY", *[f"- {label}: {value}" for label, value in stability_rows]]
    if not record.get("stability"):
        sections.append("- (The user can compute it with the Stability analysis below the results.)")

    sections += ["", "CLUSTERS (sizes use all rows)",
                 *_cluster_lines(characterization, stability_details, labels_sizes or {}, 12 if compact else MAX_DETAIL_CLUSTERS)]
    if characterization and not compact:
        tests = (characterization.get("feature_tests") or {}).get("rows") or []
        if tests:
            sections += ["", "FEATURES THAT DIFFER MOST BETWEEN CLUSTERS (Kruskal-Wallis, epsilon squared 0..1)"]
            sections += [f"- {row['feature']}: epsilon squared {_fmt(row.get('epsilon_squared'))}, q {_q(row.get('q'))}"
                         for row in tests[:8]]
        outliers = characterization.get("outliers") or []
        if outliers:
            sections += ["", "MOST UNUSUAL ROWS (far from their cluster's median profile)"]
            sections += [f"- {row['name']} in {_cluster_label(row['cluster'])} (robust z {row['robust_z']})"
                         for row in outliers[:5]]

    warnings = record.get("warnings") or []
    sections += ["", "WARNINGS", *([f"- {w}" for w in warnings[:10]] or ["- none"])]
    if methods_text and not compact:
        sections += ["", "METHODS TEXT", methods_text]

    others = [run for run in other_runs if run.get("run_id") != record.get("run_id") and run.get("status") == "success"]
    if others:
        sections += ["", "OTHER RUNS IN THIS SESSION (for comparison)"]
        for run in others[-8:]:
            metrics = run.get("metrics") or {}
            sections.append(
                f"- Run #{run.get('number')}: {run['reduction']['method']} + {run['clustering']['algorithm']} "
                f"({_params(run['clustering'].get('params'))}); {run.get('n_clusters')} clusters, "
                f"{run.get('n_noise')} noise, silhouette {_fmt(metrics.get('silhouette_score'))}, "
                f"Davies-Bouldin {_fmt(metrics.get('davies_bouldin_score'))}, file {(run.get('dataset') or {}).get('filename')}")

    if available_settings:
        sections += ["", "AVAILABLE SETTINGS (names allowed in a suggested json block)"]
        sections += [f"- {line}" for line in available_settings]
    return "\n".join(sections)


def clean_history(messages):
    """The user/assistant turns of a request, validated and trimmed to the last MAX_HISTORY_MESSAGES."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("Send at least one message.")
    cleaned = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ValueError("Each message needs a role of 'user' or 'assistant'.")
        content = str(message.get("content", ""))[:MAX_QUESTION_CHARS * 4]
        cleaned.append({"role": message["role"], "content": content})
    if cleaned[-1]["role"] != "user" or not cleaned[-1]["content"].strip():
        raise ValueError("The last message must be a question from the user.")
    cleaned[-1]["content"] = cleaned[-1]["content"][:MAX_QUESTION_CHARS]
    cleaned = cleaned[-MAX_HISTORY_MESSAGES:]
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


_RUN_MENTION = re.compile(r"\b(?:run|lauf)\s*(?:#|no\.?|number|nr\.?)?\s*(\d{1,4})\b|#(\d{1,4})\b", re.IGNORECASE)


def mentioned_runs(history, current_number):
    """Numbers of other runs the user names ("run 5", "Run #5", "#5"), newest question first, at most
    MAX_MENTIONED_RUNS."""
    numbers = []
    for message in reversed(history):
        if message["role"] != "user":
            continue
        for match in _RUN_MENTION.finditer(message["content"]):
            number = int(match.group(1) or match.group(2))
            if number != current_number and number not in numbers:
                numbers.append(number)
    return numbers[:MAX_MENTIONED_RUNS]


def repetition_cut(text):
    """The answer cut after the first copy of a block it keeps repeating, or None if it does not repeat."""
    if len(text) < REPEAT_WINDOW * REPEAT_LIMIT:
        return None
    tail = text[-REPEAT_WINDOW:]
    if not tail.strip():
        return None
    # Overlapping matches count too: a loop shorter than the window overlaps with itself (str.count would miss it).
    first, found, start = -1, 0, 0
    while found < REPEAT_LIMIT:
        position = text.find(tail, start)
        if position < 0:
            return None
        first = position if first < 0 else first
        found += 1
        start = position + 1
    return text[:first + REPEAT_WINDOW].rstrip()


def _request(path, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{ASSISTANT_URL}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 (fixed, configured URL)


def status():
    """{"available": bool, "model": str, "reason": str|None} - quick check for the frontend."""
    try:
        with _request("/api/tags", timeout=3) as response:
            models = [m.get("name", "") for m in json.load(response).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"available": False, "model": ASSISTANT_MODEL,
                "reason": f"The assistant model server is not running ({ASSISTANT_URL}): {getattr(e, 'reason', e)}"}
    wanted = ASSISTANT_MODEL if ":" in ASSISTANT_MODEL else f"{ASSISTANT_MODEL}:latest"
    if not any(name in (ASSISTANT_MODEL, wanted) for name in models):
        return {"available": False, "model": ASSISTANT_MODEL,
                "reason": f"The model server runs, but the model '{ASSISTANT_MODEL}' is not downloaded "
                          f"(run: ollama pull {ASSISTANT_MODEL})."}
    return {"available": True, "model": ASSISTANT_MODEL, "reason": None}


def stream_chat(context, history):
    """Yields ("delta", text) pieces of the answer. If the model starts repeating itself, yields
    ("replace", shortened answer) and stops. Raises AssistantUnavailable if the server fails."""
    payload = {
        "model": ASSISTANT_MODEL,
        "stream": True,
        "messages": [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{context}"}, *history],
        # Low temperature: explanations should stick to the numbers in the context.
        # repeat_penalty over the last 256 tokens (Ollama's default looks back only 64) against repetition loops.
        "options": {"num_ctx": CONTEXT_TOKENS, "num_predict": MAX_ANSWER_TOKENS, "temperature": 0.3,
                    "repeat_penalty": 1.15, "repeat_last_n": 256},
    }
    try:
        # Reading a long context on a CPU can take a while before the first word arrives.
        response = _request("/api/chat", payload, timeout=600)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise AssistantUnavailable(f"The assistant model server answered with HTTP {e.code}: {detail}")
    except (urllib.error.URLError, OSError) as e:
        raise AssistantUnavailable(f"The assistant model server is not reachable ({ASSISTANT_URL}): {getattr(e, 'reason', e)}")

    answer = ""
    with response:  # closing the response stops the generation on the model server
        for line in response:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise AssistantUnavailable(chunk["error"])
            piece = (chunk.get("message") or {}).get("content", "")
            if piece:
                answer += piece
                cut = repetition_cut(answer)
                if cut is not None:
                    yield "replace", cut + REPEAT_NOTE
                    return
                yield "delta", piece
            if chunk.get("done"):
                return
