"""Clustering Assistant: a local language model (Qwen2.5 7B Instruct via Ollama) that explains a session's results.

The model never computes anything itself: every number it may talk about comes from what RapCluster already
calculated. build_context() writes the open run in full (metrics, clusters and their members, characterization,
stability, preprocessing, settings, warnings) and one line per run of the session into its context. For everything
else - any run, any cluster with all its members, where a row ended up in each run, how two runs agree - the model
calls the TOOLS, which the backend answers from the stored results. stream_chat() forwards the conversation to the
Ollama server, runs the tool calls and yields the answer piece by piece.
"""
import json
import os
import re
import urllib.error
import urllib.request

ASSISTANT_URL = os.environ.get("RAPCLUSTER_ASSISTANT_URL", "http://127.0.0.1:11434").rstrip("/")
ASSISTANT_MODEL = os.environ.get("RAPCLUSTER_ASSISTANT_MODEL", "qwen2.5:7b")
# Qwen2.5 7B reads up to 32K tokens; 16K covers the run context, a conversation and a few looked-up results
# (cluster members, other runs) at about 1 GB more CPU memory than 8K.
CONTEXT_TOKENS = int(os.environ.get("RAPCLUSTER_ASSISTANT_CONTEXT", 16384))
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
MAX_INLINE_MEMBERS = 400         # up to this many rows, the open run's member names are written into the context
MAX_SESSION_RUNS = 60            # runs listed one line each in the context
MAX_TOOL_ROUNDS = 6              # model turns that may call tools before it has to answer
MAX_TOOL_RESULT_CHARS = 8000     # one looked-up result
MAX_TOOL_CHARS_PER_ANSWER = 30000  # all looked-up results of one answer (they share CONTEXT_TOKENS)
MAX_LISTED_MEMBERS = 400         # members of one cluster per get_cluster call (more with offset)

SYSTEM_PROMPT = """You are the RapCluster Clustering Assistant. RapCluster is a web tool that clusters tabular data \
(rows = items such as proteins or genes, columns = measurements). The user has a session with one or more clustering \
runs, numbered Run #1, Run #2, ... The user opened you from the run named at the top of RUN RESULTS (the "open run"), \
but you can answer about ANY run of the session, any cluster and any row.

Rules:
- Use only facts from RUN RESULTS below and from tool results. Never invent numbers, clusters, features or row \
names. If something is not available, say that RapCluster did not compute it and, if useful, where in RapCluster it \
can be computed.
- Runs are different clusterings: cluster numbers are arbitrary per run, so "Cluster 2" of Run #3 is unrelated to \
"Cluster 2" of Run #5. Always say which run a fact belongs to (e.g. "Run #3, Cluster 2"). When the user does not \
name a run, they mean the open run.
- RUN RESULTS has the open run in detail and one line per run of the session. For anything else, call the tools \
instead of guessing: get_run_results (all results of any run), get_cluster (all members, features and stability of one \
cluster), find_rows (in which cluster of each run a row/gene/protein is), compare_runs (how two runs agree, which \
clusters correspond), feature_by_cluster (one measurement column across the clusters of a run). Call as many as you \
need, then answer. If a tool returns an error, tell the user what is missing.
- Explain plainly for a biologist, not a statistician. Be concise: short paragraphs or bullet points, usually under \
200 words unless the user asks for detail or for a list (then list everything they asked for).
- Interpret metrics with care: they are guides, not proof. Silhouette ranges -1..1 (above ~0.5 well separated, \
0.25-0.5 moderate, below ~0.25 weak or overlapping). Davies-Bouldin: lower is better, below ~1 is good. \
Calinski-Harabasz: higher is better, only comparable between runs on the same data. DBCV (-1..1, higher is better) \
suits density-based clusters and noise. Cluster -1 is noise (rows not assigned to any cluster). Cliff's delta \
(-1..1) compares a feature inside a cluster with all other rows: |delta| above ~0.47 is a large difference. \
Stability: Jaccard >= 0.75 stable, 0.6-0.75 doubtful, below 0.6 unstable; ARI near 1 means repeated runs agree. \
Metrics of runs on different files or with different preprocessing are not directly comparable.
- Never repeat a sentence, paragraph or json block you have already written in this answer. Say each thing once, \
then stop.
- When you suggest different settings, explain why, and add ONE fenced ```json block with only the settings that \
change (at least one changed value, never empty "params"), in this layout (all keys optional):
{"preprocessing": {"options": {"scaling": "row_zscore"}}, "reduction": {"method": "UMAP", "params": {"n_neighbors": 30}}, \
"clustering": {"algorithm": "HDBSCAN", "params": {"min_cluster_size": 20}}}
Use only algorithm, method, parameter and option names that appear in the results or in AVAILABLE SETTINGS. \
The user can apply that block to the form with one click.
"""

_RUN_ARG = {"type": "integer", "description": "Run number, e.g. 3 for Run #3. Leave out for the open run."}
TOOLS = [
    {"type": "function", "function": {
        "name": "get_run_results",
        "description": "All results of one run: data, preprocessing, settings, metrics, every cluster with size, "
                       "distinguishing features and stability, warnings.",
        "parameters": {"type": "object", "properties": {"run": _RUN_ARG}}}},
    {"type": "function", "function": {
        "name": "get_cluster",
        "description": "One cluster of one run: all member names (row names), size, all distinguishing features, "
                       "most central and representative members, stability. Use cluster -1 for noise.",
        "parameters": {"type": "object", "properties": {
            "run": _RUN_ARG,
            "cluster": {"type": "integer", "description": "Cluster number of that run (-1 = noise)."},
            "offset": {"type": "integer", "description": "Skip this many members (for very large clusters)."}},
            "required": ["cluster"]}}},
    {"type": "function", "function": {
        "name": "find_rows",
        "description": "Finds rows (genes, proteins, ...) whose name contains the text and tells in which cluster "
                       "they are in each run of the session (or only in the given run).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Name or part of a name; several names separated by commas."},
            "run": {"type": "integer", "description": "Only this run. Leave out for all runs."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "compare_runs",
        "description": "Agreement of two runs on their common rows: adjusted Rand index, which clusters correspond "
                       "(overlap, Jaccard), clusters without partner, rows that changed cluster.",
        "parameters": {"type": "object", "properties": {
            "run_a": {"type": "integer", "description": "First run number."},
            "run_b": {"type": "integer", "description": "Second run number. Leave out for the open run."}},
            "required": ["run_a"]}}},
    {"type": "function", "function": {
        "name": "feature_by_cluster",
        "description": "One measurement column (feature) of a run: median, quartiles and mean in each cluster "
                       "(preprocessed values that were clustered). Without a feature it lists the feature names.",
        "parameters": {"type": "object", "properties": {
            "run": _RUN_ARG,
            "feature": {"type": "string", "description": "Column name (case-insensitive)."}}}}},
]


class AssistantUnavailable(Exception):
    """The model server cannot be reached or does not have the model."""


class ToolsUnsupported(Exception):
    """The configured model cannot call tools; it then answers from the context alone."""


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


def cluster_label(cluster):
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
        lines.append(f"- {cluster_label(entry['cluster'])}: " + "; ".join(parts))
    if len(entries) > max_clusters:
        rest = entries[max_clusters:]
        lines.append(f"- ... and {len(rest)} smaller clusters with {sum(e['size'] for e in rest)} rows in total "
                     f"(listed in cluster_summary.tsv of the ZIP download).")
    return lines


def _member_lines(members):
    """Member names of each cluster ({cluster: [names]}), real clusters first, noise last."""
    return [f"- {cluster_label(c)} ({len(names)}): " + ", ".join(names)
            for c, names in sorted(members.items(), key=lambda item: (int(item[0]) == -1, int(item[0])))]


def run_summary_line(run, open_number=None):
    """One line per run of the session, for the overview and for find_rows."""
    number = run.get("number")
    mark = " [OPEN RUN]" if number == open_number else ""
    dataset = run.get("dataset") or {}
    head = (f"- Run #{number}{mark}: file {dataset.get('filename')}; {run['reduction']['method']} "
            f"({_params(run['reduction'].get('params'))}) + {run['clustering']['algorithm']} "
            f"({_params(run['clustering'].get('params'))})")
    if run.get("status") != "success":
        return f"{head}; status {run.get('status')} (no results)"
    metrics = run.get("metrics") or {}
    stability = run.get("stability") or {}
    line = (f"{head}; {run.get('n_clusters')} clusters, {run.get('n_noise')} noise rows, "
            f"silhouette {_fmt(metrics.get('silhouette_score'))}, Davies-Bouldin {_fmt(metrics.get('davies_bouldin_score'))}, "
            f"DBCV {_fmt(metrics.get('dbcv'))}")
    if stability.get("mean_ari") is not None:
        line += f", stability ARI {_fmt(stability['mean_ari'])}"
    options = (run.get("preprocessing") or {}).get("options")
    if options:
        line += "; preprocessing " + _params(options)
    return line


def build_context(record, *, characterization=None, stability_details=None, labels_sizes=None,
                  preprocessing_rows=(), model_rows=(), stability_rows=(), methods_text="", other_runs=(),
                  available_settings=None, display_note=None, title=None, compact=False, members=None):
    """The run's results as plain text for the model (kept compact: CPU models read slowly).

    compact=True (other runs the user asks about) leaves out the outliers, the methods text and fewer clusters.
    members: {cluster: [row names]} written out in full (small data sets); otherwise the model calls get_cluster.
    other_runs: all run records of the session, listed one line each.
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

    number = record.get("number")
    sections += ["", f"CLUSTERS OF RUN #{number} (sizes use all rows)",
                 *_cluster_lines(characterization, stability_details, labels_sizes or {}, 12 if compact else MAX_DETAIL_CLUSTERS)]
    if members:
        sections += ["", f"ALL MEMBERS OF EACH CLUSTER OF RUN #{number} (row names)", *_member_lines(members)]
    else:
        sections.append(f"- (Member names of each cluster: call get_cluster with run {number}.)")
    if characterization and not compact:
        tests = (characterization.get("feature_tests") or {}).get("rows") or []
        if tests:
            sections += ["", "FEATURES THAT DIFFER MOST BETWEEN CLUSTERS (Kruskal-Wallis, epsilon squared 0..1)"]
            sections += [f"- {row['feature']}: epsilon squared {_fmt(row.get('epsilon_squared'))}, q {_q(row.get('q'))}"
                         for row in tests[:8]]
        outliers = characterization.get("outliers") or []
        if outliers:
            sections += ["", "MOST UNUSUAL ROWS (far from their cluster's median profile)"]
            sections += [f"- {row['name']} in {cluster_label(row['cluster'])} (robust z {row['robust_z']})"
                         for row in outliers[:5]]

    warnings = record.get("warnings") or []
    sections += ["", "WARNINGS", *([f"- {w}" for w in warnings[:10]] or ["- none"])]
    if methods_text and not compact:
        sections += ["", "METHODS TEXT", methods_text]

    runs = list(other_runs)
    if runs:
        sections += ["", f"ALL RUNS OF THIS SESSION ({len(runs)}; details of any run via the tools)"]
        sections += [run_summary_line(run, number) for run in runs[-MAX_SESSION_RUNS:]]
        if len(runs) > MAX_SESSION_RUNS:
            sections.append(f"- ({len(runs) - MAX_SESSION_RUNS} earlier runs not listed; get_run_results works for them too.)")

    if available_settings:
        sections += ["", "AVAILABLE SETTINGS (names allowed in a suggested json block)"]
        sections += [f"- {line}" for line in available_settings]
    return "\n".join(sections)


def cluster_text(number, cluster, names, entry=None, stability=None, membership=None, offset=0):
    """get_cluster: one cluster of one run with its member names (a page of MAX_LISTED_MEMBERS from offset)."""
    total = len(names)
    lines = [f"RUN #{number}, {cluster_label(cluster).upper()}: {total} rows"]
    if entry:
        lines.append(f"- Share of all rows: {100 * entry['fraction']:.1f}%")
        if entry.get("medoid"):
            lines.append(f"- Most central member (medoid): {entry['medoid']}")
        if entry.get("representatives"):
            lines.append("- Representative members (closest to the medoid): " + ", ".join(entry["representatives"]))
        features = [f"{f['feature']} (delta {f['cliffs_delta']:+.2f}, median {f['median_in']} vs {f['median_rest']})"
                    for f in entry.get("top_features", []) if f.get("cliffs_delta") is not None]
        if features:
            lines.append("- Distinguishing features vs. all other rows (Cliff's delta, median in vs. other rows): "
                         + "; ".join(features))
        for key, label in (("mean_membership", "Mean membership"), ("persistence", "Persistence"),
                           ("component_weight", "Component weight")):
            if entry.get(key) is not None:
                lines.append(f"- {label}: {entry[key]:.3f}")
    if membership is not None:
        lines.append(f"- Mean membership probability of its rows: {membership:.3f}")
    if stability and stability.get("mean_jaccard") is not None:
        lines.append(f"- Stability: Jaccard {stability['mean_jaccard']:.2f} ({stability['verdict']})")
    page = names[offset:offset + MAX_LISTED_MEMBERS]
    if page:
        shown = f"members {offset + 1}-{offset + len(page)} of {total}" if len(page) < total else f"all {total} members"
        lines.append(f"- Member names ({shown}): " + ", ".join(page))
        if offset + len(page) < total:
            lines.append(f"- (More members: call get_cluster again with offset {offset + len(page)}.)")
    else:
        lines.append(f"- No members from offset {offset} on.")
    return "\n".join(lines)


def comparison_text(number_a, number_b, comparison, max_changed=40):
    """compare_runs: the agreement of two runs (see app.compare_labelings)."""
    lines = [f"COMPARISON RUN #{number_a} (A) vs. RUN #{number_b} (B)",
             f"- Common rows: {comparison['n_common']} (only in A: {comparison['n_only_a']}, only in B: {comparison['n_only_b']})",
             f"- Adjusted Rand index: {comparison['adjusted_rand_index']:.3f} (1 = identical, ~0 = unrelated)",
             f"- Normalized mutual information: {comparison['normalized_mutual_info']:.3f}",
             "- Corresponding clusters (matched by largest Jaccard overlap):"]
    lines += [f"  - Run #{number_a} {cluster_label(m['cluster_a'])} ({m['size_a']} rows) <-> Run #{number_b} "
              f"{cluster_label(m['cluster_b'])} ({m['size_b']} rows): {m['overlap']} rows in both, Jaccard {m['jaccard']:.2f}"
              for m in comparison["matches"]] or ["  - none"]
    if comparison["unmatched_a"]:
        lines.append(f"- Clusters of Run #{number_a} without partner: " + ", ".join(map(str, comparison["unmatched_a"])))
    if comparison["unmatched_b"]:
        lines.append(f"- Clusters of Run #{number_b} without partner: " + ", ".join(map(str, comparison["unmatched_b"])))
    lines.append(f"- Rows not in the corresponding cluster: {comparison['n_changed']}")
    changed = comparison["changed_samples"][:max_changed]
    if changed:
        lines.append("- Examples (row: cluster in A -> cluster in B): " + ", ".join(
            f"{row['name']}: {row['cluster_a']} -> {row['cluster_b']}" for row in changed))
    return "\n".join(lines)


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


def _post_chat(payload):
    try:
        # Reading a long context on a CPU can take a while before the first word arrives.
        return _request("/api/chat", payload, timeout=600)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        if "does not support tools" in detail:
            raise ToolsUnsupported(detail)
        raise AssistantUnavailable(f"The assistant model server answered with HTTP {e.code}: {detail}")
    except (urllib.error.URLError, OSError) as e:
        raise AssistantUnavailable(f"The assistant model server is not reachable ({ASSISTANT_URL}): {getattr(e, 'reason', e)}")


def _tool_arguments(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def describe_call(name, arguments):
    """Short note for the user on what the assistant looked up."""
    run = f"Run #{arguments['run']}" if arguments.get("run") not in (None, "") else "the open run"
    if name == "get_run_results":
        return f"Read the results of {run}"
    if name == "get_cluster":
        return f"Read cluster {arguments.get('cluster')} of {run}"
    if name == "find_rows":
        where = "all runs" if arguments.get("run") in (None, "") else run
        return f"Searched rows matching \"{arguments.get('query')}\" in {where}"
    if name == "compare_runs":
        other = f"Run #{arguments['run_b']}" if arguments.get("run_b") not in (None, "") else "the open run"
        return f"Compared Run #{arguments.get('run_a')} with {other}"
    if name == "feature_by_cluster":
        return f"Read feature {arguments.get('feature') or '(list)'} by cluster in {run}"
    return f"Called {name}"


def stream_chat(context, history, call_tool=None):
    """Yields ("delta", text) pieces of the answer and ("tool", note) for each looked-up result. If the model
    starts repeating itself, yields ("replace", shortened answer) and stops. Raises AssistantUnavailable if the
    server fails.

    call_tool(name, arguments) -> text answers the model's tool calls (see TOOLS); without it the model answers
    from the context alone."""
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{context}"}, *history]
    use_tools = call_tool is not None
    tool_chars = 0
    answer = ""
    for round_number in range(MAX_TOOL_ROUNDS + 1):
        payload = {
            "model": ASSISTANT_MODEL,
            "stream": True,
            "messages": messages,
            # Low temperature: explanations should stick to the numbers in the context.
            # repeat_penalty over the last 256 tokens (Ollama's default looks back only 64) against repetition loops.
            "options": {"num_ctx": CONTEXT_TOKENS, "num_predict": MAX_ANSWER_TOKENS, "temperature": 0.3,
                        "repeat_penalty": 1.15, "repeat_last_n": 256},
        }
        # In the last round the model has to answer with what it looked up so far.
        if use_tools and round_number < MAX_TOOL_ROUNDS:
            payload["tools"] = TOOLS
        try:
            response = _post_chat(payload)
        except ToolsUnsupported:
            use_tools = False
            payload.pop("tools", None)
            response = _post_chat(payload)

        content, calls = "", []
        with response:  # closing the response stops the generation on the model server
            for line in response:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise AssistantUnavailable(chunk["error"])
                message = chunk.get("message") or {}
                calls += message.get("tool_calls") or []
                piece = message.get("content", "")
                if piece:
                    content += piece
                    answer += piece
                    cut = repetition_cut(answer)
                    if cut is not None:
                        yield "replace", cut + REPEAT_NOTE
                        return
                    yield "delta", piece
                if chunk.get("done"):
                    break
        if not calls:
            return
        if content.strip():  # text written before the tool calls; the answer continues after them
            answer += "\n\n"
            yield "delta", "\n\n"
        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        for call in calls:
            function = call.get("function") or {}
            name, arguments = function.get("name", ""), _tool_arguments(function.get("arguments"))
            yield "tool", describe_call(name, arguments)
            if tool_chars >= MAX_TOOL_CHARS_PER_ANSWER:
                result = "Not looked up: enough results were read for this answer. Answer with what you have."
            else:
                try:
                    result = call_tool(name, arguments)
                except (LookupError, ValueError, TypeError) as e:
                    result = f"Error: {e}"
                if len(result) > MAX_TOOL_RESULT_CHARS:
                    result = result[:MAX_TOOL_RESULT_CHARS] + "\n... (cut here; ask for less, e.g. with offset)"
                tool_chars += len(result)
            messages.append({"role": "tool", "tool_name": name, "content": result})
