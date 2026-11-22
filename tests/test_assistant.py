"""Results assistant: a real clustering run is explained through a fake Ollama server (no model download needed)."""
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class FakeOllama(BaseHTTPRequestHandler):
    """Answers like Ollama: /api/tags lists qwen2.5:7b, /api/chat streams NDJSON and remembers the request."""
    last_chat = None
    pieces = ("The run found ", "3 clusters.")

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen2.5:7b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        FakeOllama.last_chat = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for piece in FakeOllama.pieces:
            self.wfile.write((json.dumps({"message": {"role": "assistant", "content": piece}, "done": False}) + "\n").encode())
            self.wfile.flush()
        self.wfile.write((json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}) + "\n").encode())


@pytest.fixture(scope="module")
def fake_ollama():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["RAPCLUSTER_DATA_DIR"] = str(tmp_path_factory.mktemp("sessions"))
    import app as rapcluster
    rapcluster.app.config["TESTING"] = True
    return rapcluster.app.test_client()


@pytest.fixture(scope="module")
def finished_run(client):
    """Runs PCA + KMeans on the iris example through the API and waits for the result."""
    session_id = uuid.uuid4().hex
    with open(os.path.join(ROOT, "examples", "iris.tsv"), "rb") as fh:
        upload = client.post(f"/api/sessions/{session_id}/datasets", data={"file": (fh, "iris.tsv")},
                             content_type="multipart/form-data")
    assert upload.status_code in (200, 201), upload.get_json()
    response = client.post("/api/cluster", data={
        "sessionId": session_id, "datasetId": upload.get_json()["dataset_id"], "nameColumn": "Name",
        "intensityStartIndex": "2", "sheetName": "",
        "preprocessing": json.dumps({"transform": "none", "scaling": "column_zscore"}),
        "reductionMethod": "PCA", "reductionParams": json.dumps({"n_components": 2, "random_state": 42}),
        "clusteringAlgorithm": "KMeans", "clusteringParams": json.dumps({"n_clusters": 3, "random_state": 42}),
    })
    assert response.status_code == 202, response.get_json()
    run = response.get_json()["run"]
    for _ in range(240):
        status = client.get("/api" + run["links"]["status"]).get_json()
        if status["state"] != "running":
            break
        time.sleep(0.5)
    assert status["state"] == "success", status
    return session_id, run["run_id"]


def test_status_reports_missing_server(client, monkeypatch):
    import assistant
    monkeypatch.setattr(assistant, "ASSISTANT_URL", "http://127.0.0.1:9")  # nothing listens here
    data = client.get("/api/assistant/status").get_json()
    assert data["available"] is False and "not running" in data["reason"]


def test_status_with_model(client, fake_ollama, monkeypatch):
    import assistant
    monkeypatch.setattr(assistant, "ASSISTANT_URL", fake_ollama)
    assert client.get("/api/assistant/status").get_json() == {"available": True, "model": "qwen2.5:7b", "reason": None}


def test_run_is_explained_with_all_results(client, fake_ollama, finished_run, monkeypatch):
    import assistant
    monkeypatch.setattr(assistant, "ASSISTANT_URL", fake_ollama)
    session_id, run_id = finished_run
    response = client.post(f"/api/sessions/{session_id}/runs/{run_id}/assistant",
                           json={"messages": [{"role": "user", "content": "Summarize the results."}]})
    assert response.status_code == 200
    assert response.headers["X-Accel-Buffering"] == "no"
    lines = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line.strip()]
    assert "".join(line.get("delta", "") for line in lines) == "The run found 3 clusters."
    assert lines[-1] == {"done": True}

    sent = FakeOllama.last_chat
    assert sent["model"] == "qwen2.5:7b" and sent["stream"] is True
    assert sent["messages"][-1] == {"role": "user", "content": "Summarize the results."}
    context = sent["messages"][0]["content"]
    for expected in ("Run #1", "iris.tsv", "KMeans", "n_clusters=3", "PCA", "Silhouette", "Davies-Bouldin",
                     "Cluster 0", "Cluster 2", "most distinguishing features", "STABILITY", "WARNINGS",
                     "METHODS TEXT", "AVAILABLE SETTINGS", "HDBSCAN"):
        assert expected in context, expected


def test_invalid_conversation_is_rejected(client, finished_run):
    session_id, run_id = finished_run
    response = client.post(f"/api/sessions/{session_id}/runs/{run_id}/assistant",
                           json={"messages": [{"role": "assistant", "content": "hi"}]})
    assert response.status_code == 400


def test_unavailable_model_is_reported_in_the_stream(client, finished_run, monkeypatch):
    import assistant
    monkeypatch.setattr(assistant, "ASSISTANT_URL", "http://127.0.0.1:9")
    session_id, run_id = finished_run
    response = client.post(f"/api/sessions/{session_id}/runs/{run_id}/assistant",
                           json={"messages": [{"role": "user", "content": "Hello?"}]})
    lines = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line.strip()]
    assert lines[-1]["code"] == "assistant_unavailable"


def test_clean_history_trims_and_validates():
    import assistant
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(21)]
    cleaned = assistant.clean_history(history)
    assert len(cleaned) <= assistant.MAX_HISTORY_MESSAGES
    assert cleaned[0]["role"] == "user" and cleaned[-1]["content"] == "m20"
    with pytest.raises(ValueError):
        assistant.clean_history([])


def test_repeating_answer_is_stopped_and_shortened(client, fake_ollama, finished_run, monkeypatch):
    import assistant
    monkeypatch.setattr(assistant, "ASSISTANT_URL", fake_ollama)
    block = "If you want more groups, increase n_clusters. Keep the other parameters as they are.\n\n"
    monkeypatch.setattr(FakeOllama, "pieces", ("Run #1 has 3 clusters.\n\n",) + (block,) * 10)
    session_id, run_id = finished_run
    response = client.post(f"/api/sessions/{session_id}/runs/{run_id}/assistant",
                           json={"messages": [{"role": "user", "content": "What next?"}]})
    lines = [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line.strip()]
    replaced = [line["replace"] for line in lines if "replace" in line]
    assert len(replaced) == 1 and lines[-1] == {"done": True}
    assert replaced[0].count("increase n_clusters") == 1
    assert replaced[0].startswith("Run #1 has 3 clusters.") and "repeat itself" in replaced[0]
    sent_options = FakeOllama.last_chat["options"]
    assert sent_options["repeat_last_n"] >= 256 and sent_options["repeat_penalty"] > 1.1


def test_named_runs_get_their_full_results(client, fake_ollama, finished_run, monkeypatch):
    import assistant
    monkeypatch.setattr(assistant, "ASSISTANT_URL", fake_ollama)
    session_id, run_id = finished_run
    # A second run (Run #2) in the same session, explained from Run #1's assistant by naming it.
    runs = client.get(f"/api/sessions/{session_id}/runs").get_json()["runs"]
    first = runs[0]
    response = client.post("/api/cluster", data={
        "sessionId": session_id, "datasetId": first["dataset"]["dataset_id"], "nameColumn": "Name",
        "intensityStartIndex": "2", "sheetName": "", "preprocessing": json.dumps(first["preprocessing"]["options"]),
        "reductionMethod": "None", "reductionParams": "{}",
        "clusteringAlgorithm": "KMeans", "clusteringParams": json.dumps({"n_clusters": 5, "random_state": 1}),
    })
    second = response.get_json()["run"]
    for _ in range(240):
        if client.get("/api" + second["links"]["status"]).get_json()["state"] != "running":
            break
        time.sleep(0.5)
    client.post(f"/api/sessions/{session_id}/runs/{run_id}/assistant",
                json={"messages": [{"role": "user", "content": "Now focus on Run #2 and run 9, please."}]}).get_data()
    context = FakeOllama.last_chat["messages"][0]["content"]
    assert f"FULL RESULTS OF RUN #{second['number']}" in context
    assert "n_clusters=5" in context
    assert "no successful run with that number" in context  # run 9 does not exist


def test_mentions_and_repetition_helpers():
    import assistant
    history = [{"role": "user", "content": "compare run 5 with Run #7"}, {"role": "assistant", "content": "run 3"},
               {"role": "user", "content": "and #4? and run 1"}]
    assert assistant.mentioned_runs(history, current_number=1) == [4, 5]
    assert assistant.repetition_cut("A normal answer without loops. " * 2) is None
    looping = "Intro.\n" + "Same paragraph that keeps coming back again and again in this answer. " * 5
    cut = assistant.repetition_cut(looping)
    assert cut is not None and len(cut) < len(looping) and cut.startswith("Intro.")
