"""Smoke tests for the backend API and the reproduce script (run with: python -m pytest tests)."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["RAPCLUSTER_DATA_DIR"] = str(tmp_path_factory.mktemp("sessions"))
    import app as rapcluster
    rapcluster.app.config["TESTING"] = True
    return rapcluster.app.test_client()


def test_algorithms(client):
    response = client.get("/api/algorithms")
    assert response.status_code == 200
    data = response.get_json()
    assert "KMeans" in data["clustering_algorithms"]
    assert "PCA" in data["dimensionality_reduction_methods"]
    for algorithm in data["clustering_algorithms"]:
        assert algorithm in data["algorithm_params"]


def test_examples_and_limits(client):
    examples = client.get("/api/examples").get_json()["examples"]
    assert any(example["id"] == "iris" for example in examples)
    assert client.get("/api/limits").status_code == 200


def test_reproduce_script(tmp_path):
    config = {
        "preprocessing": {"name_column": "Name", "intensity_start_index": 2,
                          "options": {"transform": "none", "scaling": "column_zscore"}},
        "reduction": {"method": "PCA", "params": {"n_components": 2, "random_state": 42}},
        "clustering": {"algorithm": "KMeans", "params": {"n_clusters": 3, "random_state": 42}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    subprocess.run([sys.executable, os.path.join(ROOT, "rapcluster_reproduce.py"),
                    os.path.join(ROOT, "examples", "iris.tsv"), str(config_path)],
                   cwd=tmp_path, check=True)
    assignments = pd.read_csv(tmp_path / "reproduced_assignments.tsv", sep="\t")
    assert len(assignments) == 150
    assert assignments["Cluster"].nunique() == 3
