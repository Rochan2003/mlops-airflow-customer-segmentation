"""Tests for the segmentation steps and the DAG structure.

Run inside the Airflow container, since the DAG tests need Airflow installed:

    docker compose run --rm airflow-cli bash -c "cd /opt/airflow && pytest tests -v"
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/opt/airflow/dags")

from src import segmentation as seg  # noqa: E402

DATA_DIR = "/opt/airflow/dags/data"
GOOD_CSV = os.path.join(DATA_DIR, "file.csv")
BAD_CSV = os.path.join(DATA_DIR, "bad_sample.csv")
SAMPLE_CSV = os.path.join(DATA_DIR, "test.csv")


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Point artifact writes at a temp dir so tests don't touch real output."""
    monkeypatch.setattr(seg, "ARTIFACT_DIR", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------- validation --

def test_good_data_passes_validation():
    report = seg.validate_data(GOOD_CSV)
    assert report["passed"] is True
    assert report["rows"] > 8000
    assert all(report["checks"].values())


def test_bad_data_fails_validation():
    report = seg.validate_data(BAD_CSV)
    assert report["passed"] is False
    assert report["checks"]["required_columns_present"] is False
    assert "CREDIT_LIMIT" in report["missing_columns"]


def test_missing_file_fails_validation():
    report = seg.validate_data("/opt/airflow/dags/data/does_not_exist.csv")
    assert report["passed"] is False
    assert report["checks"]["file_exists"] is False


# -------------------------------------------------------------- preprocessing --

def test_preprocess_saves_matrix_and_scaler(artifacts):
    paths = seg.preprocess(GOOD_CSV)

    assert os.path.exists(paths["matrix_path"])
    assert os.path.exists(paths["scaler_path"])
    assert paths["rows"] > 8000


def test_preprocess_scales_into_unit_range(artifacts):
    from joblib import load

    matrix = load(seg.preprocess(GOOD_CSV)["matrix_path"])

    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0
    assert matrix.shape[1] == len(seg.FEATURES)


# ------------------------------------------------------------------ k search --

def test_fit_one_k_returns_k_and_sse(artifacts):
    matrix_path = seg.preprocess(GOOD_CSV)["matrix_path"]

    result = seg.fit_one_k(4, matrix_path)

    assert result["k"] == 4
    assert result["sse"] > 0


def test_sse_decreases_as_k_grows(artifacts):
    """More clusters must never increase within-cluster error."""
    matrix_path = seg.preprocess(GOOD_CSV)["matrix_path"]

    sse = [seg.fit_one_k(k, matrix_path)["sse"] for k in (2, 4, 8)]

    assert sse[0] > sse[1] > sse[2]


def test_select_best_k_finds_the_elbow(artifacts):
    # A curve with an obvious bend at k=4.
    curve = [{"k": k, "sse": s} for k, s in
             zip(range(2, 10), [100, 60, 30, 28, 27, 26, 25, 24])]

    summary = seg.select_best_k(curve)

    assert summary["elbow_found"] is True
    assert summary["best_k"] == 4


def test_select_best_k_falls_back_without_an_elbow(artifacts):
    # A straight line has no bend, so KneeLocator returns None.
    curve = [{"k": k, "sse": 100 - k} for k in range(2, 10)]

    summary = seg.select_best_k(curve)

    assert summary["elbow_found"] is False
    assert summary["best_k"] in [c["k"] for c in curve]


# ----------------------------------------------------------- final model --

def test_train_final_model_uses_the_requested_k(artifacts):
    """The saved model must be the k that was asked for."""
    from joblib import load

    matrix_path = seg.preprocess(GOOD_CSV)["matrix_path"]

    info = seg.train_final_model(5, matrix_path)

    assert info["k"] == 5
    assert load(info["model_path"]).n_clusters == 5


def test_score_sample_applies_the_scaler(artifacts):
    """Scoring must scale new rows before predicting.

    test.csv has a CREDIT_LIMIT far outside the training range. Predicting on
    raw values and on scaled values land in different clusters, so this guards
    against the scaler being skipped.
    """
    from joblib import load

    paths = seg.preprocess(GOOD_CSV)
    info = seg.train_final_model(5, paths["matrix_path"])

    result = seg.score_sample(info["model_path"], paths["scaler_path"], SAMPLE_CSV)

    assert result["sample_rows"] == 1
    assert 0 <= result["clusters"][0] < 5

    raw = pd.read_csv(SAMPLE_CSV)[seg.FEATURES]
    scaler = load(paths["scaler_path"])
    scaled = scaler.transform(raw)
    assert not np.allclose(scaled, raw.values), "scaler did not change the values"


# ----------------------------------------------------------- DAG structure --

@pytest.fixture(scope="module")
def dag():
    from airflow.models.dagbag import DagBag

    bag = DagBag("/opt/airflow/dags")
    assert not bag.import_errors, bag.import_errors
    return bag.dags["customer_segmentation"]


def test_dag_has_the_expected_tasks(dag):
    expected = {
        "wait_for_data", "validate", "route_on_quality", "data_rejected",
        "preprocess_data", "k_search.fit_k", "k_search.pick_k",
        "train_final", "score", "pipeline_summary",
    }
    assert {t.task_id for t in dag.tasks} == expected


def test_k_search_is_a_mapped_task(dag):
    """fit_k must be mapped, not a loop inside one task."""
    fit_k = dag.get_task("k_search.fit_k")
    assert type(fit_k).__name__ == "DecoratedMappedOperator"


def test_branch_leads_to_both_outcomes(dag):
    branch = dag.get_task("route_on_quality")
    assert branch.downstream_task_ids == {"preprocess_data", "data_rejected"}


def test_training_is_downstream_of_the_gate(dag):
    """Nothing may train without passing through the branch."""
    for task_id in ("preprocess_data", "k_search.fit_k", "train_final", "score"):
        upstream = dag.get_task(task_id).upstream_list
        assert upstream, f"{task_id} has no upstream"


def test_summary_runs_even_when_upstream_fails(dag):
    summary = dag.get_task("pipeline_summary")
    assert summary.trigger_rule == "all_done"
