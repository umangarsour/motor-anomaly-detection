"""Regression tests for data, artifacts and the prediction API.

Run:  python -m pytest -q
Artifact-dependent tests are skipped (not failed) until `python -m src.pipeline` has run.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.risk import RISK_LEVELS, risk_level
from src.config import ALL_FEATURES, MODEL_DIR, SCORED_PATH, TEST_CYCLE, split_train_test

warnings.filterwarnings("ignore", message=".*No runtime found.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

HAVE_ARTIFACTS = (MODEL_DIR / "model_metadata.json").exists() and SCORED_PATH.exists()
needs_artifacts = pytest.mark.skipif(not HAVE_ARTIFACTS, reason="run `python -m src.pipeline` first")


# ---------------------------------------------------------------- no artifacts needed
def test_risk_level_bands():
    t = {"high": -0.05, "moderate": 0.0, "watch": 0.03}
    got = risk_level([-0.2, -0.05, -0.01, 0.0, 0.029, 0.03, 0.2], t)
    assert list(got) == ["High Risk", "Moderate Risk", "Moderate Risk", "Slightly Abnormal",
                         "Slightly Abnormal", "Normal", "Normal"]
    assert set(got) <= set(RISK_LEVELS)


def test_split_holds_out_whole_cycles():
    df = pd.DataFrame({"motor_id": ["M1"] * 10, "cycle_id": [1, 2, 3, 4, 5] * 2})
    train, test = split_train_test(df)
    assert set(test.cycle_id) == {TEST_CYCLE}
    assert TEST_CYCLE not in set(train.cycle_id)
    assert len(train) + len(test) == len(df)


# ---------------------------------------------------------------- artifacts
@pytest.fixture(scope="module")
def meta():
    return json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_parquet(SCORED_PATH)


@pytest.fixture(scope="module")
def pred():
    from dashboard.utils import prediction
    return prediction


@needs_artifacts
def test_scored_dataset_is_in_original_units(data):
    assert len(data) == 100_000
    assert data.health_score.between(0, 100).all()          # was z-scored in the old anomaly_results.csv
    assert data.rpm.between(1000, 2000).all()
    assert set(data.anomaly_label.unique()) <= {0, 1}
    assert ((data.anomaly_score < 0) == (data.anomaly_label == 1)).all()
    assert set(data.risk_level.astype(str)) <= set(RISK_LEVELS)
    assert set(data.loc[data.split == "test", "cycle_id"]) == {TEST_CYCLE}


@needs_artifacts
def test_metadata_feature_lists(meta, data):
    clf, iso = meta["classifier"], meta["isolation_forest"]
    for feats in (clf["features"], iso["features"]):
        assert feats and set(feats) <= set(ALL_FEATURES) and len(set(feats)) == len(feats)
        assert set(feats) <= set(data.columns)
    assert len(clf["classes"]) == 7 and clf["classes"] == sorted(clf["classes"])
    assert set(meta["reference"]) == set(ALL_FEATURES)


@needs_artifacts
def test_models_load_and_match_metadata(meta, pred):
    from dashboard.utils.load_data import load_fault_classifier, load_isolation_forest
    clf, iso = load_fault_classifier(), load_isolation_forest()
    assert clf.n_features_in_ == len(meta["classifier"]["features"])
    assert iso.n_features_in_ == len(meta["isolation_forest"]["features"])
    assert list(clf.classes_) == list(range(len(meta["classifier"]["classes"])))


@needs_artifacts
def test_metrics_are_plausible(meta):
    """Leakage canary: no honest model can beat the generator's own noisy labelling rule by much."""
    m = meta["classifier"]["test_metrics"]
    ceiling = meta["classifier"].get("theoretical_ceiling", {}).get("expected_bayes_accuracy", 0.84)
    assert 0.70 <= m["accuracy"] <= ceiling + 0.02
    assert m["f1_macro"] >= 0.45
    assert all(0 <= v["recall"] <= 1 for v in m["per_class"].values())


@needs_artifacts
def test_predict_motor_schema(meta, data, pred):
    row = data.iloc[123][pred.required_features()].to_dict()
    r = pred.predict_motor(row)
    assert {"fault", "confidence", "anomaly", "anomaly_score", "risk_level", "probabilities"} <= set(r)
    assert r["fault"] in meta["classifier"]["classes"]
    assert 0 <= r["confidence"] <= 100
    assert r["anomaly"] in {"Anomaly", "Normal"}
    assert isinstance(r["anomaly_score"], float) and np.isfinite(r["anomaly_score"])
    assert r["risk_level"] in RISK_LEVELS
    assert (r["anomaly"] == "Anomaly") == (r["anomaly_score"] < 0)
    assert abs(sum(r["probabilities"].values()) - 100) < 0.1
    assert r["confidence"] == pytest.approx(max(r["probabilities"].values()), abs=0.01)


@needs_artifacts
@pytest.mark.parametrize("n", [1, 10, 1000])
def test_batch_matches_single_row(data, pred, n):
    rows = data.sample(n, random_state=n)
    batch = pred.predict_batch(rows)
    assert list(batch.columns) == pred.OUTPUT_COLUMNS and batch.index.equals(rows.index)
    for idx in rows.index[:5]:                      # identical to the old one-row-at-a-time loop
        single = pred.predict_motor(rows.loc[idx, pred.required_features()].to_dict())
        for k in pred.OUTPUT_COLUMNS:
            assert batch.loc[idx, k] == single[k]


@needs_artifacts
def test_batch_matches_stored_scores(data, pred):
    rows = data.sample(500, random_state=0)
    out = pred.predict_batch(rows)
    assert np.allclose(out.anomaly_score, rows.anomaly_score.round(3), atol=1e-3)
    assert (out.risk_level.values == rows.risk_level.astype(str).values).all()


@needs_artifacts
def test_column_order_and_extra_columns_do_not_matter(data, pred):
    rows = data.sample(50, random_state=3)
    shuffled = rows[pred.required_features()[::-1]].assign(notes="x", unit_id=7)
    pd.testing.assert_frame_equal(pred.predict_batch(rows), pred.predict_batch(shuffled))


@needs_artifacts
def test_missing_feature_message(data, pred):
    rows = data.head(3).drop(columns=pred.required_features()[:3])
    with pytest.raises(pred.PredictionInputError, match="missing 3 required features"):
        pred.predict_batch(rows)


@needs_artifacts
@pytest.mark.parametrize("bad", [np.nan, np.inf, "abc", ""])
def test_invalid_values_rejected(data, pred, bad):
    rows = data.head(3)[pred.required_features()].astype(object)
    rows.iloc[1, 0] = bad
    with pytest.raises(pred.PredictionInputError, match="1 row"):
        pred.predict_batch(rows)


@needs_artifacts
def test_empty_input_rejected(pred):
    with pytest.raises(pred.PredictionInputError, match="no data rows"):
        pred.predict_batch(pd.DataFrame(columns=pred.required_features()))


@needs_artifacts
def test_unusual_readings_flags_extreme_value(meta, pred):
    ref = meta["reference"]
    values = {f: ref[f]["q50"] for f in ref}
    values["vibration_rms"] = ref["vibration_rms"]["q99"] * 2
    got = pred.unusual_readings(values, ref)
    assert got and got[0][0] == "vibration_rms" and "99th" in got[0][2]
