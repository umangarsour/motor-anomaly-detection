"""Fault classification + anomaly detection for single records and whole CSV files.

Both entry points share one vectorised path (predict_batch), so a CSV of 100k rows
costs one predict_proba and one decision_function call instead of 100k of each.

Output schema (one row / one dict per record):
    fault          str    predicted fault class
    confidence     float  max class probability x 100 (0-100); see metadata
                          classifier.probability_label for how to name it
    anomaly        str    "Anomaly" | "Normal"  (Isolation Forest decision_function < 0)
    anomaly_score  float  Isolation Forest decision_function; lower = more anomalous,
                          negative = flagged. Not a probability or a percentage.
    risk_level     str    one of RISK_LEVELS, from score bands fixed at training time
"""

import numpy as np
import pandas as pd

from src.anomaly_detection.risk import RISK_LEVELS, risk_level  # noqa: F401 (re-exported)
from dashboard.utils.load_data import (
    load_fault_classifier,
    load_isolation_forest,
    load_model_metadata,
)

OUTPUT_COLUMNS = ["fault", "confidence", "anomaly", "anomaly_score", "risk_level"]


class PredictionInputError(ValueError):
    """Input problem the user can fix (shown as a message, never a traceback)."""


def required_features():
    meta = load_model_metadata()
    # union, classifier order first; the detector may use a subset
    feats = list(meta["classifier"]["features"])
    return feats + [f for f in meta["isolation_forest"]["features"] if f not in feats]


def validate_features(df, features):
    """Return a float matrix of `features` in the required order, or raise PredictionInputError."""
    if len(df) == 0:
        raise PredictionInputError("The input contains no data rows.")
    missing = [f for f in features if f not in df.columns]
    if missing:
        s = "s" if len(missing) > 1 else ""
        raise PredictionInputError(
            f"Your file is missing {len(missing)} required feature{s}: {', '.join(missing)}."
        )
    X = df[features].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(X.to_numpy(dtype=float))
    if bad.any():
        cols = X.columns[bad.any(axis=0)].tolist()
        rows = int(bad.any(axis=1).sum())
        raise PredictionInputError(
            f"{rows} row(s) have empty, non-numeric or infinite values in: {', '.join(cols)}. "
            "All feature values must be finite numbers."
        )
    return X.astype(float)


def predict_batch(df):
    """Predict every row of `df`. Extra columns are ignored. Returns OUTPUT_COLUMNS, same index."""
    return _predict(df)[0]


def _predict(df):
    meta = load_model_metadata()
    clf_meta, iso_meta = meta["classifier"], meta["isolation_forest"]
    X = validate_features(df, required_features())

    proba = load_fault_classifier().predict_proba(X[clf_meta["features"]])
    classes = np.asarray(clf_meta["classes"])
    score = load_isolation_forest().decision_function(X[iso_meta["features"]])

    result = pd.DataFrame(
        {
            "fault": classes[proba.argmax(axis=1)],
            "confidence": np.round(proba.max(axis=1) * 100, 2),
            "anomaly": np.where(score < 0, "Anomaly", "Normal"),
            "anomaly_score": np.round(score, 3),
            "risk_level": risk_level(score, iso_meta["risk_thresholds"]),
        },
        index=df.index,
    )
    return result, proba


def predict_motor(sensor_data):
    """Single-record wrapper around predict_batch.

    Returns the OUTPUT_COLUMNS keys plus `probabilities` ({class: 0-100}).
    """
    result, proba = _predict(pd.DataFrame([sensor_data]))
    row, proba = result.iloc[0], proba[0]
    classes = load_model_metadata()["classifier"]["classes"]
    return {
        "fault": str(row["fault"]),
        "confidence": float(row["confidence"]),
        "anomaly": str(row["anomaly"]),
        "anomaly_score": float(row["anomaly_score"]),
        "risk_level": str(row["risk_level"]),
        "probabilities": {c: round(float(p) * 100, 2) for c, p in zip(classes, proba)},
    }


def unusual_readings(sensor_data, reference, limit=3):
    """Inputs outside the fleet's 5th-95th percentile, most extreme first.

    Purely descriptive (how unusual each value is relative to training data);
    it does not claim that these readings caused the prediction.
    """
    out = []
    for feat, ref in reference.items():
        v = sensor_data.get(feat)
        if v is None:
            continue
        spread = (ref["q95"] - ref["q05"]) or 1.0
        if v > ref["q95"]:
            band = "above 99th" if v > ref["q99"] else "above 95th"
            out.append((feat, v, f"{band} percentile", (v - ref["q95"]) / spread))
        elif v < ref["q05"]:
            band = "below 1st" if v < ref["q01"] else "below 5th"
            out.append((feat, v, f"{band} percentile", (ref["q05"] - v) / spread))
    out.sort(key=lambda t: -t[3])
    return [t[:3] for t in out[:limit]]
