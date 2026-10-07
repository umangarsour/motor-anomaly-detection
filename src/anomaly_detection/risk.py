"""Isolation Forest decision_function -> risk level (shared by training and the dashboard).

Bands are quantiles of the TRAINING decision scores, fixed at training time and stored
in models/model_metadata.json:
    High Risk          score < q01            (most anomalous 1% of training data)
    Moderate Risk      q01 <= score < 0       (rest of the flagged 5%; 0 is the IF threshold)
    Slightly Abnormal  0 <= score < q15       (not flagged, next 10%: watch list)
    Normal             score >= q15
So "Anomaly" (score < 0) is exactly {Moderate Risk, High Risk}.
"""

import numpy as np

RISK_LEVELS = ["Normal", "Slightly Abnormal", "Moderate Risk", "High Risk"]


def risk_thresholds(train_scores):
    return {
        "high": float(np.quantile(train_scores, 0.01)),
        "moderate": 0.0,
        "watch": float(np.quantile(train_scores, 0.15)),
    }


def risk_level(scores, thresholds):
    scores = np.asarray(scores, dtype=float)
    return np.select(
        [scores < thresholds["high"], scores < thresholds["moderate"], scores < thresholds["watch"]],
        ["High Risk", "Moderate Risk", "Slightly Abnormal"],
        default="Normal",
    )
