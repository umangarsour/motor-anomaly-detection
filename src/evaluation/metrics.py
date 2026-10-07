"""Shared metric helpers so baseline and final evaluations are computed identically."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, class_names):
    """Overall + per-class metrics as a JSON-serialisable dict."""
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    p_mac, r_mac, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": {
            name: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def anomaly_proxy_metrics(decision_scores, predicted_anomaly, proxy_target):
    """Score an unsupervised detector against a *proxy* binary target.

    decision_scores: IsolationForest.decision_function (lower = more anomalous).
    """
    proxy_target = np.asarray(proxy_target).astype(int)
    predicted_anomaly = np.asarray(predicted_anomaly).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        proxy_target, predicted_anomaly, average="binary", zero_division=0
    )
    return {
        "proxy_positive_rate": proxy_target.mean(),
        "precision": p,
        "recall": r,
        "f1": f,
        "roc_auc": roc_auc_score(proxy_target, -np.asarray(decision_scores)),
        "pr_auc": average_precision_score(proxy_target, -np.asarray(decision_scores)),
    }


def to_builtin(obj):
    """Recursively convert numpy scalars so json.dump works."""
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
