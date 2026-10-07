"""Train the production fault classifier and evaluate it ONCE on the held-out test cycle.

Methodology (justified in reports/refactor_and_optimization_report.md):
- Train = operating cycles 1-4 of every motor, test = cycle 5 (whole cycles held out).
- Model selection was done beforehand with GroupKFold(4) on cycle_id inside the training
  cycles (src/experiments/run_experiments.py); the parameters below are its outcome.
- This script re-runs that CV for the record, fits on all training cycles, then
  touches the test cycle exactly once.

Outputs:
    models/fault_classifier.pkl, models/model_metadata.json ("classifier", "reference")
    reports/model_metrics.csv, classification_report.csv, confusion_matrix.csv,
    reports/feature_importance.csv

Usage: python -m src.classification.fault_classifier
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate

from src.config import (
    ALL_FEATURES, CLEAN_PATH, CLF_FEATURES, GROUP_COLUMN, MODEL_DIR, REPORT_DIR,
    RF_PARAMS, SEED, TARGET, TEST_CYCLE, split_train_test,
)
from src.evaluation.metrics import classification_metrics, to_builtin

META_PATH = MODEL_DIR / "model_metadata.json"
CEILING_PATH = REPORT_DIR / "experiments" / "ceiling.json"


def build_model():
    # Uncalibrated on purpose: isotonic calibration cut CV macro-F1 by 2.4 points
    # (reports/experiments/calibration.json); vote shares are shown as "Model Confidence".
    return RandomForestClassifier(**RF_PARAMS, random_state=SEED, n_jobs=-1)


def reliability(prob, y, bins=10):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "mean_confidence": conf[m].mean(), "accuracy": (pred[m] == y[m]).mean()})
    ece = sum(r["n"] * abs(r["accuracy"] - r["mean_confidence"]) for r in rows) / len(y)
    return rows, ece


def main():
    df = pd.read_csv(CLEAN_PATH)
    classes = sorted(df[TARGET].unique())
    df["y"] = np.searchsorted(classes, df[TARGET])
    train, test = split_train_test(df)
    X_train, y_train = train[CLF_FEATURES], train["y"].to_numpy()
    X_test, y_test = test[CLF_FEATURES], test["y"].to_numpy()
    groups = train[GROUP_COLUMN].to_numpy()
    cv = list(GroupKFold(n_splits=4).split(X_train, groups=groups))
    print(f"Train {X_train.shape} (cycles != {TEST_CYCLE}), test {X_test.shape}, "
          f"{len(CLF_FEATURES)} features")

    # ---- cross-validation on training cycles (for the record; selection already done) ----
    t = time.perf_counter()
    cv_res = cross_validate(build_model(), X_train, y_train, cv=cv, groups=groups,
                            scoring=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"])
    oof = cross_val_predict(build_model(), X_train, y_train, cv=cv, groups=groups)
    print(f"CV done in {time.perf_counter() - t:.0f}s")
    cv_metrics = {k[5:]: {"mean": v.mean(), "std": v.std()} for k, v in cv_res.items() if k.startswith("test_")}
    cv_oof = classification_metrics(y_train, oof, classes)

    # ---- final fit on all training cycles ----
    model = build_model()
    t = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t
    assert list(model.classes_) == list(range(len(classes)))

    # ---- single evaluation on the held-out test cycle ----
    t = time.perf_counter()
    prob = model.predict_proba(X_test)
    batch_s = time.perf_counter() - t
    pred = prob.argmax(1)
    test_metrics = classification_metrics(y_test, pred, classes)
    rel, ece = reliability(prob, y_test)
    lat = []
    for i in range(30):
        t = time.perf_counter()
        model.predict_proba(X_test.iloc[[i]])
        lat.append(time.perf_counter() - t)

    oob = model.oob_score_
    perm = permutation_importance(model, X_test, y_test, scoring="f1_macro", n_repeats=5,
                                  random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({
        "Feature": CLF_FEATURES,
        "Importance": model.feature_importances_,           # mean decrease in impurity (train)
        "Permutation_F1_Macro": perm.importances_mean,   # drop in test macro-F1 when shuffled
        "Permutation_Std": perm.importances_std,
    }).sort_values("Permutation_F1_Macro", ascending=False)

    joblib.dump(model, MODEL_DIR / "fault_classifier.pkl", compress=3)
    size_mb = (MODEL_DIR / "fault_classifier.pkl").stat().st_size / 1e6

    # ---- reports ----
    names = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced Accuracy",
             "f1_macro": "Macro F1", "f1_weighted": "Weighted F1",
             "precision_macro": "Macro Precision", "recall_macro": "Macro Recall"}
    rows = [(v, test_metrics[k]) for k, v in names.items()]
    rows += [("CV Macro F1", cv_metrics["f1_macro"]["mean"]),
             ("CV Accuracy", cv_metrics["accuracy"]["mean"]),
             ("OOB Score", oob),
             ("Training Time (s)", train_s), ("Model Size (MB)", size_mb)]
    pd.DataFrame(rows, columns=["Metric", "Value"]).to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(test_metrics["per_class"]).T.to_csv(REPORT_DIR / "classification_report.csv")
    pd.DataFrame(test_metrics["confusion_matrix"], index=classes, columns=classes).to_csv(
        REPORT_DIR / "confusion_matrix.csv")
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    ref = train[ALL_FEATURES].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).T
    ref.columns = ["q01", "q05", "q50", "q95", "q99"]
    ref["min"], ref["max"] = train[ALL_FEATURES].min(), train[ALL_FEATURES].max()

    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    meta["classifier"] = {
        "algorithm": "RandomForestClassifier",
        "params": {**RF_PARAMS, "random_state": SEED},
        "calibration": None,
        "probability_label": "Model Confidence",
        "features": CLF_FEATURES,
        "classes": classes,
        "split": f"train = cycles 1-4 of every motor ({len(train):,} rows); "
                 f"test = cycle {TEST_CYCLE} ({len(test):,} rows); CV = GroupKFold(4) on {GROUP_COLUMN}",
        "train_size": len(train), "test_size": len(test),
        "class_distribution_train": train[TARGET].value_counts().to_dict(),
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "cv_oof_per_class": cv_oof["per_class"],
        "oob_score": oob,
        "training_time_s": train_s,
        "inference_ms_per_row_batch": batch_s / len(X_test) * 1000,
        "inference_ms_single_row_median": float(np.median(lat) * 1000),
        "model_size_mb": size_mb,
        "reliability_test": rel,
        "ece_test": ece,
        "feature_importance": importance.to_dict("records"),
        "sklearn_version": sklearn.__version__,
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    if CEILING_PATH.exists():
        c = json.loads(CEILING_PATH.read_text())
        meta["classifier"]["theoretical_ceiling"] = {
            "expected_bayes_accuracy": c["expected_bayes_accuracy_all"],
            "bayes_rule_test": {k: c["bayes_rule_vs_labels_test_cycle"][k]
                                for k in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted")},
        }
    meta["reference"] = ref.to_dict("index")
    META_PATH.write_text(json.dumps(to_builtin(meta), indent=2), encoding="utf-8")

    print(f"\nTraining time {train_s:.1f}s, model {size_mb:.1f} MB, OOB {oob}")
    print("CV  (train cycles):", {k: round(v["mean"], 4) for k, v in cv_metrics.items()})
    print("TEST (cycle 5):    ", {k: round(test_metrics[k], 4) for k in names})
    print(f"ECE (test) {ece:.4f}")
    print(pd.DataFrame(test_metrics["per_class"]).T.round(3))


if __name__ == "__main__":
    main()
