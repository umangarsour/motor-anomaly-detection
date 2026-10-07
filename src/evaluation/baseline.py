"""Phase 1 baseline: evaluate the ORIGINAL (pre-refactor) model artifacts.

Re-creates the exact split used by the original fault_classifier.py
(test_size=0.2, random_state=42, stratified, row-level), evaluates the original
pickled Random Forest and Isolation Forest, and re-runs the original training
configuration once to measure training time / CV under the current environment.

Writes reports/baseline/baseline_metrics.json and reports/baseline/baseline_report.md.

Usage: python -m src.evaluation.baseline   (original artifacts live in models/legacy/)
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split

from src.evaluation.metrics import anomaly_proxy_metrics, classification_metrics, to_builtin

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "motor_telemetry_clean.csv"
OUT_DIR = BASE_DIR / "reports" / "baseline"

LEGACY_RF_PARAMS = dict(
    n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2,
    max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
)
DROP = ["motor_id", "motor_type", "cycle_id", "timestamp", "fault_type", "severity",
        "fault_type_encoded", "severity_encoded"]


def main(legacy_dir):
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP)
    y = df["fault_type_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    encoder = joblib.load(legacy_dir / "fault_label_encoder.pkl")
    classes = list(encoder.classes_)
    features = joblib.load(legacy_dir / "selected_features.pkl")
    assert features == list(X.columns), "feature order differs from original training"

    out = {
        "dataset": {
            "samples": len(df),
            "n_features": len(features),
            "features": features,
            "class_distribution": df["fault_type"].value_counts().to_dict(),
            "severity_distribution": df["severity"].value_counts().to_dict(),
            "train_size": len(X_train),
            "validation_size": 0,
            "test_size": len(X_test),
            "split": "row-level stratified train_test_split(test_size=0.2, random_state=42); "
                     "no separate validation set (5-fold CV on train only)",
        }
    }

    # ---------------- Original Random Forest artifact ----------------
    print("Loading original Random Forest (large pickle)...")
    t = time.perf_counter()
    rf = joblib.load(legacy_dir / "fault_classifier.pkl")
    load_s = time.perf_counter() - t

    t = time.perf_counter()
    pred = rf.predict(X_test)
    batch_s = time.perf_counter() - t
    single = X_test.iloc[[0]]
    lat = []
    for _ in range(20):
        t = time.perf_counter()
        rf.predict_proba(single)
        lat.append(time.perf_counter() - t)

    rf_metrics = classification_metrics(y_test, pred, classes)
    rf_metrics.update({
        "oob_score": rf.oob_score_,
        "params": {k: v for k, v in rf.get_params().items() if k in LEGACY_RF_PARAMS},
        "model_file_mb": (legacy_dir / "fault_classifier.pkl").stat().st_size / 1e6,
        "total_tree_nodes": int(sum(e.tree_.node_count for e in rf.estimators_)),
        "load_time_s": load_s,
        "inference_batch_test_s": batch_s,
        "inference_per_row_ms_batch": batch_s / len(X_test) * 1000,
        "inference_single_row_ms_median": float(np.median(lat) * 1000),
    })
    del rf

    # ---------------- Re-run original config: CV + training time ----------------
    print("Re-running original configuration (5-fold CV + final fit)...")
    cv = cross_validate(
        RandomForestClassifier(**LEGACY_RF_PARAMS), X_train, y_train, cv=5,
        scoring=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
    )
    rf_metrics["cv_5fold_rowlevel"] = {
        k.replace("test_", ""): {"mean": v.mean(), "std": v.std()}
        for k, v in cv.items() if k.startswith("test_")
    }
    refit = RandomForestClassifier(**LEGACY_RF_PARAMS)
    t = time.perf_counter()
    refit.fit(X_train, y_train)
    rf_metrics["training_time_s_current_env"] = time.perf_counter() - t
    rf_metrics["refit_test_prediction_agreement"] = float((refit.predict(X_test) == pred).mean())
    del refit
    out["random_forest"] = rf_metrics

    # ---------------- Original Isolation Forest artifact ----------------
    print("Evaluating original Isolation Forest...")
    iso = joblib.load(legacy_dir / "isolation_forest.pkl")
    scaler = joblib.load(legacy_dir / "feature_scaler.pkl")
    Xs = pd.DataFrame(scaler.transform(X[features]), columns=features)
    score = iso.decision_function(Xs)
    label = (iso.predict(Xs) == -1).astype(int)
    test_idx = X_test.index
    proxies = {
        "fault_type != Normal": (df["fault_type"] != "Normal").to_numpy(),
        "severity in {High, Critical} (health < 50)": df["severity"].isin(["High", "Critical"]).to_numpy(),
        "severity == Critical (health < 25)": (df["severity"] == "Critical").to_numpy(),
    }
    out["isolation_forest"] = {
        "params": {k: iso.get_params()[k] for k in
                   ["n_estimators", "contamination", "max_samples", "max_features", "random_state"]},
        "fit_on": "all 100,000 rows (no hold-out), StandardScaler fit on all rows",
        "n_anomalies": int(label.sum()),
        "anomaly_pct": label.mean() * 100,
        "decision_function": pd.Series(score).describe(percentiles=[.01, .05, .25, .5, .75, .95]).to_dict(),
        "score_samples": pd.Series(iso.score_samples(Xs)).describe().to_dict(),
        "risk_level_share_pct": {
            "Normal (>0.05)": float((score > 0.05).mean() * 100),
            "Slightly Abnormal (0..0.05]": float(((score > 0) & (score <= 0.05)).mean() * 100),
            "Moderate Risk (-0.05..0]": float(((score > -0.05) & (score <= 0)).mean() * 100),
            "High Risk (<=-0.05)": float((score <= -0.05).mean() * 100),
        },
        "ground_truth_anomaly_labels": "none in dataset - proxy targets only",
        "proxy_eval_all_rows": {k: anomaly_proxy_metrics(score, label, v) for k, v in proxies.items()},
        "proxy_eval_test_rows": {k: anomaly_proxy_metrics(score[test_idx], label[test_idx], v[test_idx])
                                 for k, v in proxies.items()},
        "model_file_mb": (legacy_dir / "isolation_forest.pkl").stat().st_size / 1e6,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = to_builtin(out)
    (OUT_DIR / "baseline_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT_DIR / "baseline_report.md").write_text(render_md(out, classes), encoding="utf-8")
    print("Saved", OUT_DIR)


def render_md(o, classes):
    rf, iso, ds = o["random_forest"], o["isolation_forest"], o["dataset"]
    L = ["# Baseline (pre-refactor) evaluation", "",
         "Generated by `python -m src.evaluation.baseline` from the original model artifacts.", "",
         "## Dataset", f"- Samples: {ds['samples']:,}; features: {ds['n_features']}",
         f"- Split: {ds['split']}",
         f"- Train / validation / test: {ds['train_size']:,} / {ds['validation_size']} / {ds['test_size']:,}",
         f"- Features: {', '.join(ds['features'])}", "",
         "| Class | Count | Share |", "|---|---|---|"]
    tot = sum(ds["class_distribution"].values())
    L += [f"| {k} | {v:,} | {v / tot:.2%} |" for k, v in ds["class_distribution"].items()]
    L += ["", "## Random Forest (original artifact, original test split)", "",
          "| Metric | Value |", "|---|---|"]
    for k in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "oob_score"]:
        L.append(f"| {k} | {rf[k]:.4f} |")
    for k, v in rf["cv_5fold_rowlevel"].items():
        L.append(f"| CV {k} (5-fold, train) | {v['mean']:.4f} ± {v['std']:.4f} |")
    L += [f"| training time (s, current env, 12 cores) | {rf['training_time_s_current_env']:.1f} |",
          f"| batch inference, 20k rows (s) | {rf['inference_batch_test_s']:.2f} |",
          f"| single-row predict_proba median (ms) | {rf['inference_single_row_ms_median']:.1f} |",
          f"| model file (MB) | {rf['model_file_mb']:.0f} |",
          f"| model load time (s) | {rf['load_time_s']:.1f} |",
          f"| refit reproduces original predictions | {rf['refit_test_prediction_agreement']:.4%} |",
          "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c, m in rf["per_class"].items():
        L.append(f"| {c} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    L += ["", "Confusion matrix (rows = true, cols = predicted):", "",
          "| | " + " | ".join(classes) + " |", "|---" * (len(classes) + 1) + "|"]
    for c, row in zip(classes, rf["confusion_matrix"]):
        L.append(f"| **{c}** | " + " | ".join(str(v) for v in row) + " |")
    L += ["", "## Isolation Forest (original artifact)", "",
          f"- Params: {iso['params']}", f"- Fit on: {iso['fit_on']}",
          f"- Anomalies: {iso['n_anomalies']:,} ({iso['anomaly_pct']:.2f}%)",
          f"- Ground-truth anomaly labels: {iso['ground_truth_anomaly_labels']}", "",
          "decision_function distribution:", "",
          "| stat | value |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in iso["decision_function"].items()]
    L += ["", "Share of records per original risk band:", "", "| Band | % |", "|---|---|"]
    L += [f"| {k} | {v:.2f} |" for k, v in iso["risk_level_share_pct"].items()]
    L += ["", "Proxy evaluation (test rows; NOTE: the IF was fit on these rows too):", "",
          "| Proxy target | Pos. rate | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|---|"]
    for k, m in iso["proxy_eval_test_rows"].items():
        L.append(f"| {k} | {m['proxy_positive_rate']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                 f"{m['f1']:.3f} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-dir", default=str(BASE_DIR / "models" / "legacy"))
    main(Path(ap.parse_args().legacy_dir))
