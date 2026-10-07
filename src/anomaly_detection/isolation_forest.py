"""Train the Isolation Forest anomaly detector and score the whole fleet.

- Fit on the TRAINING cycles only (the held-out test cycle stays unseen), on raw
  features: tree splits are invariant to per-feature scaling, so the old
  StandardScaler step changed nothing (verified: identical labels).
- contamination=0.05 is an alert budget (5% of training records flagged); there are
  no ground-truth anomaly labels, so it is not tuned against any target.
- The detector is evaluated against PROXY targets on the test cycle only.

Outputs:
    models/isolation_forest.pkl, models/model_metadata.json ("isolation_forest")
    data/processed/motor_telemetry_scored.parquet  (all rows, original units)
    reports/anomaly_summary.csv, anomaly_statistics.csv, top_anomalies.csv

Usage: python -m src.anomaly_detection.isolation_forest
"""

import json
import time

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.anomaly_detection.risk import RISK_LEVELS, risk_level, risk_thresholds
from src.config import CLEAN_PATH, IF_FEATURES, IF_PARAMS, MODEL_DIR, REPORT_DIR, SCORED_PATH, SEED, split_train_test
from src.evaluation.metrics import anomaly_proxy_metrics, to_builtin

META_PATH = MODEL_DIR / "model_metadata.json"


def main():
    df = pd.read_csv(CLEAN_PATH).drop(columns=["fault_type_encoded", "severity_encoded"], errors="ignore")
    train, test = split_train_test(df)

    iso = IsolationForest(**IF_PARAMS, random_state=SEED, n_jobs=-1)
    t = time.perf_counter()
    iso.fit(train[IF_FEATURES])
    fit_s = time.perf_counter() - t

    train_scores = iso.decision_function(train[IF_FEATURES])
    thresholds = risk_thresholds(train_scores)

    df["anomaly_score"] = iso.decision_function(df[IF_FEATURES])
    df["anomaly_label"] = (df["anomaly_score"] < 0).astype(int)
    df["risk_level"] = pd.Categorical(risk_level(df["anomaly_score"], thresholds), RISK_LEVELS, ordered=True)
    df["split"] = "train"
    df.loc[test.index, "split"] = "test"

    t_idx = test.index
    proxies = {
        "fault_type != Normal": df["fault_type"] != "Normal",
        "severity in {High, Critical} (health < 50)": df["severity"].isin(["High", "Critical"]),
        "severity == Critical (health < 25)": df["severity"] == "Critical",
    }
    proxy_eval = {k: anomaly_proxy_metrics(df.loc[t_idx, "anomaly_score"], df.loc[t_idx, "anomaly_label"],
                                           v[t_idx]) for k, v in proxies.items()}
    test_rows = df.loc[t_idx]
    by_band = (test_rows.groupby("risk_level", observed=False)
               .agg(share=("anomaly_score", lambda s: len(s) / len(test_rows)),
                    mean_health=("health_score", "mean"),
                    critical_share=("severity", lambda s: (s == "Critical").mean()))
               .reset_index().to_dict("records"))

    # ---- artifacts ----
    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCORED_PATH, index=False)
    joblib.dump(iso, MODEL_DIR / "isolation_forest.pkl", compress=3)

    summary = pd.DataFrame({
        "Metric": ["Total Samples", "Normal", "Anomalies", "Anomaly Percentage",
                   "Test-cycle Anomaly Percentage"],
        "Value": [len(df), int((df.anomaly_label == 0).sum()), int(df.anomaly_label.sum()),
                  round(df.anomaly_label.mean() * 100, 2), round(test_rows.anomaly_label.mean() * 100, 2)],
    })
    summary.to_csv(REPORT_DIR / "anomaly_summary.csv", index=False)
    df["anomaly_score"].describe().to_csv(REPORT_DIR / "anomaly_statistics.csv")
    df.nsmallest(100, "anomaly_score").to_csv(REPORT_DIR / "top_anomalies.csv", index=False)

    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    meta["isolation_forest"] = to_builtin({
        "algorithm": "IsolationForest",
        "params": {**IF_PARAMS, "random_state": SEED},
        "features": IF_FEATURES,
        "fit_on": f"training cycles only ({len(train):,} rows), unscaled features",
        "fit_time_s": fit_s,
        "risk_thresholds": thresholds,
        "anomaly_pct_train": float((train_scores < 0).mean() * 100),
        "anomaly_pct_test": float(test_rows.anomaly_label.mean() * 100),
        "decision_function_test": test_rows["anomaly_score"].describe().to_dict(),
        "risk_bands_test": by_band,
        "proxy_eval_test": proxy_eval,
    })
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Isolation Forest fit on {len(train):,} rows in {fit_s:.1f}s")
    print(f"Anomalies: train {meta['isolation_forest']['anomaly_pct_train']:.2f}%  "
          f"test {meta['isolation_forest']['anomaly_pct_test']:.2f}%")
    for k, m in proxy_eval.items():
        print(f"  proxy [{k}]: ROC-AUC {m['roc_auc']:.3f}  PR-AUC {m['pr_auc']:.3f}  "
              f"precision {m['precision']:.3f}  recall {m['recall']:.3f}")
    print("Saved", SCORED_PATH)


if __name__ == "__main__":
    main()
