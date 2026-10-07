"""Describe what the Isolation Forest flags, on the held-out test cycle.

Prints health / severity / fault-label breakdowns of flagged vs. normal samples and
the most anomalous samples. Requires src.anomaly_detection.isolation_forest to have run.

Usage: python -m src.anomaly_detection.evaluate_isolation_forest
"""

import pandas as pd

from src.config import SCORED_PATH


def main():
    df = pd.read_parquet(SCORED_PATH)
    test = df[df.split == "test"]
    line = "=" * 60

    print(f"\n{line}\nMEAN HEALTH SCORE BY ANOMALY LABEL (test cycle)\n{line}")
    print(test.groupby("anomaly_label")["health_score"].mean())

    print(f"\n{line}\nSEVERITY x ANOMALY LABEL (test cycle)\n{line}")
    print(pd.crosstab(test["severity"], test["anomaly_label"]))

    print(f"\n{line}\nSHARE FLAGGED BY RISK LEVEL / FAULT LABEL (test cycle)\n{line}")
    print(test["risk_level"].value_counts(normalize=True).sort_index())
    print(test.groupby("fault_type")["anomaly_label"].mean().sort_values(ascending=False))

    print(f"\n{line}\nTOP 20 MOST ANOMALOUS SAMPLES (all cycles)\n{line}")
    cols = ["motor_id", "cycle_id", "timestamp", "health_score", "fault_type", "severity",
            "risk_level", "anomaly_score"]
    print(df.nsmallest(20, "anomaly_score")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
