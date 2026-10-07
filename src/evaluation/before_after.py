"""Phase 17 regression comparison: original configuration vs. final model on the SAME split.

The original model was evaluated on a row-level split; the final one on the cycle
hold-out. To compare like with like, the original configuration (28 features,
original hyper-parameters, 500 trees) is retrained here on training cycles 1-4 and
evaluated on test cycle 5, next to the production model from models/.

Run after the pipeline:  python -m src.evaluation.before_after
Writes reports/before_after.json and reports/before_after.md.
"""

import io
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate

from src.config import ALL_FEATURES, CLEAN_PATH, GROUP_COLUMN, MODEL_DIR, REPORT_DIR, SEED, TARGET, split_train_test
from src.evaluation.metrics import classification_metrics, to_builtin

LEGACY_PARAMS = dict(n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2,
                     max_features="sqrt", class_weight="balanced", oob_score=True)


def timed_eval(model, X_test, y_test, classes):
    t = time.perf_counter()
    prob = model.predict_proba(X_test)
    batch = time.perf_counter() - t
    lat = []
    for i in range(30):
        t = time.perf_counter()
        model.predict_proba(X_test.iloc[[i]])
        lat.append(time.perf_counter() - t)
    m = classification_metrics(y_test, prob.argmax(1), classes)
    m.update(inference_s_test=batch, inference_ms_single_median=float(np.median(lat) * 1000))
    return m


def main():
    df = pd.read_csv(CLEAN_PATH)
    classes = sorted(df[TARGET].unique())
    df["y"] = np.searchsorted(classes, df[TARGET])
    train, test = split_train_test(df)
    cv = list(GroupKFold(4).split(train, groups=train[GROUP_COLUMN]))
    meta = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))["classifier"]
    out = {}

    print("Original configuration on the cycle hold-out...")
    legacy = RandomForestClassifier(**LEGACY_PARAMS, random_state=SEED, n_jobs=-1)
    cvr = cross_validate(RandomForestClassifier(**{**LEGACY_PARAMS, "oob_score": False}, random_state=SEED, n_jobs=-1),
                         train[ALL_FEATURES], train.y, cv=cv, scoring=["accuracy", "f1_macro", "balanced_accuracy"])
    t = time.perf_counter()
    legacy.fit(train[ALL_FEATURES], train.y)
    fit_s = time.perf_counter() - t
    buf = io.BytesIO()
    joblib.dump(legacy, buf)
    out["before"] = {**timed_eval(legacy, test[ALL_FEATURES], test.y.to_numpy(), classes),
                     "features": len(ALL_FEATURES), "oob_score": legacy.oob_score_, "training_time_s": fit_s,
                     "model_size_mb_uncompressed": buf.tell() / 1e6,
                     "cv": {k[5:]: v.mean() for k, v in cvr.items() if k.startswith("test_")}}
    del legacy

    print("Final production model...")
    final = joblib.load(MODEL_DIR / "fault_classifier.pkl")
    feats = meta["features"]
    t = time.perf_counter()
    clone(final).fit(train[feats], train.y)
    out["after"] = {**timed_eval(final, test[feats], test.y.to_numpy(), classes),
                    "features": len(feats), "oob_score": meta["oob_score"],
                    "training_time_s": time.perf_counter() - t,
                    "model_size_mb_compressed": (MODEL_DIR / "fault_classifier.pkl").stat().st_size / 1e6,
                    "cv": {k: v["mean"] for k, v in meta["cv_metrics"].items()}}
    base = json.loads((REPORT_DIR / "baseline" / "baseline_metrics.json").read_text(encoding="utf-8"))["random_forest"]
    out["original_artifact_row_split"] = {k: base[k] for k in
                                          ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "oob_score",
                                           "model_file_mb", "inference_single_row_ms_median")}

    (REPORT_DIR / "before_after.json").write_text(json.dumps(to_builtin(out), indent=2), encoding="utf-8")
    b, a = out["before"], out["after"]
    rows = [("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"), ("Macro F1", "f1_macro"),
            ("Weighted F1", "f1_weighted"), ("Macro precision", "precision_macro"), ("Macro recall", "recall_macro")]
    L = ["# Before / after (same split: train cycles 1-4, test cycle 5)", "",
         "| Metric | Original config | Final model | Change |", "|---|---|---|---|"]
    L += [f"| {n} | {b[k]:.4f} | {a[k]:.4f} | {a[k] - b[k]:+.4f} |" for n, k in rows]
    L += [f"| CV macro F1 (GroupKFold 4) | {b['cv']['f1_macro']:.4f} | {a['cv']['f1_macro']:.4f} | "
          f"{a['cv']['f1_macro'] - b['cv']['f1_macro']:+.4f} |",
          f"| CV accuracy | {b['cv']['accuracy']:.4f} | {a['cv']['accuracy']:.4f} | "
          f"{a['cv']['accuracy'] - b['cv']['accuracy']:+.4f} |",
          f"| OOB score | {b['oob_score']:.4f} | {a['oob_score']:.4f} | |",
          f"| Features | {b['features']} | {a['features']} | |",
          f"| Training time (s) | {b['training_time_s']:.1f} | {a['training_time_s']:.1f} | |",
          f"| Batch inference, {len(test):,} rows (s) | {b['inference_s_test']:.2f} | {a['inference_s_test']:.2f} | |",
          f"| Single-row latency, median (ms) | {b['inference_ms_single_median']:.1f} | "
          f"{a['inference_ms_single_median']:.1f} | |",
          f"| Model size (MB) | {b['model_size_mb_uncompressed']:.0f} (uncompressed, as originally saved) | "
          f"{a['model_size_mb_compressed']:.1f} (compressed) | |", "",
          "| Class | Recall before | Recall after | F1 before | F1 after | Support |", "|---|---|---|---|---|---|"]
    for c in classes:
        pb, pa = b["per_class"][c], a["per_class"][c]
        L.append(f"| {c} | {pb['recall']:.3f} | {pa['recall']:.3f} | {pb['f1']:.3f} | {pa['f1']:.3f} | {pa['support']} |")
    (REPORT_DIR / "before_after.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
