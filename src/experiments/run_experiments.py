"""Research experiments behind the model decisions (Phases 2-7 of the refactor report).

All experiments use ONLY the training cycles (cycle_id != TEST_CYCLE) except
`ceiling`, which needs no fitting. The held-out test cycle is touched exactly once,
by src/classification/fault_classifier.py.

Usage:
    python -m src.experiments.run_experiments ceiling leakage cv tune compare calibration features
Each stage writes reports/experiments/<stage>.json.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import (
    GroupKFold,
    ParameterSampler,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    ALL_FEATURES, CLEAN_PATH, GROUP_COLUMN, LATENT_FEATURES, REPORT_DIR, SEED, TARGET,
    split_train_test,
)
from src.evaluation.metrics import classification_metrics, to_builtin

OUT = REPORT_DIR / "experiments"
SCORING = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
LEGACY_RF = dict(n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2,
                 max_features="sqrt", class_weight="balanced")
# Ablations use 200 trees: tree count changes variance, not the comparison.
FAST_RF = {**LEGACY_RF, "n_estimators": 200}

SENSOR_FEATURES = [f for f in ALL_FEATURES if f not in LATENT_FEATURES]


def load():
    df = pd.read_csv(CLEAN_PATH)
    train, _ = split_train_test(df)
    enc = LabelEncoder().fit(df[TARGET])
    return train.reset_index(drop=True), enc


def rf(**kw):
    return RandomForestClassifier(**{**FAST_RF, **kw}, random_state=SEED, n_jobs=-1)


def group_cv(train):
    return list(GroupKFold(n_splits=4).split(train, groups=train[GROUP_COLUMN]))


def cv_scores(model, X, y, cv):
    t = time.perf_counter()
    r = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    out = {k[5:]: {"mean": r[k].mean(), "std": r[k].std()} for k in r if k.startswith("test_")}
    out["seconds"] = time.perf_counter() - t
    return out


def cv_per_class(model, X, y, cv, classes):
    """Out-of-fold predictions -> full metric dict incl. per-class recall/F1."""
    pred = cross_val_predict(model, X, y, cv=cv)
    return classification_metrics(y, pred, classes)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(to_builtin(obj), indent=2))
    print(f"saved {OUT / name}.json")


# --------------------------------------------------------------------------
# Theoretical ceiling: the generator's label rule applied to the exported
# features, with its N(0, 5) score noise integrated out by Monte Carlo.
# --------------------------------------------------------------------------
NOISE_SD, ACTIVE, MULTI = 5.0, 68, 85
STATES = ["Bearing_Wear", "Misalignment", "Overheating", "Electrical_Fault", "Overload", "Normal"]


def deterministic_scores(d):
    vib15 = np.clip(d.vibration_rms * 15, 0, 100)
    return np.column_stack([
        0.5 * d.bearing_wear_index + 0.3 * vib15 + 0.2 * np.clip(d.bearing_temperature, 0, 100),
        0.5 * d.misalignment_index + 0.4 * vib15 + 0.1 * d.load_percentage,
        0.5 * d.thermal_stress_index + 0.5 * np.clip((d.motor_temperature - 30) * 1.5, 0, 100),
        0.6 * d.electrical_stress_index + 0.4 * np.clip((1 - d.power_factor) * 500, 0, 100),
        0.6 * d.load_percentage + 0.4 * np.clip(d.current * 8, 0, 100),
        d.health_score * 0.4 + d.lubrication_index * 0.2 + d.efficiency_index * 100 * 0.3
        - d.vibration_rms * 5 - (d.motor_temperature - 30) * 0.2,
    ])


def label_rule(scores):
    """Vectorised copy of generate_health_and_faults' assignment logic."""
    order = np.argsort(-scores, axis=1)
    top, second = order[:, 0], order[:, 1]
    rows = np.arange(len(scores))
    s1, s2 = scores[rows, top], scores[rows, second]
    normal = STATES.index("Normal")
    multi = (top != normal) & (second != normal) & (s1 > MULTI) & (s2 > MULTI)
    labels = np.array(STATES + ["Multi_Fault"])[np.where(multi, len(STATES), top)]
    return labels


def stage_ceiling(draws=400):
    df = pd.read_csv(CLEAN_PATH)
    rng = np.random.default_rng(SEED)
    base = deterministic_scores(df)
    classes = sorted(df[TARGET].unique())
    counts = np.zeros((len(df), len(classes)))
    for _ in range(draws):
        lab = label_rule(base + rng.normal(0, NOISE_SD, base.shape))
        counts[np.arange(len(df)), np.searchsorted(classes, lab)] += 1
    prob = counts / draws
    bayes_pred = prob.argmax(1)
    y = np.searchsorted(classes, df[TARGET])
    _, test = split_train_test(df)
    t = test.index.to_numpy()
    save("ceiling", {
        "method": f"Bayes-optimal rule: generator label logic + Monte Carlo over its N(0,{NOISE_SD}) "
                  f"score noise ({draws} draws/row). Uses all 28 features (incl. latent ones).",
        "expected_bayes_accuracy_all": prob.max(1).mean(),
        "noise_free_rule_matches_label": (label_rule(base) == df[TARGET].to_numpy()).mean(),
        "bayes_rule_vs_labels_test_cycle": classification_metrics(y[t], bayes_pred[t], classes),
    })


# --------------------------------------------------------------------------
def stage_leakage():
    train, enc = load()
    y = enc.transform(train[TARGET])
    cv = group_cv(train)
    sets = {
        "A_all_28": ALL_FEATURES,
        "B_minus_health_score": [f for f in ALL_FEATURES if f != "health_score"],
        "C_minus_degradation_indices": [f for f in ALL_FEATURES if f not in LATENT_FEATURES[:4]],
        "D_sensors_only (minus all 5 latent)": SENSOR_FEATURES,
        "E_D_minus_maintenance_quality": [f for f in SENSOR_FEATURES if f != "maintenance_quality"],
        "F_D_minus_acoustic_anomaly_score": [f for f in SENSOR_FEATURES if f != "acoustic_anomaly_score"],
        "G_D_minus_misalignment_imbalance": [f for f in SENSOR_FEATURES
                                             if f not in ("misalignment_index", "imbalance_index")],
    }
    res = {}
    for f in LATENT_FEATURES:  # single-feature drops from the full set
        sets[f"A_minus_{f}"] = [c for c in ALL_FEATURES if c != f]
    for name, feats in sets.items():
        print("leakage:", name)
        res[name] = {"n_features": len(feats), **cv_scores(rf(), train[feats], y, cv)}
    # how well can sensors reconstruct each latent variable? (R^2, grouped CV)
    from sklearn.ensemble import HistGradientBoostingRegressor
    recon = {}
    for f in LATENT_FEATURES:
        r = cross_validate(HistGradientBoostingRegressor(random_state=SEED), train[SENSOR_FEATURES],
                           train[f], cv=cv, scoring="r2")
        recon[f] = r["test_score"].mean()
    corr = train[ALL_FEATURES + [TARGET]].assign(
        **{TARGET: enc.transform(train[TARGET])}).corr()[LATENT_FEATURES].drop(LATENT_FEATURES)
    save("leakage", {"cv": "GroupKFold(4) on cycle_id, training cycles only, RF(200 trees, legacy params)",
                     "results": res, "latent_reconstructable_from_sensors_r2": recon,
                     "max_abs_corr_latent_vs_sensor": corr.abs().max().to_dict()})


def stage_cv():
    """Does row-level CV over-estimate vs. grouped CV?"""
    train, enc = load()
    y = enc.transform(train[TARGET])
    res = {}
    for feats_name, feats in [("all_28", ALL_FEATURES), ("sensors_only", SENSOR_FEATURES)]:
        for cv_name, cv in [
            ("row_level_StratifiedKFold(4, shuffle)", StratifiedKFold(4, shuffle=True, random_state=SEED)),
            ("GroupKFold(4) on cycle_id", group_cv(train)),
            ("GroupKFold(4) on motor_id (unseen motor type)",
             list(GroupKFold(4).split(train, groups=train["motor_id"]))),
        ]:
            print("cv:", feats_name, cv_name)
            res[f"{feats_name} | {cv_name}"] = cv_scores(rf(), train[feats], y, cv)
    save("cv", res)


# --------------------------------------------------------------------------
def feature_set():
    """Feature set chosen after the leakage stage (see report)."""
    return json.loads((OUT / "decision.json").read_text())["features"]


def stage_tune(n_iter=24):
    train, enc = load()
    y = enc.transform(train[TARGET])
    feats = feature_set()
    cv = group_cv(train)
    space = {
        "max_depth": [None, 12, 16, 20, 28],
        "min_samples_leaf": [1, 2, 5, 10, 20, 40],
        "min_samples_split": [2, 5, 10, 20],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "bootstrap": [True, False],
    }
    trials = [LEGACY_RF | {"bootstrap": True}] + list(ParameterSampler(space, n_iter, random_state=SEED))
    res = []
    for i, p in enumerate(trials):
        p = {k: p[k] for k in space}
        print(f"tune {i + 1}/{len(trials)}: {p}")
        res.append({"params": p, **cv_scores(rf(n_estimators=200, **p), train[feats], y, cv)})
    res.sort(key=lambda r: -r["f1_macro"]["mean"])
    save("tune", {"features": feats, "selection_metric": "f1_macro (GroupKFold(4) on cycle_id)",
                  "trials": res})


def best_rf_params():
    best = json.loads((OUT / "tune.json").read_text())["trials"][0]["params"]
    return best


def stage_trees():
    """n_estimators curve for the tuned config (OOB is unavailable if bootstrap=False)."""
    train, enc = load()
    y = enc.transform(train[TARGET])
    feats, p = feature_set(), best_rf_params()
    cv = group_cv(train)
    save("trees", {str(n): cv_scores(rf(**{**p, "n_estimators": n}), train[feats], y, cv)
                   for n in [50, 100, 200, 300, 500]})


def stage_compare():
    train, enc = load()
    y = enc.transform(train[TARGET])
    classes = list(enc.classes_)
    feats = feature_set()
    cv = group_cv(train)
    p = best_rf_params()
    n_trees = json.loads((OUT / "decision.json").read_text()).get("n_estimators", 300)
    models = {
        "RandomForest (legacy params)": RandomForestClassifier(**LEGACY_RF, random_state=SEED, n_jobs=-1),
        "RandomForest (tuned)": rf(**{**p, "n_estimators": n_trees}),
        "ExtraTrees (tuned RF params)": ExtraTreesClassifier(**{**p, "n_estimators": n_trees},
                                                             random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "HistGradientBoosting (balanced)": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED),
        "GradientBoosting (100 trees, subsample 0.5)": GradientBoostingClassifier(
            n_estimators=100, subsample=0.5, random_state=SEED),
        "LogisticRegression (scaled, reference)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "LogisticRegression (scaled, balanced, reference)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    }
    res = {}
    import io
    import joblib
    for name, m in models.items():
        print("compare:", name)
        t = time.perf_counter()
        pred = cross_val_predict(m, train[feats], y, cv=cv)
        cv_s = time.perf_counter() - t
        metrics = classification_metrics(y, pred, classes)
        # fit once on 3/4 of training cycles to time fit / inference / size
        tr, va = cv[0]
        t = time.perf_counter()
        m.fit(train[feats].iloc[tr], y[tr])
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        m.predict_proba(train[feats].iloc[va])
        inf_s = time.perf_counter() - t
        buf = io.BytesIO()
        joblib.dump(m, buf, compress=3)
        res[name] = {**metrics, "cv_wall_s": cv_s, "fit_s": fit_s,
                     "inference_ms_per_1k_rows": inf_s / len(va) * 1e6,
                     "size_mb_compressed": buf.tell() / 1e6}
    save("compare", {"features": feats, "cv": "out-of-fold predictions, GroupKFold(4) on cycle_id",
                     "results": res})


def ece(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def reliability(prob, y, bins=10):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "mean_confidence": conf[m].mean(), "accuracy": (pred[m] == y[m]).mean()})
    return rows


def stage_calibration():
    train, enc = load()
    y = enc.transform(train[TARGET])
    feats = feature_set()
    dec = json.loads((OUT / "decision.json").read_text())
    base = rf(**{**best_rf_params(), "n_estimators": dec.get("n_estimators", 300)})
    cv = group_cv(train)
    res = {}
    # Outer grouped CV; calibrators use inner grouped CV on the outer-train cycles only.
    for name, make in {
        "uncalibrated": lambda: base,
        "sigmoid": lambda: CalibratedClassifierCV(base, method="sigmoid", cv=3, ensemble=False),
        "isotonic": lambda: CalibratedClassifierCV(base, method="isotonic", cv=3, ensemble=False),
    }.items():
        print("calibration:", name)
        prob = np.zeros((len(y), len(enc.classes_)))
        for tr, va in cv:
            m = make()
            if name == "uncalibrated":
                m.fit(train[feats].iloc[tr], y[tr])
            else:
                inner = list(GroupKFold(3).split(tr, groups=train[GROUP_COLUMN].iloc[tr]))
                m.set_params(cv=inner).fit(train[feats].iloc[tr], y[tr])
            prob[va] = m.predict_proba(train[feats].iloc[va])
        onehot = np.eye(len(enc.classes_))[y]
        res[name] = {
            "log_loss": log_loss(y, prob),
            "brier_multiclass": float(((prob - onehot) ** 2).sum(1).mean()),
            "brier_top_label": brier_score_loss((prob.argmax(1) == y).astype(int), prob.max(1)),
            "ece_top_label": ece(prob, y),
            "reliability": reliability(prob, y),
            **{k: v for k, v in classification_metrics(y, prob.argmax(1), list(enc.classes_)).items()
               if k in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "per_class")},
        }
    save("calibration", res)


def stage_features():
    train, enc = load()
    y = enc.transform(train[TARGET])
    feats = feature_set()
    cv = group_cv(train)
    p = {**best_rf_params(), "n_estimators": 200}
    tr, va = cv[0]
    m = rf(**p).fit(train[feats].iloc[tr], y[tr])
    perm = permutation_importance(m, train[feats].iloc[va], y[va], scoring="f1_macro",
                                  n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": feats, "mdi": m.feature_importances_,
                        "perm_f1_macro_mean": perm.importances_mean,
                        "perm_f1_macro_std": perm.importances_std}).sort_values("perm_f1_macro_mean",
                                                                                 ascending=False)
    corr = train[feats].corr().abs()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:]
             if corr.loc[a, b] > 0.9]
    low = imp[imp.perm_f1_macro_mean <= 0]["feature"].tolist()
    # redundant = second member of each >0.9 pair with the lower permutation importance
    pi = imp.set_index("feature").perm_f1_macro_mean
    redundant = sorted({b if pi[a] >= pi[b] else a for a, b, _ in pairs})
    variants = {
        "all": feats,
        "drop_nonpositive_permutation": [f for f in feats if f not in low],
        "drop_redundant_corr>0.9": [f for f in feats if f not in redundant],
        "top_12_permutation": imp.feature.head(12).tolist(),
    }
    res = {k: {"features": v, "n": len(v), **cv_scores(rf(**p), train[v], y, cv)}
           for k, v in variants.items()}
    save("features", {"importance": imp.to_dict("records"),
                      "high_corr_pairs_gt_0.9": pairs, "variants": res})


STAGES = {f.__name__[6:]: f for f in [stage_ceiling, stage_leakage, stage_cv, stage_tune,
                                     stage_trees, stage_compare, stage_calibration, stage_features]}

if __name__ == "__main__":
    for s in sys.argv[1:]:
        STAGES[s]()
