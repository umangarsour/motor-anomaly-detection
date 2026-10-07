# Refactor and Optimization Report

**Project:** Industrial Motor Health Monitoring & Predictive Maintenance System
**Date:** 2026-10-07 · **Environment:** Windows 11, Python 3.13.7, scikit-learn 1.9.0, 12 CPU cores

Every number in this report comes from a script in this repository, and its raw output is saved next
to this file:

| Source | Script |
|---|---|
| `reports/baseline/` | `python -m src.evaluation.baseline` (original artifacts) |
| `reports/experiments/*.json` | `python -m src.experiments.run_experiments <stage>` (training cycles only) |
| `models/model_metadata.json`, `reports/model_metrics.csv` | `python -m src.classification.fault_classifier` |
| `reports/before_after.md` | `python -m src.evaluation.before_after` |

---

## Summary

1. **The original 82.8% accuracy was roughly honest, but it hid a broken minority class.** Bearing_Wear
   had recall and F1 of **0.00**. Overheating had F1 0.30, Misalignment 0.17.
2. **The labels contain irreducible noise.** The generator picks each label as `argmax(scores + N(0, 5))`.
   Applying its exact rule to the exact features gives **≈ 84.0%** expected accuracy. No model can
   do much better, and the rare classes are mostly noise.
3. **No classic target leakage was found.** No feature is computed from the label, and row-level
   splitting did not inflate scores. `health_score` was removed from the model inputs at zero cost in
   validation. The four degradation indices stay, documented as simulator-only indicators.
4. **The final model trades 0.5 points of accuracy for +2.9 balanced accuracy and +0.9 macro F1.**
   Minority recall rose: Bearing_Wear 0.07 → 0.18, Overheating 0.41 → 0.49, Misalignment 0.20 → 0.25.
   Comparison is on the same split.
5. **Engineering:**
   - Model size 678 MB → 34 MB; single-row latency 52 → 29 ms; model load 2.7 s → 0.4 s.
   - CSV batch prediction is vectorised: 100k rows take 1.9 s instead of about 79 min, with identical
     outputs.
   - The pipeline runs in the correct order with one command.
   - 28 automated tests.
6. **Dashboard bugs fixed:**
   - Health scores were displayed as z-scores.
   - Motor trends zig-zagged across 5 interleaved cycles.
   - 21.6% of records were labelled "Slightly Abnormal" by fixed thresholds.
   - The README's health bands contradicted the data's severity labels.

---

## Phase 0: Audit (before any change)

> Note: the session started in `Desktop/motor-anomaly-system`, an older, simpler project: 1 motor,
> 4 features, Isolation Forest only. The brief described `Desktop/final_motor_project`. With your
> confirmation, all work was done in `final_motor_project`. `motor-anomaly-system` was not modified.
> **Both local repos point to the same GitHub remote with unrelated histories** (see Phase 19).

**A. Architecture.**
- Script-style pipeline: `src/` scripts run top to bottom, one per step.
- Streamlit multipage app: `app.py` landing page plus `pages/1-5`.
- Shared helpers in `dashboard/utils/`.

**B. Data pipeline.**
- Digital twin v5_10 → raw CSV (100,000 × 34).
- `preprocess_dataset.py` validates (no NaN or duplicates), label-encodes fault and severity, and
  writes the clean CSV.
- `scale_dataset.py` fits a StandardScaler on **all** rows and writes the scaled CSV.

**C. Features.**
- 28 numeric columns, all used: operating conditions, electrical, mechanical, thermal, vibration,
  acoustic, 4 degradation indices and `health_score`.
- No separate feature-engineering step; every feature comes from the generator.

**D. Random Forest.**
- 500 trees, depth 20, min_samples_split 5, min_samples_leaf 2, `sqrt`, `class_weight="balanced"`,
  OOB on.
- Row-level stratified 80/20 split, 5-fold CV on the training split (accuracy only).

**E. Isolation Forest.**
- 300 trees, contamination 0.05, fit on all 100k scaled rows (no hold-out).
- Exports `anomaly_results.csv` with **scaled** feature values.

**F. Prediction.**
- `predict_motor(dict)` runs RF on unscaled inputs and IF on scaled inputs.
- Risk bands are fixed at ±0.05 on `decision_function`, with emoji in the labels.
- The CSV page called it once per row in a Python loop.

**G. Streamlit.**
- Each page calls `apply_theme()`, which calls `set_page_config` inside a try/except.
- Loaders are cached correctly: `cache_data` for frames, `cache_resource` for models.
- Charts styled per call; inconsistent colours.

**H. Dependencies.**
- numpy 2.4.6, pandas 3.0.3, scipy, scikit-learn 1.9.0, joblib, matplotlib, seaborn, plotly,
  streamlit 1.58.0.
- **The pickles were saved with scikit-learn 1.8.0**, so they raised `InconsistentVersionWarning`.
- Refitting the original config under 1.9.0 reproduced only 94.2% of the original test predictions.

**I. Duplicate and dead code.**
- 9 empty placeholder packages under `src/` (only `__init__.py`).
- Empty files: `src/preprocessing/feature_selection.py`, `tests/test_digital_twin.py`,
  `docs/project_architecture.md`, `data/raw/generated/motor_telemetry_clean.csv`.
- Unused artifacts: `standard_scaler.pkl` (24 features, from an archived script),
  `severity_label_encoder.pkl` loader, `prediction_probabilities.csv` loader.
- `data/results/` duplicates old reports.
- `scratch/` held ad-hoc checks.

**J. Bugs.**
- **J1** Fleet Overview "Latest Detected Anomalies", the Anomaly Monitoring "Health vs Anomaly"
  scatter, and the top-anomalies table showed `health_score` (and all sensors) **z-scored**, with a
  range of -2.2 to 1.2, labelled as health 0-100.
- **J2** Motor Diagnostics sorted a motor's 10,000 rows by `timestamp`, but each of the 5 cycles
  restarts at 0. Every trend chart interleaved 5 cycles.
- **J3** Pipeline order was broken. `scale_dataset.py` loads `selected_features.pkl`, which only the
  later classifier step writes, so the README order fails on a fresh clone.
- **J4** The generator wrote to a CWD-relative path.
- **J5** The landing page and Fleet Overview hard-coded "82.79%".
- **J6** Fault Analytics hard-coded "80,000 / 20,000 / 28 features".
- **J7** The README health bands (70/40) disagreed with the severity labels in the data (75/50/25).
- **J8** The Live Prediction "System Status" mapped every anomaly to "Critical".

**K. Leakage.** See Phase 2.

**L. Imbalance.** Overload 40.1% vs Bearing_Wear 0.20%, a 200:1 ratio.

**M. Split.** Row-level, although rows are time series grouped by motor and cycle. See Phase 3.

**N. Scaling.**
- The RF used unscaled inputs, which is correct.
- The IF used a scaler fit on all rows (including test). It was also **redundant**: IF splits are
  invariant to per-feature affine transforms. Verified: identical labels, max score difference 0.001.

**O. Hyperparameters.** Hand-set. No search was recorded.

**P. Evaluation.**
- Accuracy, OOB and CV accuracy only.
- No balanced accuracy or macro-F1 reasoning.
- The weakest class (F1 0.00) went unnoticed in the UI.

**Q. UI/UX.**
- Very long single-column pages and emoji everywhere.
- Status shown by colour only.
- 28 unlabelled number inputs with no units.
- No units on charts.
- "Anomaly Index" presented without saying what it is.
- Gauge bands inconsistent with the data.

**R. Performance.**
- A 678 MB model took 2.7 s to load and 134 ms per single prediction.
- CSV prediction looped per row (est. 79 min for 100k rows).
- The dashboard read two 55-60 MB CSVs.

**S. Reproducibility.**
- Seeds were fixed, and the generator is bit-for-bit deterministic (verified by regenerating).
- Problems: broken step order (J3), CWD-relative output (J4), and model and library version mismatch
  (H).

**T. Security and privacy.**
- No secrets, credentials or personal data.
- Pickles are loaded only from the repo's own `models/`. Note that pickle loading executes code, so
  do not load untrusted pickles.
- CSV uploads are parsed with pandas and never executed.
- Uploads are limited to 200 MB (`.streamlit/config.toml`).

---

## Phase 1: Baseline (`reports/baseline/baseline_report.md`)

The original pickled models were evaluated on the exact original split (row-level stratified 80/20,
seed 42): 80,000 train / 20,000 test, 28 features.

| Metric | Original RF |
|---|---|
| Accuracy | 0.8279 |
| Balanced accuracy | 0.5393 |
| Macro precision / recall | 0.5677 / 0.5393 |
| Macro F1 | 0.5447 |
| Weighted F1 | 0.8230 |
| OOB | 0.8308 |
| CV (5-fold, train) accuracy / macro F1 / bal. acc. | 0.8218 / 0.5610 / 0.5756 |
| Model file | 678 MB |
| Single-row predict_proba (median) | 134 ms |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bearing_Wear | 0.000 | 0.000 | 0.000 | 40 |
| Electrical_Fault | 0.753 | 0.830 | 0.789 | 4,872 |
| Misalignment | 0.216 | 0.143 | 0.172 | 56 |
| Multi_Fault | 0.756 | 0.852 | 0.801 | 1,720 |
| Normal | 0.882 | 0.872 | 0.877 | 4,587 |
| Overheating | 0.483 | 0.220 | 0.302 | 701 |
| Overload | 0.885 | 0.858 | 0.871 | 8,024 |

The baseline run's training time (75.8 s) was distorted by another job running in parallel. The
quiet-machine timing of the original configuration is in Phase 17.

**Isolation Forest (original):** contamination 0.05 → 5,000 anomalies (5.00%). Decision function:
mean 0.064, sd 0.032, range -0.133 to 0.129. Original risk bands: Normal 73.4%, Slightly Abnormal
21.6%, Moderate 4.6%, High 0.4%. Proxy ROC-AUC on test rows (in-sample, because the IF was fit on
them): fault ≠ Normal 0.554, health < 50 0.856, health < 25 0.943.

---

## Phase 2: Leakage audit (`reports/experiments/leakage.json`, `ceiling.json`)

**How the labels are made.** `generate_health_and_faults` computes six scores per row from:
- `bearing_wear_index`, `thermal_stress_index`, `electrical_stress_index`, `lubrication_index` (the
  hidden degradation states);
- `health_score` (itself 100 − a weighted sum of those states and `misalignment_index` + noise);
- vibration RMS, temperatures, power factor, load, current and efficiency.

It then adds independent N(0, 5) noise to each score and takes the argmax, with a Multi_Fault rule.
`severity` is a threshold on `health_score`.

**Theoretical ceiling.** I vectorised that rule, applied it to the exported features, and integrated
out the noise with 400 Monte Carlo draws per row. The Bayes-optimal classifier then reaches an
expected accuracy of **0.8398**. On test cycle 5 it scores: accuracy 0.8369, balanced accuracy 0.525,
macro F1 0.538, Bearing_Wear recall 0.00, Misalignment recall 0.05.

**Per-feature verdicts** (CV on training cycles, GroupKFold(4) by cycle, RF with the original
parameters and 200 trees):

| Feature set | n | Accuracy | Bal. acc. | Macro F1 |
|---|---|---|---|---|
| All 28 | 28 | 0.8218 | 0.5785 | 0.5642 |
| − health_score | 27 | 0.8206 | 0.5807 | **0.5652** |
| − bearing_wear_index | 27 | 0.8212 | 0.5738 | 0.5590 |
| − thermal_stress_index | 27 | 0.8147 | 0.5661 | 0.5505 |
| − electrical_stress_index | 27 | 0.8169 | 0.5695 | 0.5552 |
| − lubrication_index | 27 | 0.8207 | 0.5788 | 0.5637 |
| − all 4 degradation indices | 24 | 0.8099 | 0.5489 | 0.5373 |
| Sensors only (− all 5) | 23 | 0.8053 | 0.5427 | 0.5306 |
| Sensors only − maintenance_quality | 22 | 0.8051 | 0.5404 | 0.5281 |
| Sensors only − acoustic_anomaly_score | 22 | 0.8054 | 0.5451 | 0.5324 |
| Sensors only − misalignment/imbalance | 21 | 0.8054 | 0.5359 | 0.5217 |

How well a gradient-boosting model can rebuild each latent variable from the 23 sensor features
(grouped CV R²):

| Variable | R² |
|---|---|
| bearing_wear_index | 0.985 |
| thermal_stress_index | 0.979 |
| electrical_stress_index | 0.981 |
| lubrication_index | 0.973 |
| health_score | 0.973 |

| Feature | Verdict |
|---|---|
| `health_score` | **Removed from model inputs.** It is not computed from the label, but it enters the "Normal" score directly and defines `severity`. It is the most label-adjacent feature. Removing it costs nothing (macro F1 +0.001). It is still shown everywhere in the dashboard. |
| 4 degradation indices | **Kept, documented.** They are the label's *inputs*, not its outputs, and they are present in every record, so this is not target leakage. They are hidden simulator state that a real motor cannot measure, but they are 97-99% recoverable from sensors. Cost of removal: −1.2 accuracy, −2.7 macro F1 points. The sensors-only result above is the realistic estimate for a real deployment, and the README states it. Switching is one line: `CLF_FEATURES` in `src/config.py`. |
| `acoustic_anomaly_score` | Engineered from sound level + vibration RMS + noise; no label input. Kept. |
| `maintenance_quality`, `misalignment_index`, `imbalance_index` | Simulated operating and mechanical indicators, not label-derived. Kept. |
| `severity`, `fault_type_encoded`, IDs, `timestamp` | Excluded, as in the original. |

**Future information:** none. All features are same-timestamp values, and no feature uses
aggregates across time.

**Sanity check:** every model in this report scores *below* the 84% Bayes ceiling. A leaking feature
would let a model exceed it. `tests/test_pipeline.py::test_metrics_are_plausible` enforces this
permanently.

---

## Phase 3: Evaluation methodology (`reports/experiments/cv.json`)

| Features | CV scheme | Accuracy | Macro F1 | Bal. acc. |
|---|---|---|---|---|
| all 28 | row-level StratifiedKFold(4) | 0.8212 | 0.5664 | 0.5817 |
| all 28 | GroupKFold(4) by cycle | 0.8218 | 0.5642 | 0.5785 |
| all 28 | GroupKFold(4) by motor (unseen motor type) | 0.8221 | 0.5430 | 0.5516 |
| sensors only | row-level | 0.8052 | 0.5305 | 0.5438 |
| sensors only | by cycle | 0.8053 | 0.5306 | 0.5427 |
| sensors only | by motor | 0.8032 | 0.5157 | 0.5316 |

**Finding.** Row-level and cycle-grouped CV agree, so the original row split did *not* inflate
scores. The label noise is independent per row, so a neighbouring row's label carries no extra
information.

**Decision.** Use a cycle hold-out anyway, because it is cleaner and costs nothing:
- **Test:** cycle 5 of every motor (20,000 rows, 20%).
- **Training:** cycles 1-4.
- **Model selection:** GroupKFold(4) on `cycle_id` inside the training cycles.
- **Test set:** touched once, by `fault_classifier.py`.

All 10 motors (and so all 10 motor types) are in both train and test. The leave-motors-out row above
(−2 macro F1 points) is the stress test for a new, unseen motor type. Seeds are fixed (42)
everywhere.

---

## Phase 4: Random Forest optimisation (`tune.json`, `trees.json`)

- **Search:** 25 configurations (the original parameters plus 24 random draws), 200 trees each,
  GroupKFold(4) by cycle, selected on **macro F1**.
- **Search space:** max_depth {None, 12, 16, 20, 28}; min_samples_leaf {1, 2, 5, 10, 20, 40};
  min_samples_split {2, 5, 10, 20}; max_features {sqrt, log2, 0.3, 0.5};
  class_weight {None, balanced, balanced_subsample}; bootstrap {True, False}.

| Rank | Params | Macro F1 | Bal. acc. | Accuracy |
|---|---|---|---|---|
| 1 (**chosen**) | depth 16, leaf 10, split 10, max_features 0.5, balanced_subsample, bootstrap | **0.5724 ± 0.0064** | 0.6048 | 0.8174 |
| 2 | depth 16, leaf 10, split 20, max_features 0.3, balanced_subsample | 0.5721 | 0.6004 | 0.8174 |
| 5 | original parameters | 0.5652 | 0.5807 | 0.8206 |
| best accuracy | depth 28, leaf 5, max_features 0.5, **class_weight=None** | 0.5151 | 0.5095 | **0.8346** |

**Imbalance.** At 200:1, `class_weight=None` maximises accuracy (0.835) but drops macro F1 to 0.515,
because the rare classes are never predicted. `balanced_subsample` with min_samples_leaf 10 gives the
best macro F1. Very large leaves (40) push balanced accuracy to 0.67-0.70 but cut accuracy to
0.77-0.79, which over-predicts minorities.

SMOTE was not used. The minority labels are mostly noise (Phase 2), so synthetic oversampling would
only amplify noise. Class weighting already happens inside each fold.

**Tree count** (tuned params, macro F1):

| Trees | 50 | 100 | 200 | 300 | 500 |
|---|---|---|---|---|---|
| Macro F1 | 0.5707 | 0.5726 | 0.5724 | 0.5726 | 0.5736 |
| CV time (s) | 17 | 31 | 60 | 93 | 159 |

The curve plateaus from 100 trees. **200** was chosen: 500 is within noise and 2.6× slower.

---

## Phase 5: Feature analysis (`features.json`)

Permutation importance (drop in macro F1, fold-0 validation):

| Feature | Importance |
|---|---|
| electrical_stress_index | 0.106 |
| load_percentage | 0.063 |
| thermal_stress_index | 0.053 |
| current | 0.033 |
| motor_temperature | 0.029 |
| vibration_rms | 0.023 |
| power_factor | 0.022 |
| misalignment_index | 0.018 |
| … | … |
| lubrication_index | −0.001 (the only non-positive one) |

There are 33 feature pairs with |r| > 0.9. The degradation indices are 0.97-0.98 correlated with each
other, and power factor, kurtosis, dominant frequency and vibration amplitude correlate 0.90-0.96
with them.

| Variant | n | Macro F1 | Accuracy |
|---|---|---|---|
| All 27 | 27 | 0.5724 ± 0.0064 | 0.8174 |
| Drop non-positive permutation | 26 | 0.5744 ± 0.0061 | 0.8174 |
| Drop redundant (r > 0.9) | 16 | 0.5449 ± 0.0011 | 0.8013 |
| Top-12 by permutation | 12 | 0.5771 ± 0.0083 | 0.8134 |

**Decision: keep all 27.**
- Dropping the "redundant" features costs 2.7 points, so correlated features still carry signal.
- The other two variants are within one standard deviation.
- The top-12 ranking was computed on data inside the CV, so its small gain is selection-biased.

Test-set permutation importance for the final model is saved in `reports/feature_importance.csv` and
shown on the dashboard.

---

## Phase 6: Model comparison (`compare.json`)

Same features and folds throughout; out-of-fold predictions on the training cycles.

| Model | Acc. | Bal. acc. | Macro F1 | Wtd F1 | Fit (s) | ms / 1k rows | Size (MB) |
|---|---|---|---|---|---|---|---|
| **Random Forest (tuned)** | 0.8174 | 0.6060 | **0.5741** | 0.8219 | 15.0 | 6.5 | 24.9 |
| Random Forest (original params) | 0.8209 | 0.5805 | 0.5666 | 0.8221 | 12.8 | 16.6 | 80.5 |
| Extra Trees | 0.8078 | 0.6228 | 0.5691 | 0.8152 | 2.1 | 6.0 | 30.6 |
| HistGradientBoosting | 0.8292 | 0.5449 | 0.5487 | 0.8262 | 1.4 | 4.7 | 0.5 |
| HistGradientBoosting (balanced) | 0.7914 | 0.6554 | 0.5576 | 0.8059 | 6.8 | 3.7 | 0.4 |
| GradientBoosting (100, subsample 0.5) | 0.8314 | 0.5296 | 0.5369 | 0.8269 | 129.9 | 4.1 | 0.3 |
| Logistic regression (reference) | 0.8322 | 0.5236 | 0.5372 | 0.8256 | 2.4 | 0.5 | <0.1 |
| Logistic regression (balanced) | 0.7480 | 0.7239 | 0.5239 | 0.7804 | 2.0 | 0.5 | <0.1 |

(The "Size" column is a compressed in-memory dump of a model fit on 3/4 of the training cycles. The
saved production model trained on all training cycles is 34.2 MB.)

**Random Forest stays.** It has the best macro F1 and the best Bearing_Wear and Overheating F1. The
higher-accuracy models get there by predicting minority classes less. XGBoost and LightGBM were not
added: HistGradientBoosting is sklearn's equivalent and did not win, so a new dependency was not
justified.

Viva note: plain logistic regression reaches 83.2%, close to the ceiling, because the generator's
scoring rule is almost linear in the features.

---

## Phase 7: Calibration (`calibration.json`)

Outer GroupKFold(4); the calibrators were fit with an inner GroupKFold(3) on the outer-train cycles
only.

| | Log loss | Brier (multi) | Top-label ECE | Macro F1 | Bal. acc. | Acc. |
|---|---|---|---|---|---|---|
| Uncalibrated | 0.480 | 0.260 | 0.037 | **0.574** | **0.606** | 0.817 |
| Sigmoid | 0.485 | 0.248 | 0.047 | 0.571 | 0.569 | 0.827 |
| Isotonic | **0.427** | **0.238** | **0.009** | 0.550 | 0.543 | 0.831 |

Isotonic calibration produces almost perfect probabilities. But it re-learns the true class
frequencies and undoes the class balancing: macro F1 −2.4, balanced accuracy −6.3. **It was not
adopted.**

The forest's vote share is already fairly reliable. On the test cycle, ECE is 0.031, and it is mildly
**under-confident**: stated 0.55 → 0.60 observed, 0.75 → 0.82. The dashboard therefore calls the
value **"Model Confidence"**, says it is uncalibrated, and shows the test-cycle reliability diagram.
"Prediction Probability" is never used.

---

## Phase 8: Isolation Forest

| Aspect | Before | After |
|---|---|---|
| Fit data | All 100k rows, incl. test | Training cycles only (80k) |
| Scaling | StandardScaler fit on all rows | None. Verified invariant (identical labels); removes the 59 MB scaled CSV and the J3 bug. |
| Features | 28 incl. `health_score` | Same 27 as the classifier, so `health_score` is not used to then be evaluated against `severity` (which is derived from it) |
| n_estimators / contamination / max_samples / max_features | 300 / 0.05 / auto / 1.0 | Unchanged |
| Risk bands | Fixed ±0.05 thresholds; 21.6% "Slightly Abnormal" | Quantiles of training scores, stored in metadata: High < q01 = −0.033, Moderate < 0, Slightly Abnormal < q15 = 0.034 |
| Labels | Emoji strings | "Normal", "Slightly Abnormal", "Moderate Risk", "High Risk" (same names, no emoji); icons and text in the UI |

**Contamination.** There are no anomaly labels, so contamination is an *alert budget*, not a tuned
value. It was kept at 5%; tuning it against the proxy targets would be optimising against
`severity`. Training flag rate: 5.00%. Test cycle: 3.69%.

**Interpretation.** `decision_function` is shown as a "decision score": below 0 = flagged. It is not
a probability or a percentage.

**Test cycle (out-of-sample proxy evaluation):**

| Proxy target | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|
| fault_type ≠ Normal | 0.580 | 0.836 | 0.984 | 0.047 |
| health < 50 | 0.850 | 0.749 | 0.980 | 0.118 |
| health < 25 | 0.927 | 0.709 | 0.903 | 0.253 |

The original's 0.943 for health < 25 was in-sample and used `health_score` as an input. The new
bands are monotone in health on the test cycle:

| Band | Share | Mean health | Critical share |
|---|---|---|---|
| Normal | 86.7% | 70.5 | 5% |
| Slightly Abnormal | 9.6% | 32.6 | 54% |
| Moderate Risk | 3.0% | 12.9 | 89% |
| High Risk | 0.7% | 7.8 | 96% |

---

## Phase 9: Health score audit

- **Formula** (generator):
  `health = clip(100 − clip(0.30·bearing + 0.25·thermal + 0.20·electrical + 0.15·(100 − lubrication) + 0.10·misalignment, 0, 100) + N(0, 4), 0, 100)`.
- **What it is:** an engineered indicator, not an ML prediction. Its inputs are the four hidden
  degradation states plus `misalignment_index`.
- **Labels:** it does not use the fault label. It does feed the label: it is 40% of the "Normal"
  score, and `severity` is a threshold on it.
- **Correlation:** |r| ≈ 0.98 with each degradation index.

**No formula change.** No technical defect was found. The only change is that it is no longer a model
input (Phase 2). The gauge and diagnostics bands now follow the data's severity thresholds (75/50/25)
instead of the README's inconsistent 70/40.

---

## Phases 10-11: Prediction pipeline and batch prediction

`dashboard/utils/prediction.py`:

- **One vectorised path**, `predict_batch(df)`. `predict_motor(dict)` is a one-row wrapper, so manual
  and CSV predictions can never diverge.
- **Feature order** comes from `model_metadata.json` and is enforced by indexing. Column order in the
  CSV does not matter, and extra columns are kept but ignored. Both are tested.
- **Validation** raises `PredictionInputError` with a readable message:
  - missing features: "Your file is missing 3 required features: …";
  - NaN, empty, non-numeric or ±inf: "N row(s) have empty, non-numeric or infinite values in: …";
  - empty file.
- **Unchanged output keys:** `fault`, `confidence` (0-100), `anomaly` ("Anomaly"/"Normal"),
  `anomaly_score` (3 decimals), `risk_level`. `predict_motor` also returns `probabilities`.
- **Models and metadata** are loaded once (`st.cache_resource` / `st.cache_data`). A missing artifact
  shows "Required artifact not found … Run the training pipeline first" plus the command, not a
  traceback.
- **Version safety:** the metadata records the scikit-learn version, and `requirements.txt` pins the
  same one.

**Batch benchmark** (same models; old = loop over `predict_motor`):

| Rows | Vectorised | Old loop | Identical outputs |
|---|---|---|---|
| 1 | 0.05 s | 0.05 s | yes |
| 10 | 0.05 s | 0.5 s | yes |
| 1,000 | 0.08 s | ~48 s | yes |
| 100,000 | 1.9 s | ~79 min (extrapolated from 200 rows) | yes |

---

## Phase 12: Streamlit performance

- **One dataset:** a parquet file with clean telemetry in original units plus IF score, label, risk
  level and split. It replaces the clean CSV + the 60 MB z-scored `anomaly_results.csv`. Load time
  0.80 s → 0.07 s.
- **Classifier load:** 2.7 s → 0.4 s.
- **Cold render after cache warm-up:** 0.03-0.4 s per page (the first page of a session ~2.4 s, mostly
  imports).
- **`session_state`** holds the selected motor, cycle and sample, the manual prediction result, and
  the CSV results. CSV results are keyed by upload `file_id`, so a new file invalidates them. Download
  buttons use `on_click="ignore"`, so results survive a download.
- **No stale caching:** all cached loaders read immutable artifacts; restart the app after retraining.

---

## Phases 13-15: UI/UX, accessibility, errors

- **Theme and layout:**
  - Native dark theme in `.streamlit/config.toml`, replacing the per-page CSS.
  - `st.navigation` with sections (Monitoring / Models / Project); Fleet Overview is the landing page.
  - One Plotly template for all charts.
  - Bordered KPI cards with help tooltips.
- **Fleet Overview:**
  - KPIs: fleet health, motors, anomalous samples, critical samples.
  - Gauge (bands = severity thresholds).
  - Anomaly-rate trend across the operating cycle.
  - Distributions in tabs.
  - Motors ranked by risk, with progress bars.
- **Motor Diagnostics:**
  - Motor + cycle + sample selectors (fixes J2).
  - Status cards with severity, risk and above-fleet-p95/p99 badges, plus a warning banner.
  - Trend tabs with units, threshold lines and anomaly markers.
  - Filterable, rounded history table with friendly column names and a CSV download.
- **Anomaly Monitoring:**
  - Filters for risk level, fault, severity and motor, applied to everything on the page.
  - KPIs; score histogram with the threshold lines.
  - Score vs health (now in real units, fixes J1).
  - Flag *rate* by fault, most anomalous motors.
  - Top-N table with highlighted risk and a download.
  - "How to read the score" section with the proxy evaluation.
- **Fault Analytics:**
  - Accuracy, balanced accuracy, macro F1, weighted F1 (each with its CV value) and training time.
  - Ceiling explanation.
  - Per-class chart and table sorted weakest-first, with Weak/Fair/Good status.
  - Confusion matrix with a normalise toggle.
  - Permutation and MDI importance.
  - Reliability diagram.
  - Class distribution on a log scale.
- **Live Prediction:**
  - Inputs in 7 tabs, all labelled with units; tooltips give the description, typical 1st-99th
    percentile range and training range.
  - Defaults are training medians.
  - "Start from" a real held-out example of any class; reset to median.
  - Out-of-range warning.
  - Result cards: fault, Model Confidence, anomaly status, risk level, anomaly score.
  - Top-3 class bar.
  - A "Why this result?" panel lists inputs outside the fleet's 5th-95th percentile and flags close
    top-two classes. It is labelled descriptive, not causal.
  - CSV tab: template download, readable validation errors, summary KPIs, flagged-only filter,
    download.
- **About page** with methodology and limitations. All hard-coded metrics were replaced by values read
  from metadata (fixes J5, J6).
- **Accessibility:**
  - Okabe-Ito colour-blind-safe fault palette.
  - Every status badge carries text plus a distinct icon; colour is never the only signal.
  - Light text (#E2E8F0) on #0F172A.
  - Units on every axis and input.
  - Responsive: on a phone the sidebar auto-collapses and columns stack.
- **Emoji:** removed from labels, risk strings and titles.

---

## Phase 16: Tests (`python -m pytest -q` → **28 passed**)

| Area | Tests |
|---|---|
| Risk banding (pure logic) | boundaries map to the 4 levels |
| Split | test = cycle 5 only, no overlap |
| Data loading | 100k rows; health 0-100 and RPM in range (J1 regression); anomaly ⇔ score < 0; valid risk levels |
| Feature / metadata loading | feature lists valid and unique, 7 sorted classes, reference stats for all 28 |
| Model loading | `n_features_in_` matches metadata; class indices align |
| Metrics plausibility | accuracy between 0.70 and ceiling + 0.02 (leakage canary); macro F1 ≥ 0.45 |
| `predict_motor` schema | expected keys; fault ∈ classes; confidence ∈ [0, 100] and = max probability; anomaly ∈ {Anomaly, Normal}; score finite float; risk ∈ levels; probabilities sum to 100 |
| Batch | 1 / 10 / 1,000 rows equal the single-row results; matches stored scores and risk on 500 rows |
| Robustness | shuffled columns + extra columns give identical output; missing 3 features → clear message; NaN / inf / "abc" / "" rejected; empty input rejected |
| Explanation | an extreme vibration value is reported first as "above 99th percentile" |
| Dashboard (AppTest) | all 6 pages + the entrypoint render without exception or error; diagnostics with another motor and sample; Live Prediction end to end (load example → predict → 5 result cards) |

---

## Phase 17: Regression, before vs after (`reports/before_after.md`)

Same split (train cycles 1-4, test cycle 5), quiet machine.

| Metric | Original config (28 feat., 500 trees) | Final model | Change |
|---|---|---|---|
| Accuracy | 0.8140 | 0.8087 | −0.0053 |
| Balanced accuracy | 0.5755 | 0.6042 | **+0.0288** |
| Macro F1 | 0.5610 | 0.5699 | **+0.0090** |
| Weighted F1 | 0.8165 | 0.8154 | −0.0010 |
| Macro precision / recall | 0.5540 / 0.5755 | 0.5493 / 0.6042 | −0.0047 / +0.0288 |
| CV macro F1 | 0.5632 | 0.5724 | +0.0092 |
| CV accuracy | 0.8219 | 0.8174 | −0.0046 |
| OOB | 0.8212 | 0.8152 | |
| Training time | 27.2 s | 24.9 s | |
| Batch inference, 20k rows | 0.26 s | 0.10 s | |
| Single-row latency (median) | 51.9 ms | 29.0 ms | |
| Model size | 529 MB (uncompressed, as originally saved) | 34.2 MB (compressed) | |

| Class | Recall before → after | F1 before → after |
|---|---|---|
| Bearing_Wear | 0.068 → **0.182** | 0.091 → **0.160** |
| Misalignment | 0.197 → **0.246** | 0.182 → 0.176 |
| Overheating | 0.405 → **0.485** | 0.369 → 0.371 |
| Multi_Fault | 0.860 → 0.851 | 0.769 → 0.773 |
| Normal | 0.879 → 0.870 | 0.871 → 0.870 |
| Electrical_Fault | 0.798 → 0.774 | 0.785 → 0.780 |
| Overload | 0.822 → 0.822 | 0.860 → 0.860 |

On this split the original configuration scores 81.4% accuracy, not the 82.8% it scored on its own
row split. That reflects the different test rows, not leakage: CV accuracy is 0.822 in both schemes.

**Is this an improvement?**
- The primary objectives (macro F1, balanced accuracy, minority recall) improved in both CV and test.
- Every minority class gained recall.
- Majority classes lost at most 2.4 recall points (Electrical_Fault).
- Headline accuracy fell 0.5 points. That is a deliberate trade, not hidden.
- Weighted F1 is unchanged within noise.

Against the theoretical rule on the same test cycle (0.837 accuracy, 0.538 macro F1), the final model
gives up 2.8 accuracy points to gain 3.2 macro-F1 points.

---

## Phases 18-19: Reproducibility and GitHub

- `python -m src.pipeline` rebuilds everything from scratch in **3 min 26 s**, and the generated
  dataset is identical to the committed one.
- **Paths:** all relative to the project root (`src/config.py`); the generator path was fixed.
- **Seeds:** fixed (42).
- **Versions:** library versions are pinned (now including `pyarrow` and `pytest`), and the models
  were retrained under the pinned scikit-learn 1.9.0.
- **Metadata:** human-readable JSON with features, classes, params, metrics, split, version and
  timestamp.
- **`.gitignore`:** now excludes `data/raw/`, `data/processed/*.csv`, `data/results/`,
  `models/legacy/` and `reports/legacy/`.
- **Committed artifacts:** small, so a clone works immediately: model 34 MB, detector 1 MB, scored
  parquet 28 MB (all under GitHub's 50 MB warning size).

**⚠ Outstanding (needs your decision): git history.**
- The only local commit `8a3e8f8` already contains `models/fault_classifier.pkl` (678 MB, over
  GitHub's 100 MB hard limit) and about 400 MB of CSVs. **It can never be pushed as is**, and
  `.gitignore` does not remove files from existing commits.
- The local `main` has no upstream, and the remote `umangarsour/motor-anomaly-detection` currently
  holds the *other* project's history (`95162f4` from `motor-anomaly-system`).
- Publishing therefore needs both:
  1. a fresh history, e.g. an orphan branch committing the current tree;
  2. a decision to replace the remote's history (force push) or use a new repository.
- Neither was done, because both are irreversible and outward-facing. Commands are in the hand-over
  message.

---

## Deliverables index

| # | Deliverable | Where |
|---|---|---|
| 1 | Summary of changes | Summary, Phases 0-19 |
| 2-4 | Files modified / deleted / added | Below |
| 5 | Final model configuration | Phase 4, `src/config.py`, `models/model_metadata.json` |
| 6 | Baseline metrics | Phase 1, `reports/baseline/` |
| 7-8 | Final metrics, before/after | Phase 17, `reports/before_after.md` |
| 9 | Feature set before/after | 28 → 27 (− `health_score`), Phases 2 and 5 |
| 10 | Leakage findings | Phase 2 |
| 11-12 | Tests and results | Phase 16 |
| 13-14 | UI/UX and performance | Phases 11-15 |
| 15 | Remaining issues | Below |
| 16 | Reproduction commands | README "Reproduce the results" |

### Files added

| Area | Files |
|---|---|
| Pipeline (`src/`) | `config.py`, `pipeline.py`, `anomaly_detection/risk.py` |
| Evaluation (`src/evaluation/`) | `metrics.py`, `baseline.py`, `before_after.py` |
| Experiments | `src/experiments/run_experiments.py` |
| Dashboard | `dashboard/utils/features.py`, `pages/6_About.py`, `.streamlit/config.toml` |
| Tests | `pytest.ini`, `tests/__init__.py`, `tests/test_pipeline.py`, `tests/test_dashboard.py` |
| Models | `models/model_metadata.json` |
| Data | `data/processed/motor_telemetry_scored.parquet` |
| Reports | `reports/baseline/*`, `reports/experiments/*.json`, `reports/before_after.{json,md}`, this report |

### Files modified

| Area | Files |
|---|---|
| App | `app.py` |
| Pages | `pages/1_Fleet_Overview.py` … `pages/5_Live_Prediction.py` |
| Dashboard utils | `dashboard/utils/load_data.py`, `prediction.py`, `plotly_theme.py`, `ui.py` |
| Pipeline (`src/`) | `classification/fault_classifier.py`, `anomaly_detection/isolation_forest.py`, `anomaly_detection/evaluate_isolation_forest.py`, `data_generation/motor_digital_twin_v5_10.py` (output path only) |
| Project files | `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_STRUCTURE.md`, `docs/architecture.md` (now pointers) |
| Regenerated outputs | the models and all report CSVs/plots |

### Files moved (preserved, git-ignored)

- `models/*.pkl` → `models/legacy/`: the original models, used by the baseline script.
- `reports/anomaly_results.csv`, `prediction_probabilities.csv` and the three root-level PNGs →
  `reports/legacy/`.

### Files deleted

All deleted files remain in git history (commit `8a3e8f8`). Nothing in the active code imported them.

| Path | Reason |
|---|---|
| `src/preprocessing/scale_dataset.py` | Superseded; would fail without the old pickles |
| `src/preprocessing/feature_selection.py`, `tests/test_digital_twin.py`, `docs/project_architecture.md`, `data/raw/generated/motor_telemetry_clean.csv` | Empty |
| 9 placeholder packages under `src/` | `dashboard`, `data_ingestion`, `diagnosis`, `fault_simulation`, `feature_engineering`, `health_monitoring`, `prediction`, `reporting`, `utils` held only an empty `__init__.py` |
| `scratch/` | Ad-hoc checks against the removed loaders; superseded by `tests/` |
| `models/checkpoints/`, `models/trained/`, `notebooks/`, `images/` | Empty directories |
| `data/processed/motor_telemetry_scaled.csv` | 59 MB; scaling step removed (Phase 8) |
| `data/raw/generated/motor_telemetry.csv`, `motor_telemetry_v5.csv` … `v5_5.csv` | Outputs of the archived simulators |
| `data/results/` | Stale copies of old reports |

---

## Remaining issues and limitations

1. **Synthetic data only.** Nothing has been validated on a real motor.
2. **Label noise sets the ceiling.** Bearing_Wear and Misalignment stay weak (F1 0.16 / 0.18), because
   the generator labels them almost only through noise. Improving them means changing the simulator,
   which this work deliberately did not do.
3. **Degradation indices are simulator state.** The sensors-only estimate is about 1.7 accuracy and 3
   macro-F1 points lower (CV).
4. **The anomaly detector has no ground truth.** Contamination is an alert budget.
5. **Confidence is uncalibrated** (ECE 0.031, mildly under-confident). The trade-off is documented in
   Phase 7.
6. **Git history must be rewritten before publishing** (Phase 19).
7. **Older code remains.** `archive/` holds earlier simulator and preprocessing versions, kept as you
   left them. The current pipeline does not use it.
