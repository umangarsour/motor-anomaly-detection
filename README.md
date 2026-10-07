# Industrial Motor Health Monitoring & Predictive Maintenance System

An end-to-end machine-learning system for **fault classification** and **anomaly detection** on a
simulated fleet of industrial induction motors, with a Streamlit monitoring dashboard.

- **Digital twin**: 10 motors × 5 run-to-failure cycles × 2,000 samples = 100,000 records
- **Random Forest** fault classifier (7 classes)
- **Isolation Forest** unsupervised anomaly detector with risk levels
- **Dashboard**: Fleet Overview, Motor Diagnostics, Anomaly Monitoring, Fault Analytics,
  Live Prediction (manual entry and CSV batch with download)

> All data is synthetic. Results show that the method works on simulated data, not how it would
> perform on a real plant. See [Limitations](#limitations).

---

## Results (held-out test data)

Test set = the last operating cycle of every motor (20,000 rows), never seen during training or
model selection. Full methodology: [`reports/refactor_and_optimization_report.md`](reports/refactor_and_optimization_report.md).

| Metric | Test (cycle 5) | Cross-validation (cycles 1-4, GroupKFold 4) |
|---|---|---|
| Accuracy | **80.9%** | 81.7% ± 0.3 |
| Balanced accuracy | **60.4%** | 60.5% ± 0.8 |
| Macro F1 | **57.0%** | 57.2% ± 0.6 |
| Weighted F1 | **81.5%** | 82.2% ± 0.3 |
| OOB score | 81.5% | |

| Class | Precision | Recall | F1 | Test samples |
|---|---|---|---|---|
| Normal | 0.869 | 0.870 | 0.870 | 4,614 |
| Overload | 0.901 | 0.822 | 0.860 | 8,126 |
| Electrical_Fault | 0.787 | 0.774 | 0.780 | 4,951 |
| Multi_Fault | 0.707 | 0.851 | 0.773 | 1,517 |
| Overheating | 0.300 | 0.485 | 0.371 | 687 |
| Misalignment | 0.138 | 0.246 | 0.176 | 61 |
| Bearing_Wear | 0.143 | 0.182 | 0.160 | 44 |

**Why accuracy is not higher.** The digital twin picks each label as the highest of six scores
*plus Gaussian noise*. Applying the generator's own rule to the exact features, with that noise
integrated out, gives the best achievable accuracy: **≈ 84.0%**. On the test cycle that rule scores
83.7% accuracy but only 53.8% macro F1. The rare classes are decided almost entirely by noise, so
**no model can predict them reliably from these features**. The classifier is tuned for macro F1 and
balanced accuracy, not raw accuracy, so it does not silently ignore minority classes.

**Anomaly detector.** There are no ground-truth anomaly labels. The detector is fit on the training
cycles and checked on the test cycle against proxy targets: ROC-AUC **0.93** for "health < 25"
(Critical severity), **0.85** for "health < 50", and **0.58** for "any fault label". It finds
degraded motors but is not a fault classifier.

---

## Architecture

```
src/data_generation/motor_digital_twin_v5_10.py   -> data/raw/generated/motor_telemetry_v5_10.csv
src/preprocessing/preprocess_dataset.py           -> data/processed/motor_telemetry_clean.csv (+ label encoders)
src/eda/eda_analysis.py                           -> reports/*.csv, reports/plots/*.png
src/classification/fault_classifier.py            -> models/fault_classifier.pkl, models/model_metadata.json
src/anomaly_detection/isolation_forest.py         -> models/isolation_forest.pkl,
                                                     data/processed/motor_telemetry_scored.parquet
app.py + pages/ + dashboard/utils/                -> Streamlit dashboard (reads the parquet + models)
```

`src/config.py` is the single source of truth for paths, the random seed (42), feature lists, model
parameters and the train/test split. `python -m src.pipeline` runs the five steps in order.

### Dataset

| | |
|---|---|
| Records | 100,000 (10 motors × 5 cycles × 2,000 samples) |
| Motor profiles | Normal, High_Load, High_RPM, Thermal_Stress, Electrical_Stress, Fast_Wear, Efficient, Heavy_Duty, Low_Maintenance, Aging |
| Fault classes | Overload 40.1%, Electrical_Fault 24.4%, Normal 22.9%, Multi_Fault 8.6%, Overheating 3.5%, Misalignment 0.28%, Bearing_Wear 0.20% |
| Severity (from health score) | Low ≥ 75, Medium ≥ 50, High ≥ 25, Critical < 25 |

### Features (27 model inputs)

| Group | Features |
|---|---|
| Operating conditions | load_percentage (%), ambient_temperature (°C), supply_voltage (V), maintenance_quality |
| Electrical | current (A), voltage (V), active_power (kW), power_factor, efficiency_index |
| Mechanical | rpm, imbalance_index, misalignment_index |
| Thermal | motor_temperature (°C), bearing_temperature (°C) |
| Vibration | vibration_amplitude, vibration_rms, vibration_peak, crest_factor, kurtosis (simulated units) |
| Acoustic | sound_level_db (dB), dominant_frequency (Hz), spectral_energy, acoustic_anomaly_score |
| Condition indicators | bearing_wear_index, thermal_stress_index, electrical_stress_index, lubrication_index |

`health_score` is **not** a model input. It is an engineered indicator (100 minus a weighted sum of the
hidden degradation states, plus noise) that feeds the generator's label rule and defines severity.
Removing it cost nothing in cross-validation. The four condition indicators are the simulator's
hidden degradation states, and a real motor cannot measure them directly. A sensors-only model
(23 features) scores about 1.7 points lower accuracy and 3 points lower macro F1 in cross-validation.

### Models

| | Random Forest classifier | Isolation Forest detector |
|---|---|---|
| Parameters | 200 trees, max_depth 16, min_samples_leaf 10, min_samples_split 10, max_features 0.5, class_weight balanced_subsample | 300 trees, contamination 0.05 |
| Selected by | Macro F1, GroupKFold(4) on training cycles, 25-config random search | Alert budget (no labels to tune against) |
| Size / train time | 34 MB / 24 s | 1 MB / 1.4 s |
| Output | fault class + "Model Confidence" (uncalibrated vote share, ECE 0.03) | decision score (< 0 = flagged) + risk level |

Compared against Extra Trees, HistGradientBoosting, GradientBoosting and logistic regression (same
cross-validation). The Random Forest had the best macro F1. The boosting models and logistic
regression reach ~83% accuracy by rarely predicting the minority classes.

**Risk levels** come from quantiles of the training decision scores: *High Risk* (most anomalous 1%),
*Moderate Risk* (the rest of the flagged 5%), *Slightly Abnormal* (next 10%, not flagged), *Normal*.
"Anomaly" is exactly Moderate or High Risk. The decision score is **not** a probability or percentage.

---

## Installation

Python 3.13 (tested on Windows 11).

```bash
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the results

The repository includes the trained models and the scored dataset, so the dashboard runs straight
after installation. To rebuild everything from scratch (about 4 minutes; the generator is deterministic):

```bash
python -m src.pipeline
```

or step by step:

```bash
python -m src.data_generation.motor_digital_twin_v5_10   # 1. generate dataset
python -m src.preprocessing.preprocess_dataset           # 2. validate + encode labels
python -m src.eda.eda_analysis                           # 3. EDA reports and plots
python -m src.classification.fault_classifier            # 4. train + evaluate Random Forest
python -m src.anomaly_detection.isolation_forest         # 5. train Isolation Forest, score fleet
python -m src.evaluation.before_after                    # 6. before/after comparison report
```

The model-selection experiments behind the settings in `src/config.py` take about 2 hours:

```bash
python -m src.experiments.run_experiments ceiling leakage cv tune trees compare calibration features
```

## Run the dashboard

```bash
streamlit run app.py
```

**Live Prediction**: choose *Manual entry*, start from the fleet median or a real held-out reading of
any class, edit values in the grouped tabs (hover the ⓘ for units and typical ranges), then press
*Run prediction*. For *CSV batch*, download the template, fill in rows (extra columns are kept but
ignored), upload, then download the results. 100,000 rows take about 2 seconds.

## Tests

```bash
python -m pytest -q
```

28 tests cover data loading, artifact/metadata consistency, the prediction schema, batch-vs-single
equivalence, missing/invalid input handling, a leakage canary (accuracy must stay below the
theoretical ceiling) and headless rendering of every dashboard page.

---

## Project structure

```
app.py                     Streamlit entrypoint (navigation)
pages/                     Fleet Overview, Motor Diagnostics, Anomaly Monitoring,
                           Fault Analytics, Live Prediction, About
dashboard/utils/           load_data (cached loaders), prediction (vectorised inference),
                           features (names/units), plotly_theme, ui
src/config.py              paths, seed, features, split, model parameters
src/pipeline.py            one-command rebuild
src/data_generation/       digital twin simulator
src/preprocessing/         validation + label encoding
src/eda/                   exploratory analysis
src/classification/        Random Forest training + evaluation
src/anomaly_detection/     Isolation Forest, risk bands, evaluation
src/evaluation/            shared metrics, baseline and before/after reports
src/experiments/           model-selection experiments
models/                    trained models + model_metadata.json (legacy/ = pre-refactor, git-ignored)
reports/                   metrics, EDA, baseline/, experiments/, final report
tests/                     pytest suite
archive/                   earlier simulator and preprocessing versions (not used)
```

## Limitations

- **Synthetic data only.** There is no validation on real motors.
- **Label noise sets an ≈ 84% accuracy ceiling.** Bearing_Wear and Misalignment (under 0.3% of
  samples each) stay weak, with F1 below 0.2.
- **Condition indicators are simulator state.** Real deployments would need them estimated from
  sensors. That is feasible here (R² ≈ 0.97-0.99 from the other features) but not implemented.
- **The anomaly detector has no ground truth.** The 5% flag rate is a chosen alert budget.
- **"Model Confidence" is uncalibrated.** Isotonic calibration reduced macro F1 by 2.4 points, so
  it was not adopted. The reliability diagram is on the Fault Analytics page.
- **Explanations are descriptive.** They list unusual inputs, not causes.
- **Generalisation to a new motor type** is about 2 points lower in macro F1 (leave-motors-out CV).

## Future work

- Estimate condition indicators from raw sensors and retrain on sensors only
- Sequence models over the time axis (e.g. rolling-window features, LSTM) and remaining-useful-life estimation
- SHAP-based per-prediction explanations
- Real sensor integration (MQTT / OPC-UA) and live alerting

---

**Author:** Umang Raj Jaiswal, B.Tech Computer Science & Engineering, Birla Institute of Technology, Mesra.
Developed for educational and research purposes.
