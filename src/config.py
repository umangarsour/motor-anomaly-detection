"""Single source of truth for paths, seeds, feature lists and the train/test split."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "generated" / "motor_telemetry_v5_10.csv"
CLEAN_PATH = BASE_DIR / "data" / "processed" / "motor_telemetry_clean.csv"
# Clean telemetry + Isolation Forest outputs, in original units. Read by the dashboard.
SCORED_PATH = BASE_DIR / "data" / "processed" / "motor_telemetry_scored.parquet"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

SEED = 42
TARGET = "fault_type"

# Hold out the last operating cycle of every motor as the test set (20% of rows).
# Whole cycles are held out so near-identical neighbouring samples of one run can
# never sit on both sides of the split; CV on the training cycles uses GroupKFold
# on cycle_id for the same reason. See reports/refactor_and_optimization_report.md.
GROUP_COLUMN = "cycle_id"
TEST_CYCLE = 5

ID_COLUMNS = ["motor_id", "motor_type", "cycle_id", "timestamp"]

# The 28 numeric columns produced by the digital twin (original training order).
ALL_FEATURES = [
    "load_percentage", "ambient_temperature", "supply_voltage", "maintenance_quality",
    "rpm", "current", "voltage", "active_power", "power_factor", "efficiency_index",
    "motor_temperature", "bearing_temperature", "imbalance_index", "misalignment_index",
    "vibration_amplitude", "vibration_rms", "vibration_peak", "crest_factor", "kurtosis",
    "sound_level_db", "dominant_frequency", "spectral_energy", "acoustic_anomaly_score",
    "bearing_wear_index", "thermal_stress_index", "electrical_stress_index",
    "lubrication_index", "health_score",
]

# Hidden simulator state that the fault label is computed from (see Phase 2 of the
# report). Real sensors cannot observe these directly.
LATENT_FEATURES = [
    "bearing_wear_index", "thermal_stress_index", "electrical_stress_index",
    "lubrication_index", "health_score",
]


def split_train_test(df):
    """Return (train, test) frames using the cycle hold-out described above."""
    is_test = df[GROUP_COLUMN] == TEST_CYCLE
    return df[~is_test], df[is_test]

# ---------------------------------------------------------------------------
# Production model settings (chosen by src/experiments/run_experiments.py; see report)
# ---------------------------------------------------------------------------
# health_score is excluded: dropping it costs nothing in CV and it is the feature
# closest to the labels (it feeds the generator's "Normal" score and defines severity).
CLF_FEATURES = [f for f in ALL_FEATURES if f != "health_score"]
IF_FEATURES = CLF_FEATURES
IF_PARAMS = dict(n_estimators=300, contamination=0.05)
# Selected by grouped-CV macro-F1 (reports/experiments/tune.json, trees.json).
RF_PARAMS = dict(
    n_estimators=200, max_depth=16, min_samples_leaf=10, min_samples_split=10,
    max_features=0.5, class_weight="balanced_subsample", bootstrap=True, oob_score=True,
)
# Probabilities stay uncalibrated: see src/classification/fault_classifier.py::build_model.
