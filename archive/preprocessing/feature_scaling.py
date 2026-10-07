import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# LOAD DATA
# ==========================================

file_path = Path(
    "data/processed/motor_telemetry_clean.csv"
)

df = pd.read_csv(file_path)

print("\nDataset Loaded")
print(df.shape)

# ==========================================
# FEATURES FOR ML
# ==========================================

feature_columns = [

    "load_percentage",
    "rpm",

    "current",
    "voltage",
    "active_power",
    "power_factor",
    "efficiency_index",

    "motor_temperature",
    "bearing_temperature",

    "imbalance_index",
    "misalignment_index",

    "vibration_amplitude",
    "vibration_rms",
    "vibration_peak",
    "crest_factor",
    "kurtosis",

    "sound_level_db",
    "dominant_frequency",
    "spectral_energy",
    "acoustic_anomaly_score",

    "bearing_wear_index",
    "thermal_stress_index",
    "electrical_stress_index",
    "lubrication_index"
]

X = df[feature_columns]

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_columns
)

# ==========================================
# SAVE SCALER
# ==========================================

Path("models").mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    scaler,
    "models/standard_scaler.pkl"
)

# ==========================================
# SAVE SCALED DATA
# ==========================================

scaled_path = Path(
    "data/processed/motor_telemetry_scaled.csv"
)

scaled_df.to_csv(
    scaled_path,
    index=False
)

# ==========================================
# OUTPUT
# ==========================================

print("\nScaling Complete")

print("\nScaled Dataset Shape:")
print(scaled_df.shape)

print("\nScaler Saved:")
print("models/standard_scaler.pkl")

print("\nScaled Dataset Saved:")
print(scaled_path)