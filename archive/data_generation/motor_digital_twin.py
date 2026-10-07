import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

NUM_MOTORS = 10
SAMPLES_PER_MOTOR = 10000

MOTOR_TYPES = [
    "Normal",
    "High_Load",
    "High_RPM",
    "Thermal_Stress",
    "Electrical_Stress",
    "Fast_Wear",
    "Efficient",
    "Heavy_Duty",
    "Low_Maintenance",
    "Aging"
]

all_motor_data = []

def get_motor_profile(motor_type):

    profile = {
        "load_multiplier": 1.0,
        "rpm_multiplier": 1.0,
        "wear_multiplier": 1.0,
        "thermal_multiplier": 1.0,
        "electrical_multiplier": 1.0
    }

    if motor_type == "High_Load":
        profile["load_multiplier"] = 1.25

    elif motor_type == "High_RPM":
        profile["rpm_multiplier"] = 1.15

    elif motor_type == "Thermal_Stress":
        profile["thermal_multiplier"] = 1.40

    elif motor_type == "Electrical_Stress":
        profile["electrical_multiplier"] = 1.40

    elif motor_type == "Fast_Wear":
        profile["wear_multiplier"] = 1.60

    elif motor_type == "Efficient":
        profile["load_multiplier"] = 0.90
        profile["wear_multiplier"] = 0.80

    elif motor_type == "Heavy_Duty":
        profile["load_multiplier"] = 1.35
        profile["thermal_multiplier"] = 1.20

    elif motor_type == "Low_Maintenance":
        profile["wear_multiplier"] = 1.30

    elif motor_type == "Aging":
        profile["wear_multiplier"] = 1.50
        profile["thermal_multiplier"] = 1.30

    return profile

RATED_RPM = 1500
RATED_VOLTAGE = 415
RATED_CURRENT = 10
RATED_EFFICIENCY = 0.90

np.random.seed(42)

# ==========================================
# TIMESTAMP
# ==========================================

timestamp = np.arange(N_SAMPLES)

# ==========================================
# DEGRADATION STATES
# ==========================================

life_progress = np.linspace(0, 1, N_SAMPLES)

bearing_wear_index = (
    100 * (life_progress ** 2.2)
    + np.random.normal(0, 2, N_SAMPLES)
)

bearing_wear_index = np.clip(
    bearing_wear_index,
    0,
    100
)

thermal_stress_index = (
    100 * (life_progress ** 1.8)
    + np.random.normal(0, 2, N_SAMPLES)
)

thermal_stress_index = np.clip(
    thermal_stress_index,
    0,
    100
)

electrical_stress_index = (
    100 * (life_progress ** 2.0)
    + np.random.normal(0, 2, N_SAMPLES)
)

electrical_stress_index = np.clip(
    electrical_stress_index,
    0,
    100
)

lubrication_index = (
    100
    - (100 * (life_progress ** 2.1))
    + np.random.normal(0, 2, N_SAMPLES)
)

lubrication_index = np.clip(
    lubrication_index,
    0,
    100
)

# ==========================================
# MECHANICAL DEGRADATION
# ==========================================

imbalance_index = (
    0.6 * bearing_wear_index
    + np.random.normal(0, 3, N_SAMPLES)
)

imbalance_index = np.clip(
    imbalance_index,
    0,
    100
)

misalignment_index = (
    0.5 * bearing_wear_index
    + np.random.normal(0, 3, N_SAMPLES)
)

misalignment_index = np.clip(
    misalignment_index,
    0,
    100
)

# ==========================================
# LOAD PERCENTAGE
# ==========================================

load_percentage = (
    60
    + 20 * np.sin(np.linspace(0, 100, N_SAMPLES))
    + np.random.normal(0, 5, N_SAMPLES)
)

load_percentage = np.clip(
    load_percentage,
    20,
    120
)

# ==========================================
# RPM
# ==========================================

rpm = (
    RATED_RPM
    - 0.6 * load_percentage
    + np.random.normal(0, 8, N_SAMPLES)
)

# ==========================================
# CURRENT
# ==========================================

current = (
    (load_percentage / 100)
    * RATED_CURRENT
    + np.random.normal(0, 0.25, N_SAMPLES)
)

# ==========================================
# VOLTAGE
# ==========================================

voltage = (
    RATED_VOLTAGE
    + np.random.normal(0, 3, N_SAMPLES)
)

# ==========================================
# POWER FACTOR
# ==========================================

power_factor = (
    0.92
    - (load_percentage / 1000)
    + np.random.normal(0, 0.01, N_SAMPLES)
)

power_factor = np.clip(
    power_factor,
    0.75,
    0.95
)

# ==========================================
# ACTIVE POWER
# ==========================================

active_power = (
    voltage
    * current
    * power_factor
) / 1000

# ==========================================
# EFFICIENCY
# ==========================================

efficiency_index = (
    RATED_EFFICIENCY
    - (load_percentage / 1000)
    + np.random.normal(0, 0.01, N_SAMPLES)
)

efficiency_index = np.clip(
    efficiency_index,
    0.70,
    0.92
)

# ==========================================
# TEMPERATURES
# ==========================================

motor_temperature = (
    30
    + 0.35 * load_percentage
    + 0.08 * thermal_stress_index
    + np.random.normal(0, 1.5, N_SAMPLES)
)

bearing_temperature = (
    motor_temperature
    - 4
    + 0.05 * bearing_wear_index
    + np.random.normal(0, 1, N_SAMPLES)
)

# ==========================================
# VIBRATION FEATURES
# ==========================================

vibration_amplitude = (
    0.10
    + 0.015 * bearing_wear_index
    + 0.010 * imbalance_index
    + 0.008 * misalignment_index
    + 0.003 * (100 - lubrication_index)
    + np.random.normal(0, 0.05, N_SAMPLES)
)

vibration_amplitude = np.clip(
    vibration_amplitude,
    0.05,
    None
)

vibration_rms = (
    vibration_amplitude * 0.707
)

vibration_peak = (
    vibration_amplitude * 1.414
)

crest_factor = (
    vibration_peak / vibration_rms
)

kurtosis = (
    3
    + (bearing_wear_index / 30)
    + np.random.normal(0, 0.3, N_SAMPLES)
)

kurtosis = np.clip(
    kurtosis,
    2,
    10
)

# ==========================================
# DATAFRAME
# ==========================================

df = pd.DataFrame({
    "timestamp": timestamp,
    "load_percentage": load_percentage,
    "rpm": rpm,
    "motor_temperature": motor_temperature,
    "bearing_temperature": bearing_temperature,
    "current": current,
    "voltage": voltage,
    "active_power": active_power,
    "power_factor": power_factor,
    "efficiency_index": efficiency_index,

    "bearing_wear_index": bearing_wear_index,
    "thermal_stress_index": thermal_stress_index,
    "electrical_stress_index": electrical_stress_index,
    "lubrication_index": lubrication_index,

    "imbalance_index": imbalance_index,
    "misalignment_index": misalignment_index,

    "vibration_amplitude": vibration_amplitude,
    "vibration_rms": vibration_rms,
    "vibration_peak": vibration_peak,

    "crest_factor": crest_factor,
    "kurtosis": kurtosis
})

# ==========================================
# SAVE DATASET
# ==========================================

output_path = Path(
    "data/raw/generated/motor_telemetry.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output_path,
    index=False
)

# ==========================================
# OUTPUT
# ==========================================

print("\nDataset Generated Successfully!\n")

print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)

print("\nSummary Statistics:")
print(df.describe())

print("\nMaximum Values\n")

print(
    "Bearing Wear:",
    round(df["bearing_wear_index"].max(), 2)
)

print(
    "Thermal Stress:",
    round(df["thermal_stress_index"].max(), 2)
)

print(
    "Electrical Stress:",
    round(df["electrical_stress_index"].max(), 2)
)

print(
    "Minimum Lubrication:",
    round(df["lubrication_index"].min(), 2)
)

print(
    "Maximum Vibration:",
    round(df["vibration_amplitude"].max(), 2)
)

print(
    "Maximum Kurtosis:",
    round(df["kurtosis"].max(), 2)
)

print(
    "Maximum Imbalance:",
    round(df["imbalance_index"].max(), 2)
)

print(
    "Maximum Misalignment:",
    round(df["misalignment_index"].max(), 2)
)