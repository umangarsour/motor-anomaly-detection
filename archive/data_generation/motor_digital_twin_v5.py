import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

NUM_MOTORS = 10
SAMPLES_PER_MOTOR = 10000

RATED_RPM = 1500

np.random.seed(42)

# ==========================================
# MOTOR TYPES
# ==========================================

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

# ==========================================
# MOTOR PROFILES
# ==========================================

def get_motor_profile(motor_type):

    profile = {
        "load_multiplier": 1.0,
        "rpm_multiplier": 1.0,
        "wear_multiplier": 1.0
    }

    if motor_type == "High_Load":
        profile["load_multiplier"] = 1.25

    elif motor_type == "High_RPM":
        profile["rpm_multiplier"] = 1.15

    elif motor_type == "Fast_Wear":
        profile["wear_multiplier"] = 1.50

    elif motor_type == "Efficient":
        profile["load_multiplier"] = 0.90

    elif motor_type == "Heavy_Duty":
        profile["load_multiplier"] = 1.35

    elif motor_type == "Aging":
        profile["wear_multiplier"] = 1.40

    return profile

# ==========================================
# DATA GENERATION
# ==========================================

all_motor_data = []

for motor_index, motor_type in enumerate(MOTOR_TYPES, start=1):

    profile = get_motor_profile(motor_type)

    timestamp = np.arange(SAMPLES_PER_MOTOR)

    load_percentage = (
        60
        + 20 * np.sin(np.linspace(0, 100, SAMPLES_PER_MOTOR))
        + np.random.normal(0, 5, SAMPLES_PER_MOTOR)
    )

    load_percentage = (
        load_percentage
        * profile["load_multiplier"]
    )

    load_percentage = np.clip(
        load_percentage,
        20,
        120
    )

    rpm = (
        RATED_RPM
        * profile["rpm_multiplier"]
        - 0.6 * load_percentage
        + np.random.normal(0, 8, SAMPLES_PER_MOTOR)
    )

    df_motor = pd.DataFrame({
        "motor_id": f"M{motor_index}",
        "motor_type": motor_type,
        "timestamp": timestamp,
        "load_percentage": load_percentage,
        "rpm": rpm
    })

    all_motor_data.append(df_motor)

# ==========================================
# FINAL DATASET
# ==========================================

df = pd.concat(
    all_motor_data,
    ignore_index=True
)

# ==========================================
# SAVE
# ==========================================

output_path = Path(
    "data/raw/generated/motor_telemetry_v5.csv"
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

print("\nV5 DATASET GENERATED\n")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nMotor Counts:")
print(df["motor_type"].value_counts())

print("\nColumns:")
print(df.columns.tolist())