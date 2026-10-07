import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

NUM_MOTORS = 10
NUM_CYCLES = 5
SAMPLES_PER_CYCLE = 2000

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
        "wear_multiplier": 1.0,
        "thermal_multiplier": 1.0,
        "electrical_multiplier": 1.0,
    }

    if motor_type == "Fast_Wear":
        profile["wear_multiplier"] = 1.6

    elif motor_type == "Thermal_Stress":
        profile["thermal_multiplier"] = 1.5

    elif motor_type == "Electrical_Stress":
        profile["electrical_multiplier"] = 1.5

    elif motor_type == "Efficient":
        profile["wear_multiplier"] = 0.8

    elif motor_type == "Heavy_Duty":
        profile["wear_multiplier"] = 1.3
        profile["thermal_multiplier"] = 1.2

    elif motor_type == "Low_Maintenance":
        profile["wear_multiplier"] = 1.25

    elif motor_type == "Aging":
        profile["wear_multiplier"] = 1.5
        profile["thermal_multiplier"] = 1.3

    return profile

# ==========================================
# FAULT LOGIC
# ==========================================

def assign_fault(motor_type, health_score):

    if health_score >= 80:
        return "Normal"

    if motor_type == "Fast_Wear":
        return np.random.choice(
            ["Bearing_Wear", "Misalignment"],
            p=[0.8, 0.2]
        )

    if motor_type == "Thermal_Stress":
        return np.random.choice(
            ["Overheating", "Bearing_Wear"],
            p=[0.8, 0.2]
        )

    if motor_type == "Electrical_Stress":
        return np.random.choice(
            ["Electrical_Fault", "Overload"],
            p=[0.8, 0.2]
        )

    if motor_type == "Aging":
        return np.random.choice(
            ["Multi_Fault", "Bearing_Wear"],
            p=[0.7, 0.3]
        )

    return np.random.choice(
        [
            "Bearing_Wear",
            "Misalignment",
            "Overheating",
            "Electrical_Fault"
        ]
    )

# ==========================================
# DATA GENERATION
# ==========================================

all_data = []

for motor_number, motor_type in enumerate(MOTOR_TYPES, start=1):

    profile = get_motor_profile(motor_type)

    for cycle in range(1, NUM_CYCLES + 1):

        timestamp = np.arange(SAMPLES_PER_CYCLE)

        life_progress = np.linspace(
            0,
            1,
            SAMPLES_PER_CYCLE
        )

        # ------------------------------
        # Degradation
        # ------------------------------

        bearing_wear_index = (
            100
            * (life_progress ** 2.2)
            * profile["wear_multiplier"]
            + np.random.normal(0, 2, SAMPLES_PER_CYCLE)
        )

        bearing_wear_index = np.clip(
            bearing_wear_index,
            0,
            100
        )

        thermal_stress_index = (
            100
            * (life_progress ** 1.8)
            * profile["thermal_multiplier"]
            + np.random.normal(0, 2, SAMPLES_PER_CYCLE)
        )

        thermal_stress_index = np.clip(
            thermal_stress_index,
            0,
            100
        )

        electrical_stress_index = (
            100
            * (life_progress ** 2.0)
            * profile["electrical_multiplier"]
            + np.random.normal(0, 2, SAMPLES_PER_CYCLE)
        )

        electrical_stress_index = np.clip(
            electrical_stress_index,
            0,
            100
        )

        lubrication_index = (
            100
            - (100 * (life_progress ** 2.1))
            + np.random.normal(0, 2, SAMPLES_PER_CYCLE)
        )

        lubrication_index = np.clip(
            lubrication_index,
            0,
            100
        )

        # ------------------------------
        # Load
        # ------------------------------

        load_percentage = (
            60
            + 20 * np.sin(
                np.linspace(
                    0,
                    20,
                    SAMPLES_PER_CYCLE
                )
            )
            + np.random.normal(
                0,
                5,
                SAMPLES_PER_CYCLE
            )
        )

        load_percentage = np.clip(
            load_percentage,
            20,
            120
        )

        # ------------------------------
        # Health Score
        # ------------------------------

        degradation_score = (
            bearing_wear_index
            + thermal_stress_index
            + electrical_stress_index
            + (100 - lubrication_index)
        ) / 4

        health_score = (
            100 - degradation_score
        )

        health_score = np.clip(
            health_score,
            0,
            100
        )

        # ------------------------------
        # Severity
        # ------------------------------

        severity = []

        for score in degradation_score:

            if score < 25:
                severity.append("Low")

            elif score < 50:
                severity.append("Medium")

            elif score < 75:
                severity.append("High")

            else:
                severity.append("Critical")

        # ------------------------------
        # Fault Type
        # ------------------------------

        fault_type = [
            assign_fault(
                motor_type,
                hs
            )
            for hs in health_score
        ]

        # ------------------------------
        # DataFrame
        # ------------------------------

        df_cycle = pd.DataFrame({

            "motor_id":
            f"M{motor_number}",

            "motor_type":
            motor_type,

            "cycle_id":
            cycle,

            "timestamp":
            timestamp,

            "load_percentage":
            load_percentage,

            "bearing_wear_index":
            bearing_wear_index,

            "thermal_stress_index":
            thermal_stress_index,

            "electrical_stress_index":
            electrical_stress_index,

            "lubrication_index":
            lubrication_index,

            "health_score":
            health_score,

            "fault_type":
            fault_type,

            "severity":
            severity

        })

        all_data.append(df_cycle)

# ==========================================
# FINAL DATASET
# ==========================================

df = pd.concat(
    all_data,
    ignore_index=True
)

# ==========================================
# SAVE
# ==========================================

output_path = Path(
    "data/raw/generated/motor_telemetry_v5_1.csv"
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

print("\nV5.1 DATASET GENERATED\n")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMotor Types:")
print(df["motor_type"].value_counts())

print("\nFault Types:")
print(df["fault_type"].value_counts())

print("\nSeverity Distribution:")
print(df["severity"].value_counts())