import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

NUM_MOTORS = 10
NUM_CYCLES = 5
SAMPLES_PER_CYCLE = 2000

RATED_RPM = 1500
RATED_VOLTAGE = 415
RATED_CURRENT = 10
RATED_EFFICIENCY = 0.90

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
        "load_multiplier": 1.0,
        "rpm_multiplier": 1.0
    }

    if motor_type == "High_Load":
        profile["load_multiplier"] = 1.25

    elif motor_type == "High_RPM":
        profile["rpm_multiplier"] = 1.15

    elif motor_type == "Thermal_Stress":
        profile["thermal_multiplier"] = 1.50

    elif motor_type == "Electrical_Stress":
        profile["electrical_multiplier"] = 1.50

    elif motor_type == "Fast_Wear":
        profile["wear_multiplier"] = 1.60

    elif motor_type == "Efficient":
        profile["wear_multiplier"] = 0.80
        profile["load_multiplier"] = 0.90

    elif motor_type == "Heavy_Duty":
        profile["wear_multiplier"] = 1.30
        profile["load_multiplier"] = 1.35

    elif motor_type == "Low_Maintenance":
        profile["wear_multiplier"] = 1.25

    elif motor_type == "Aging":
        profile["wear_multiplier"] = 1.50
        profile["thermal_multiplier"] = 1.30

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

        # ==================================
        # DEGRADATION
        # ==================================

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
        
        # ==================================
        # MECHANICAL DEGRADATION
        # ==================================

        imbalance_index = (
            0.6 * bearing_wear_index
            + np.random.normal(0, 3, SAMPLES_PER_CYCLE)
        )

        imbalance_index = np.clip(
            imbalance_index,
            0,
            100
        )

        misalignment_index = (
            0.5 * bearing_wear_index
            + np.random.normal(0, 3, SAMPLES_PER_CYCLE)
        )

        misalignment_index = np.clip(
            misalignment_index,
            0,
            100
        )

        # ==================================
        # LOAD
        # ==================================

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

        load_percentage *= profile["load_multiplier"]

        load_percentage = np.clip(
            load_percentage,
            20,
            120
        )

        # ==================================
        # RPM
        # ==================================

        rpm = (
            (RATED_RPM * profile["rpm_multiplier"])
            - 0.6 * load_percentage
            - 0.2 * bearing_wear_index
            + np.random.normal(
                0,
                8,
                SAMPLES_PER_CYCLE
            )
        )

        # ==================================
        # CURRENT
        # ==================================

        current = (
            (load_percentage / 100)
            * RATED_CURRENT
            + (bearing_wear_index / 100)
            * 2
            + np.random.normal(
                0,
                0.25,
                SAMPLES_PER_CYCLE
            )
        )

        current = np.clip(
            current,
            0,
            None
        )

        # ==================================
        # VOLTAGE
        # ==================================

        voltage = (
            RATED_VOLTAGE
            - (electrical_stress_index * 0.05)
            + np.random.normal(
                0,
                3,
                SAMPLES_PER_CYCLE
            )
        )

        # ==================================
        # POWER FACTOR
        # ==================================

        power_factor = (
            0.92
            - (electrical_stress_index / 500)
            + np.random.normal(
                0,
                0.01,
                SAMPLES_PER_CYCLE
            )
        )

        power_factor = np.clip(
            power_factor,
            0.70,
            0.95
        )

        # ==================================
        # ACTIVE POWER
        # ==================================

        active_power = (
            voltage
            * current
            * power_factor
        ) / 1000

        # ==================================
        # EFFICIENCY
        # ==================================

        efficiency_index = (
            RATED_EFFICIENCY
            - (bearing_wear_index / 500)
            - (thermal_stress_index / 1000)
            + np.random.normal(
                0,
                0.01,
                SAMPLES_PER_CYCLE
            )
        )

        efficiency_index = np.clip(
            efficiency_index,
            0.50,
            0.92
        )
        
        # ==================================
        # THERMAL PARAMETERS
        # ==================================

        motor_temperature = (
            30
            + 0.35 * load_percentage
            + 0.08 * thermal_stress_index
            + np.random.normal(0, 1.5, SAMPLES_PER_CYCLE)
        )

        bearing_temperature = (
            motor_temperature
            - 4
            + 0.05 * bearing_wear_index
            + np.random.normal(0, 1, SAMPLES_PER_CYCLE)
        )

        # ==================================
        # VIBRATION FEATURES
        # ==================================

        vibration_amplitude = (
            0.10
            + 0.015 * bearing_wear_index
            + 0.010 * imbalance_index
            + 0.008 * misalignment_index
            + 0.003 * (100 - lubrication_index)
            + np.random.normal(0, 0.05, SAMPLES_PER_CYCLE)
        )

        vibration_amplitude = np.clip(
            vibration_amplitude,
            0.05,
            None
        )

        vibration_rms = (
            vibration_amplitude
            * (
                0.65
                + np.random.normal(0, 0.03, SAMPLES_PER_CYCLE)
            )
        )

        vibration_rms = np.clip(
            vibration_rms,
            0.01,
            None
        )

        vibration_peak = (
            vibration_amplitude
            * (
                1.4
                + 0.02 * (bearing_wear_index / 10)
                + np.random.normal(0, 0.05, SAMPLES_PER_CYCLE)
            )
        )

        crest_factor = vibration_peak / vibration_rms

        crest_factor = np.clip(
            crest_factor,
            1.5,
            10
        )

        kurtosis = (
            3
            + (bearing_wear_index / 30)
            + np.random.normal(0, 0.3, SAMPLES_PER_CYCLE)
        )

        kurtosis = np.clip(
            kurtosis,
            2,
            10
        )

        # ==================================
        # HEALTH SCORE
        # ==================================

        degradation_score = (
            bearing_wear_index
            + thermal_stress_index
            + electrical_stress_index
            + (100 - lubrication_index)
        ) / 4

        health_score = (
            100
            - degradation_score
        )

        health_score = np.clip(
            health_score,
            0,
            100
        )

        # ==================================
        # SEVERITY
        # ==================================

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

        # ==================================
        # FAULT TYPE
        # ==================================

        fault_type = [
            assign_fault(
                motor_type,
                hs
            )
            for hs in health_score
        ]

        # ==================================
        # DATAFRAME
        # ==================================

        df_cycle = pd.DataFrame({

            "motor_id": f"M{motor_number}",
            "motor_type": motor_type,
            "cycle_id": cycle,
            "timestamp": timestamp,

            "load_percentage": load_percentage,
            "rpm": rpm,

            "current": current,
            "voltage": voltage,
            "active_power": active_power,
            "power_factor": power_factor,
            "efficiency_index": efficiency_index,
            
            "motor_temperature": motor_temperature,
            "bearing_temperature": bearing_temperature,

            "imbalance_index": imbalance_index,
            "misalignment_index": misalignment_index,

            "vibration_amplitude": vibration_amplitude,
            "vibration_rms": vibration_rms,
            "vibration_peak": vibration_peak,

            "crest_factor": crest_factor,
            "kurtosis": kurtosis,

            "bearing_wear_index": bearing_wear_index,
            "thermal_stress_index": thermal_stress_index,
            "electrical_stress_index": electrical_stress_index,
            "lubrication_index": lubrication_index,

            "health_score": health_score,

            "fault_type": fault_type,
            "severity": severity

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
    "data/raw/generated/motor_telemetry_v5_2.csv"
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

print("\nV5.2 DATASET GENERATED\n")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFault Distribution:")
print(df["fault_type"].value_counts())

print("\nSeverity Distribution:")
print(df["severity"].value_counts())