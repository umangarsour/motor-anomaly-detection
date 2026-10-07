import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

NUM_MOTORS = 10
NUM_CYCLES = 5
SAMPLES_PER_CYCLE = 2000

TOTAL_SAMPLES = (
    NUM_MOTORS
    * NUM_CYCLES
    * SAMPLES_PER_CYCLE
)

# ==========================================
# MOTOR RATINGS
# ==========================================

RATED_RPM = 1500          # RPM
RATED_VOLTAGE = 415       # Volts
RATED_CURRENT = 10        # Amperes
RATED_POWER_FACTOR = 0.92
RATED_EFFICIENCY = 0.90

# ==========================================
# ENVIRONMENT
# ==========================================

BASE_AMBIENT_TEMPERATURE = 30      # °C

MAX_HEALTH = 100
MAX_DEGRADATION = 100

# ==========================================
# RANDOM SEED
# ==========================================

SEED = 42
np.random.seed(SEED)

# ==========================================
# SENSOR NOISE
# ==========================================

TEMPERATURE_NOISE = 2.5
CURRENT_NOISE = 0.45
VOLTAGE_NOISE = 4.0
RPM_NOISE = 12.0
VIBRATION_NOISE = 0.15
SOUND_NOISE = 3.0
POWER_FACTOR_NOISE = 0.02
EFFICIENCY_NOISE = 0.02
DEGRADATION_NOISE = 3.0

# ==========================================
# FAULT SCORING THRESHOLDS
# ==========================================

ACTIVE_FAULT_THRESHOLD = 68
MULTI_FAULT_THRESHOLD = 85

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
# OUTPUT PATH
# ==========================================

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/raw/generated/motor_telemetry_v5_10.csv"
)

# ==========================================
# MOTOR PROFILE DEFINITIONS
# ==========================================

def get_motor_profile(motor_type):

    profile = {
        # Operating Characteristics
        "load_factor": 1.00,
        "rpm_factor": 1.00,

        # Degradation Tendencies
        "bearing_factor": 1.00,
        "thermal_factor": 1.00,
        "electrical_factor": 1.00,
        "lubrication_factor": 1.00,

        # Environmental Influence
        "cooling_factor": 1.00,

        # Maintenance Quality
        "maintenance_factor": 1.00
    }

    # ======================================
    # MOTOR-SPECIFIC BEHAVIOUR
    # ======================================

    if motor_type == "High_Load":
        profile["load_factor"] = 1.15
        profile["thermal_factor"] = 1.05

    elif motor_type == "High_RPM":
        profile["rpm_factor"] = 1.10
        profile["bearing_factor"] = 1.05

    elif motor_type == "Thermal_Stress":
        profile["thermal_factor"] = 1.15
        profile["cooling_factor"] = 0.90

    elif motor_type == "Electrical_Stress":
        profile["electrical_factor"] = 1.15

    elif motor_type == "Fast_Wear":
        profile["bearing_factor"] = 1.20
        profile["lubrication_factor"] = 1.10

    elif motor_type == "Efficient":
        profile["load_factor"] = 0.95
        profile["bearing_factor"] = 0.90
        profile["thermal_factor"] = 0.90
        profile["electrical_factor"] = 0.95

    elif motor_type == "Heavy_Duty":
        profile["load_factor"] = 1.15
        profile["thermal_factor"] = 1.10
        profile["bearing_factor"] = 1.10

    elif motor_type == "Low_Maintenance":
        profile["lubrication_factor"] = 1.15
        profile["maintenance_factor"] = 0.90

    elif motor_type == "Aging":
        profile["bearing_factor"] = 1.20
        profile["thermal_factor"] = 1.10
        profile["electrical_factor"] = 1.10
        profile["maintenance_factor"] = 0.95

    return profile

# ==========================================
# OPERATING CONDITIONS ENGINE
# ==========================================

def generate_operating_conditions(profile):

    life_scale = np.random.normal(1, 0.08)

    life_progress = np.clip(
        np.linspace(0, 1, SAMPLES_PER_CYCLE) * life_scale,
        0,
        1
    )

    base_load = (
        60
        + 20 * np.sin(np.linspace(0, 6 * np.pi, SAMPLES_PER_CYCLE))
        + 8 * np.sin(np.linspace(0, 18 * np.pi, SAMPLES_PER_CYCLE))
    )

    random_load = np.random.normal(0, 5, SAMPLES_PER_CYCLE)

    load_percentage = base_load + random_load
    load_percentage *= profile["load_factor"]
    load_percentage = np.clip(load_percentage, 20, 120)

    ambient_temperature = (
        BASE_AMBIENT_TEMPERATURE
        + 5 * np.sin(np.linspace(0, 2 * np.pi, SAMPLES_PER_CYCLE))
        + np.random.normal(0, 1.5, SAMPLES_PER_CYCLE)
    )

    supply_voltage = (
        RATED_VOLTAGE
        + np.random.normal(0, 2.5, SAMPLES_PER_CYCLE)
    )
    supply_voltage = np.clip(supply_voltage, 390, 435)

    maintenance_quality = np.clip(
        np.random.normal(profile["maintenance_factor"], 0.05, SAMPLES_PER_CYCLE),
        0.7,
        1.2
    )

    return {
        "life_progress": life_progress,
        "load_percentage": load_percentage,
        "ambient_temperature": ambient_temperature,
        "supply_voltage": supply_voltage,
        "maintenance_quality": maintenance_quality
    }

# ==========================================
# DEGRADATION ENGINE
# ==========================================

def generate_degradation(profile, operating):

    life = operating["life_progress"]
    load = operating["load_percentage"]
    ambient = operating["ambient_temperature"]
    maintenance = operating["maintenance_quality"]
    voltage = operating["supply_voltage"]

    load_stress = load / 100
    thermal_environment = (ambient - BASE_AMBIENT_TEMPERATURE) / 10
    voltage_stress = np.abs(voltage - RATED_VOLTAGE) / 20

    bearing_wear = (
        100
        * (life ** 2.1)
        * profile["bearing_factor"]
        * (0.8 + 0.2 * load_stress)
        * (2 - maintenance)
        + np.random.normal(0, DEGRADATION_NOISE, SAMPLES_PER_CYCLE)
    )

    thermal_stress = (
        100
        * (life ** 1.8)
        * profile["thermal_factor"]
        * (1 + 0.15 * thermal_environment + 0.15 * load_stress)
        / profile["cooling_factor"]
        + np.random.normal(0, DEGRADATION_NOISE, SAMPLES_PER_CYCLE)
    )

    electrical_stress = (
        100
        * (life ** 2.0)
        * profile["electrical_factor"]
        * (1 + 0.25 * voltage_stress)
        + np.random.normal(0, DEGRADATION_NOISE, SAMPLES_PER_CYCLE)
    )

    lubrication = (
        100
        - (100 * (life ** 2.0) * profile["lubrication_factor"] * (2 - maintenance))
        + np.random.normal(0, DEGRADATION_NOISE, SAMPLES_PER_CYCLE)
    )

    bearing_wear = np.clip(bearing_wear, 0, 100)
    thermal_stress = np.clip(thermal_stress, 0, 100)
    electrical_stress = np.clip(electrical_stress, 0, 100)
    lubrication = np.clip(lubrication, 0, 100)

    return {
        "bearing_wear": bearing_wear,
        "thermal_stress": thermal_stress,
        "electrical_stress": electrical_stress,
        "lubrication": lubrication
    }

# ==========================================
# SENSOR GENERATION ENGINE
# ==========================================

def generate_sensor_data(profile, operating, degradation):

    load = operating["load_percentage"]
    ambient = operating["ambient_temperature"]
    voltage_supply = operating["supply_voltage"]
    life = operating["life_progress"]

    bearing = degradation["bearing_wear"]
    thermal = degradation["thermal_stress"]
    electrical = degradation["electrical_stress"]
    lubrication = degradation["lubrication"]

    indep_osc1 = np.sin(np.linspace(0, 14 * np.pi, SAMPLES_PER_CYCLE))
    indep_osc2 = np.cos(np.linspace(0, 22 * np.pi, SAMPLES_PER_CYCLE))

    # --------------------------------------
    # Core Electrical/Mechanical
    # --------------------------------------
    rpm = (
        (RATED_RPM * profile["rpm_factor"])
        - 0.65 * load
        - 0.18 * bearing
        + np.random.normal(0, RPM_NOISE, SAMPLES_PER_CYCLE)
    )

    current = (
        (load / 100) * RATED_CURRENT
        + 0.020 * bearing
        + 0.015 * thermal
        + 0.020 * electrical
        + np.random.normal(0, CURRENT_NOISE, SAMPLES_PER_CYCLE)
        + indep_osc1 * 0.2 
    )
    current = np.clip(current, 0, None)

    voltage = (
        voltage_supply
        - 0.050 * electrical
        + np.random.normal(0, VOLTAGE_NOISE, SAMPLES_PER_CYCLE)
    )

    power_factor = (
        RATED_POWER_FACTOR
        - (electrical / 500)
        - (load / 1200)
        + np.random.normal(0, POWER_FACTOR_NOISE, SAMPLES_PER_CYCLE)
        + indep_osc2 * 0.01
    )
    power_factor = np.clip(power_factor, 0.65, 0.95)

    active_power = (
        voltage 
        * current 
        * power_factor 
        * np.random.normal(1, 0.03, SAMPLES_PER_CYCLE)
    ) / 1000

    efficiency = (
        RATED_EFFICIENCY
        - 0.08 * (load / 100)
        - 0.06 * (bearing / 100)
        - 0.05 * (thermal / 100)
        - 0.03 * (electrical / 100)
        + indep_osc2 * 0.02
        + np.random.normal(0, 0.03, SAMPLES_PER_CYCLE)
    )
    efficiency = np.clip(efficiency, 0.50, 0.92)

    # --------------------------------------
    # Temperatures
    # --------------------------------------
    motor_temperature = (
        ambient
        + 20
        + 0.25 * load
        + 0.22 * thermal
        + np.random.normal(0, TEMPERATURE_NOISE, SAMPLES_PER_CYCLE)
    )

    bearing_temperature = (
        ambient
        + 15
        + 0.25 * bearing
        + 0.20 * thermal
        + 0.15 * load
        + np.random.normal(0, TEMPERATURE_NOISE, SAMPLES_PER_CYCLE)
        + indep_osc2 * 1.5
    )

    # --------------------------------------
    # Mechanical / Vibration
    # --------------------------------------
    imbalance_index = (
        0.20 * bearing
        + 0.10 * load
        + 10 * life
        + np.random.normal(0, 3, SAMPLES_PER_CYCLE)
        + indep_osc1 * 2
    )

    misalignment_index = (
        15
        + 45 * life
        + 0.08 * load
        + np.random.normal(0, 6, SAMPLES_PER_CYCLE)
    )
    misalignment_index = np.clip(misalignment_index, 0, 100)

    vibration_amplitude = (
        1
        + 0.015 * bearing
        + 0.015 * misalignment_index
        + 0.010 * load
        + 0.005 * thermal
        + np.random.normal(0, VIBRATION_NOISE, SAMPLES_PER_CYCLE)
    )

    vibration_rms = (
        vibration_amplitude
        * (0.90 + np.random.normal(0, 0.12, SAMPLES_PER_CYCLE))
        + indep_osc1 * 0.15
    )

    vibration_peak = (
        vibration_amplitude
        * (1.45 + np.random.normal(0, 0.25, SAMPLES_PER_CYCLE))
        + indep_osc2 * 0.2
    )

    crest_factor = (
        vibration_peak / np.maximum(vibration_rms, 0.01)
        + np.random.normal(0, 0.2, SAMPLES_PER_CYCLE)
    )

    kurtosis = (
        3
        + bearing / 50
        + misalignment_index / 50
        + electrical / 100
        + np.random.normal(0, 0.30, SAMPLES_PER_CYCLE)
    )

    # --------------------------------------
    # Acoustics
    # --------------------------------------
    sound_level_db = (
        50
        + (rpm / 1500) * 5
        + 0.08 * bearing
        + 0.06 * load
        + 0.05 * electrical
        + np.random.normal(0, SOUND_NOISE, SAMPLES_PER_CYCLE)
        + indep_osc1 * 1.5
    )

    dominant_frequency = (
        50
        + (rpm / 1500) * 10
        + 0.30 * bearing
        + 0.20 * misalignment_index
        + np.random.normal(0, 3, SAMPLES_PER_CYCLE)
    )

    spectral_energy = (
        vibration_rms
        * dominant_frequency
        * np.random.normal(1, 0.15, SAMPLES_PER_CYCLE)
    )

    acoustic_anomaly_score = (
        (sound_level_db - 60)
        + vibration_rms * 4
        + np.random.normal(0, 4, SAMPLES_PER_CYCLE)
    )

    return {
        "rpm": rpm,
        "current": current,
        "voltage": voltage,
        "active_power": active_power,
        "power_factor": power_factor,
        "efficiency_index": efficiency,
        "motor_temperature": motor_temperature,
        "bearing_temperature": bearing_temperature,
        "imbalance_index": imbalance_index,
        "misalignment_index": misalignment_index,
        "vibration_amplitude": vibration_amplitude,
        "vibration_rms": vibration_rms,
        "vibration_peak": vibration_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis,
        "sound_level_db": sound_level_db,
        "dominant_frequency": dominant_frequency,
        "spectral_energy": spectral_energy,
        "acoustic_anomaly_score": acoustic_anomaly_score
    }

# ==========================================
# HEALTH & FAULT ENGINE (Scoring vs Normal)
# ==========================================

def generate_health_and_faults(degradation, sensors, operating):

    bearing = degradation["bearing_wear"]
    thermal = degradation["thermal_stress"]
    electrical = degradation["electrical_stress"]
    lubrication = degradation["lubrication"]

    motor_temp = sensors["motor_temperature"]
    bearing_temp = sensors["bearing_temperature"]
    efficiency = sensors["efficiency_index"]
    
    load = operating["load_percentage"]
    current = sensors["current"]
    
    vibration = sensors["vibration_rms"]
    power_factor = sensors["power_factor"]
    misalignment = sensors["misalignment_index"]

    # --------------------------------------
    # HEALTH SCORE
    # --------------------------------------
    degradation_score = (
        0.30 * bearing
        + 0.25 * thermal
        + 0.20 * electrical
        + 0.15 * (100 - lubrication)
        + 0.10 * misalignment
    )

    degradation_score = np.clip(degradation_score, 0, 100)
    
    health_score = (
        100 
        - degradation_score 
        + np.random.normal(0, 4, SAMPLES_PER_CYCLE)
    )
    health_score = np.clip(health_score, 0, 100)

    # --------------------------------------
    # SEVERITY
    # --------------------------------------
    severity = []
    for health in health_score:
        if health >= 75:
            severity.append("Low")
        elif health >= 50:
            severity.append("Medium")
        elif health >= 25:
            severity.append("High")
        else:
            severity.append("Critical")

    # --------------------------------------
    # SCORE-BASED FAULT ASSIGNMENT
    # --------------------------------------
    fault_type = []

    for i in range(SAMPLES_PER_CYCLE):
        
        scores = {
            "Bearing_Wear": (
                0.50 * bearing[i] 
                + 0.30 * np.clip(vibration[i] * 15, 0, 100) 
                + 0.20 * np.clip(bearing_temp[i], 0, 100) 
                + np.random.normal(0, 5)
            ),
            "Misalignment": (
                0.50 * misalignment[i] 
                + 0.40 * np.clip(vibration[i] * 15, 0, 100) 
                + 0.10 * load[i] 
                + np.random.normal(0, 5)
            ),
            "Overheating": (
                0.50 * thermal[i] 
                + 0.50 * np.clip((motor_temp[i] - 30) * 1.5, 0, 100) 
                + np.random.normal(0, 5)
            ),
            "Electrical_Fault": (
                0.60 * electrical[i] 
                + 0.40 * np.clip((1 - power_factor[i]) * 500, 0, 100) 
                + np.random.normal(0, 5)
            ),
            "Overload": (
                0.60 * load[i] 
                + 0.40 * np.clip(current[i] * 8, 0, 100) 
                + np.random.normal(0, 5)
            ),
            "Normal": (
                health_score[i] * 0.40
                + lubrication[i] * 0.20
                + (efficiency[i] * 100) * 0.30
                - (vibration[i] * 5)
                - (motor_temp[i] - 30) * 0.20
                + np.random.normal(0, 5)
            )
        }

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_state, top_score = sorted_scores[0]
        second_state, second_score = sorted_scores[1]

        if top_state == "Normal":
            fault_type.append("Normal")
        else:
            if second_state != "Normal" and top_score > ACTIVE_FAULT_THRESHOLD and second_score > ACTIVE_FAULT_THRESHOLD:
                if top_score > MULTI_FAULT_THRESHOLD and second_score > MULTI_FAULT_THRESHOLD:
                    fault_type.append("Multi_Fault")
                else:
                    fault_type.append(top_state)
            else:
                fault_type.append(top_state)

    return {
        "health_score": health_score,
        "severity": severity,
        "fault_type": fault_type
    }

# ==========================================
# DATASET GENERATION
# ==========================================

all_data = []

for motor_number, motor_type in enumerate(MOTOR_TYPES, start=1):
    print(f"Generating {motor_type}...")

    profile = get_motor_profile(motor_type)

    for cycle in range(1, NUM_CYCLES + 1):

        operating = generate_operating_conditions(profile)
        degradation = generate_degradation(profile, operating)
        sensors = generate_sensor_data(profile, operating, degradation)
        health = generate_health_and_faults(degradation, sensors, operating)

        df_cycle = pd.DataFrame({
            "motor_id": f"M{motor_number}",
            "motor_type": motor_type,
            "cycle_id": cycle,
            "timestamp": np.arange(SAMPLES_PER_CYCLE),

            # Operating Conditions
            "load_percentage": operating["load_percentage"],
            "ambient_temperature": operating["ambient_temperature"],
            "supply_voltage": operating["supply_voltage"],
            "maintenance_quality": operating["maintenance_quality"],

            # Electrical
            "rpm": sensors["rpm"],
            "current": sensors["current"],
            "voltage": sensors["voltage"],
            "active_power": sensors["active_power"],
            "power_factor": sensors["power_factor"],
            "efficiency_index": sensors["efficiency_index"],

            # Thermal
            "motor_temperature": sensors["motor_temperature"],
            "bearing_temperature": sensors["bearing_temperature"],

            # Mechanical
            "imbalance_index": sensors["imbalance_index"],
            "misalignment_index": sensors["misalignment_index"],

            # Vibration
            "vibration_amplitude": sensors["vibration_amplitude"],
            "vibration_rms": sensors["vibration_rms"],
            "vibration_peak": sensors["vibration_peak"],
            "crest_factor": sensors["crest_factor"],
            "kurtosis": sensors["kurtosis"],

            # Acoustic
            "sound_level_db": sensors["sound_level_db"],
            "dominant_frequency": sensors["dominant_frequency"],
            "spectral_energy": sensors["spectral_energy"],
            "acoustic_anomaly_score": sensors["acoustic_anomaly_score"],

            # Degradation
            "bearing_wear_index": degradation["bearing_wear"],
            "thermal_stress_index": degradation["thermal_stress"],
            "electrical_stress_index": degradation["electrical_stress"],
            "lubrication_index": degradation["lubrication"],

            # Health
            "health_score": health["health_score"],
            "fault_type": health["fault_type"],
            "severity": health["severity"]
        })

        all_data.append(df_cycle)

# ==========================================
# FINAL DATAFRAME
# ==========================================

df = pd.concat(all_data, ignore_index=True)

# ==========================================
# DATA QUALITY VALIDATION
# ==========================================

print("\n" + "=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

print(f"\nDataset Shape : {df.shape}")

duplicate_count = df.duplicated().sum()
print(f"\nDuplicate Rows : {duplicate_count}")

missing_count = df.isnull().sum().sum()
print(f"\nMissing Values : {missing_count}")

if missing_count > 0:
    raise ValueError("Dataset contains missing values.")

if duplicate_count > 0:
    raise ValueError("Dataset contains duplicate rows.")

# ------------------------------------------
# SENSOR RANGE CHECKS
# ------------------------------------------
validation_checks = {
    "RPM": ((df["rpm"] >= 0) & (df["rpm"] <= 2000)).all(),
    "Voltage": ((df["voltage"] >= 350) & (df["voltage"] <= 450)).all(),
    "Current": ((df["current"] >= 0) & (df["current"] <= 20)).all(),
    "Power Factor": ((df["power_factor"] >= 0) & (df["power_factor"] <= 1)).all(),
    "Efficiency": ((df["efficiency_index"] >= 0) & (df["efficiency_index"] <= 1)).all(),
    "Health Score": ((df["health_score"] >= 0) & (df["health_score"] <= 100)).all()
}

print("\nRange Validation")
for parameter, status in validation_checks.items():
    print(f"{parameter:<20} {'PASS' if status else 'FAIL'}")

# ------------------------------------------
# HEALTH STATISTICS
# ------------------------------------------
print("\nHealth Score Statistics")
print(df["health_score"].describe())

# ------------------------------------------
# FAULT DISTRIBUTION
# ------------------------------------------
print("\nFault Distribution")
print(df["fault_type"].value_counts())

# ------------------------------------------
# SEVERITY DISTRIBUTION
# ------------------------------------------
print("\nSeverity Distribution")
print(df["severity"].value_counts())

# ------------------------------------------
# MOTOR DISTRIBUTION
# ------------------------------------------
print("\nMotor Distribution")
print(df["motor_type"].value_counts())

# ==========================================
# EXPORT DATASET
# ==========================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print("\nDataset Successfully Saved")
print(OUTPUT_PATH)
print("\nGeneration Complete")
print("=" * 60)

# ==========================================
# DATASET QUALITY AUDIT
# ==========================================

print("\n" + "=" * 60)
print("DATASET QUALITY AUDIT")
print("=" * 60)

quality_score = 100

# ------------------------------------------
# HIGHLY CORRELATED FEATURES
# ------------------------------------------
numeric_df = df.select_dtypes(include="number")
corr_matrix = numeric_df.corr().abs()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Define mathematically coupled features that are expected to be highly correlated
EXPECTED_HIGH_CORRELATION = {
    ("current", "active_power"),
    ("active_power", "current"),
    ("vibration_rms", "spectral_energy"),
    ("spectral_energy", "vibration_rms"),
    ("vibration_amplitude", "vibration_rms"),
    ("vibration_rms", "vibration_amplitude"),
    ("vibration_amplitude", "vibration_peak"),
    ("vibration_peak", "vibration_amplitude"),
    ("vibration_rms", "vibration_peak"),
    ("vibration_peak", "vibration_rms")
}

unexpected_high_corr = []
expected_corr = []

for column in upper_triangle.columns:
    for index in upper_triangle.index:
        value = upper_triangle.loc[index, column]
        if pd.notna(value) and value > 0.95:
            if (index, column) in EXPECTED_HIGH_CORRELATION:
                expected_corr.append((index, column, round(value, 3)))
            else:
                unexpected_high_corr.append((index, column, round(value, 3)))

print("\nHighly Correlated Features (>0.95)")

if len(unexpected_high_corr) == 0 and len(expected_corr) == 0:
    print("PASS")
else:
    if len(expected_corr) > 0:
        print("\nExpected Physical Correlations (No Penalty):")
        for item in expected_corr:
            print(f"  {item[0]} <-> {item[1]} : {item[2]}")
            
    if len(unexpected_high_corr) > 0:
        print("\nUnexpected High Correlations (Penalty Applied):")
        quality_score -= min(len(unexpected_high_corr) * 2, 20)
        for item in unexpected_high_corr:
            print(f"  {item[0]} <-> {item[1]} : {item[2]}")
    else:
        print("\nNo unexpected high correlations. PASS.")

# ------------------------------------------
# CLASS BALANCE
# ------------------------------------------
print("\nClass Balance")
fault_distribution = df["fault_type"].value_counts(normalize=True)
print(fault_distribution)

small_classes = fault_distribution[fault_distribution < 0.05]
if len(small_classes) == 0:
    print("\nPASS")
else:
    quality_score -= 10
    print("\nMinor Classes Detected")
    print(small_classes)

# ------------------------------------------
# CONSTANT FEATURES
# ------------------------------------------
print("\nConstant Features")
constant_columns = []
for column in numeric_df.columns:
    if numeric_df[column].nunique() <= 1:
        constant_columns.append(column)

if len(constant_columns) == 0:
    print("PASS")
else:
    quality_score -= 15
    print(constant_columns)

# ------------------------------------------
# HEALTH SCORE
# ------------------------------------------
print("\nHealth Score Range")
print(f"Minimum : {df['health_score'].min():.2f}")
print(f"Maximum : {df['health_score'].max():.2f}")
print(f"Mean    : {df['health_score'].mean():.2f}")

# ------------------------------------------
# SENSOR SUMMARY
# ------------------------------------------
sensor_columns = [
    "current",
    "voltage",
    "motor_temperature",
    "bearing_temperature",
    "vibration_rms",
    "sound_level_db"
]

print("\nSensor Summary")
print(df[sensor_columns].describe())

# ------------------------------------------
# OVERALL RESULT
# ------------------------------------------
print("\n" + "=" * 60)
print("QUALITY SCORE")
print(f"{quality_score}/100")

if quality_score >= 90:
    print("Excellent Dataset")
elif quality_score >= 75:
    print("Good Dataset")
elif quality_score >= 60:
    print("Fair Dataset")
else:
    print("Needs Improvement")
print("=" * 60)