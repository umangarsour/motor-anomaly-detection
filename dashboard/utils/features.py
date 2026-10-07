"""Human-readable names, units, groups and help text for every telemetry feature.

Units are the ones the digital twin defines; where it defines none (vibration,
spectral energy, indices) the unit says so instead of inventing one.
"""

GROUPS = [
    "Operating Conditions", "Electrical", "Mechanical", "Thermal",
    "Vibration", "Acoustic", "Condition Indicators",
]

# name: (label, unit, group, help)
FEATURE_INFO = {
    "load_percentage": ("Load", "%", "Operating Conditions", "Mechanical load as a percentage of rated load."),
    "ambient_temperature": ("Ambient temperature", "°C", "Operating Conditions", "Air temperature around the motor."),
    "supply_voltage": ("Supply voltage", "V", "Operating Conditions", "Line voltage at the supply (rated 415 V)."),
    "maintenance_quality": ("Maintenance quality", "factor", "Operating Conditions",
                            "Simulated maintenance quality factor; 1.0 is nominal, higher is better."),
    "current": ("Current", "A", "Electrical", "Phase current (rated 10 A)."),
    "voltage": ("Terminal voltage", "V", "Electrical", "Voltage at the motor terminals."),
    "active_power": ("Active power", "kW", "Electrical", "Real power drawn: V × I × PF / 1000."),
    "power_factor": ("Power factor", "", "Electrical", "Ratio of real to apparent power (rated 0.92)."),
    "efficiency_index": ("Efficiency", "fraction", "Electrical", "Output / input power, 0–1 (rated 0.90)."),
    "rpm": ("Shaft speed", "RPM", "Mechanical", "Rotational speed (rated 1500 RPM)."),
    "imbalance_index": ("Imbalance index", "index", "Mechanical", "Simulated rotor imbalance indicator."),
    "misalignment_index": ("Misalignment index", "index 0–100", "Mechanical",
                           "Simulated shaft misalignment indicator."),
    "motor_temperature": ("Motor temperature", "°C", "Thermal", "Stator / housing temperature."),
    "bearing_temperature": ("Bearing temperature", "°C", "Thermal", "Bearing housing temperature."),
    "vibration_amplitude": ("Vibration amplitude", "sim. units", "Vibration",
                            "Simulated vibration level; the digital twin defines no physical unit."),
    "vibration_rms": ("Vibration RMS", "sim. units", "Vibration", "Root-mean-square vibration level."),
    "vibration_peak": ("Vibration peak", "sim. units", "Vibration", "Peak vibration level."),
    "crest_factor": ("Crest factor", "peak/RMS", "Vibration", "Peak divided by RMS; spikes raise it."),
    "kurtosis": ("Kurtosis", "", "Vibration", "Impulsiveness of the vibration signal (3 = Gaussian)."),
    "sound_level_db": ("Sound level", "dB", "Acoustic", "Acoustic noise level."),
    "dominant_frequency": ("Dominant frequency", "Hz", "Acoustic", "Strongest frequency component."),
    "spectral_energy": ("Spectral energy", "sim. units", "Acoustic", "Vibration RMS × dominant frequency."),
    "acoustic_anomaly_score": ("Acoustic anomaly score", "sim. units", "Acoustic",
                               "Engineered from sound level and vibration RMS."),
    "bearing_wear_index": ("Bearing wear index", "index 0–100", "Condition Indicators",
                           "Simulator's hidden bearing degradation state (not a real sensor)."),
    "thermal_stress_index": ("Thermal stress index", "index 0–100", "Condition Indicators",
                             "Simulator's hidden thermal degradation state (not a real sensor)."),
    "electrical_stress_index": ("Electrical stress index", "index 0–100", "Condition Indicators",
                                "Simulator's hidden electrical degradation state (not a real sensor)."),
    "lubrication_index": ("Lubrication index", "index 0–100", "Condition Indicators",
                          "Simulator's hidden lubricant condition; 100 = fresh (not a real sensor)."),
    "health_score": ("Health score", "0–100", "Condition Indicators",
                     "Engineered: 100 minus a weighted sum of the hidden degradation states, plus noise."),
}


def label(name, with_unit=True):
    """'vibration_rms' -> 'Vibration RMS (sim. units)'."""
    text, unit, *_ = FEATURE_INFO.get(name, (name.replace("_", " ").capitalize(), ""))
    return f"{text} ({unit})" if with_unit and unit else text


def by_group(features):
    """{group: [features in that group]} for the given features, in GROUPS order."""
    out = {g: [] for g in GROUPS}
    for f in features:
        out[FEATURE_INFO.get(f, ("", "", "Condition Indicators"))[2]].append(f)
    return {g: fs for g, fs in out.items() if fs}
