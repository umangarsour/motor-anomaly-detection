"""Preprocesses the raw simulated telemetry dataset for machine learning.

Validates the raw data, encodes labels for fault classification and severity,
saves the fitted label encoders to models/, and outputs mappings and cleaned data.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==========================================
# PATH RESOLUTION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "generated" / "motor_telemetry_v5_10.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "motor_telemetry_clean.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================
print("\nLoading Dataset...")
df = pd.read_csv(RAW_DATA_PATH)
print("Dataset Loaded Successfully")
print("Shape :", df.shape)

# ==========================================
# DATA VALIDATION
# ==========================================
print("\nRunning Validation...")
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

print("Missing Values :", missing_values)
print("Duplicate Rows :", duplicate_rows)

if missing_values > 0:
    raise ValueError("Dataset contains missing values.")

if duplicate_rows > 0:
    raise ValueError("Dataset contains duplicate rows.")

# ==========================================
# LABEL ENCODING
# ==========================================
print("\nEncoding Labels...")
fault_encoder = LabelEncoder()
severity_encoder = LabelEncoder()

df["fault_type_encoded"] = fault_encoder.fit_transform(df["fault_type"])
df["severity_encoded"] = severity_encoder.fit_transform(df["severity"])

# ==========================================
# SAVE ENCODERS
# ==========================================
joblib.dump(fault_encoder, MODEL_DIR / "fault_label_encoder.pkl")
joblib.dump(severity_encoder, MODEL_DIR / "severity_label_encoder.pkl")
print("Label Encoders Saved")

# ==========================================
# SAVE LABEL MAPPINGS (for reports)
# ==========================================
fault_mapping = pd.DataFrame({
    "Encoded Label": range(len(fault_encoder.classes_)),
    "Fault Type": fault_encoder.classes_
})

severity_mapping = pd.DataFrame({
    "Encoded Label": range(len(severity_encoder.classes_)),
    "Severity": severity_encoder.classes_
})

fault_mapping.to_csv(REPORT_DIR / "fault_mapping.csv", index=False)
severity_mapping.to_csv(REPORT_DIR / "severity_mapping.csv", index=False)
print("Mappings Saved")

# ==========================================
# DATASET INFORMATION PRINT
# ==========================================
print("\nDataset Information")
print("\nColumns:")
for column in df.columns:
    print(column)

print("\nData Types:")
print(df.dtypes)

# ==========================================
# SAVE PROCESSED DATASET
# ==========================================
df.to_csv(PROCESSED_DATA_PATH, index=False)
print("\nProcessed Dataset Saved")

# ==========================================
# SUMMARY REPORT
# ==========================================
print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"Final Shape : {df.shape}")
print(f"Number of Features : {len(df.columns)}")

print("\nFault Classes:")
for i, cls in enumerate(fault_encoder.classes_):
    print(f"  {i} -> {cls}")

print("\nSeverity Classes:")
for i, cls in enumerate(severity_encoder.classes_):
    print(f"  {i} -> {cls}")

print(f"\nMissing Values : {df.isnull().sum().sum()}")
print(f"Duplicate Rows : {df.duplicated().sum()}")
print(f"Processed Dataset : {PROCESSED_DATA_PATH}")
print(f"Fault Encoder : {MODEL_DIR / 'fault_label_encoder.pkl'}")
print(f"Severity Encoder : {MODEL_DIR / 'severity_label_encoder.pkl'}")
print("Preprocessing Completed Successfully")
print("=" * 60)