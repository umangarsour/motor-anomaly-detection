import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ==========================================
# LOAD DATA
# ==========================================

file_path = Path(
    "data/raw/generated/motor_telemetry_v5_4.csv"
)

df = pd.read_csv(file_path)

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)

# ==========================================
# MISSING VALUES
# ==========================================

print("\nMissing Values:\n")
print(df.isnull().sum())

# ==========================================
# DUPLICATES
# ==========================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows:", duplicate_count)

# ==========================================
# LABEL ENCODING
# ==========================================

fault_encoder = LabelEncoder()
severity_encoder = LabelEncoder()

df["fault_type_encoded"] = fault_encoder.fit_transform(
    df["fault_type"]
)

df["severity_encoded"] = severity_encoder.fit_transform(
    df["severity"]
)

# ==========================================
# SAVE CLEAN DATASET
# ==========================================

output_path = Path(
    "data/processed/motor_telemetry_clean.csv"
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

print("\nProcessed Dataset Saved")

print("\nFinal Shape:")
print(df.shape)

print("\nFault Mapping")

for i, label in enumerate(
    fault_encoder.classes_
):
    print(i, "->", label)

print("\nSeverity Mapping")

for i, label in enumerate(
    severity_encoder.classes_
):
    print(i, "->", label)