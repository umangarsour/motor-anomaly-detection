"""Performs Exploratory Data Analysis (EDA) on the clean simulated telemetry dataset.

Generates descriptive statistics summaries, identifies highly correlated features,
and exports visualizations (health score distribution, fault pie charts, severity bars,
heatmaps) to reports/ and reports/plots/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# PATH RESOLUTION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "motor_telemetry_clean.csv"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD DATA & VALIDATE
# ==========================================
print("\nLoading Dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset Loaded Successfully")
print("Shape:", df.shape)

print("\nRunning Validation...")
print("Missing Values :", df.isnull().sum().sum())
print("Duplicate Rows :", df.duplicated().sum())

# ==========================================
# DATASET SUMMARY REPORT
# ==========================================
summary = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Missing Values",
        "Duplicate Rows"
    ],
    "Value": [
        len(df),
        len(df.columns),
        df.isnull().sum().sum(),
        df.duplicated().sum()
    ]
})
summary.to_csv(REPORT_DIR / "dataset_summary.csv", index=False)

# ==========================================
# FEATURE STATISTICS
# ==========================================
numeric_df = df.select_dtypes(include=np.number)
feature_stats = numeric_df.describe().T
feature_stats.to_csv(REPORT_DIR / "feature_statistics.csv")
print("\nFeature Statistics Saved")

# ==========================================
# HEALTH STATISTICS
# ==========================================
health_stats = df.groupby("fault_type")["health_score"].describe()
health_stats.to_csv(REPORT_DIR / "health_statistics.csv")
print("\nHealth Statistics Saved")

# ==========================================
# FAULT & SEVERITY DISTRIBUTIONS
# ==========================================
fault_distribution = df["fault_type"].value_counts()
fault_distribution.to_csv(REPORT_DIR / "fault_distribution.csv")
print("\nFault Distribution saved")

severity_distribution = df["severity"].value_counts()
severity_distribution.to_csv(REPORT_DIR / "severity_distribution.csv")
print("\nSeverity Distribution saved")

# ==========================================
# CORRELATION MATRIX & HIGHEST CORRELATIONS
# ==========================================
corr_matrix = numeric_df.corr()
corr_matrix.to_csv(REPORT_DIR / "correlation_matrix.csv")
print("\nCorrelation Matrix Saved")

high_corr = []
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

for column in upper.columns:
    for row in upper.index:
        value = upper.loc[row, column]
        if pd.notna(value) and abs(value) > 0.90:
            high_corr.append([row, column, round(value, 3)])

high_corr_df = pd.DataFrame(
    high_corr,
    columns=["Feature 1", "Feature 2", "Correlation"]
)
high_corr_df.to_csv(REPORT_DIR / "high_correlation_pairs.csv", index=False)
print("High Correlation Report Saved")

# ==========================================
# MATPLOTLIB / SEABORN VISUALIZATIONS
# ==========================================
# 1. Health Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["health_score"], bins=40, kde=True, color="royalblue")
plt.title("Health Score Distribution")
plt.xlabel("Health Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOT_DIR / "health_distribution.png", dpi=300)
plt.close()

# 2. Fault Distribution
plt.figure(figsize=(10, 6))
fault_distribution.plot(kind="bar", color="tomato")
plt.title("Fault Distribution")
plt.xlabel("Fault Type")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(PLOT_DIR / "fault_distribution.png", dpi=300)
plt.close()

# 3. Severity Distribution
plt.figure(figsize=(8, 5))
severity_distribution.plot(kind="bar", color="darkorange")
plt.title("Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOT_DIR / "severity_distribution.png", dpi=300)
plt.close()

# 4. Health Score vs Fault Type Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="fault_type", y="health_score")
plt.xticks(rotation=20)
plt.title("Health Score vs Fault Type")
plt.tight_layout()
plt.savefig(PLOT_DIR / "health_vs_fault_boxplot.png", dpi=300)
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.1,
    cbar=True
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

# 6. Feature Correlation Preview (with Health Score)
health_corr = (
    corr_matrix["health_score"]
    .drop("health_score")
    .abs()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 7))
health_corr.sort_values().plot(kind="barh", color="mediumseagreen")
plt.title("Top Features Correlated with Health Score")
plt.xlabel("Absolute Correlation")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance_preview.png", dpi=300)
plt.close()

# ==========================================
# EXPORT STATS SUMMARIES
# ==========================================
sensor_columns = [
    "current",
    "voltage",
    "motor_temperature",
    "bearing_temperature",
    "vibration_rms",
    "sound_level_db"
]
sensor_summary = df[sensor_columns].describe()
sensor_summary.to_csv(REPORT_DIR / "sensor_statistics.csv")

motor_summary = df.groupby("motor_type").agg({
    "health_score": "mean",
    "motor_temperature": "mean",
    "bearing_temperature": "mean",
    "current": "mean"
})
motor_summary.to_csv(REPORT_DIR / "motor_type_summary.csv")

# ==========================================
# TERMINAL OUTPUT AUDIT
# ==========================================
print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"Dataset Shape : {df.shape}")
print(f"Number of Features : {len(df.columns)}")
print(f"Missing Values : {df.isnull().sum().sum()}")
print(f"Duplicate Rows : {df.duplicated().sum()}")

print("\nTop Correlations With Health Score:")
print(corr_matrix["health_score"].sort_values(ascending=False))

print("\nGenerated Reports:")
for file in sorted(REPORT_DIR.glob("*.csv")):
    print(f"  {file.name}")

print("\nGenerated Plots:")
for file in sorted(PLOT_DIR.glob("*.png")):
    print(f"  {file.name}")

print("\nEDA Completed Successfully")
print("=" * 60)