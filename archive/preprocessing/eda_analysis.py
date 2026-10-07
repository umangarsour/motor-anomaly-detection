import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
# CORRELATION MATRIX
# ==========================================

numeric_df = df.select_dtypes(
    include=["number"]
)

corr_matrix = numeric_df.corr()

plt.figure(figsize=(18, 12))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "reports/correlation_matrix.png"
)

plt.close()

# ==========================================
# HEALTH SCORE CORRELATION
# ==========================================

health_corr = corr_matrix[
    "health_score"
].sort_values(
    ascending=False
)

print("\nTop Correlations With Health Score\n")
print(health_corr)

# ==========================================
# FAULT DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="fault_type"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "reports/fault_distribution.png"
)

plt.close()

# ==========================================
# HEALTH SCORE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["health_score"],
    bins=30,
    kde=True
)

plt.tight_layout()

plt.savefig(
    "reports/health_score_distribution.png"
)

plt.close()

# ==========================================
# SUMMARY
# ==========================================

print(
    "\nEDA Reports Saved Successfully"
)