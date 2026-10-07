"""Fleet Overview: fleet-level KPIs, distributions and the motors that need attention first."""

import plotly.express as px
import streamlit as st

from dashboard.utils.load_data import load_scored_dataset
from dashboard.utils.plotly_theme import (
    FAULT_COLORS, OK, SEVERITY_COLORS, SEVERITY_ORDER, WARN, health_gauge,
)
from dashboard.utils.ui import apply_theme, chart, footer, guard, page_header

apply_theme()
with guard():
    df = load_scored_dataset()

page_header(
    "Fleet Overview",
    f"All {len(df):,} recorded samples: {df.motor_id.nunique()} motors × "
    f"{df.cycle_id.nunique()} run-to-failure operating cycles.",
)

# ---------------- KPIs ----------------
n = len(df)
anomalies = int(df.anomaly_label.sum())
critical = int((df.severity == "Critical").sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fleet health (0–100)", f"{df.health_score.mean():.1f}", border=True,
          help="Mean engineered health score over all samples (0 = failed, 100 = new).")
c2.metric("Motors monitored", df.motor_id.nunique(), border=True,
          help=f"{df.motor_type.nunique()} distinct operating profiles.")
c3.metric("Anomalous samples", f"{anomalies:,}", f"{anomalies / n:.1%} of samples", delta_color="off", delta_arrow="off",
          border=True, help="Samples the Isolation Forest flags (decision score < 0).")
c4.metric("Critical-severity samples", f"{critical:,}", f"{critical / n:.1%} of samples",
          delta_color="off", delta_arrow="off", border=True, help="Samples with health score below 25.")

# ---------------- gauge + anomaly trend ----------------
left, right = st.columns([1, 2])
with left:
    with st.container(border=True):
        chart(health_gauge(df.health_score.mean(), "Mean fleet health (0–100)"))
        st.caption("Bands follow the severity labels: <25 Critical, <50 High, <75 Medium.")
with right:
    with st.container(border=True):
        trend = (df.assign(position=(df.timestamp // 100) * 100)
                 .groupby("position").agg(anomaly_rate=("anomaly_label", "mean"),
                                          health=("health_score", "mean")).reset_index())
        fig = px.line(trend, x="position", y="anomaly_rate", markers=True,
                      title="Anomaly rate across the operating cycle",
                      labels={"position": "Sample index within cycle (bins of 100)",
                              "anomaly_rate": "Share of samples flagged"})
        fig.update_traces(line_color=WARN)
        fig.update_yaxes(tickformat=".0%")
        chart(fig)
        st.caption("Each cycle runs from a new motor (sample 0) to end of life (sample 1999); "
                   "anomalies concentrate late in the cycle as degradation builds up.")

# ---------------- distributions ----------------
st.subheader("Distributions")
t1, t2, t3, t4 = st.tabs(["Fault types", "Severity", "Health score", "Motor temperature"])
with t1:
    faults = df.fault_type.value_counts().rename_axis("Fault").reset_index(name="Samples")
    faults["Share"] = faults.Samples / n
    fig = px.bar(faults, x="Samples", y="Fault", orientation="h", color="Fault",
                 color_discrete_map=FAULT_COLORS, text=faults.Share.map("{:.1%}".format))
    fig.update_layout(showlegend=False, yaxis=dict(categoryorder="total ascending", title=""))
    chart(fig)
    st.caption("Labels are heavily imbalanced: Bearing_Wear and Misalignment are each under 0.3% of samples.")
with t2:
    sev = df.severity.value_counts().reindex(SEVERITY_ORDER).rename_axis("Severity").reset_index(name="Samples")
    fig = px.bar(sev, x="Severity", y="Samples", color="Severity", color_discrete_map=SEVERITY_COLORS,
                 text="Samples")
    fig.update_layout(showlegend=False)
    chart(fig)
with t3:
    fig = px.histogram(df, x="health_score", nbins=50, labels={"health_score": "Health score (0–100)"},
                       color_discrete_sequence=[OK])
    fig.update_layout(yaxis_title="Samples", bargap=0.05)
    chart(fig)
with t4:
    fig = px.histogram(df, x="motor_temperature", nbins=50, color_discrete_sequence=[WARN],
                       labels={"motor_temperature": "Motor temperature (°C)"})
    fig.update_layout(yaxis_title="Samples", bargap=0.05)
    chart(fig)

# ---------------- motors needing attention ----------------
st.subheader("Motors ranked by risk")
per_motor = (df.groupby(["motor_id", "motor_type"])
             .agg(health=("health_score", "mean"),
                  anomaly_rate=("anomaly_label", "mean"),
                  high_risk=("risk_level", lambda s: int((s == "High Risk").sum())),
                  critical_rate=("severity", lambda s: (s == "Critical").mean()),
                  worst_score=("anomaly_score", "min"),
                  top_fault=("fault_type", lambda s: s.mode().iat[0]))
             .reset_index().sort_values(["anomaly_rate", "health"], ascending=[False, True]))
per_motor.insert(0, "Rank", range(1, len(per_motor) + 1))
st.dataframe(
    per_motor, hide_index=True,
    column_config={
        "motor_id": "Motor", "motor_type": "Profile",
        "health": st.column_config.ProgressColumn("Mean health", min_value=0, max_value=100, format="%.1f"),
        "anomaly_rate": st.column_config.ProgressColumn("Anomaly rate", min_value=0, max_value=float(
            max(per_motor.anomaly_rate.max(), 0.01)), format="percent"),
        "high_risk": st.column_config.NumberColumn("High-risk samples"),
        "critical_rate": st.column_config.NumberColumn("Critical share", format="percent"),
        "worst_score": st.column_config.NumberColumn("Lowest anomaly score", format="%.3f",
                                                     help="More negative = more anomalous."),
        "top_fault": "Most frequent label",
    },
)
st.caption("Ranked by the share of samples the anomaly detector flags, then by mean health. "
           "Open Motor Diagnostics for per-cycle trends.")
footer()
