"""Anomaly Monitoring: what the Isolation Forest flags, where, and how much to trust it."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils.features import label
from dashboard.utils.load_data import load_model_metadata, load_scored_dataset
from dashboard.utils.plotly_theme import FAULT_COLORS, RISK_COLORS, RISK_ORDER, SEVERITY_ORDER, WARN
from dashboard.utils.ui import apply_theme, chart, footer, guard, page_header

apply_theme()
with guard():
    df = load_scored_dataset()
    iso = load_model_metadata()["isolation_forest"]

page_header("Anomaly Monitoring",
            "Unsupervised Isolation Forest scores for every sample. Negative score = flagged.")

# ---------------- filters ----------------
with st.container(border=True):
    f1, f2, f3, f4 = st.columns(4)
    risk = f1.multiselect("Risk level", RISK_ORDER, default=RISK_ORDER, key="am_risk")
    faults = f2.multiselect("Fault label", sorted(df.fault_type.unique()), key="am_fault",
                            placeholder="All faults")
    sev = f3.multiselect("Severity", SEVERITY_ORDER, key="am_sev", placeholder="All severities")
    motors = f4.multiselect("Motor", sorted(df.motor_id.unique(), key=lambda m: int(m[1:])), key="am_motor",
                            placeholder="All motors")
mask = df.risk_level.isin(risk)
if faults:
    mask &= df.fault_type.isin(faults)
if sev:
    mask &= df.severity.isin(sev)
if motors:
    mask &= df.motor_id.isin(motors)
view = df[mask]
if view.empty:
    st.info("No samples match these filters.", icon=":material/filter_alt_off:")
    st.stop()

# ---------------- KPIs ----------------
flag = view.anomaly_label == 1
c1, c2, c3, c4 = st.columns(4)
c1.metric("Anomaly rate", f"{flag.mean():.2%}", border=True,
          help="Training-time budget: 5% of training samples (contamination = 0.05).")
c2.metric("Anomalous samples", f"{int(flag.sum()):,}", border=True)
c3.metric("Normal samples", f"{int((~flag).sum()):,}", border=True)
c4.metric("High-risk samples", f"{int((view.risk_level == 'High Risk').sum()):,}", border=True,
          help="Score below the 1st percentile of training scores.")

# ---------------- charts ----------------
t = iso["risk_thresholds"]
a, b = st.columns(2)
with a, st.container(border=True):
    fig = px.histogram(view, x="anomaly_score", color="risk_level", nbins=80,
                       color_discrete_map=RISK_COLORS, category_orders={"risk_level": RISK_ORDER},
                       title="Anomaly score distribution", labels={"anomaly_score": "Decision score",
                                                                   "risk_level": "Risk level"})
    for (x, txt), y in zip([(t["high"], "High risk"), (0, "Flag threshold (0)"), (t["watch"], "Watch")],
                           [0.98, 0.88, 0.78]):
        fig.add_vline(x=x, line=dict(dash="dot", width=1))
        fig.add_annotation(x=x, y=y, yref="paper", text=txt, showarrow=False, xanchor="left", xshift=4,
                           font=dict(size=11), bgcolor="rgba(15,23,42,0.8)")
    fig.update_layout(yaxis_title="Samples", bargap=0.02)
    chart(fig)
with b, st.container(border=True):
    sample = view.sample(min(5000, len(view)), random_state=42)
    fig = px.scatter(sample, x="health_score", y="anomaly_score", color="risk_level", opacity=0.6,
                     color_discrete_map=RISK_COLORS, category_orders={"risk_level": RISK_ORDER},
                     title=f"Anomaly score vs health ({len(sample):,} sampled points)",
                     labels={"health_score": "Health score (0–100)", "anomaly_score": "Decision score",
                             "risk_level": "Risk level"})
    fig.update_traces(marker_size=4)
    chart(fig)

a, b = st.columns(2)
with a, st.container(border=True):
    by_fault = view.groupby("fault_type").agg(rate=("anomaly_label", "mean"), n=("anomaly_label", "size")).reset_index()
    fig = px.bar(by_fault.sort_values("rate"), x="rate", y="fault_type", orientation="h", color="fault_type",
                 color_discrete_map=FAULT_COLORS, title="Share of samples flagged, by fault label",
                 hover_data={"n": True}, labels={"rate": "Share flagged", "fault_type": "", "n": "Samples"})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickformat=".0%")
    chart(fig)
with b, st.container(border=True):
    by_motor = (view.groupby(["motor_id", "motor_type"]).anomaly_label.mean().reset_index()
                .sort_values("anomaly_label"))
    by_motor["name"] = by_motor.motor_id + " · " + by_motor.motor_type.str.replace("_", " ")
    fig = px.bar(by_motor, x="anomaly_label", y="name", orientation="h", title="Most anomalous motors",
                 labels={"anomaly_label": "Share flagged", "name": ""}, color_discrete_sequence=[WARN])
    fig.update_xaxes(tickformat=".0%")
    chart(fig)

# ---------------- top anomalies ----------------
st.subheader("Most anomalous samples")
top_n = st.select_slider("Rows", [25, 50, 100, 250, 500], value=100, key="am_top")
top = view.nsmallest(top_n, "anomaly_score")
cols = ["motor_id", "cycle_id", "timestamp", "risk_level", "anomaly_score", "health_score", "severity",
        "fault_type", "motor_temperature", "bearing_temperature", "vibration_rms", "current"]
st.dataframe(
    top[cols].style.map(lambda v: f"color: {RISK_COLORS.get(v, '')}; font-weight: 600", subset=["risk_level"]),
    hide_index=True, height=380,
    column_config={
        "motor_id": "Motor", "cycle_id": "Cycle", "timestamp": "Sample", "risk_level": "Risk level",
        "anomaly_score": st.column_config.NumberColumn("Decision score", format="%.3f"),
        "health_score": st.column_config.NumberColumn("Health", format="%.1f"),
        "severity": "Severity", "fault_type": "Fault label",
        **{c: st.column_config.NumberColumn(label(c), format="%.2f") for c in cols[-4:]},
    },
)
st.download_button("Download these rows (CSV)", top[cols].to_csv(index=False).encode(), "top_anomalies.csv",
                   "text/csv", on_click="ignore", icon=":material/download:")

# ---------------- how much to trust it ----------------
with st.expander("How to read the score, and how well does the detector work?", icon=":material/help:"):
    st.markdown(
        f"""
- **Decision score** is the Isolation Forest `decision_function`: below **0** the sample is flagged.
  It is a relative isolation score, **not a probability or a percentage**.
- **Risk level** bands come from the training-score distribution: *High Risk* < {t['high']:.3f}
  (most anomalous 1%), *Moderate Risk* < 0 (rest of the flagged 5%), *Slightly Abnormal* < {t['watch']:.3f}
  (next 10%), otherwise *Normal*.
- The dataset has **no ground-truth anomaly labels**, so the detector is checked against proxy targets on
  the held-out test cycle (it was fit on cycles 1-4 only):
"""
    )
    st.dataframe(
        pd.DataFrame(iso["proxy_eval_test"]).T.reset_index(names="Proxy target")[
            ["Proxy target", "proxy_positive_rate", "roc_auc", "pr_auc", "precision", "recall"]],
        hide_index=True,
        column_config={"proxy_positive_rate": st.column_config.NumberColumn("Positive rate", format="%.3f"),
                       "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                       "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
                       "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                       "recall": st.column_config.NumberColumn("Recall", format="%.3f")},
    )
    st.caption("ROC-AUC 0.5 = no better than chance. The detector ranks low-health samples well but "
               "is not a fault classifier: most labelled faults are ordinary-looking operating points.")
footer()
