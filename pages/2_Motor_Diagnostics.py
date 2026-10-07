"""Motor Diagnostics: one motor, one operating cycle, inspected sample by sample."""

import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.features import label
from dashboard.utils.load_data import load_model_metadata, load_scored_dataset
from dashboard.utils.plotly_theme import (
    ACCENT, CRIT, FAULT_COLORS, MUTED, OK, RISK_ORDER, SEVERITY_ORDER, WARN, WATCH,
)
from dashboard.utils.ui import apply_theme, badge, chart, footer, guard, page_header

apply_theme()
with guard():
    df = load_scored_dataset()
    ref = load_model_metadata()["reference"]

page_header("Motor Diagnostics", "Health, sensor trends and detector output for a single motor and cycle.")

# ---------------- selection ----------------
motors = df[["motor_id", "motor_type"]].drop_duplicates()
motors = motors.iloc[motors.motor_id.str[1:].astype(int).argsort()]
types = dict(zip(motors.motor_id, motors.motor_type))
s1, s2, s3 = st.columns([1.2, 1.4, 2])
motor = s1.selectbox("Motor", motors.motor_id, format_func=lambda m: f"{m} · {types[m].replace('_', ' ')}",
                     key="diag_motor")
cycle = s2.segmented_control("Operating cycle", sorted(df.cycle_id.unique()), default=df.cycle_id.max(),
                             key="diag_cycle", required=True,
                             help="Each cycle is one run of the motor from new (sample 0) to end of life.")
mdf = df[(df.motor_id == motor) & (df.cycle_id == cycle)].sort_values("timestamp")
t_sel = s3.slider("Inspect sample", 0, int(mdf.timestamp.max()), int(mdf.timestamp.max()), key="diag_t",
                  help="Status cards show this sample; charts mark it with a vertical line.")
row = mdf[mdf.timestamp == t_sel].iloc[0]


def level(feat, value):
    """'crit' above the fleet's 99th percentile, 'warn' above the 95th, else None."""
    r = ref[feat]
    return "crit" if value > r["q99"] else "warn" if value > r["q95"] else None


# ---------------- status ----------------
st.subheader(f"Status at sample {t_sel} of cycle {cycle}")
c = st.columns([1, 1.5, 1, 1, 1, 1])
with c[0]:
    st.metric("Health score", f"{row.health_score:.1f}", border=True)
    badge(f"Severity: {row.severity}", {"Low": "ok", "Medium": "watch", "High": "warn", "Critical": "crit"}[row.severity])
with c[1]:
    st.metric("Fault label", row.fault_type.replace("_", " "), border=True,
              help="Label stored in the dataset for this sample.")
with c[2]:
    st.metric("Anomaly score", f"{row.anomaly_score:.3f}", border=True,
              help="Isolation Forest decision function. Negative = flagged as anomalous.")
    badge(str(row.risk_level))
for col, (feat, name, unit) in zip(c[3:], [("motor_temperature", "Motor temp", " °C"),
                                           ("bearing_temperature", "Bearing temp", " °C"),
                                           ("vibration_rms", "Vibration RMS", "")]):
    with col:
        st.metric(name, f"{row[feat]:.1f}{unit}", border=True, help=label(feat))
        lv = level(feat, row[feat])
        if lv:
            badge("> fleet p99" if lv == "crit" else "> fleet p95", lv)
        else:
            badge("Typical", "ok")

HIGH_IS_BAD = ["motor_temperature", "bearing_temperature", "vibration_rms", "vibration_peak", "current",
               "sound_level_db", "kurtosis", "imbalance_index", "misalignment_index"]
alerts = [label(f, False) for f in HIGH_IS_BAD if level(f, row[f]) == "crit"]
if row.anomaly_label == 1 or alerts:
    msg = "**Attention:** "
    msg += "the anomaly detector flags this sample. " if row.anomaly_label == 1 else ""
    msg += f"Readings above the fleet 99th percentile: {', '.join(alerts)}." if alerts else ""
    st.warning(msg, icon=":material/warning:")

# ---------------- trends ----------------
st.subheader("Trends")
flagged = mdf[mdf.anomaly_label == 1]


def trend(features, title, colors, band=None):
    fig = go.Figure()
    for feat, color in zip(features, colors):
        fig.add_scatter(x=mdf.timestamp, y=mdf[feat], name=label(feat, False), mode="lines",
                        line=dict(color=color, width=1.6))
    if band:
        for y, txt, col in band:
            fig.add_hline(y=y, line=dict(color=col, dash="dot", width=1), annotation_text=txt,
                          annotation_font_color=col, annotation_position="top left")
    fig.add_scatter(x=flagged.timestamp, y=flagged[features[0]], mode="markers", name="Flagged anomaly",
                    marker=dict(color=CRIT, size=6, symbol="x"))
    fig.add_vline(x=t_sel, line=dict(color=MUTED, dash="dash", width=1))
    fig.update_layout(title=title, xaxis_title="Sample index within cycle", height=380,
                      yaxis_title=label(features[0]) if len(features) == 1 else None)
    chart(fig)


tabs = st.tabs(["Health", "Temperature", "Vibration", "Electrical", "Power"])
with tabs[0]:
    trend(["health_score"], "Health score (0–100)", [OK],
          band=[(50, "High severity below 50", WARN), (25, "Critical below 25", CRIT)])
with tabs[1]:
    trend(["motor_temperature", "bearing_temperature"], "Temperatures (°C)", [WARN, WATCH],
          band=[(ref["motor_temperature"]["q99"], "Motor temp fleet 99th pct", CRIT)])
with tabs[2]:
    trend(["vibration_rms", "vibration_peak"], "Vibration (simulated units)", [ACCENT, FAULT_COLORS["Misalignment"]],
          band=[(ref["vibration_rms"]["q99"], "RMS fleet 99th pct", CRIT)])
with tabs[3]:
    st.caption("Current (A) and terminal voltage (V) use different scales, so they are shown separately.")
    trend(["current"], "Current (A)", [ACCENT])
    trend(["voltage"], "Terminal voltage (V)", [WATCH])
with tabs[4]:
    trend(["active_power"], "Active power (kW)", [OK])
    trend(["power_factor"], "Power factor", [FAULT_COLORS["Misalignment"]])

# ---------------- history ----------------
st.subheader("History")
f1, f2, f3 = st.columns([2, 2, 1])
sev = f1.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER, key="diag_sev")
risk = f2.multiselect("Risk level", RISK_ORDER, default=RISK_ORDER, key="diag_risk")
only_anom = f3.toggle("Flagged only", key="diag_flag")
hist = mdf[mdf.severity.isin(sev) & mdf.risk_level.isin(risk)]
if only_anom:
    hist = hist[hist.anomaly_label == 1]
cols = ["timestamp", "health_score", "severity", "fault_type", "risk_level", "anomaly_score",
        "load_percentage", "rpm", "current", "voltage", "motor_temperature", "bearing_temperature",
        "vibration_rms", "sound_level_db"]
if hist.empty:
    st.info("No samples match these filters.", icon=":material/filter_alt_off:")
else:
    st.dataframe(
        hist[cols].sort_values("timestamp", ascending=False), hide_index=True, height=360,
        column_config={
            "timestamp": "Sample", "severity": "Severity", "fault_type": "Fault label",
            "risk_level": "Risk level",
            "anomaly_score": st.column_config.NumberColumn("Anomaly score", format="%.3f"),
            **{c: st.column_config.NumberColumn(label(c), format="%.2f") for c in cols
               if c not in ("timestamp", "severity", "fault_type", "risk_level", "anomaly_score")},
        },
    )
    st.download_button("Download filtered history (CSV)", hist[cols].to_csv(index=False).encode(),
                       f"{motor}_cycle{cycle}_history.csv", "text/csv", on_click="ignore",
                       icon=":material/download:")
st.caption(f"{len(hist):,} of {len(mdf):,} samples shown.")
footer()
