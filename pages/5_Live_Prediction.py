"""Live Prediction: score one manually entered reading, or a whole CSV file."""

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils.features import FEATURE_INFO, by_group, label
from dashboard.utils.load_data import load_model_metadata, load_scored_dataset
from dashboard.utils.plotly_theme import FAULT_COLORS
from dashboard.utils.prediction import (
    PredictionInputError, predict_batch, predict_motor, required_features, unusual_readings,
)
from dashboard.utils.ui import apply_theme, badge, chart, footer, guard, page_header

apply_theme()
with guard():
    meta = load_model_metadata()
    features = required_features()
    df = load_scored_dataset()
ref = meta["reference"]
prob_label = meta["classifier"]["probability_label"]

page_header("Live Prediction", "Predict the fault class and anomaly status of a motor reading.")
mode = st.segmented_control("Input", ["Manual entry", "CSV batch"], default="Manual entry", key="lp_mode",
                            required=True)


def fmt(v):
    return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"


# =====================================================================
# Manual entry
# =====================================================================
def set_inputs(values):
    for f in features:
        st.session_state[f"lp_{f}"] = float(round(values[f], 4))
    st.session_state.pop("lp_result", None)


def preset_values(name):
    if name == "Fleet median":
        return {f: ref[f]["q50"] for f in features}
    # a real held-out reading of that class (test cycle, never seen in training)
    pool = df[(df.split == "test") & (df.fault_type == name)]
    return pool.sample(1, random_state=0).iloc[0][features].to_dict()


if mode == "Manual entry":
    if f"lp_{features[0]}" not in st.session_state:
        set_inputs(preset_values("Fleet median"))

    p1, p2, p3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    preset = p1.selectbox("Start from", ["Fleet median"] + meta["classifier"]["classes"], key="lp_preset",
                          help="Fill the form with the fleet median, or with a real held-out reading of a class.")
    p2.button("Load values", on_click=lambda: set_inputs(preset_values(st.session_state.lp_preset)),
              icon=":material/input:", width="stretch")
    p3.button("Reset to median", on_click=lambda: set_inputs(preset_values("Fleet median")),
              icon=":material/restart_alt:", width="stretch")

    with st.form("lp_form", border=True):
        groups = by_group(features)
        for tab, (group, feats) in zip(st.tabs(list(groups)), groups.items()):
            with tab:
                cols = st.columns(3)
                for i, f in enumerate(feats):
                    r, (_, unit, _, desc) = ref[f], FEATURE_INFO[f]
                    cols[i % 3].number_input(
                        label(f), key=f"lp_{f}", format="%.4f" if abs(r["q50"]) < 10 else "%.2f",
                        help=f"{desc} Typical range (1st–99th percentile): {fmt(r['q01'])} – {fmt(r['q99'])}"
                             f"{' ' + unit if unit else ''}. Training data range: {fmt(r['min'])} – {fmt(r['max'])}.",
                    )
        submitted = st.form_submit_button("Run prediction", type="primary", icon=":material/play_arrow:",
                                          width="stretch")

    if submitted:
        values = {f: st.session_state[f"lp_{f}"] for f in features}
        with guard():
            st.session_state["lp_result"] = (predict_motor(values), values)

    if "lp_result" in st.session_state:
        result, values = st.session_state["lp_result"]
        outside = [label(f, False) for f in features if not ref[f]["min"] <= values[f] <= ref[f]["max"]]
        if outside:
            st.warning(f"Outside the training data range: {', '.join(outside)}. "
                       "The models extrapolate here, so treat the result with caution.", icon=":material/warning:")

        st.subheader("Result")
        c = st.columns(5)
        c[0].metric("Predicted fault", result["fault"].replace("_", " "), border=True)
        c[1].metric(prob_label, f"{result['confidence']:.1f}%", border=True,
                    help="Highest class probability from the classifier."
                         + ("" if meta["classifier"]["calibration"] else
                            " These are uncalibrated forest vote shares; see Fault Analytics → Confidence reliability."))
        with c[2]:
            st.metric("Anomaly status", result["anomaly"], border=True)
            badge(result["anomaly"])
        with c[3]:
            st.metric("Risk level", result["risk_level"], border=True)
            badge(result["risk_level"])
        c[4].metric("Anomaly score", f"{result['anomaly_score']:.3f}", border=True,
                    help="Isolation Forest decision score. Below 0 = flagged. Not a probability.")

        probs = pd.Series(result["probabilities"]).sort_values(ascending=False)
        left, right = st.columns([3, 2])
        with left, st.container(border=True):
            top = probs.head(3).rename_axis("Class").reset_index(name="p")
            fig = px.bar(top, x="p", y="Class", orientation="h", color="Class", color_discrete_map=FAULT_COLORS,
                         text=top.p.map("{:.1f}%".format), title=f"Top classes by {prob_label.lower()}",
                         labels={"p": "%", "Class": ""}, range_x=[0, 100])
            fig.update_layout(showlegend=False, height=220, yaxis=dict(categoryorder="total ascending"))
            chart(fig)
        with right, st.container(border=True):
            st.markdown("**Why this result?**")
            unusual = unusual_readings(values, {f: ref[f] for f in features})
            lines = [
                f"Anomaly detector: score **{result['anomaly_score']:.3f}** "
                + ("is below 0, so the reading is **flagged**." if result["anomaly"] == "Anomaly"
                   else "is 0 or above, so the reading is **not flagged**.")
            ]
            if unusual:
                lines.append("Readings furthest from the fleet's typical values: "
                             + "; ".join(f"{label(f, False)} = {fmt(v)} ({where})" for f, v, where in unusual) + ".")
            else:
                lines.append("Every reading is within the fleet's 5th–95th percentile range; the detector scores "
                             "the combination of readings, not any single value.")
            gap = probs.iloc[0] - probs.iloc[1]
            if gap < 15:
                lines.append(f"The top two classes are close ({probs.index[0]} {probs.iloc[0]:.0f}% vs "
                             f"{probs.index[1]} {probs.iloc[1]:.0f}%), so treat the fault label as uncertain.")
            st.markdown("\n\n".join(lines))
            st.caption("Descriptive only: this lists unusual inputs relative to the training data. "
                       "It does not establish what caused the prediction.")

# =====================================================================
# CSV batch
# =====================================================================
else:
    template = df[df.split == "test"].sample(5, random_state=1)[features]
    st.markdown(f"Upload a CSV with these **{len(features)} columns** (any order; extra columns are kept "
                "but ignored).")
    st.download_button("Download a template with 5 example rows", template.to_csv(index=False).encode(),
                       "prediction_template.csv", "text/csv", on_click="ignore", icon=":material/download:")
    uploaded = st.file_uploader("CSV file", type="csv", key="lp_upload")

    if uploaded is None:
        st.session_state.pop("lp_batch", None)
        st.info("No file uploaded yet.", icon=":material/upload_file:")
    else:
        try:
            data = pd.read_csv(uploaded)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            st.error(f"Could not read this file as CSV ({type(e).__name__}). Check the delimiter and encoding.")
            st.stop()

        cached = st.session_state.get("lp_batch")
        if cached is None or cached[0] != uploaded.file_id:
            try:
                t = time.perf_counter()
                out = predict_batch(data)
                st.session_state["lp_batch"] = (uploaded.file_id, pd.concat([data, out], axis=1),
                                                time.perf_counter() - t)
            except PredictionInputError as e:
                st.error(str(e), icon=":material/error:")
                st.stop()
        _, result_df, secs = st.session_state["lp_batch"]

        st.success(f"Scored {len(result_df):,} rows in {secs:.2f} s.", icon=":material/check_circle:")
        c = st.columns(4)
        c[0].metric("Rows", f"{len(result_df):,}", border=True)
        c[1].metric("Flagged anomalies", f"{(result_df.anomaly == 'Anomaly').sum():,}",
                    f"{(result_df.anomaly == 'Anomaly').mean():.1%}", delta_color="off", delta_arrow="off", border=True)
        c[2].metric("High-risk rows", f"{(result_df.risk_level == 'High Risk').sum():,}", border=True)
        c[3].metric("Most frequent fault", result_df.fault.mode().iat[0].replace("_", " "), border=True)

        counts = result_df.fault.value_counts().rename_axis("Fault").reset_index(name="Rows")
        fig = px.bar(counts, x="Rows", y="Fault", orientation="h", color="Fault", color_discrete_map=FAULT_COLORS,
                     title="Predicted faults", text="Rows")
        fig.update_layout(showlegend=False, height=280, yaxis=dict(categoryorder="total ascending", title=""))
        chart(fig)

        only_flag = st.toggle("Show flagged anomalies only", key="lp_only_flag")
        shown = result_df[result_df.anomaly == "Anomaly"] if only_flag else result_df
        st.dataframe(
            shown, hide_index=True, height=400,
            column_order=["fault", "confidence", "anomaly", "risk_level", "anomaly_score"]
            + [c for c in shown.columns if c not in ("fault", "confidence", "anomaly", "risk_level", "anomaly_score")],
            column_config={
                "fault": "Predicted fault",
                "confidence": st.column_config.ProgressColumn(prob_label, min_value=0, max_value=100, format="%.1f%%"),
                "anomaly": "Anomaly", "risk_level": "Risk level",
                "anomaly_score": st.column_config.NumberColumn("Anomaly score", format="%.3f"),
            },
        )
        st.download_button("Download results (CSV)", result_df.to_csv(index=False).encode(),
                           "prediction_results.csv", "text/csv", on_click="ignore", type="primary",
                           icon=":material/download:")
footer()
