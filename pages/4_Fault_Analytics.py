"""Fault Analytics: held-out performance of the fault classifier, weakest classes first."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.features import label
from dashboard.utils.load_data import load_model_metadata, load_scored_dataset
from dashboard.utils.plotly_theme import ACCENT, FAULT_COLORS, MUTED, OK, WARN
from dashboard.utils.ui import apply_theme, badge, chart, footer, guard, page_header

apply_theme()
with guard():
    meta = load_model_metadata()["classifier"]
    df = load_scored_dataset()

m = meta["test_metrics"]
classes = meta["classes"]
page_header("Fault Analytics", f"{meta['algorithm']} · evaluated once on held-out data. {meta['split']}.")

# ---------------- KPIs ----------------
c = st.columns(5)
cv = meta["cv_metrics"]
for col, (key, name, hlp) in zip(c, [
    ("accuracy", "Accuracy", "Share of test samples labelled correctly."),
    ("balanced_accuracy", "Balanced accuracy", "Mean per-class recall; every class counts equally."),
    ("f1_macro", "Macro F1", "Unweighted mean F1 over the 7 classes: exposes weak minority classes."),
    ("f1_weighted", "Weighted F1", "F1 weighted by class size: dominated by the large classes."),
]):
    col.metric(name, f"{m[key]:.1%}", f"CV {cv[key]['mean']:.1%} ± {cv[key]['std']:.1%}", delta_color="off", delta_arrow="off",
               border=True, help=hlp + " Delta line: cross-validation on the training cycles.")
c[4].metric("Training time", f"{meta['training_time_s']:.1f} s", f"{meta['model_size_mb']:.1f} MB model",
            delta_color="off", delta_arrow="off", border=True)

ceiling = meta.get("theoretical_ceiling")
if ceiling:
    st.info(
        f"**Accuracy ceiling ≈ {ceiling['expected_bayes_accuracy']:.1%}.** The digital twin assigns each "
        "label from scores with random noise added, so even the generator's own rule cannot predict "
        f"labels perfectly. On the same test data, that rule scores {ceiling['bayes_rule_test']['accuracy']:.1%} "
        f"accuracy and {ceiling['bayes_rule_test']['f1_macro']:.1%} macro F1.",
        icon=":material/info:",
    )

# ---------------- per-class ----------------
pc = pd.DataFrame(m["per_class"]).T.reset_index(names="Class")
pc["status"] = pd.cut(pc.f1, [-1, 0.4, 0.7, 1.01], labels=["Weak", "Fair", "Good"])
tabs = st.tabs(["Per-class performance", "Confusion matrix", "Feature importance", "Confidence reliability",
                "Class distribution"])
with tabs[0]:
    long = pc.melt(id_vars="Class", value_vars=["precision", "recall", "f1"], var_name="Metric")
    fig = px.bar(long, x="Class", y="value", color="Metric", barmode="group",
                 color_discrete_map={"precision": MUTED, "recall": ACCENT, "f1": OK},
                 category_orders={"Class": pc.sort_values("f1").Class.tolist()},
                 labels={"value": "Score", "Class": ""}, title="Test-set metrics per class (weakest left)")
    fig.update_yaxes(range=[0, 1])
    chart(fig)
    st.dataframe(
        pc.sort_values("f1")[["Class", "status", "precision", "recall", "f1", "support"]], hide_index=True,
        column_config={"status": "Status",
                       **{k: st.column_config.ProgressColumn(k.capitalize() if k != "f1" else "F1",
                                                             min_value=0, max_value=1, format="%.2f")
                          for k in ["precision", "recall", "f1"]},
                       "support": st.column_config.NumberColumn("Test samples")},
    )
    weak = pc[pc.status == "Weak"].Class.tolist()
    if weak:
        badge(f"Weak classes (F1 < 0.4): {', '.join(weak)}", "warn")
        st.caption("These classes are rare (under 4% of samples) and their labels are mostly decided by "
                   "the generator's random noise, so no model can recognise them reliably from these features.")
with tabs[1]:
    norm = st.toggle("Normalise by true class (show recall)", value=True, key="fa_norm")
    cm = pd.DataFrame(m["confusion_matrix"], index=classes, columns=classes)
    z = cm.div(cm.sum(1), axis=0) if norm else cm
    fig = go.Figure(go.Heatmap(z=z.values, x=classes, y=classes, colorscale="Blues", zmin=0,
                               text=z.map(lambda v: f"{v:.0%}" if norm else f"{v:,}").values,
                               texttemplate="%{text}", hovertemplate="True %{y}<br>Predicted %{x}<extra></extra>"))
    fig.update_layout(height=520, xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed",
                      title="Confusion matrix (test cycle)")
    chart(fig)
with tabs[2]:
    imp = pd.DataFrame(meta["feature_importance"])
    imp["Label"] = imp.Feature.map(lambda f: label(f, False))
    kind = st.segmented_control("Importance type", ["Permutation (test macro-F1 drop)", "Impurity (MDI)"],
                                default="Permutation (test macro-F1 drop)", key="fa_imp", required=True)
    col = "Permutation_F1_Macro" if kind.startswith("Perm") else "Importance"
    top = imp.sort_values(col, ascending=False).head(15)
    fig = px.bar(top, x=col, y="Label", orientation="h", color_discrete_sequence=[ACCENT],
                 error_x="Permutation_Std" if col == "Permutation_F1_Macro" else None,
                 labels={col: "Importance", "Label": ""}, title="Top 15 features")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=480)
    chart(fig)
    st.caption("Permutation importance: how much test macro-F1 drops when one feature is shuffled. "
               "Impurity importance is computed on training data and favours high-cardinality features.")
with tabs[3]:
    rel = pd.DataFrame(meta["reliability_test"])
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=MUTED), name="Perfect calibration")
    fig.add_scatter(x=rel.mean_confidence, y=rel.accuracy, mode="lines+markers", name="Model",
                    line=dict(color=WARN), text=rel.n, hovertemplate="confidence %{x:.2f}<br>accuracy %{y:.2f}<br>n=%{text}")
    fig.update_layout(xaxis_title=f"{meta['probability_label']} (top class)", yaxis_title="Observed accuracy",
                      title=f"Reliability on the test cycle · ECE {meta['ece_test']:.3f}", height=420)
    chart(fig)
    st.caption(
        ("Probabilities were calibrated and are shown as 'Prediction Probability'. "
         if meta["calibration"] else
         "Probabilities are the forest's vote shares, not calibrated; they are shown as 'Model Confidence'. ")
        + "Points on the diagonal mean that a stated 80% is right about 80% of the time. "
          "ECE is the average gap between the two."
    )
with tabs[4]:
    dist = df.fault_type.value_counts().rename_axis("Fault").reset_index(name="Samples")
    fig = px.bar(dist, x="Fault", y="Samples", color="Fault", color_discrete_map=FAULT_COLORS, log_y=True,
                 title="Samples per fault label (log scale)", text="Samples")
    fig.update_layout(showlegend=False)
    chart(fig)

with st.expander("Model configuration", icon=":material/tune:"):
    st.json({"algorithm": meta["algorithm"], "params": meta["params"], "features": len(meta["features"]),
             "train_size": meta["train_size"], "test_size": meta["test_size"], "oob_score": meta["oob_score"],
             "sklearn_version": meta["sklearn_version"], "trained_at": meta["trained_at"]})
footer()
