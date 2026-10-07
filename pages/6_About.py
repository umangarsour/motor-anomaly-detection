"""About: what the system does, how it was built and evaluated, and its limits."""

import streamlit as st

from dashboard.utils.load_data import load_model_metadata
from dashboard.utils.ui import apply_theme, footer, guard, page_header

apply_theme()
with guard():
    meta = load_model_metadata()
clf, iso = meta["classifier"], meta["isolation_forest"]

page_header("About this system", "Industrial Motor Health Monitoring & Predictive Maintenance")

st.markdown(f"""
A machine-learning monitoring system for a simulated fleet of industrial induction motors.

**Pipeline**

```
Digital twin (10 motors x 5 run-to-failure cycles x 2000 samples)
  -> validation + label encoding
  -> Random Forest fault classifier        (supervised, {len(clf['classes'])} classes)
  -> Isolation Forest anomaly detector      (unsupervised)
  -> this dashboard
```

**Evaluation protocol.** {clf['split']}. Hyper-parameters were chosen with cross-validation inside the
training cycles only; the test cycle was used once, for the final numbers on the Fault Analytics page.

**Results on held-out data.** Accuracy {clf['test_metrics']['accuracy']:.1%}, balanced accuracy
{clf['test_metrics']['balanced_accuracy']:.1%}, macro F1 {clf['test_metrics']['f1_macro']:.1%}.
""")

st.subheader("Limitations")
st.markdown(f"""
- **Synthetic data.** All telemetry comes from a simulator, not from real motors. Results show the method
  works, not how it would perform in a plant.
- **Label noise sets a ceiling.** The simulator picks each fault label as the highest of several scores
  plus random noise, so about 16% of labels cannot be predicted from any features. Rare classes
  (Bearing_Wear, Misalignment) are almost entirely noise-driven.
- **No anomaly ground truth.** The Isolation Forest is unsupervised. Its 5% flag rate is an alert
  budget, and it is checked only against proxy targets (low health, fault labels).
- **{clf['probability_label']}.** {"Probabilities were calibrated on the training cycles." if clf['calibration'] else
  "The displayed probability is the forest's vote share. It is not calibrated."}
- **Explanations are descriptive.** Live Prediction lists inputs that are unusual relative to the fleet;
  this is not a causal diagnosis.
""")
st.caption(f"Classifier trained {clf['trained_at']} with scikit-learn {clf['sklearn_version']}. "
           f"Detector fit on {iso['fit_on']}.")
st.markdown("**Author:** Umang Raj Jaiswal · B.Tech CSE, Birla Institute of Technology, Mesra")
footer()
