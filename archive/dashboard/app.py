import streamlit as st
from pathlib import Path

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Motor Health Monitoring System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Motor Health Monitoring")

st.sidebar.markdown("---")

st.sidebar.success("System Status\n\nOnline")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Available Modules**

    • Fleet Overview

    • Motor Diagnostics

    • Anomaly Monitoring

    • Fault Analytics

    • Live Prediction
    """
)

st.sidebar.markdown("---")

st.sidebar.caption("Developed using Streamlit, Scikit-learn & Python")

# ==========================================
# MAIN PAGE
# ==========================================

st.title("Industrial Motor Health Monitoring & Predictive Maintenance")

st.markdown(
"""
This dashboard provides real-time monitoring and machine learning based
predictive maintenance for industrial motors.

The system integrates:

- Fleet-wide health monitoring
- Isolation Forest based anomaly detection
- Random Forest fault classification
- Sensor analytics
- Live fault prediction
"""
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Machine Learning Models",
        value="2"
    )

with col2:
    st.metric(
        label="Motors Simulated",
        value="10"
    )

with col3:
    st.metric(
        label="Dataset Size",
        value="100,000"
    )

st.divider()

st.subheader("Dashboard Modules")

st.markdown(
"""
### Fleet Overview
Overall fleet health, KPIs and operational summary.

### Motor Diagnostics
Individual motor analysis with sensor trends.

### Anomaly Monitoring
Isolation Forest based anomaly detection and anomaly ranking.

### Fault Analytics
Random Forest model performance, confusion matrix,
feature importance and fault distribution.

### Live Prediction
Predict motor faults from new sensor readings.
"""
)

st.divider()

st.success("Select a page from the left sidebar to begin.")