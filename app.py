"""Streamlit entrypoint: `streamlit run app.py`. Fleet Overview is the landing page."""

import streamlit as st

st.set_page_config(
    page_title="Motor Health Monitoring",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
    initial_sidebar_state="auto",
)

nav = st.navigation({
    "Monitoring": [
        st.Page("pages/1_Fleet_Overview.py", title="Fleet Overview", icon=":material/dashboard:", default=True),
        st.Page("pages/2_Motor_Diagnostics.py", title="Motor Diagnostics", icon=":material/monitor_heart:"),
        st.Page("pages/3_Anomaly_Monitoring.py", title="Anomaly Monitoring", icon=":material/troubleshoot:"),
    ],
    "Models": [
        st.Page("pages/4_Fault_Analytics.py", title="Fault Analytics", icon=":material/analytics:"),
        st.Page("pages/5_Live_Prediction.py", title="Live Prediction", icon=":material/online_prediction:"),
    ],
    "Project": [
        st.Page("pages/6_About.py", title="About", icon=":material/info:"),
    ],
})
st.sidebar.caption("Synthetic digital-twin fleet · 10 motors · 100,000 samples")
nav.run()
