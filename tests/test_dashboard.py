"""Headless smoke tests: every dashboard page renders without an exception."""

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_pipeline import needs_artifacts

PAGES = [
    "pages/1_Fleet_Overview.py", "pages/2_Motor_Diagnostics.py", "pages/3_Anomaly_Monitoring.py",
    "pages/4_Fault_Analytics.py", "pages/5_Live_Prediction.py", "pages/6_About.py",
]


@needs_artifacts
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(page, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@needs_artifacts
def test_entrypoint_renders_landing_page():
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Fleet Overview"


@needs_artifacts
def test_diagnostics_other_motor_and_cycle():
    at = AppTest.from_file("pages/2_Motor_Diagnostics.py", default_timeout=120).run()
    at.selectbox(key="diag_motor").set_value("M8")
    at.slider(key="diag_t").set_value(500)
    at.run()
    assert not at.exception
    assert at.subheader[0].value.startswith("Status at sample 500")


@needs_artifacts
def test_live_prediction_manual_flow():
    at = AppTest.from_file("pages/5_Live_Prediction.py", default_timeout=120).run()
    at.selectbox(key="lp_preset").set_value("Overheating")
    at.button[0].click().run()                                   # "Load values"
    next(b for b in at.button if b.label == "Run prediction").click().run()
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert {"Predicted fault", "Model Confidence", "Anomaly status", "Risk level", "Anomaly score"} <= set(labels)
    assert "lp_result" in at.session_state
