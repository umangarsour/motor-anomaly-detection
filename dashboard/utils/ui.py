"""Shared page furniture: header, status badges, chart wrapper, friendly errors."""

import contextlib

import streamlit as st

from dashboard.utils.prediction import PredictionInputError

# level -> (st.badge colour, icon). Icons differ per level so status never relies on colour alone.
_STATUS = {
    "ok": ("green", ":material/check_circle:"),
    "watch": ("yellow", ":material/visibility:"),
    "warn": ("orange", ":material/warning:"),
    "crit": ("red", ":material/error:"),
}
STATUS_OF = {
    "Normal": "ok", "Low": "ok", "Healthy": "ok",
    "Slightly Abnormal": "watch", "Medium": "watch",
    "Moderate Risk": "warn", "High": "warn",
    "High Risk": "crit", "Critical": "crit", "Anomaly": "crit",
}


def apply_theme():
    """Colours come from .streamlit/config.toml; this only tightens spacing."""
    st.markdown(
        """<style>
        .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1400px;}
        [data-testid="stMetricValue"] {font-size: 1.7rem;}
        [data-testid="stMetricLabel"] p {color: #94A3B8;}
        </style>""",
        unsafe_allow_html=True,
    )


def page_header(title, subtitle=""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def badge(text, level=None):
    """Coloured badge with an icon; `level` defaults to the mapping for `text`."""
    color, icon = _STATUS.get(level or STATUS_OF.get(text, ""), ("gray", ":material/info:"))
    st.badge(text, icon=icon, color=color)


def chart(fig, **kwargs):
    st.plotly_chart(fig, theme=None, config={"displaylogo": False}, **kwargs)


@contextlib.contextmanager
def guard():
    """Show missing-artifact and bad-input problems as messages, not tracebacks."""
    try:
        yield
    except FileNotFoundError as e:
        st.error(str(e), icon=":material/folder_off:")
        st.code("python -m src.pipeline", language="bash")
        st.stop()
    except PredictionInputError as e:
        st.error(str(e), icon=":material/error:")
        st.stop()


def footer():
    st.divider()
    st.caption(
        "Industrial Motor Health Monitoring & Predictive Maintenance · synthetic digital-twin data · "
        "Random Forest fault classifier + Isolation Forest anomaly detector"
    )
