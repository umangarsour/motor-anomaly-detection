"""Cached loaders for the dashboard's data and model artifacts.

Paths come from src.config so training scripts and the dashboard can never disagree.
A missing artifact raises FileNotFoundError with an actionable message; pages turn
that into a friendly error via dashboard.utils.ui.guard().
"""

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import BASE_DIR, MODEL_DIR, SCORED_PATH

PIPELINE_HINT = (
    "Run the training pipeline first (see README, section 'Reproduce the results')."
)


def _require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path.relative_to(BASE_DIR)}. {PIPELINE_HINT}"
        )
    return path


@st.cache_data(show_spinner="Loading telemetry...")
def load_scored_dataset():
    """Clean telemetry (original units) + Isolation Forest score, label and risk level."""
    return pd.read_parquet(_require(SCORED_PATH))


@st.cache_data
def load_model_metadata():
    """Feature lists, class names, parameters, metrics and reference statistics."""
    return json.loads(_require(MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading fault classifier...")
def load_fault_classifier():
    return joblib.load(_require(MODEL_DIR / "fault_classifier.pkl"))


@st.cache_resource(show_spinner="Loading anomaly detector...")
def load_isolation_forest():
    return joblib.load(_require(MODEL_DIR / "isolation_forest.pkl"))
