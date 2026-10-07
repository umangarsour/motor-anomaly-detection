import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# HEALTH GAUGE
# ==========================================

def health_gauge(value):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Fleet Health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.35},
                "steps": [
                    {"range": [0, 40], "color": "#d62728"},
                    {"range": [40, 70], "color": "#ffb000"},
                    {"range": [70, 100], "color": "#2ca02c"}
                ]
            }
        )
    )

    fig.update_layout(height=350)

    return fig


# ==========================================
# HEALTH DISTRIBUTION
# ==========================================

def health_distribution(df):

    fig = px.histogram(
        df,
        x="health_score",
        nbins=40,
        title="Health Score Distribution"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# ==========================================
# FAULT PIE
# ==========================================

def fault_distribution(df):

    fig = px.pie(
        df,
        names="fault_type",
        title="Fault Distribution"
    )

    return fig


# ==========================================
# SEVERITY BAR
# ==========================================

def severity_distribution(df):

    severity = (
        df["severity"]
        .value_counts()
        .reset_index()
    )

    severity.columns = [
        "Severity",
        "Count"
    ]

    fig = px.bar(
        severity,
        x="Severity",
        y="Count",
        color="Severity",
        title="Severity Distribution"
    )

    return fig


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def feature_importance_chart(df):

    fig = px.bar(

        df.head(15),

        x="Importance",

        y="Feature",

        orientation="h",

        title="Top Feature Importance"

    )

    fig.update_layout(

        yaxis={"categoryorder": "total ascending"}

    )

    return fig


# ==========================================
# ANOMALY SCORE
# ==========================================

def anomaly_score_distribution(df):

    fig = px.histogram(

        df,

        x="anomaly_score",

        nbins=40,

        title="Anomaly Score Distribution"

    )

    return fig