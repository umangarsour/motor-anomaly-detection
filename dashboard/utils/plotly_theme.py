"""One Plotly template and one set of colour maps for every chart in the dashboard.

Category colours use the Okabe-Ito palette (distinguishable with colour-vision
deficiency). Status colours are always paired with a text label in the UI, so
colour is never the only carrier of meaning.
"""

import plotly.graph_objects as go
import plotly.io as pio

BACKGROUND = "#0F172A"
GRID = "#334155"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
ACCENT = "#3B82F6"

OK, WATCH, WARN, CRIT = "#22C55E", "#EAB308", "#F97316", "#EF4444"

FAULT_COLORS = {
    "Normal": "#009E73",
    "Overload": "#0072B2",
    "Electrical_Fault": "#F0E442",
    "Multi_Fault": "#E69F00",
    "Overheating": "#D55E00",
    "Misalignment": "#56B4E9",
    "Bearing_Wear": "#CC79A7",
}
SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
SEVERITY_COLORS = dict(zip(SEVERITY_ORDER, [OK, WATCH, WARN, CRIT]))
RISK_ORDER = ["Normal", "Slightly Abnormal", "Moderate Risk", "High Risk"]
RISK_COLORS = dict(zip(RISK_ORDER, [OK, WATCH, WARN, CRIT]))

pio.templates["industrial"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=13),
        title=dict(font=dict(size=15, color=TEXT), x=0, xanchor="left"),
        colorway=list(FAULT_COLORS.values()),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2, title_text=""),
        hoverlabel=dict(bgcolor="#1E293B", font_color=TEXT),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font_color=MUTED),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font_color=MUTED),
        height=360,
    )
)
pio.templates.default = "plotly_dark+industrial"


def health_gauge(value, title="Health score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(valueformat=".1f", font=dict(size=30)),
        title=dict(text=title, font=dict(size=14, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED),
            bar=dict(color=TEXT, thickness=0.25),
            # bands match the severity labels: Critical <25, High <50, Medium <75
            steps=[
                dict(range=[0, 25], color=CRIT),
                dict(range=[25, 50], color=WARN),
                dict(range=[50, 75], color=WATCH),
                dict(range=[75, 100], color=OK),
            ],
        ),
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=0))
    return fig
