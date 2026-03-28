from __future__ import annotations

import altair as alt
import streamlit as st

PALETTE = [
    "#245F73",
    "#733E24",
    "#3d8fa8",
    "#9e5633",
    "#5aadcc",
    "#c4855a",
    "#1a4557",
    "#4a2618",
]


def register_chart_theme() -> None:
    def _theme() -> dict:
        return {
            "config": {
                "background": "#ffffff",
                "font": "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "title": {
                    "fontSize": 13,
                    "fontWeight": 600,
                    "color": "#111827",
                    "anchor": "start",
                    "offset": 4,
                },
                "axis": {
                    "labelColor": "#6b7280",
                    "labelFontSize": 11,
                    "titleColor": "#374151",
                    "titleFontSize": 12,
                    "titleFontWeight": 500,
                    "gridColor": "#f3f4f6",
                    "gridOpacity": 1,
                    "domainColor": "#e5e7eb",
                    "tickColor": "#e5e7eb",
                    "labelPadding": 6,
                    "titlePadding": 8,
                },
                "legend": {
                    "labelColor": "#374151",
                    "labelFontSize": 11,
                    "titleColor": "#111827",
                    "titleFontSize": 11,
                    "titleFontWeight": 600,
                    "symbolSize": 100,
                    "padding": 8,
                },
                "range": {"category": PALETTE},
                "view": {"stroke": "transparent", "fill": "white"},
                "bar": {"cornerRadiusEnd": 4},
                "arc": {"stroke": "white", "strokeWidth": 1.5},
                "point": {"filled": True, "size": 60},
                "line": {"strokeWidth": 2},
            }
        }

    alt.themes.register("climate_dash", _theme)
    alt.themes.enable("climate_dash")


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Page background ── */
.stApp {
    background-color: #F2F0EF;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"]    { display: none; }

/* ── Block container ── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1280px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #BBBDBC !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}
[data-testid="stSidebarNavLink"] {
    border-radius: 8px;
    margin: 2px 0;
    font-size: 0.875rem;
    font-weight: 500;
    color: #374151;
    padding: 0.45rem 0.75rem !important;
    transition: background 0.12s ease;
}
[data-testid="stSidebarNavLink"]:hover {
    background: #f3f4f6 !important;
}
[data-testid="stSidebarNavLink"][aria-selected="true"],
[data-testid="stSidebarNavLink"][data-active="true"] {
    background: #e6f0f3 !important;
    color: #245F73 !important;
    font-weight: 600;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #BBBDBC !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #BBBDBC !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
    word-break: break-word !important;
}
[data-testid="stMetricValue"] div {
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    white-space: normal !important;
}
[data-testid="stMetricDeltaIcon-Up"]   { color: #245F73 !important; }
[data-testid="stMetricDeltaIcon-Down"] { color: #733E24 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 2px solid #BBBDBC;
    margin-bottom: 1.25rem;
}
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.875rem;
    font-weight: 500;
    color: #6b7280;
    padding: 0.5rem 1rem;
    border-radius: 6px 6px 0 0;
    border: none;
    background: transparent;
    transition: color 0.12s ease;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #374151;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #245F73 !important;
    font-weight: 600;
    border-bottom: 2px solid #245F73;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] > div {
    border: 1px solid #BBBDBC;
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: #f9fafb !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #374151 !important;
}

/* ── Inputs ── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important;
    border: 1px solid #BBBDBC !important;
    background: #ffffff !important;
}
[data-testid="stMultiSelect"] > div > div {
    border-radius: 8px !important;
    border: 1px solid #BBBDBC !important;
    background: #ffffff !important;
}
[data-baseweb="slider"] [role="slider"] {
    background: #245F73 !important;
    border-color: #245F73 !important;
}
[data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
    background: #245F73 !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.25rem !important;
    border: 1px solid #BBBDBC !important;
    background: #ffffff !important;
    color: #374151 !important;
    transition: all 0.12s ease !important;
}
[data-testid="stButton"] > button:hover {
    border-color: #245F73 !important;
    color: #245F73 !important;
    background: #e6f0f3 !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: #245F73 !important;
    color: white !important;
    border-color: #245F73 !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1a4557 !important;
    border-color: #1a4557 !important;
}

/* ── Custom components ── */

.page-header {
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #BBBDBC;
}
.page-header h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 0.3rem;
    line-height: 1.2;
}
.page-header p {
    font-size: 0.9rem;
    color: #6b7280;
    margin: 0;
    line-height: 1.5;
}

.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: #733E24;
    display: block;
    margin-bottom: 0.6rem;
}

.divider {
    border: none;
    border-top: 1px solid #BBBDBC;
    margin: 1.75rem 0;
}

.info-box {
    background: #e6f0f3;
    border: 1px solid #b3d0da;
    border-left: 3px solid #245F73;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #1a4557;
    line-height: 1.5;
    margin: 0.75rem 0;
}

/* ── Model performance cards ── */
.model-card {
    background: white;
    border: 1px solid #BBBDBC;
    border-radius: 12px;
    padding: 1.25rem 1.5rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    height: 100%;
}
.model-card-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
    color: #BBBDBC;
    margin-bottom: 1rem;
}
.model-card-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem 1.5rem;
    align-items: flex-start;
}
.model-stat {
    min-width: 80px;
}
.model-stat.wide {
    flex: 1 1 180px;
}
.ms-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    color: #BBBDBC;
    margin-bottom: 3px;
}
.ms-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.2;
    display: flex;
    align-items: baseline;
    gap: 6px;
    flex-wrap: wrap;
    word-break: break-word;
}
.ms-value.sm {
    font-size: 0.95rem;
    font-weight: 600;
    color: #374151;
    line-height: 1.3;
}
.ms-delta {
    font-size: 0.78rem;
    font-weight: 600;
    color: #245F73;
    background: #e6f0f3;
    padding: 1px 6px;
    border-radius: 99px;
}
.ms-delta.neg {
    color: #733E24;
    background: #fdf0ea;
}

/* ── Hero section ── */
.hero {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 2rem;
    min-height: 320px;
    background: linear-gradient(135deg, rgba(36,95,115,0.92) 0%, rgba(115,62,36,0.78) 100%), url('https://images.unsplash.com/photo-1574629810360-7efbbe195018?auto=format&w=1920&q=80') center/cover no-repeat;
    padding: 3rem 3.5rem 2.5rem;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: white;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    padding: 0.3rem 0.8rem;
    border-radius: 99px;
    border: 1px solid rgba(255,255,255,0.3);
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin: 0 0 0.75rem;
    line-height: 1.1;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.85);
    margin: 0 0 2rem;
    max-width: 640px;
    line-height: 1.6;
}

.hero-stats {
    display: flex;
    gap: 0;
    flex-wrap: wrap;
}

.hero-stat {
    padding: 0.875rem 1.75rem;
    border-right: 1px solid rgba(255,255,255,0.2);
}
.hero-stat:last-child {
    border-right: none;
}
.hs-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
    line-height: 1;
}
.hs-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
    margin-top: 4px;
}

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 0.2rem 0.65rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.4;
}
.badge-teal  { background: #e6f0f3; color: #245F73; }
.badge-rust  { background: #fdf0ea; color: #733E24; }
.badge-gray  { background: #f3f4f6; color: #374151; }
.badge-green { background: #ecfdf5; color: #059669; }
.badge-red   { background: #fef2f2; color: #dc2626; }

/* ── Result pills ── */
.result-pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 99px;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-win  { background: #ecfdf5; color: #059669; }
.result-draw { background: #fffbeb; color: #d97706; }
.result-loss { background: #fef2f2; color: #dc2626; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    register_chart_theme()
