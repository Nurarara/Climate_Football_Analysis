from __future__ import annotations
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import altair as alt
import pandas as pd
import streamlit as st

from utils.constants import (
    ANALYSIS_DIR,
    MODELS_DIR,
    FEATURE_COLUMNS,
    COLD_TEMP,
    HOT_TEMP,
    WINDY_SPEED,
    HUMIDITY_HIGH,
    HEAVY_RAIN,
    STRONG_WIND,
    FREEZING_APPARENT,
    HEATWAVE_APPARENT,
    RESULT_LABELS,
    MONTH_ORDER,
    TEMP_BAND_ORDER,
)
from utils.data_loader import check_artifacts, load_feature_data, load_json, load_csv, load_models
from utils.model_utils import predict_fixture
from utils.styles import inject_styles

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_styles()

# ── Artifact check ─────────────────────────────────────────────────────────────
missing = check_artifacts()
if missing:
    st.error("Missing pipeline artifacts. Run `python scripts/run_pipeline.py` first.")
    st.code("\n".join(str(p) for p in missing))
    st.stop()

# ── Data ───────────────────────────────────────────────────────────────────────
df = load_feature_data()
metrics = load_json(MODELS_DIR / "evaluation_metrics.json")
clf_m = metrics["classification"]
reg_m = metrics["regression"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Climate Football Intelligence")
    st.markdown(
        '<p style="font-size:0.8rem;color:#6b7280;margin-top:-0.25rem;line-height:1.5;">'
        "Premier League weather analytics · 5 seasons · 1,900 matches"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption(
        "Use the tabs on the main page to explore: Climate Overview, "
        "Team Rankings, Match Simulator, Future Scenarios, and Extreme Weather."
    )

# ── Hero section ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <span class="hero-tag">Premier League · 2019–2024 · Five Seasons</span>
      <h1 class="hero-title">Climate Football Intelligence</h1>
      <p class="hero-sub">Does weather measurably affect how Premier League matches are played? Five seasons of results, stadium geography, and match-day weather — combined into predictive models and multi-phase analysis.</p>
      <div class="hero-stats">
        <div class="hero-stat"><div class="hs-value">{len(df):,}</div><div class="hs-label">Matches</div></div>
        <div class="hero-stat"><div class="hs-value">{df['home_team'].nunique()}</div><div class="hs-label">Clubs</div></div>
        <div class="hero-stat"><div class="hs-value">{clf_m['model_accuracy']:.1%}</div><div class="hs-label">Outcome Accuracy</div></div>
        <div class="hero-stat"><div class="hs-value">{clf_m['weather_delta']:+.1%}</div><div class="hs-label">Weather Lift</div></div>
        <div class="hero-stat"><div class="hs-value">{df['rain_flag'].mean():.1%}</div><div class="hs-label">Rain Share</div></div>
        <div class="hero-stat"><div class="hs-value">{reg_m['model_mae']:.3f}</div><div class="hs-label">Goals MAE</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Model performance section ──────────────────────────────────────────────────
def _fmt_feature(key: str) -> str:
    return " ".join(w.capitalize() for w in key.split("_"))


clf_feat = _fmt_feature(clf_m["top_feature"])
reg_feat = _fmt_feature(reg_m["top_feature"])

left_mc, right_mc = st.columns(2)

with left_mc:
    delta_sign = "+" if clf_m["model_vs_baseline_delta"] >= 0 else ""
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-card-title">Match Outcome · Classification Model</div>
            <div class="model-card-row">
                <div class="model-stat">
                    <div class="ms-label">Accuracy</div>
                    <div class="ms-value">{clf_m['model_accuracy']:.1%}</div>
                </div>
                <div class="model-stat">
                    <div class="ms-label">Baseline</div>
                    <div class="ms-value">
                        {clf_m['baseline_accuracy']:.1%}
                        <span class="ms-delta">{delta_sign}{clf_m['model_vs_baseline_delta']:.1%}</span>
                    </div>
                </div>
                <div class="model-stat">
                    <div class="ms-label">Weather Lift</div>
                    <div class="ms-value">{clf_m['weather_delta']:+.1%}</div>
                </div>
                <div class="model-stat wide">
                    <div class="ms-label">Top Driver</div>
                    <div class="ms-value sm">{clf_feat}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_mc:
    reg_delta_class = "neg" if clf_m["model_vs_baseline_delta"] >= 0 else ""
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-card-title">Total Goals · Regression Model</div>
            <div class="model-card-row">
                <div class="model-stat">
                    <div class="ms-label">MAE</div>
                    <div class="ms-value">{reg_m['model_mae']:.4f}</div>
                </div>
                <div class="model-stat">
                    <div class="ms-label">Baseline MAE</div>
                    <div class="ms-value">
                        {reg_m['baseline_mae']:.4f}
                        <span class="ms-delta neg">-{abs(reg_m['model_vs_baseline_delta']):.4f}</span>
                    </div>
                </div>
                <div class="model-stat">
                    <div class="ms-label">Weather Δ</div>
                    <div class="ms-value">{reg_m['weather_delta']:+.4f}</div>
                </div>
                <div class="model-stat wide">
                    <div class="ms-label">Top Driver</div>
                    <div class="ms-value sm">{reg_feat}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Climate Overview",
    "🏆 Team Rankings",
    "⚽ Match Simulator",
    "🌡️ Future Scenarios",
    "⛈️ Extreme Weather",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Climate Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Inline filters
    f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
    with f_col1:
        all_seasons = sorted(df["season_year_start"].unique().tolist())
        season_labels = [f"{int(s)}-{int(s)+1}" for s in all_seasons]
        season_map = {f"{int(s)}-{int(s)+1}": int(s) for s in all_seasons}
        selected_seasons_labels = st.multiselect(
            "Season",
            options=season_labels,
            default=season_labels,
            key="t1_seasons",
        )
        selected_seasons = [season_map[lb] for lb in selected_seasons_labels] if selected_seasons_labels else all_seasons
    with f_col2:
        rain_filter = st.radio(
            "Rain",
            options=["All matches", "Rain only", "Dry only"],
            horizontal=True,
            key="t1_rain",
        )

    filt = df[df["season_year_start"].isin(selected_seasons)].copy()
    if rain_filter == "Rain only":
        filt = filt[filt["rain_flag"] == 1]
    elif rain_filter == "Dry only":
        filt = filt[filt["rain_flag"] == 0]

    # Row 1
    row1_left, row1_right = st.columns(2)

    with row1_left:
        temp_df = (
            filt.groupby("temperature_band", observed=False)
            .agg(matches=("date", "count"), avg_goals=("total_goals", "mean"))
            .reset_index()
            .dropna(subset=["temperature_band"])
        )
        temp_df["temperature_band"] = temp_df["temperature_band"].astype(str)
        chart_temp = (
            alt.Chart(temp_df)
            .mark_bar(color="#245F73")
            .encode(
                x=alt.X(
                    "temperature_band:N",
                    title="Temperature Band",
                    sort=TEMP_BAND_ORDER,
                ),
                y=alt.Y(
                    "avg_goals:Q",
                    title="Avg Goals",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip("temperature_band:N", title="Band"),
                    alt.Tooltip("matches:Q", title="Matches"),
                    alt.Tooltip("avg_goals:Q", format=".2f", title="Avg Goals"),
                ],
            )
            .properties(height=260, title="Goals by Temperature Band")
        )
        st.altair_chart(chart_temp, use_container_width=True)

    with row1_right:
        rain_ws_df = load_csv(ANALYSIS_DIR / "phase11_rain_winter_vs_summer.csv")
        rain_ws_df["condition_label"] = rain_ws_df["seasonal_indicator"].str.capitalize() + " / " + rain_ws_df["condition"]
        chart_rain_ws = (
            alt.Chart(rain_ws_df)
            .mark_bar()
            .encode(
                x=alt.X("condition_label:N", title="Condition", sort=None),
                y=alt.Y("avg_total_goals:Q", title="Avg Goals", scale=alt.Scale(zero=False)),
                color=alt.condition(
                    alt.datum.condition == "Rainy",
                    alt.value("#245F73"),
                    alt.value("#BBBDBC"),
                ),
                tooltip=[
                    alt.Tooltip("seasonal_indicator:N", title="Season"),
                    alt.Tooltip("condition:N", title="Rain"),
                    alt.Tooltip("matches:Q", title="Matches"),
                    alt.Tooltip("avg_total_goals:Q", format=".3f", title="Avg Goals"),
                ],
            )
            .properties(height=260, title="Rain Impact — Winter vs Summer")
        )
        st.altair_chart(chart_rain_ws, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Row 2
    row2_left, row2_right = st.columns([1.3, 1])

    with row2_left:
        perm_df = load_csv(ANALYSIS_DIR / "phase19_permutation_importance.csv").head(12).copy()
        perm_df = perm_df.sort_values("permutation_importance_mean", ascending=True)
        chart_perm = (
            alt.Chart(perm_df)
            .mark_bar()
            .encode(
                y=alt.Y("feature:N", title=None, sort=None),
                x=alt.X("permutation_importance_mean:Q", title="Permutation Importance (mean)"),
                color=alt.condition(
                    alt.datum.feature_group == "weather",
                    alt.value("#245F73"),
                    alt.value("#BBBDBC"),
                ),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("feature_group:N", title="Group"),
                    alt.Tooltip("permutation_importance_mean:Q", format=".5f", title="Mean Importance"),
                    alt.Tooltip("permutation_importance_std:Q", format=".5f", title="Std"),
                ],
            )
            .properties(height=320, title="Feature Importance (Permutation) — Top 12")
        )
        st.altair_chart(chart_perm, use_container_width=True)

    with row2_right:
        scenario_df = load_csv(ANALYSIS_DIR / "phase17_climate_change_scenarios.csv")
        display_cols = {
            "scenario": "Scenario",
            "predicted_total_goals": "Avg Goals",
            "goals_change_vs_current": "Goals Δ",
            "predicted_home_win_probability": "Home Win %",
        }
        scen_display = scenario_df[list(display_cols.keys())].rename(columns=display_cols).copy()
        scen_display["Scenario"] = scen_display["Scenario"].str.replace("_", " ").str.title()
        scen_display["Avg Goals"] = scen_display["Avg Goals"].map("{:.3f}".format)
        scen_display["Goals Δ"] = scen_display["Goals Δ"].map("{:+.4f}".format)
        scen_display["Home Win %"] = scen_display["Home Win %"].map("{:.1%}".format)
        st.markdown('<span class="section-label">Climate Change Scenarios</span>', unsafe_allow_html=True)
        st.dataframe(scen_display, height=340, hide_index=True, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Monthly heatmap
    st.markdown('<span class="section-label">Monthly Temperature × Goals Heatmap</span>', unsafe_allow_html=True)
    monthly_df = load_csv(ANALYSIS_DIR / "phase11_monthly_temperature_goals.csv").copy()
    monthly_df["temperature_band"] = monthly_df["temperature_band"].astype(str)

    base_heat = alt.Chart(monthly_df)

    rect_layer = base_heat.mark_rect().encode(
        x=alt.X("month_name:N", title="Month", sort=MONTH_ORDER),
        y=alt.Y("temperature_band:N", title="Temperature Band", sort=TEMP_BAND_ORDER),
        color=alt.Color(
            "avg_total_goals:Q",
            scale=alt.Scale(scheme="tealblues", domain=[2.0, 5.0]),
            title="Avg Goals",
        ),
        tooltip=[
            alt.Tooltip("month_name:N", title="Month"),
            alt.Tooltip("temperature_band:N", title="Temp Band"),
            alt.Tooltip("matches:Q", title="Matches"),
            alt.Tooltip("avg_total_goals:Q", format=".2f", title="Avg Goals"),
        ],
    )

    text_layer = (
        base_heat.transform_filter(alt.datum.matches >= 10)
        .mark_text(fontSize=10, fontWeight=600)
        .encode(
            x=alt.X("month_name:N", sort=MONTH_ORDER),
            y=alt.Y("temperature_band:N", sort=TEMP_BAND_ORDER),
            text=alt.Text("avg_total_goals:Q", format=".1f"),
            color=alt.condition(
                alt.datum.avg_total_goals > 3.5,
                alt.value("white"),
                alt.value("#374151"),
            ),
        )
    )

    heatmap_chart = (rect_layer + text_layer).properties(
        height=300,
        title="Average Total Goals by Month and Temperature Band",
    )
    st.altair_chart(heatmap_chart, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Team Rankings
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    csi_df = load_csv(ANALYSIS_DIR / "phase12_climate_sensitivity_index.csv")
    geo_df = load_csv(ANALYSIS_DIR / "phase18_team_geography.csv")

    geo_merge_cols = [c for c in geo_df.columns if c not in ["climate_sensitivity_index"]]
    merged_df = csi_df.merge(geo_df[geo_merge_cols], on="home_team", how="left")

    # Inline controls
    ctrl1, ctrl2, ctrl3 = st.columns([1.5, 1.5, 1])
    with ctrl1:
        view_mode = st.radio(
            "View",
            options=["Most Affected", "Most Resilient"],
            horizontal=True,
            key="t2_view_mode",
        )
    with ctrl2:
        total_teams = len(merged_df)
        n_teams = st.slider(
            "Number of teams",
            min_value=5,
            max_value=total_teams,
            value=min(15, total_teams),
            step=1,
            key="t2_n_teams",
        )
    with ctrl3:
        region_options = ["All"] + sorted(merged_df["region_group"].dropna().unique().tolist())
        region_sel = st.selectbox("Region", options=region_options, key="t2_region")

    display_df = merged_df.copy()
    if region_sel != "All":
        display_df = display_df[display_df["region_group"] == region_sel]

    if view_mode == "Most Affected":
        display_df = display_df.sort_values("climate_sensitivity_index", ascending=False).head(n_teams)
    else:
        display_df = display_df.sort_values("climate_sensitivity_index", ascending=True).head(n_teams)

    # Summary metrics
    s1, s2, s3, s4 = st.columns(4)
    most_affected_row = merged_df.loc[merged_df["climate_sensitivity_index"].idxmax()]
    most_resilient_row = merged_df.loc[merged_df["climate_sensitivity_index"].idxmin()]
    highly_affected_count = int((merged_df["climate_sensitivity_index"] >= 5).sum())
    resilient_count = int((merged_df["climate_sensitivity_index"] < 0).sum())

    s1.metric(
        "Most Affected",
        most_affected_row["home_team"],
        f"Index {most_affected_row['climate_sensitivity_index']:.1f}",
    )
    s2.metric(
        "Most Resilient",
        most_resilient_row["home_team"],
        f"Index {most_resilient_row['climate_sensitivity_index']:.1f}",
    )
    s3.metric("Highly Affected (≥5)", f"{highly_affected_count}")
    s4.metric("Resilient (<0)", f"{resilient_count}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    chart_col, donut_col = st.columns([2, 1])

    with chart_col:
        bar_df = display_df.copy()
        bar_df["region_group"] = bar_df["region_group"].fillna("Other")

        region_domain = ["North", "Midlands", "South", "Other"]
        region_range = ["#245F73", "#733E24", "#3d8fa8", "#9e5633"]

        bar_chart = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("climate_sensitivity_index:Q", title="Climate Sensitivity Index"),
                y=alt.Y(
                    "home_team:N",
                    title=None,
                    sort=alt.EncodingSortField(
                        field="climate_sensitivity_index",
                        order="descending" if view_mode == "Most Affected" else "ascending",
                    ),
                ),
                color=alt.Color(
                    "region_group:N",
                    scale=alt.Scale(domain=region_domain, range=region_range),
                    title="Region",
                ),
                tooltip=[
                    alt.Tooltip("home_team:N", title="Club"),
                    alt.Tooltip("region_group:N", title="Region"),
                    alt.Tooltip("climate_sensitivity_index:Q", format=".2f", title="Sensitivity Index"),
                    alt.Tooltip("climate_sensitivity_label:N", title="Label"),
                ],
            )
            .properties(height=max(280, n_teams * 22), title="Climate Sensitivity Index by Club")
        )

        zero_line_df = pd.DataFrame({"x": [0]})
        zero_rule = (
            alt.Chart(zero_line_df)
            .mark_rule(color="#374151", strokeDash=[4, 3], strokeWidth=1.5)
            .encode(x="x:Q")
        )

        st.altair_chart(bar_chart + zero_rule, use_container_width=True)

    with donut_col:
        label_counts = (
            merged_df["climate_sensitivity_label"]
            .value_counts()
            .reset_index()
            .rename(columns={"climate_sensitivity_label": "label", "count": "count"})
        )
        if "label" not in label_counts.columns:
            label_counts.columns = ["label", "count"]

        donut_domain = ["highly affected", "moderately affected", "weather-resilient"]
        donut_range = ["#733E24", "#9e5633", "#245F73"]

        donut_chart = (
            alt.Chart(label_counts)
            .mark_arc(innerRadius=52)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(domain=donut_domain, range=donut_range),
                    title="Sensitivity",
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Category"),
                    alt.Tooltip("count:Q", title="Teams"),
                ],
            )
            .properties(height=220, title="Sensitivity Distribution")
        )
        st.altair_chart(donut_chart, use_container_width=True)

        regional_stats = (
            merged_df.groupby("region_group")
            .agg(
                mean_index=("climate_sensitivity_index", "mean"),
                count=("home_team", "count"),
            )
            .reset_index()
            .rename(columns={"region_group": "Region", "mean_index": "Mean Index", "count": "Teams"})
            .sort_values("Mean Index", ascending=False)
        )
        regional_stats["Mean Index"] = regional_stats["Mean Index"].map("{:.2f}".format)
        st.dataframe(regional_stats, hide_index=True, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Map
    st.markdown('<span class="section-label">Stadium Locations</span>', unsafe_allow_html=True)
    map_df = geo_df[["latitude", "longitude"]].dropna().copy()
    map_df = map_df.rename(columns={"latitude": "lat", "longitude": "lon"})
    st.map(map_df, zoom=5)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.expander("Club Detail Table"):
        detail_cols = [
            c for c in [
                "home_team", "stadium_name", "city", "region_group",
                "climate_sensitivity_index", "climate_sensitivity_label",
            ]
            if c in merged_df.columns
        ]
        detail_display = merged_df[detail_cols].sort_values(
            "climate_sensitivity_index", ascending=False
        ).copy()
        if "climate_sensitivity_index" in detail_display.columns:
            detail_display["climate_sensitivity_index"] = detail_display["climate_sensitivity_index"].map("{:.2f}".format)
        st.dataframe(detail_display, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Match Simulator
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    _MONTH_NAMES = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }

    all_teams = sorted(df["home_team"].unique().tolist())

    left_sim, right_sim = st.columns([1, 1.4])

    with left_sim:
        st.markdown('<span class="section-label">Fixture Setup</span>', unsafe_allow_html=True)

        home_default = all_teams.index("Arsenal") if "Arsenal" in all_teams else 0
        away_default = all_teams.index("Liverpool") if "Liverpool" in all_teams else 1

        home_team = st.selectbox("Home Team", options=all_teams, index=home_default, key="sim_home")
        away_team = st.selectbox("Away Team", options=all_teams, index=away_default, key="sim_away")

        match_month = st.select_slider(
            "Match Month",
            options=list(range(1, 13)),
            value=1,
            format_func=lambda m: _MONTH_NAMES[m],
            key="sim_month",
        )

        st.markdown('<span class="section-label">Weather Conditions</span>', unsafe_allow_html=True)
        sim_temperature = st.slider(
            "Temperature (°C)", min_value=-5.0, max_value=35.0, value=12.0, step=0.5, key="sim_temp"
        )
        sim_rain = st.slider(
            "Rainfall (mm)", min_value=0.0, max_value=40.0, value=3.0, step=0.5, key="sim_rain"
        )
        sim_wind = st.slider(
            "Wind Speed (km/h)", min_value=0.0, max_value=80.0, value=20.0, step=1.0, key="sim_wind"
        )
        sim_humidity = st.slider(
            "Humidity (%)", min_value=30.0, max_value=100.0, value=75.0, step=1.0, key="sim_humidity"
        )

    # Run predictions
    input_row, pred_result, pred_goals, prob_df = predict_fixture(
        df,
        home_team=home_team,
        away_team=away_team,
        temperature=sim_temperature,
        rain=sim_rain,
        wind=sim_wind,
        humidity=sim_humidity,
        month=match_month,
    )

    _, base_result, base_goals, base_prob_df = predict_fixture(
        df,
        home_team=home_team,
        away_team=away_team,
        temperature=15.0,
        rain=0.0,
        wind=15.0,
        humidity=65.0,
        month=match_month,
    )

    comparison_df = prob_df.merge(
        base_prob_df.rename(columns={"probability": "baseline_probability"}),
        on="result",
    )
    comparison_df["delta"] = comparison_df["probability"] - comparison_df["baseline_probability"]

    _result_pill_class = {
        "win": "result-win",
        "draw": "result-draw",
        "loss": "result-loss",
    }
    _result_display = RESULT_LABELS.get(pred_result, pred_result)
    _pill_class = _result_pill_class.get(pred_result, "result-draw")

    home_win_prob = prob_df.loc[
        prob_df["result"] == RESULT_LABELS.get("win", "Home Win"), "probability"
    ].values
    home_win_prob_val = float(home_win_prob[0]) if len(home_win_prob) > 0 else 0.0

    base_home_win_prob = base_prob_df.loc[
        base_prob_df["result"] == RESULT_LABELS.get("win", "Home Win"), "probability"
    ].values
    base_home_win_prob_val = float(base_home_win_prob[0]) if len(base_home_win_prob) > 0 else 0.0

    goals_delta = pred_goals - base_goals
    hw_delta = home_win_prob_val - base_home_win_prob_val

    with right_sim:
        st.markdown(
            f'<span class="result-pill {_pill_class}">{_result_display}</span>',
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Expected Goals",
            f"{pred_goals:.2f}",
            f"{goals_delta:+.2f} vs baseline",
        )
        m2.metric(
            "Home Win Prob",
            f"{home_win_prob_val:.1%}",
            f"{hw_delta:+.1%} vs baseline",
        )
        m3.metric(
            "Baseline Result",
            RESULT_LABELS.get(base_result, base_result),
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Probability delta chart
        delta_chart = (
            alt.Chart(comparison_df)
            .mark_bar()
            .encode(
                x=alt.X("result:N", title="Outcome"),
                y=alt.Y("delta:Q", title="Probability Δ vs Baseline"),
                color=alt.condition(
                    alt.datum.delta >= 0,
                    alt.value("#245F73"),
                    alt.value("#733E24"),
                ),
                tooltip=[
                    alt.Tooltip("result:N", title="Outcome"),
                    alt.Tooltip("delta:Q", format="+.3f", title="Probability Δ"),
                    alt.Tooltip("probability:Q", format=".3f", title="Current Prob"),
                    alt.Tooltip("baseline_probability:Q", format=".3f", title="Baseline Prob"),
                ],
            )
            .properties(height=200, title="Probability Δ vs Mild Baseline")
        )
        zero_line_df2 = pd.DataFrame({"y": [0]})
        zero_rule2 = (
            alt.Chart(zero_line_df2)
            .mark_rule(color="#374151", strokeDash=[4, 3], strokeWidth=1.5)
            .encode(y="y:Q")
        )
        st.altair_chart(delta_chart + zero_rule2, use_container_width=True)

        # Full outcome probabilities
        folded_df = comparison_df.copy()
        prob_chart_df = pd.concat([
            comparison_df[["result", "probability"]].assign(scenario="Current"),
            comparison_df[["result", "baseline_probability"]].rename(
                columns={"baseline_probability": "probability"}
            ).assign(scenario="Baseline"),
        ])

        full_prob_chart = (
            alt.Chart(prob_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("result:N", title="Outcome"),
                y=alt.Y("probability:Q", title="Probability"),
                color=alt.Color(
                    "scenario:N",
                    scale=alt.Scale(
                        domain=["Current", "Baseline"],
                        range=["#245F73", "#BBBDBC"],
                    ),
                    title="Scenario",
                ),
                xOffset="scenario:N",
                tooltip=[
                    alt.Tooltip("result:N", title="Outcome"),
                    alt.Tooltip("scenario:N", title="Scenario"),
                    alt.Tooltip("probability:Q", format=".3f", title="Probability"),
                ],
            )
            .properties(height=220, title="Full Outcome Probabilities")
        )
        st.altair_chart(full_prob_chart, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Weather flags badges
    apparent_temp = float(input_row["apparent_temperature"].iloc[0])
    apparent_gap = float(input_row["apparent_temperature_gap"].iloc[0])

    badges_html = '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin:0.5rem 0;">'
    if sim_rain > 0:
        badges_html += '<span class="badge badge-teal">🌧 Rain</span>'
    if sim_rain >= HEAVY_RAIN:
        badges_html += '<span class="badge badge-rust">⛈ Heavy Rain</span>'
    if sim_temperature < COLD_TEMP:
        badges_html += '<span class="badge badge-teal">🥶 Cold</span>'
    if sim_temperature >= HOT_TEMP:
        badges_html += '<span class="badge badge-rust">🌡 Hot</span>'
    if sim_wind >= WINDY_SPEED:
        badges_html += '<span class="badge badge-gray">💨 Windy</span>'
    if sim_wind >= STRONG_WIND:
        badges_html += '<span class="badge badge-rust">🌬 Strong Wind</span>'
    if sim_humidity >= HUMIDITY_HIGH:
        badges_html += '<span class="badge badge-gray">💧 High Humidity</span>'
    if apparent_temp <= FREEZING_APPARENT:
        badges_html += '<span class="badge badge-rust">❄ Freezing (Apparent)</span>'
    if apparent_temp >= HEATWAVE_APPARENT:
        badges_html += '<span class="badge badge-rust">🔥 Heatwave (Apparent)</span>'
    if sim_rain == 0 and sim_wind < WINDY_SPEED and sim_temperature >= COLD_TEMP and sim_temperature < HOT_TEMP:
        badges_html += '<span class="badge badge-green">✓ Mild Conditions</span>'
    badges_html += "</div>"
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Weather detail metrics
    wm1, wm2, wm3, wm4, wm5 = st.columns(5)
    wm1.metric("Temperature", f"{sim_temperature:.1f}°C")
    wm2.metric("Apparent Temp", f"{apparent_temp:.1f}°C", f"{apparent_gap:+.1f}°C gap")
    wm3.metric("Rainfall", f"{sim_rain:.1f} mm")
    wm4.metric("Wind Speed", f"{sim_wind:.0f} km/h")
    wm5.metric("Humidity", f"{sim_humidity:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Future Scenarios
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<span class="section-label">Custom Climate Scenario</span>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        temp_delta = st.slider(
            "Temperature Δ (°C)",
            min_value=-2.0,
            max_value=6.0,
            value=2.0,
            step=0.5,
            key="fs_temp_delta",
        )
    with sc2:
        rain_mult = st.slider(
            "Rainfall Multiplier",
            min_value=0.3,
            max_value=3.0,
            value=1.3,
            step=0.05,
            key="fs_rain_mult",
        )
    with sc3:
        humidity_delta = st.slider(
            "Humidity Δ (%)",
            min_value=-15,
            max_value=20,
            value=4,
            step=1,
            key="fs_humidity_delta",
        )

    # Build scenario df
    scenario_df_fs = df.copy()
    scenario_df_fs["temperature"] = scenario_df_fs["temperature"] + temp_delta
    scenario_df_fs["apparent_temperature"] = scenario_df_fs["apparent_temperature"] + temp_delta * 0.9
    scenario_df_fs["rain"] = scenario_df_fs["rain"] * rain_mult
    scenario_df_fs["humidity"] = (scenario_df_fs["humidity"] + humidity_delta).clip(0, 100)

    # Recompute flag columns
    scenario_df_fs["rain_flag"] = (scenario_df_fs["rain"] > 0).astype(int)
    scenario_df_fs["cold_flag"] = (scenario_df_fs["temperature"] < COLD_TEMP).astype(int)
    scenario_df_fs["hot_flag"] = (scenario_df_fs["temperature"] >= HOT_TEMP).astype(int)
    scenario_df_fs["windy_flag"] = (scenario_df_fs["wind"] >= WINDY_SPEED).astype(int)
    scenario_df_fs["humidity_flag"] = (scenario_df_fs["humidity"] >= HUMIDITY_HIGH).astype(int)
    scenario_df_fs["apparent_cold_flag"] = (scenario_df_fs["apparent_temperature"] < COLD_TEMP).astype(int)
    scenario_df_fs["apparent_hot_flag"] = (scenario_df_fs["apparent_temperature"] >= HOT_TEMP).astype(int)
    scenario_df_fs["apparent_temperature_gap"] = (
        scenario_df_fs["apparent_temperature"] - scenario_df_fs["temperature"]
    )
    scenario_df_fs["heavy_rain_flag"] = (scenario_df_fs["rain"] >= HEAVY_RAIN).astype(int)
    scenario_df_fs["strong_wind_flag"] = (scenario_df_fs["wind"] >= STRONG_WIND).astype(int)
    scenario_df_fs["freezing_flag"] = (scenario_df_fs["apparent_temperature"] <= FREEZING_APPARENT).astype(int)
    scenario_df_fs["heatwave_flag"] = (scenario_df_fs["apparent_temperature"] >= HEATWAVE_APPARENT).astype(int)
    scenario_df_fs["extreme_weather_flag"] = (
        (scenario_df_fs["heavy_rain_flag"] == 1)
        | (scenario_df_fs["strong_wind_flag"] == 1)
        | (scenario_df_fs["freezing_flag"] == 1)
        | (scenario_df_fs["heatwave_flag"] == 1)
    ).astype(int)

    # Load models and compute predictions
    clf_model, reg_model = load_models()

    baseline_feat = df[FEATURE_COLUMNS].copy()
    scenario_feat = scenario_df_fs[FEATURE_COLUMNS].copy()

    baseline_probs_all = clf_model.predict_proba(baseline_feat)
    scenario_probs_all = clf_model.predict_proba(scenario_feat)

    classes = list(clf_model.classes_)
    hw_idx = classes.index("win") if "win" in classes else 0
    draw_idx = classes.index("draw") if "draw" in classes else 1
    aw_idx = classes.index("loss") if "loss" in classes else 2

    baseline_goals = float(reg_model.predict(baseline_feat).mean())
    scenario_goals = float(reg_model.predict(scenario_feat).mean())

    baseline_hw = float(baseline_probs_all[:, hw_idx].mean())
    scenario_hw = float(scenario_probs_all[:, hw_idx].mean())
    baseline_draw = float(baseline_probs_all[:, draw_idx].mean())
    scenario_draw = float(scenario_probs_all[:, draw_idx].mean())
    baseline_aw = float(baseline_probs_all[:, aw_idx].mean())
    scenario_aw = float(scenario_probs_all[:, aw_idx].mean())

    baseline_extreme = float(df["extreme_weather_flag"].mean())
    scenario_extreme = float(scenario_df_fs["extreme_weather_flag"].mean())

    goals_delta_fs = scenario_goals - baseline_goals
    hw_delta_fs = scenario_hw - baseline_hw
    draw_delta_fs = scenario_draw - baseline_draw
    extreme_delta_fs = scenario_extreme - baseline_extreme

    # Delta metrics
    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("Predicted Goals Δ", f"{goals_delta_fs:+.3f}", f"{goals_delta_fs/max(baseline_goals,0.001):+.1%}")
    dm2.metric("Home Win Prob Δ", f"{hw_delta_fs:+.1%}", f"From {baseline_hw:.1%}")
    dm3.metric("Draw Prob Δ", f"{draw_delta_fs:+.1%}", f"From {baseline_draw:.1%}")
    dm4.metric("Extreme Weather Share Δ", f"{extreme_delta_fs:+.1%}", f"From {baseline_extreme:.1%}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Comparison dataframe
    comparison_rows = {
        "Metric": [
            "Avg Predicted Goals",
            "Home Win Probability",
            "Draw Probability",
            "Away Win Probability",
            "Extreme Weather Share",
            "Rain Share",
            "Cold Match Share",
        ],
        "Current Climate": [
            f"{baseline_goals:.3f}",
            f"{baseline_hw:.1%}",
            f"{baseline_draw:.1%}",
            f"{baseline_aw:.1%}",
            f"{baseline_extreme:.1%}",
            f"{df['rain_flag'].mean():.1%}",
            f"{df['cold_flag'].mean():.1%}",
        ],
        "Your Scenario": [
            f"{scenario_goals:.3f}",
            f"{scenario_hw:.1%}",
            f"{scenario_draw:.1%}",
            f"{scenario_aw:.1%}",
            f"{scenario_extreme:.1%}",
            f"{scenario_df_fs['rain_flag'].mean():.1%}",
            f"{scenario_df_fs['cold_flag'].mean():.1%}",
        ],
    }
    st.markdown('<span class="section-label">Full Comparison</span>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Outcome probability delta chart
    outcome_delta_df = pd.DataFrame({
        "outcome": ["Home Win", "Draw", "Away Win"],
        "delta": [hw_delta_fs, draw_delta_fs, scenario_aw - baseline_aw],
    })

    out_delta_chart = (
        alt.Chart(outcome_delta_df)
        .mark_bar()
        .encode(
            x=alt.X("outcome:N", title="Outcome"),
            y=alt.Y("delta:Q", title="Probability Δ"),
            color=alt.condition(
                alt.datum.delta >= 0,
                alt.value("#245F73"),
                alt.value("#733E24"),
            ),
            tooltip=[
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("delta:Q", format="+.4f", title="Probability Δ"),
            ],
        )
        .properties(height=240, title="Outcome Probability Δ: Your Scenario vs Current Climate")
    )

    zero_line_df_fs = pd.DataFrame({"y": [0]})
    zero_rule_fs = (
        alt.Chart(zero_line_df_fs)
        .mark_rule(color="#374151", strokeDash=[4, 3], strokeWidth=1.5)
        .encode(y="y:Q")
    )

    st.altair_chart(out_delta_chart + zero_rule_fs, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.expander("Pre-computed Climate Change Scenarios"):
        precomp_df = load_csv(ANALYSIS_DIR / "phase17_climate_change_scenarios.csv")
        precomp_display = precomp_df.copy()
        precomp_display["scenario"] = precomp_display["scenario"].str.replace("_", " ").str.title()
        if "predicted_total_goals" in precomp_display.columns:
            precomp_display["predicted_total_goals"] = precomp_display["predicted_total_goals"].map("{:.3f}".format)
        if "predicted_home_win_probability" in precomp_display.columns:
            precomp_display["predicted_home_win_probability"] = precomp_display["predicted_home_win_probability"].map("{:.1%}".format)
        if "goals_change_vs_current" in precomp_display.columns:
            precomp_display["goals_change_vs_current"] = precomp_display["goals_change_vs_current"].map("{:+.4f}".format)
        if "home_win_probability_change_vs_current" in precomp_display.columns:
            precomp_display["home_win_probability_change_vs_current"] = precomp_display["home_win_probability_change_vs_current"].map("{:+.4f}".format)
        st.dataframe(precomp_display, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="info-box">Scenario probabilities are computed by re-running the trained models over all historical matches with adjusted weather inputs. '
        "This simulates how outcomes would change under a systematic climate shift across the league.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Extreme Weather
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    summary_df = load_csv(ANALYSIS_DIR / "phase15_extreme_weather_summary.csv")

    ew_df = df.copy()
    ew_df["very_low_temperature_flag"] = (ew_df["temperature"] <= 5.0).astype(int)

    CONDITION_MAP = {
        "Heavy Rain (≥10mm)": "heavy_rain_flag",
        "Strong Wind (≥50km/h)": "strong_wind_flag",
        "Very Low Temperature (≤5°C)": "very_low_temperature_flag",
        "Any Extreme": "extreme_weather_flag",
    }

    SUMMARY_KEY_MAP = {
        "Heavy Rain (≥10mm)": "heavy_rain",
        "Strong Wind (≥50km/h)": "extreme_wind",
        "Very Low Temperature (≤5°C)": "very_low_temperature",
        "Any Extreme": "any_extreme",
    }

    # Inline filters
    filter_col1, filter_col2 = st.columns([1, 1.5])
    with filter_col1:
        condition_label = st.selectbox(
            "Extreme Condition",
            options=list(CONDITION_MAP.keys()),
            key="ew_condition",
        )
    with filter_col2:
        ew_all_teams = ["All Teams"] + sorted(ew_df["home_team"].unique().tolist())
        team_filter_ew = st.selectbox("Team (Home)", options=ew_all_teams, key="ew_team")
        all_seasons_ew = sorted(ew_df["season_year_start"].unique().tolist())
        season_labels_ew = [f"{int(s)}-{int(s)+1}" for s in all_seasons_ew]
        season_map_ew = {f"{int(s)}-{int(s)+1}": int(s) for s in all_seasons_ew}
        selected_seasons_ew_labels = st.multiselect(
            "Season",
            options=season_labels_ew,
            default=season_labels_ew,
            key="ew_seasons",
        )
        selected_seasons_ew = (
            [season_map_ew[lb] for lb in selected_seasons_ew_labels]
            if selected_seasons_ew_labels
            else all_seasons_ew
        )
        sort_by_ew = st.radio(
            "Sort by",
            options=["Date", "Goals", "Temperature"],
            horizontal=True,
            key="ew_sort",
        )

    # Apply filters
    condition_col = CONDITION_MAP[condition_label]
    filtered_ew = ew_df[ew_df[condition_col] == 1].copy()
    filtered_ew = filtered_ew[filtered_ew["season_year_start"].isin(selected_seasons_ew)]
    if team_filter_ew != "All Teams":
        filtered_ew = filtered_ew[filtered_ew["home_team"] == team_filter_ew]

    if sort_by_ew == "Date":
        filtered_ew = filtered_ew.sort_values("date", ascending=False)
    elif sort_by_ew == "Goals":
        filtered_ew = filtered_ew.sort_values("total_goals", ascending=False)
    elif sort_by_ew == "Temperature":
        filtered_ew = filtered_ew.sort_values("temperature", ascending=True)

    # Summary row from summary_df
    summary_key = SUMMARY_KEY_MAP[condition_label]
    summary_row = summary_df[summary_df["extreme_type"] == summary_key]
    if not summary_row.empty:
        sr = summary_row.iloc[0]
        extreme_matches_count = int(sr.get("extreme_matches", len(filtered_ew)))
        goals_delta_ew = float(sr.get("goals_delta_vs_baseline", 0.0))
        hw_delta_ew = float(sr.get("home_win_delta_vs_baseline", 0.0))
        draw_delta_ew = float(sr.get("draw_delta_vs_baseline", 0.0))
    else:
        extreme_matches_count = len(filtered_ew)
        goals_delta_ew = 0.0
        hw_delta_ew = 0.0
        draw_delta_ew = 0.0

    em1, em2, em3, em4 = st.columns(4)
    em1.metric("Extreme Matches", f"{extreme_matches_count:,}")
    em2.metric("Goals Δ vs Baseline", f"{goals_delta_ew:+.3f}")
    em3.metric("Home Win Rate Δ", f"{hw_delta_ew:+.1%}")
    em4.metric("Draw Rate Δ", f"{draw_delta_ew:+.1%}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Overview charts for all conditions
    ov_col1, ov_col2 = st.columns(2)

    # Build all-conditions summary
    all_cond_goals = []
    all_cond_draw = []
    for cond_lbl, s_key in SUMMARY_KEY_MAP.items():
        row = summary_df[summary_df["extreme_type"] == s_key]
        if not row.empty:
            r = row.iloc[0]
            all_cond_goals.append({
                "condition": cond_lbl,
                "goals_delta": float(r.get("goals_delta_vs_baseline", 0.0)),
            })
            all_cond_draw.append({
                "condition": cond_lbl,
                "draw_delta": float(r.get("draw_delta_vs_baseline", 0.0)),
            })

    with ov_col1:
        if all_cond_goals:
            goals_ov_df = pd.DataFrame(all_cond_goals)
            goals_ov_chart = (
                alt.Chart(goals_ov_df)
                .mark_bar()
                .encode(
                    x=alt.X("condition:N", title="Condition", sort=None),
                    y=alt.Y("goals_delta:Q", title="Goals Δ vs Baseline"),
                    color=alt.condition(
                        alt.datum.goals_delta >= 0,
                        alt.value("#059669"),
                        alt.value("#733E24"),
                    ),
                    tooltip=[
                        alt.Tooltip("condition:N", title="Condition"),
                        alt.Tooltip("goals_delta:Q", format="+.3f", title="Goals Δ"),
                    ],
                )
                .properties(height=240, title="Goals Δ vs Baseline — All Conditions")
            )
            zero_goals_df = pd.DataFrame({"y": [0]})
            zero_goals_rule = (
                alt.Chart(zero_goals_df)
                .mark_rule(color="#374151", strokeDash=[4, 3], strokeWidth=1.5)
                .encode(y="y:Q")
            )
            st.altair_chart(goals_ov_chart + zero_goals_rule, use_container_width=True)

    with ov_col2:
        if all_cond_draw:
            draw_ov_df = pd.DataFrame(all_cond_draw)
            draw_ov_chart = (
                alt.Chart(draw_ov_df)
                .mark_bar()
                .encode(
                    x=alt.X("condition:N", title="Condition", sort=None),
                    y=alt.Y("draw_delta:Q", title="Draw Rate Δ vs Baseline"),
                    color=alt.condition(
                        alt.datum.draw_delta >= 0,
                        alt.value("#245F73"),
                        alt.value("#d97706"),
                    ),
                    tooltip=[
                        alt.Tooltip("condition:N", title="Condition"),
                        alt.Tooltip("draw_delta:Q", format="+.3f", title="Draw Rate Δ"),
                    ],
                )
                .properties(height=240, title="Draw Rate Δ vs Baseline — All Conditions")
            )
            zero_draw_df = pd.DataFrame({"y": [0]})
            zero_draw_rule = (
                alt.Chart(zero_draw_df)
                .mark_rule(color="#374151", strokeDash=[4, 3], strokeWidth=1.5)
                .encode(y="y:Q")
            )
            st.altair_chart(draw_ov_chart + zero_draw_rule, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Match history
    st.markdown(
        f'<span class="section-label">Match History — {condition_label} ({len(filtered_ew):,} matches)</span>',
        unsafe_allow_html=True,
    )

    history_cols = [
        c for c in [
            "date", "home_team", "away_team", "temperature", "rain",
            "wind", "humidity", "total_goals", "result_label",
        ]
        if c in filtered_ew.columns
    ]
    history_display = filtered_ew[history_cols].head(200).copy()
    if "date" in history_display.columns:
        history_display["date"] = history_display["date"].dt.strftime("%Y-%m-%d")
    rename_hist = {
        "home_team": "Home",
        "away_team": "Away",
        "temperature": "Temp (°C)",
        "rain": "Rain (mm)",
        "wind": "Wind (km/h)",
        "humidity": "Humidity (%)",
        "total_goals": "Goals",
        "result_label": "Result",
    }
    history_display = history_display.rename(columns={k: v for k, v in rename_hist.items() if k in history_display.columns})
    if "Temp (°C)" in history_display.columns:
        history_display["Temp (°C)"] = history_display["Temp (°C)"].map("{:.1f}".format)
    if "Rain (mm)" in history_display.columns:
        history_display["Rain (mm)"] = history_display["Rain (mm)"].map("{:.1f}".format)
    if "Wind (km/h)" in history_display.columns:
        history_display["Wind (km/h)"] = history_display["Wind (km/h)"].map("{:.0f}".format)
    if "Humidity (%)" in history_display.columns:
        history_display["Humidity (%)"] = history_display["Humidity (%)"].map("{:.0f}".format)
    st.dataframe(history_display, hide_index=True, use_container_width=True)

    if len(filtered_ew) >= 10:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            f'<span class="section-label">Goals Distribution — {condition_label}</span>',
            unsafe_allow_html=True,
        )
        goals_dist_df = (
            filtered_ew["total_goals"]
            .value_counts()
            .reset_index()
            .rename(columns={"total_goals": "goals", "count": "matches"})
            .sort_values("goals")
        )
        if "goals" not in goals_dist_df.columns:
            goals_dist_df.columns = ["goals", "matches"]
        goals_dist_chart = (
            alt.Chart(goals_dist_df)
            .mark_bar(color="#245F73")
            .encode(
                x=alt.X("goals:O", title="Total Goals"),
                y=alt.Y("matches:Q", title="Matches"),
                tooltip=[
                    alt.Tooltip("goals:O", title="Goals"),
                    alt.Tooltip("matches:Q", title="Matches"),
                ],
            )
            .properties(height=220, title=f"Goals Distribution — {condition_label}")
        )
        st.altair_chart(goals_dist_chart, use_container_width=True)
