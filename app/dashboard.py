from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FEATURE_DATA_PATH = DATA_DIR / "processed" / "feature_dataset.csv"
MODELS_DIR = DATA_DIR / "processed" / "models"
ANALYSIS_DIR = DATA_DIR / "processed" / "analysis"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "match_outcome_model.joblib"
REGRESSION_MODEL_PATH = MODELS_DIR / "total_goals_model.joblib"

COLD_TEMPERATURE_THRESHOLD = 10.0
HOT_TEMPERATURE_THRESHOLD = 25.0
WINDY_SPEED_THRESHOLD = 30.0
HUMIDITY_THRESHOLD = 85.0
HEAVY_RAIN_THRESHOLD = 10.0
STRONG_WIND_THRESHOLD = 50.0
FREEZING_APPARENT_TEMPERATURE_THRESHOLD = 0.0
HEATWAVE_APPARENT_TEMPERATURE_THRESHOLD = 30.0

RESULT_LABELS = {"win": "Home Win", "draw": "Draw", "loss": "Away Win"}
FEATURE_COLUMNS = [
    "temperature",
    "apparent_temperature",
    "humidity",
    "rain",
    "wind",
    "heavy_rain_flag",
    "strong_wind_flag",
    "freezing_flag",
    "heatwave_flag",
    "storm_flag",
    "extreme_weather_flag",
    "rain_flag",
    "cold_flag",
    "hot_flag",
    "windy_flag",
    "humidity_flag",
    "apparent_cold_flag",
    "apparent_hot_flag",
    "apparent_temperature_gap",
    "home_advantage",
    "match_month",
    "season_year_start",
    "season_progress",
    "is_winter_match",
    "is_spring_match",
    "is_summer_match",
    "is_autumn_match",
    "home_team_points_avg_last_5",
    "home_team_goals_for_avg_last_5",
    "home_team_goals_against_avg_last_5",
    "home_team_goal_difference_avg_last_5",
    "away_team_points_avg_last_5",
    "away_team_goals_for_avg_last_5",
    "away_team_goals_against_avg_last_5",
    "away_team_goal_difference_avg_last_5",
    "home_team_strength_proxy",
    "away_team_strength_proxy",
    "strength_gap",
    "match_importance_proxy",
]
HOME_FORM_COLUMNS = [
    "home_team_points_avg_last_5",
    "home_team_goals_for_avg_last_5",
    "home_team_goals_against_avg_last_5",
    "home_team_goal_difference_avg_last_5",
    "home_team_strength_proxy",
]
AWAY_FORM_COLUMNS = [
    "away_team_points_avg_last_5",
    "away_team_goals_for_avg_last_5",
    "away_team_goals_against_avg_last_5",
    "away_team_goal_difference_avg_last_5",
    "away_team_strength_proxy",
]

st.set_page_config(page_title="Premier League Climate Intelligence", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #fbfcfd 0%, #f2f6fa 100%);
            border: 1px solid #d9e3ee;
            border-radius: 14px;
            padding: 0.8rem 1rem;
        }
        .section-note {
            border-left: 4px solid #1f5aa6;
            background: #f4f8fc;
            padding: 0.9rem 1rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_feature_data() -> pd.DataFrame:
    dataset_df = pd.read_csv(FEATURE_DATA_PATH)
    dataset_df["date"] = pd.to_datetime(dataset_df["date"], format="%Y-%m-%d")
    dataset_df["result_label"] = dataset_df["match_result"].map(RESULT_LABELS)
    dataset_df["rain_label"] = dataset_df["rain_flag"].map({1: "Rain", 0: "No Rain"})
    dataset_df["temperature_band"] = pd.cut(
        dataset_df["temperature"],
        bins=[-10, 5, 10, 15, 20, 25, 40],
        labels=["<5C", "5-10C", "10-15C", "15-20C", "20-25C", "25C+"],
        include_lowest=True,
    )
    return dataset_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_models():
    return joblib.load(CLASSIFICATION_MODEL_PATH), joblib.load(REGRESSION_MODEL_PATH)


def ensure_artifacts() -> None:
    required_paths = [
        FEATURE_DATA_PATH,
        CLASSIFICATION_MODEL_PATH,
        REGRESSION_MODEL_PATH,
        ANALYSIS_DIR / "phase12_climate_sensitivity_index.csv",
        ANALYSIS_DIR / "phase15_extreme_weather_summary.csv",
        ANALYSIS_DIR / "phase17_climate_change_scenarios.csv",
        ANALYSIS_DIR / "phase19_permutation_importance.csv",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        st.error("Missing required pipeline outputs.")
        st.code("\n".join(str(path) for path in missing))
        st.stop()


def compute_weather_fields(row: dict[str, float]) -> dict[str, float]:
    apparent_temperature = row["temperature"] - min(row["wind"] * 0.03, 3.0) - min(row["rain"] * 0.08, 1.8)
    row["apparent_temperature"] = apparent_temperature
    row["rain_flag"] = int(row["rain"] > 0)
    row["cold_flag"] = int(row["temperature"] < COLD_TEMPERATURE_THRESHOLD)
    row["hot_flag"] = int(row["temperature"] >= HOT_TEMPERATURE_THRESHOLD)
    row["windy_flag"] = int(row["wind"] >= WINDY_SPEED_THRESHOLD)
    row["humidity_flag"] = int(row["humidity"] >= HUMIDITY_THRESHOLD)
    row["apparent_cold_flag"] = int(apparent_temperature < COLD_TEMPERATURE_THRESHOLD)
    row["apparent_hot_flag"] = int(apparent_temperature >= HOT_TEMPERATURE_THRESHOLD)
    row["apparent_temperature_gap"] = apparent_temperature - row["temperature"]
    row["heavy_rain_flag"] = int(row["rain"] >= HEAVY_RAIN_THRESHOLD)
    row["strong_wind_flag"] = int(row["wind"] >= STRONG_WIND_THRESHOLD)
    row["freezing_flag"] = int(apparent_temperature <= FREEZING_APPARENT_TEMPERATURE_THRESHOLD)
    row["heatwave_flag"] = int(apparent_temperature >= HEATWAVE_APPARENT_TEMPERATURE_THRESHOLD)
    row["storm_flag"] = 0
    row["extreme_weather_flag"] = int(
        row["heavy_rain_flag"] or row["strong_wind_flag"] or row["freezing_flag"] or row["heatwave_flag"]
    )
    return row


def get_team_context(dataset_df: pd.DataFrame, team_name: str, side: str) -> dict[str, float]:
    column_name = "home_team" if side == "home" else "away_team"
    default_columns = HOME_FORM_COLUMNS if side == "home" else AWAY_FORM_COLUMNS
    team_df = dataset_df.loc[dataset_df[column_name] == team_name].sort_values("date")
    if team_df.empty:
        return dataset_df[default_columns].median().to_dict()
    return team_df.iloc[-1][default_columns].to_dict()


def build_prediction_row(
    dataset_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    temperature: float,
    rain: float,
    wind: float,
    humidity: float,
    month: int,
) -> pd.DataFrame:
    row: dict[str, float] = {
        "temperature": temperature,
        "rain": rain,
        "wind": wind,
        "humidity": humidity,
        "home_advantage": 1,
        "match_month": month,
        "season_year_start": int(dataset_df["season_year_start"].max()),
        "season_progress": float(dataset_df["season_progress"].median()),
    }
    row = compute_weather_fields(row)
    row["is_winter_match"] = int(month in [12, 1, 2])
    row["is_spring_match"] = int(month in [3, 4, 5])
    row["is_summer_match"] = int(month in [6, 7, 8])
    row["is_autumn_match"] = int(month in [9, 10, 11])
    row.update(get_team_context(dataset_df, home_team, "home"))
    row.update(get_team_context(dataset_df, away_team, "away"))
    row["strength_gap"] = row["home_team_strength_proxy"] - row["away_team_strength_proxy"]
    row["match_importance_proxy"] = row["season_progress"] * (
        row["home_team_strength_proxy"] + row["away_team_strength_proxy"]
    ) / 2
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_fixture(
    dataset_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    temperature: float,
    rain: float,
    wind: float,
    humidity: float,
    month: int,
) -> tuple[pd.DataFrame, str, float, pd.DataFrame]:
    classification_model, regression_model = load_models()
    input_df = build_prediction_row(dataset_df, home_team, away_team, temperature, rain, wind, humidity, month)
    probabilities = classification_model.predict_proba(input_df)[0]
    predicted_result = classification_model.predict(input_df)[0]
    predicted_goals = float(regression_model.predict(input_df)[0])
    probability_df = pd.DataFrame(
        {
            "result": [RESULT_LABELS[label] for label in classification_model.classes_],
            "probability": probabilities,
        }
    )
    return input_df, predicted_result, predicted_goals, probability_df


def render_header(dataset_df: pd.DataFrame, evaluation_metrics: dict) -> None:
    st.title("Premier League Climate Intelligence")
    st.caption("Weather-aware football analytics with team sensitivity, simulation, future scenarios, and extreme-event analysis.")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Matches", f"{len(dataset_df):,}")
    metric_cols[1].metric("Avg Goals", f"{dataset_df['total_goals'].mean():.2f}")
    metric_cols[2].metric("Rain Share", f"{dataset_df['rain_flag'].mean():.1%}")
    metric_cols[3].metric("Outcome Accuracy", f"{evaluation_metrics['classification']['model_accuracy']:.3f}")
    metric_cols[4].metric("Goals MAE", f"{evaluation_metrics['regression']['model_mae']:.3f}")


def render_climate_overview(dataset_df: pd.DataFrame) -> None:
    st.subheader("Climate Impact Overview")
    st.markdown('<div class="section-note">Global readout of how weather, seasonality, and learned model drivers interact across the league.</div>', unsafe_allow_html=True)
    temp_goals_df = (
        dataset_df.groupby("temperature_band", observed=False)
        .agg(matches=("date", "count"), avg_goals=("total_goals", "mean"))
        .reset_index()
        .dropna(subset=["temperature_band"])
    )
    seasonal_df = load_csv(ANALYSIS_DIR / "phase11_rain_winter_vs_summer.csv")
    scenario_df = load_csv(ANALYSIS_DIR / "phase17_climate_change_scenarios.csv")
    importance_df = load_csv(ANALYSIS_DIR / "phase19_permutation_importance.csv").head(10)

    chart_left = (
        alt.Chart(temp_goals_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#2f7ea1")
        .encode(
            x=alt.X("temperature_band:N", title="Temperature Band"),
            y=alt.Y("avg_goals:Q", title="Average Goals"),
            tooltip=["temperature_band", "matches", alt.Tooltip("avg_goals:Q", format=".2f")],
        )
        .properties(height=300)
    )
    chart_mid = (
        alt.Chart(seasonal_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("condition:N", title="Condition"),
            y=alt.Y("avg_total_goals:Q", title="Average Goals"),
            color=alt.Color("condition:N", legend=None),
            column=alt.Column("seasonal_indicator:N", title=None),
            tooltip=["seasonal_indicator", "condition", "matches", alt.Tooltip("avg_total_goals:Q", format=".2f")],
        )
        .properties(height=300)
    )
    chart_right = (
        alt.Chart(importance_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#165c8c")
        .encode(
            x=alt.X("permutation_importance_mean:Q", title="Permutation Importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature", alt.Tooltip("permutation_importance_mean:Q", format=".4f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart_left, use_container_width=True)
    st.altair_chart(chart_mid, use_container_width=True)
    st.altair_chart(chart_right, use_container_width=True)

    scenario_view_df = scenario_df[["scenario", "predicted_total_goals", "goals_pct_change_vs_current", "predicted_home_win_probability"]]
    st.dataframe(
        scenario_view_df.style.format(
            {
                "predicted_total_goals": "{:.3f}",
                "goals_pct_change_vs_current": "{:+.2%}",
                "predicted_home_win_probability": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_team_rankings() -> None:
    st.subheader("Team Sensitivity Rankings")
    sensitivity_df = load_csv(ANALYSIS_DIR / "phase12_climate_sensitivity_index.csv")
    geo_df = load_csv(ANALYSIS_DIR / "phase18_team_geography.csv")
    geo_columns = [
        column
        for column in geo_df.columns
        if column not in {"climate_sensitivity_index"}
    ]
    merged_df = sensitivity_df.merge(geo_df[geo_columns], on="home_team", how="left")

    ranking_mode = st.radio("Ranking view", ["Most affected", "Most resilient"], horizontal=True)
    team_count = st.slider("Teams to show", min_value=5, max_value=20, value=10)
    view_df = merged_df.sort_values("climate_sensitivity_index", ascending=(ranking_mode == "Most resilient")).head(team_count)

    ranking_chart = (
        alt.Chart(view_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("climate_sensitivity_index:Q", title="Climate Sensitivity Index"),
            y=alt.Y("home_team:N", sort="-x", title="Team"),
            color=alt.Color("region_group:N", title="Region"),
            tooltip=["home_team", "city", "region_group", alt.Tooltip("climate_sensitivity_index:Q", format=".2f")],
        )
        .properties(height=420)
    )
    st.altair_chart(ranking_chart, use_container_width=True)

    map_df = merged_df[["latitude", "longitude", "home_team", "climate_sensitivity_index"]].dropna()
    st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))


def render_simulation_tool(dataset_df: pd.DataFrame) -> None:
    st.subheader("Simulation Tool")
    team_options = sorted(dataset_df["home_team"].unique())
    controls_col, output_col = st.columns([1, 1.2])
    with controls_col:
        home_team = st.selectbox("Home team", team_options, index=team_options.index("Arsenal") if "Arsenal" in team_options else 0)
        away_options = [team for team in team_options if team != home_team]
        away_team = st.selectbox("Away team", away_options, index=away_options.index("Liverpool") if "Liverpool" in away_options else 0)
        month = st.slider("Match month", min_value=1, max_value=12, value=11)
        temperature = st.slider("Temperature", min_value=-5.0, max_value=35.0, value=12.0, step=0.5)
        rain = st.slider("Rain", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
        wind = st.slider("Wind", min_value=0.0, max_value=80.0, value=20.0, step=1.0)
        humidity = st.slider("Humidity", min_value=30.0, max_value=100.0, value=75.0, step=1.0)

    _, predicted_result, predicted_goals, probability_df = predict_fixture(
        dataset_df, home_team, away_team, temperature, rain, wind, humidity, month
    )
    _, baseline_result, baseline_goals, baseline_probability_df = predict_fixture(
        dataset_df, home_team, away_team, 15.0, 0.0, 15.0, 65.0, month
    )
    comparison_df = probability_df.merge(
        baseline_probability_df.rename(columns={"probability": "baseline_probability"}),
        on="result",
        how="left",
    )
    comparison_df["delta"] = comparison_df["probability"] - comparison_df["baseline_probability"]

    with output_col:
        metrics = st.columns(3)
        metrics[0].metric("Predicted Result", RESULT_LABELS[predicted_result], f"vs {RESULT_LABELS[baseline_result]}")
        metrics[1].metric("Predicted Goals", f"{predicted_goals:.2f}", f"{predicted_goals - baseline_goals:+.2f}")
        metrics[2].metric("Rain Flag", "Yes" if rain > 0 else "No")
        chart = (
            alt.Chart(comparison_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("result:N", title="Outcome"),
                y=alt.Y("delta:Q", title="Probability Delta vs Mild Baseline"),
                color=alt.condition(alt.datum.delta > 0, alt.value("#2e8b57"), alt.value("#c0392b")),
                tooltip=[
                    "result",
                    alt.Tooltip("probability:Q", format=".2%"),
                    alt.Tooltip("baseline_probability:Q", format=".2%"),
                    alt.Tooltip("delta:Q", format="+.2%"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)


def render_future_scenarios(dataset_df: pd.DataFrame) -> None:
    st.subheader("Future Scenario Explorer")
    st.write("Stress-test the full league under custom climate shifts while keeping team context fixed.")
    temp_shift, rain_multiplier, humidity_shift = st.columns(3)
    temperature_delta = temp_shift.slider("Temperature shift", min_value=-1.0, max_value=5.0, value=2.0, step=0.5)
    rainfall_multiplier = rain_multiplier.slider("Rain multiplier", min_value=0.5, max_value=2.0, value=1.3, step=0.05)
    humidity_delta = humidity_shift.slider("Humidity shift", min_value=-10.0, max_value=15.0, value=4.0, step=1.0)

    scenario_df = dataset_df.copy()
    scenario_df["temperature"] = scenario_df["temperature"] + temperature_delta
    scenario_df["apparent_temperature"] = scenario_df["apparent_temperature"] + temperature_delta * 0.9
    scenario_df["rain"] = scenario_df["rain"] * rainfall_multiplier
    scenario_df["humidity"] = scenario_df["humidity"] + humidity_delta
    scenario_df["rain_flag"] = (scenario_df["rain"] > 0).astype(int)
    scenario_df["cold_flag"] = (scenario_df["temperature"] < COLD_TEMPERATURE_THRESHOLD).astype(int)
    scenario_df["hot_flag"] = (scenario_df["temperature"] >= HOT_TEMPERATURE_THRESHOLD).astype(int)
    scenario_df["windy_flag"] = (scenario_df["wind"] >= WINDY_SPEED_THRESHOLD).astype(int)
    scenario_df["humidity_flag"] = (scenario_df["humidity"] >= HUMIDITY_THRESHOLD).astype(int)
    scenario_df["apparent_cold_flag"] = (scenario_df["apparent_temperature"] < COLD_TEMPERATURE_THRESHOLD).astype(int)
    scenario_df["apparent_hot_flag"] = (scenario_df["apparent_temperature"] >= HOT_TEMPERATURE_THRESHOLD).astype(int)
    scenario_df["apparent_temperature_gap"] = scenario_df["apparent_temperature"] - scenario_df["temperature"]
    scenario_df["heavy_rain_flag"] = (scenario_df["rain"] >= HEAVY_RAIN_THRESHOLD).astype(int)
    scenario_df["strong_wind_flag"] = (scenario_df["wind"] >= STRONG_WIND_THRESHOLD).astype(int)
    scenario_df["freezing_flag"] = (scenario_df["apparent_temperature"] <= FREEZING_APPARENT_TEMPERATURE_THRESHOLD).astype(int)
    scenario_df["heatwave_flag"] = (scenario_df["apparent_temperature"] >= HEATWAVE_APPARENT_TEMPERATURE_THRESHOLD).astype(int)
    scenario_df["storm_flag"] = 0
    scenario_df["extreme_weather_flag"] = scenario_df[["heavy_rain_flag", "strong_wind_flag", "freezing_flag", "heatwave_flag"]].max(axis=1)

    classification_model, regression_model = load_models()
    baseline_probs = classification_model.predict_proba(dataset_df[FEATURE_COLUMNS])
    future_probs = classification_model.predict_proba(scenario_df[FEATURE_COLUMNS])
    home_win_index = list(classification_model.classes_).index("win")
    summary_df = pd.DataFrame(
        [
            {
                "view": "Current",
                "avg_predicted_goals": regression_model.predict(dataset_df[FEATURE_COLUMNS]).mean(),
                "home_win_probability": baseline_probs[:, home_win_index].mean(),
                "extreme_weather_share": dataset_df["extreme_weather_flag"].mean(),
            },
            {
                "view": "Scenario",
                "avg_predicted_goals": regression_model.predict(scenario_df[FEATURE_COLUMNS]).mean(),
                "home_win_probability": future_probs[:, home_win_index].mean(),
                "extreme_weather_share": scenario_df["extreme_weather_flag"].mean(),
            },
        ]
    )
    st.dataframe(
        summary_df.style.format(
            {
                "avg_predicted_goals": "{:.3f}",
                "home_win_probability": "{:.2%}",
                "extreme_weather_share": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_extreme_weather_explorer(dataset_df: pd.DataFrame) -> None:
    st.subheader("Extreme Weather Explorer")
    summary_df = load_csv(ANALYSIS_DIR / "phase15_extreme_weather_summary.csv")
    condition_label = st.selectbox(
        "Extreme condition",
        ["heavy_rain", "extreme_wind", "very_low_temperature", "any_extreme"],
    )
    team_filter = st.selectbox("Team filter", ["All teams"] + sorted(dataset_df["home_team"].unique()))

    working_df = dataset_df.copy()
    working_df["very_low_temperature_flag"] = (working_df["temperature"] <= 5).astype(int)
    flag_map = {
        "heavy_rain": "heavy_rain_flag",
        "extreme_wind": "strong_wind_flag",
        "very_low_temperature": "very_low_temperature_flag",
        "any_extreme": "extreme_weather_flag",
    }
    filtered_df = working_df.loc[working_df[flag_map[condition_label]] == 1].copy()
    if team_filter != "All teams":
        filtered_df = filtered_df.loc[filtered_df["home_team"] == team_filter]

    top_row = summary_df.loc[summary_df["extreme_type"] == condition_label].iloc[0]
    metric_cols = st.columns(3)
    metric_cols[0].metric("Extreme matches", f"{int(top_row['extreme_matches']):,}")
    metric_cols[1].metric("Goals delta", f"{top_row['goals_delta_vs_baseline']:+.2f}")
    metric_cols[2].metric("Draw-rate delta", f"{top_row['draw_delta_vs_baseline']:+.2%}")
    st.dataframe(
        filtered_df[["date", "home_team", "away_team", "temperature", "rain", "wind", "total_goals", "result_label"]]
        .sort_values("date", ascending=False)
        .head(100),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    ensure_artifacts()
    inject_styles()
    dataset_df = load_feature_data()
    evaluation_metrics = load_json(MODELS_DIR / "evaluation_metrics.json")

    with st.sidebar:
        st.header("Control Panel")
        st.write("Use the tabs to move between league-level climate effects, team sensitivity, simulation, future scenarios, and rare events.")
        st.caption("Launch with `streamlit run app/dashboard.py`")

    render_header(dataset_df, evaluation_metrics)
    overview_tab, ranking_tab, simulation_tab, future_tab, extreme_tab = st.tabs(
        [
            "Climate Impact Overview",
            "Team Sensitivity Rankings",
            "Simulation Tool",
            "Future Scenario Explorer",
            "Extreme Weather Explorer",
        ]
    )
    with overview_tab:
        render_climate_overview(dataset_df)
    with ranking_tab:
        render_team_rankings()
    with simulation_tab:
        render_simulation_tool(dataset_df)
    with future_tab:
        render_future_scenarios(dataset_df)
    with extreme_tab:
        render_extreme_weather_explorer(dataset_df)


if __name__ == "__main__":
    main()
