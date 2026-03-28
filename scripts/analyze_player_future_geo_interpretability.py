from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from train_models import (
    CLASSIFICATION_MODEL_PATH,
    FEATURE_COLUMNS,
    REGRESSION_MODEL_PATH,
    load_dataset,
    split_dataset,
)


TEAM_MAPPING_PATH = Path("data/raw/team_mapping.csv")
RAW_PLAYER_OUTPUT_PATH = Path("data/raw/player_match_stats.csv")
RAW_UNDERSTAT_SCHEDULE_PATH = Path("data/raw/understat_schedule.csv")
OUTPUT_DIR = Path("data/processed/analysis")
PLOTS_DIR = OUTPUT_DIR / "plots"

PLAYER_SEASONS = ["2019", "2020", "2021", "2022", "2023"]
PLAYER_FETCH_WORKERS = 8
PLAYER_FETCH_RETRIES = 3
MIN_PLAYER_MATCHES = 8
MIN_PLAYER_MINUTES = 450

UNDERSTAT_TEAM_NAME_MAP = {
    "Brighton": "Brighton",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Luton": "Luton Town",
    "Norwich": "Norwich City",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
}
RESULT_POINTS = {"win": 3, "draw": 1, "loss": 0}
WEATHER_FEATURES = [
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
]

COLD_TEMPERATURE_THRESHOLD = 10.0
HOT_TEMPERATURE_THRESHOLD = 25.0
WINDY_SPEED_THRESHOLD = 30.0
HUMIDITY_THRESHOLD = 85.0
HEAVY_RAIN_THRESHOLD = 10.0
STRONG_WIND_THRESHOLD = 50.0
FREEZING_APPARENT_TEMPERATURE_THRESHOLD = 0.0
HEATWAVE_APPARENT_TEMPERATURE_THRESHOLD = 30.0
SEVERE_WEATHER_CODES = {75, 77, 82, 85, 86, 95, 96, 99}


def save_chart(chart: alt.Chart, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output_path)


def normalize_understat_team(team_name: str) -> str:
    return UNDERSTAT_TEAM_NAME_MAP.get(team_name, team_name)


def load_team_mapping() -> pd.DataFrame:
    mapping_df = pd.read_csv(TEAM_MAPPING_PATH)
    mapping_df["latitude"] = pd.to_numeric(mapping_df["latitude"], errors="coerce")
    mapping_df["longitude"] = pd.to_numeric(mapping_df["longitude"], errors="coerce")
    return mapping_df


def build_understat_schedule() -> pd.DataFrame:
    if RAW_UNDERSTAT_SCHEDULE_PATH.exists():
        schedule_df = pd.read_csv(RAW_UNDERSTAT_SCHEDULE_PATH)
        schedule_df["date"] = pd.to_datetime(schedule_df["date"], format="%Y-%m-%d")
        return schedule_df

    from understatapi import UnderstatClient

    client = UnderstatClient()
    schedule_rows: list[dict[str, object]] = []

    for season in PLAYER_SEASONS:
        match_rows = client.league(league="EPL").get_match_data(season=season)
        for row in match_rows:
            schedule_rows.append(
                {
                    "understat_match_id": str(row["id"]),
                    "season_year_start": int(season),
                    "date": pd.to_datetime(row["datetime"]).normalize(),
                    "home_team": normalize_understat_team(row["h"]["title"]),
                    "away_team": normalize_understat_team(row["a"]["title"]),
                }
            )

    schedule_df = pd.DataFrame(schedule_rows).drop_duplicates(
        subset=["understat_match_id"]
    )
    schedule_df = schedule_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    RAW_UNDERSTAT_SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    schedule_df.assign(date=schedule_df["date"].dt.strftime("%Y-%m-%d")).to_csv(
        RAW_UNDERSTAT_SCHEDULE_PATH, index=False
    )
    return schedule_df


def attach_understat_match_ids(dataset_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = dataset_df.merge(
        schedule_df,
        on=["date", "home_team", "away_team", "season_year_start"],
        how="left",
        validate="one_to_one",
    )
    missing_matches = merged_df.loc[merged_df["understat_match_id"].isna(), ["date", "home_team", "away_team"]]
    if not missing_matches.empty:
        raise ValueError(
            "Could not map all matches to Understat IDs. Missing examples: "
            f"{missing_matches.head(10).to_dict(orient='records')}"
        )
    return merged_df


def fetch_single_roster(match_id: str, home_team: str, away_team: str) -> list[dict[str, object]]:
    from understatapi import UnderstatClient

    last_error: Exception | None = None
    for attempt in range(1, PLAYER_FETCH_RETRIES + 1):
        try:
            roster = UnderstatClient().match(match=str(match_id)).get_roster_data()
            rows: list[dict[str, object]] = []
            for side_key, team_name, opponent_name in [
                ("h", home_team, away_team),
                ("a", away_team, home_team),
            ]:
                for roster_row in roster.get(side_key, {}).values():
                    rows.append(
                        {
                            "understat_match_id": str(match_id),
                            "player_id": str(roster_row.get("player_id", "")),
                            "player_name": roster_row.get("player", "").strip(),
                            "team_name": team_name,
                            "opponent_team": opponent_name,
                            "is_home": 1 if side_key == "h" else 0,
                            "position": roster_row.get("position", "").strip(),
                            "position_order": pd.to_numeric(
                                roster_row.get("positionOrder", 0), errors="coerce"
                            ),
                            "minutes": pd.to_numeric(roster_row.get("time", 0), errors="coerce"),
                            "goals": pd.to_numeric(roster_row.get("goals", 0), errors="coerce"),
                            "assists": pd.to_numeric(roster_row.get("assists", 0), errors="coerce"),
                            "shots": pd.to_numeric(roster_row.get("shots", 0), errors="coerce"),
                            "xG": pd.to_numeric(roster_row.get("xG", 0), errors="coerce"),
                            "xA": pd.to_numeric(roster_row.get("xA", 0), errors="coerce"),
                            "key_passes": pd.to_numeric(
                                roster_row.get("key_passes", 0), errors="coerce"
                            ),
                            "yellow_cards": pd.to_numeric(
                                roster_row.get("yellow_card", 0), errors="coerce"
                            ),
                            "red_cards": pd.to_numeric(roster_row.get("red_card", 0), errors="coerce"),
                        }
                    )
            return rows
        except Exception as exc:
            last_error = exc
            time.sleep(min(1.5 * attempt, 5))

    raise RuntimeError(f"Failed to fetch roster for match {match_id}") from last_error


def build_player_match_dataset(match_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if RAW_PLAYER_OUTPUT_PATH.exists():
        player_df = pd.read_csv(RAW_PLAYER_OUTPUT_PATH)
        player_df["date"] = pd.to_datetime(player_df["date"], format="%Y-%m-%d")
        return player_df, int(player_df["understat_match_id"].nunique())

    match_context_df = match_df[
        [
            "understat_match_id",
            "date",
            "home_team",
            "away_team",
            "temperature",
            "apparent_temperature",
            "humidity",
            "rain",
            "wind",
            "weather_code",
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
            "match_result",
            "goal_difference",
            "total_goals",
            "home_team_strength_proxy",
            "away_team_strength_proxy",
            "match_importance_proxy",
            "seasonal_indicator",
            "season_year_start",
            "season_progress",
        ]
    ].drop_duplicates(subset=["understat_match_id"])

    roster_rows: list[dict[str, object]] = []
    failed_match_ids: list[str] = []

    with ThreadPoolExecutor(max_workers=PLAYER_FETCH_WORKERS) as executor:
        future_map = {
            executor.submit(
                fetch_single_roster,
                row["understat_match_id"],
                row["home_team"],
                row["away_team"],
            ): row["understat_match_id"]
            for _, row in match_context_df.iterrows()
        }
        for future in as_completed(future_map):
            match_id = future_map[future]
            try:
                roster_rows.extend(future.result())
            except Exception:
                failed_match_ids.append(str(match_id))

    if failed_match_ids:
        raise RuntimeError(
            f"Failed to fetch {len(failed_match_ids)} match rosters. Examples: {failed_match_ids[:10]}"
        )

    player_df = pd.DataFrame(roster_rows)
    player_df = player_df.merge(match_context_df, on="understat_match_id", how="left", validate="many_to_one")
    player_df["team_strength_proxy"] = player_df.apply(
        lambda row: row["home_team_strength_proxy"]
        if row["team_name"] == row["home_team"]
        else row["away_team_strength_proxy"],
        axis=1,
    )
    player_df["team_points"] = player_df.apply(
        lambda row: RESULT_POINTS[row["match_result"]]
        if row["team_name"] == row["home_team"]
        else RESULT_POINTS[{"win": "loss", "loss": "win", "draw": "draw"}[row["match_result"]]],
        axis=1,
    )
    player_df["team_goal_difference"] = player_df.apply(
        lambda row: row["goal_difference"]
        if row["team_name"] == row["home_team"]
        else -row["goal_difference"],
        axis=1,
    )
    player_df["date"] = pd.to_datetime(player_df["date"], format="%Y-%m-%d")

    numeric_columns = [
        "position_order",
        "minutes",
        "goals",
        "assists",
        "shots",
        "xG",
        "xA",
        "key_passes",
        "yellow_cards",
        "red_cards",
        "temperature",
        "apparent_temperature",
        "humidity",
        "rain",
        "wind",
        "weather_code",
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
        "goal_difference",
        "total_goals",
        "home_team_strength_proxy",
        "away_team_strength_proxy",
        "team_strength_proxy",
        "team_points",
        "team_goal_difference",
        "match_importance_proxy",
        "season_year_start",
        "season_progress",
    ]
    player_df[numeric_columns] = player_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    player_df = player_df.fillna(
        {
            "minutes": 0,
            "goals": 0,
            "assists": 0,
            "shots": 0,
            "xG": 0,
            "xA": 0,
            "key_passes": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "position_order": 0,
        }
    )

    RAW_PLAYER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    player_df.assign(date=player_df["date"].dt.strftime("%Y-%m-%d")).to_csv(
        RAW_PLAYER_OUTPUT_PATH, index=False
    )
    return player_df, int(match_context_df["understat_match_id"].nunique())


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return 0.0
    return numerator / denominator


def build_phase16_outputs(player_df: pd.DataFrame) -> dict[str, object]:
    analysis_df = player_df.copy()
    analysis_df["xgi"] = analysis_df["xG"] + analysis_df["xA"]
    analysis_df["goal_involvements"] = analysis_df["goals"] + analysis_df["assists"]
    analysis_df["xgi_per90"] = 90 * analysis_df["xgi"] / analysis_df["minutes"].clip(lower=1)
    analysis_df["goal_involvements_per90"] = (
        90 * analysis_df["goal_involvements"] / analysis_df["minutes"].clip(lower=1)
    )

    player_base_df = (
        analysis_df.groupby(["player_id", "player_name"])
        .agg(
            appearances=("understat_match_id", "count"),
            minutes=("minutes", "sum"),
            base_xgi=("xgi", "sum"),
            base_goal_involvements=("goal_involvements", "sum"),
            avg_team_strength=("team_strength_proxy", "mean"),
        )
        .reset_index()
    )
    player_base_df["baseline_xgi_per90"] = (
        90 * player_base_df["base_xgi"] / player_base_df["minutes"].clip(lower=1)
    )
    player_base_df["baseline_goal_involvements_per90"] = (
        90 * player_base_df["base_goal_involvements"] / player_base_df["minutes"].clip(lower=1)
    )
    player_base_df = player_base_df[
        (player_base_df["appearances"] >= MIN_PLAYER_MATCHES)
        & (player_base_df["minutes"] >= MIN_PLAYER_MINUTES)
    ].copy()

    team_minutes_df = (
        analysis_df.groupby(["player_id", "team_name"])
        .agg(team_minutes=("minutes", "sum"))
        .reset_index()
        .sort_values(["player_id", "team_minutes"], ascending=[True, False])
        .drop_duplicates(subset=["player_id"])
        .rename(columns={"team_name": "primary_team"})
    )
    player_base_df = player_base_df.merge(team_minutes_df, on="player_id", how="left", validate="one_to_one")

    condition_rows: list[dict[str, object]] = []
    filtered_df = analysis_df[analysis_df["player_id"].isin(player_base_df["player_id"])].copy()
    for (player_id, player_name), player_group in filtered_df.groupby(["player_id", "player_name"]):
        baseline_row = player_base_df.loc[player_base_df["player_id"] == player_id].iloc[0]
        for condition_name, flag_name in [("rain", "rain_flag"), ("cold", "cold_flag")]:
            split_df = (
                player_group.groupby(flag_name)
                .agg(
                    appearances=("understat_match_id", "count"),
                    minutes=("minutes", "sum"),
                    xgi=("xgi", "sum"),
                    goal_involvements=("goal_involvements", "sum"),
                    team_points=("team_points", "mean"),
                    team_goal_difference=("team_goal_difference", "mean"),
                )
                .reset_index()
            )
            split_map = {int(row[flag_name]): row for _, row in split_df.iterrows()}
            off_row = split_map.get(0, {})
            on_row = split_map.get(1, {})
            off_minutes = float(off_row.get("minutes", 0) or 0)
            on_minutes = float(on_row.get("minutes", 0) or 0)
            off_xgi_per90 = 90 * safe_ratio(float(off_row.get("xgi", 0) or 0), max(off_minutes, 1))
            on_xgi_per90 = 90 * safe_ratio(float(on_row.get("xgi", 0) or 0), max(on_minutes, 1))
            off_gi_per90 = 90 * safe_ratio(
                float(off_row.get("goal_involvements", 0) or 0), max(off_minutes, 1)
            )
            on_gi_per90 = 90 * safe_ratio(
                float(on_row.get("goal_involvements", 0) or 0), max(on_minutes, 1)
            )
            sample_balance = safe_ratio(min(off_minutes, on_minutes), max(off_minutes, on_minutes, 1))
            condition_rows.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "primary_team": baseline_row["primary_team"],
                    "condition": condition_name,
                    "off_appearances": int(off_row.get("appearances", 0) or 0),
                    "on_appearances": int(on_row.get("appearances", 0) or 0),
                    "off_minutes": off_minutes,
                    "on_minutes": on_minutes,
                    "off_xgi_per90": off_xgi_per90,
                    "on_xgi_per90": on_xgi_per90,
                    "off_goal_involvements_per90": off_gi_per90,
                    "on_goal_involvements_per90": on_gi_per90,
                    "off_team_points": float(off_row.get("team_points", 0) or 0),
                    "on_team_points": float(on_row.get("team_points", 0) or 0),
                    "sample_balance": sample_balance,
                }
            )

    condition_df = pd.DataFrame(condition_rows)
    rain_df = condition_df[condition_df["condition"] == "rain"].copy()
    cold_df = condition_df[condition_df["condition"] == "cold"].copy()
    player_profiles_df = player_base_df.merge(
        rain_df.drop(columns=["condition"]).add_prefix("rain_"),
        left_on="player_id",
        right_on="rain_player_id",
        how="left",
    ).merge(
        cold_df.drop(columns=["condition"]).add_prefix("cold_"),
        left_on="player_id",
        right_on="cold_player_id",
        how="left",
    )

    player_profiles_df["strength_normalizer"] = player_profiles_df["avg_team_strength"].clip(lower=1.0)
    player_profiles_df["rain_xgi_delta"] = (
        player_profiles_df["rain_on_xgi_per90"] - player_profiles_df["rain_off_xgi_per90"]
    )
    player_profiles_df["cold_xgi_delta"] = (
        player_profiles_df["cold_on_xgi_per90"] - player_profiles_df["cold_off_xgi_per90"]
    )
    player_profiles_df["rain_points_delta"] = (
        player_profiles_df["rain_on_team_points"] - player_profiles_df["rain_off_team_points"]
    )
    player_profiles_df["cold_points_delta"] = (
        player_profiles_df["cold_on_team_points"] - player_profiles_df["cold_off_team_points"]
    )
    player_profiles_df["rain_relative_xgi_delta"] = player_profiles_df["rain_xgi_delta"] / player_profiles_df[
        "baseline_xgi_per90"
    ].clip(lower=0.1)
    player_profiles_df["cold_relative_xgi_delta"] = player_profiles_df["cold_xgi_delta"] / player_profiles_df[
        "baseline_xgi_per90"
    ].clip(lower=0.1)
    player_profiles_df["rain_sensitivity_component"] = (
        -(
            0.7 * player_profiles_df["rain_relative_xgi_delta"]
            + 0.3 * (player_profiles_df["rain_points_delta"] / 3.0)
        )
        * player_profiles_df["rain_sample_balance"].fillna(0)
        / player_profiles_df["strength_normalizer"]
        * 100
    )
    player_profiles_df["cold_sensitivity_component"] = (
        -(
            0.7 * player_profiles_df["cold_relative_xgi_delta"]
            + 0.3 * (player_profiles_df["cold_points_delta"] / 3.0)
        )
        * player_profiles_df["cold_sample_balance"].fillna(0)
        / player_profiles_df["strength_normalizer"]
        * 100
    )
    player_profiles_df["player_climate_index"] = (
        0.5 * player_profiles_df["rain_sensitivity_component"].fillna(0)
        + 0.5 * player_profiles_df["cold_sensitivity_component"].fillna(0)
    )
    player_profiles_df["player_climate_label"] = "neutral"
    player_profiles_df.loc[
        player_profiles_df["player_climate_index"] >= 15, "player_climate_label"
    ] = "highly affected"
    player_profiles_df.loc[
        player_profiles_df["player_climate_index"] <= -10, "player_climate_label"
    ] = "weather-resilient"
    player_profiles_df = player_profiles_df.sort_values(
        "player_climate_index", ascending=False
    ).reset_index(drop=True)

    condition_df.to_csv(OUTPUT_DIR / "phase16_player_condition_breakdown.csv", index=False)
    player_profiles_df.to_csv(OUTPUT_DIR / "phase16_player_climate_profiles.csv", index=False)

    sensitive_chart = (
        alt.Chart(player_profiles_df.head(15))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("player_climate_index:Q", title="Player Climate Index"),
            y=alt.Y("player_name:N", sort="-x", title="Player"),
            color=alt.Color("player_climate_label:N", title="Profile"),
            tooltip=[
                "player_name",
                "primary_team",
                alt.Tooltip("player_climate_index:Q", format=".2f"),
                alt.Tooltip("rain_xgi_delta:Q", format=".2f"),
                alt.Tooltip("cold_xgi_delta:Q", format=".2f"),
            ],
        )
        .properties(width=620, height=380, title="Phase 16 - Most Climate-Sensitive Players")
    )
    component_chart_df = player_profiles_df.head(20).melt(
        id_vars=["player_name"],
        value_vars=["rain_sensitivity_component", "cold_sensitivity_component"],
        var_name="component",
        value_name="value",
    )
    component_chart = (
        alt.Chart(component_chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("value:Q", title="Normalized Component"),
            y=alt.Y("player_name:N", sort="-x", title="Player"),
            color=alt.Color("component:N", title="Driver"),
            tooltip=["player_name", "component", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(width=620, height=420, title="Phase 16 - Rain vs Cold Player Effects")
    )
    save_chart(sensitive_chart, PLOTS_DIR / "phase16_player_climate_rankings.html")
    save_chart(component_chart, PLOTS_DIR / "phase16_player_climate_components.html")

    most_affected = player_profiles_df.iloc[0]
    least_affected = player_profiles_df.iloc[-1]
    summary = "\n".join(
        [
            "Phase 16 - Player-Level Extension",
            (
                f"Built climate profiles for {len(player_profiles_df)} players with at least "
                f"{MIN_PLAYER_MINUTES} minutes and {MIN_PLAYER_MATCHES} appearances."
            ),
            (
                f"Most climate-sensitive player: {most_affected['player_name']} "
                f"({most_affected['primary_team']}) at {most_affected['player_climate_index']:.2f}."
            ),
            (
                f"Most weather-resilient player: {least_affected['player_name']} "
                f"({least_affected['primary_team']}) at {least_affected['player_climate_index']:.2f}."
            ),
            (
                "The profile combines rainy-vs-dry and cold-vs-normal changes in xGI per 90 and team points, "
                "then normalizes by team-strength context so weak squads do not dominate the ranking by default."
            ),
        ]
    )
    (OUTPUT_DIR / "phase16_player_climate_summary.txt").write_text(summary, encoding="utf-8")

    return {"summary": summary, "player_profiles_df": player_profiles_df}


def recompute_weather_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    scenario_df = dataset_df.copy()
    scenario_df["rain"] = scenario_df["rain"].clip(lower=0)
    scenario_df["humidity"] = scenario_df["humidity"].clip(lower=0, upper=100)
    scenario_df["wind"] = scenario_df["wind"].clip(lower=0)
    scenario_df["rain_flag"] = (scenario_df["rain"] > 0).astype(int)
    scenario_df["cold_flag"] = (scenario_df["temperature"] < COLD_TEMPERATURE_THRESHOLD).astype(int)
    scenario_df["hot_flag"] = (scenario_df["temperature"] >= HOT_TEMPERATURE_THRESHOLD).astype(int)
    scenario_df["windy_flag"] = (scenario_df["wind"] >= WINDY_SPEED_THRESHOLD).astype(int)
    scenario_df["humidity_flag"] = (scenario_df["humidity"] >= HUMIDITY_THRESHOLD).astype(int)
    scenario_df["apparent_cold_flag"] = (
        scenario_df["apparent_temperature"] < COLD_TEMPERATURE_THRESHOLD
    ).astype(int)
    scenario_df["apparent_hot_flag"] = (
        scenario_df["apparent_temperature"] >= HOT_TEMPERATURE_THRESHOLD
    ).astype(int)
    scenario_df["apparent_temperature_gap"] = (
        scenario_df["apparent_temperature"] - scenario_df["temperature"]
    )
    scenario_df["heavy_rain_flag"] = (scenario_df["rain"] >= HEAVY_RAIN_THRESHOLD).astype(int)
    scenario_df["strong_wind_flag"] = (scenario_df["wind"] >= STRONG_WIND_THRESHOLD).astype(int)
    scenario_df["freezing_flag"] = (
        scenario_df["apparent_temperature"] <= FREEZING_APPARENT_TEMPERATURE_THRESHOLD
    ).astype(int)
    scenario_df["heatwave_flag"] = (
        scenario_df["apparent_temperature"] >= HEATWAVE_APPARENT_TEMPERATURE_THRESHOLD
    ).astype(int)
    scenario_df["storm_flag"] = scenario_df["weather_code"].isin(SEVERE_WEATHER_CODES).astype(int)
    scenario_df["extreme_weather_flag"] = scenario_df[
        ["heavy_rain_flag", "strong_wind_flag", "freezing_flag", "heatwave_flag", "storm_flag"]
    ].max(axis=1)
    return scenario_df


def build_phase17_outputs(dataset_df: pd.DataFrame) -> dict[str, object]:
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)

    scenario_definitions = {
        "current_climate": {
            "temp_shift": 0.0,
            "apparent_shift": 0.0,
            "rain_scale": 1.0,
            "rain_add": 0.0,
            "humidity_shift": 0.0,
        },
        "+2c_warming": {
            "temp_shift": 2.0,
            "apparent_shift": 2.2,
            "rain_scale": 1.0,
            "rain_add": 0.0,
            "humidity_shift": -1.0,
        },
        "increased_rainfall": {
            "temp_shift": 0.0,
            "apparent_shift": -0.3,
            "rain_scale": 1.35,
            "rain_add": 1.0,
            "humidity_shift": 5.0,
        },
        "warmer_and_wetter": {
            "temp_shift": 2.0,
            "apparent_shift": 1.8,
            "rain_scale": 1.35,
            "rain_add": 1.0,
            "humidity_shift": 4.0,
        },
    }

    classes = list(classification_model.classes_)
    home_win_index = classes.index("win")
    draw_index = classes.index("draw")
    away_win_index = classes.index("loss")

    scenario_rows: list[dict[str, object]] = []
    for scenario_name, settings in scenario_definitions.items():
        scenario_df = dataset_df.copy()
        scenario_df["temperature"] = scenario_df["temperature"] + settings["temp_shift"]
        scenario_df["apparent_temperature"] = (
            scenario_df["apparent_temperature"] + settings["apparent_shift"]
        )
        scenario_df["rain"] = scenario_df["rain"] * settings["rain_scale"] + settings["rain_add"]
        scenario_df["humidity"] = scenario_df["humidity"] + settings["humidity_shift"]
        scenario_df = recompute_weather_features(scenario_df)

        feature_frame = scenario_df[FEATURE_COLUMNS]
        predicted_probabilities = classification_model.predict_proba(feature_frame)
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "matches": len(scenario_df),
                "avg_temperature": scenario_df["temperature"].mean(),
                "avg_rain": scenario_df["rain"].mean(),
                "avg_humidity": scenario_df["humidity"].mean(),
                "rainy_match_share": scenario_df["rain_flag"].mean(),
                "extreme_weather_share": scenario_df["extreme_weather_flag"].mean(),
                "predicted_total_goals": regression_model.predict(feature_frame).mean(),
                "predicted_home_win_probability": predicted_probabilities[:, home_win_index].mean(),
                "predicted_draw_probability": predicted_probabilities[:, draw_index].mean(),
                "predicted_away_win_probability": predicted_probabilities[:, away_win_index].mean(),
            }
        )

    scenario_df = pd.DataFrame(scenario_rows)
    baseline_row = scenario_df.loc[scenario_df["scenario"] == "current_climate"].iloc[0]
    scenario_df["goals_change_vs_current"] = (
        scenario_df["predicted_total_goals"] - baseline_row["predicted_total_goals"]
    )
    scenario_df["goals_pct_change_vs_current"] = (
        scenario_df["goals_change_vs_current"] / baseline_row["predicted_total_goals"]
    )
    scenario_df["home_win_probability_change_vs_current"] = (
        scenario_df["predicted_home_win_probability"] - baseline_row["predicted_home_win_probability"]
    )
    scenario_df.to_csv(OUTPUT_DIR / "phase17_climate_change_scenarios.csv", index=False)

    chart = (
        alt.Chart(scenario_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("scenario:N", title="Scenario"),
            y=alt.Y("goals_pct_change_vs_current:Q", title="Predicted Goals % Change vs Current"),
            color=alt.condition(
                alt.datum.goals_pct_change_vs_current > 0,
                alt.value("#2E8B57"),
                alt.value("#C0392B"),
            ),
            tooltip=[
                "scenario",
                alt.Tooltip("predicted_total_goals:Q", format=".3f"),
                alt.Tooltip("goals_pct_change_vs_current:Q", format=".2%"),
                alt.Tooltip("predicted_home_win_probability:Q", format=".2%"),
            ],
        )
        .properties(width=620, height=320, title="Phase 17 - Future Climate Scenario Impact")
    )
    save_chart(chart, PLOTS_DIR / "phase17_climate_change_goals.html")

    future_row = scenario_df.loc[scenario_df["scenario"] == "warmer_and_wetter"].iloc[0]
    summary = "\n".join(
        [
            "Phase 17 - Climate Change Scenario Modeling",
            (
                f"In the warmer-and-wetter scenario, predicted total goals move by "
                f"{future_row['goals_pct_change_vs_current']:+.2%} versus the current climate baseline."
            ),
            (
                f"The same scenario changes average home-win probability by "
                f"{future_row['home_win_probability_change_vs_current']:+.2%}."
            ),
            (
                "This is not a forecast of future league reality; it is a controlled model stress test that keeps "
                "fixtures and team-strength context fixed while perturbing weather inputs."
            ),
        ]
    )
    (OUTPUT_DIR / "phase17_climate_change_summary.txt").write_text(summary, encoding="utf-8")
    return {"summary": summary, "scenario_df": scenario_df}


def build_phase18_outputs(dataset_df: pd.DataFrame, mapping_df: pd.DataFrame) -> dict[str, object]:
    geo_df = dataset_df.merge(
        mapping_df[["team_name", "stadium_name", "city", "latitude", "longitude"]],
        left_on="home_team",
        right_on="team_name",
        how="left",
        validate="many_to_one",
    )
    median_latitude = geo_df["latitude"].median()
    geo_df["region_group"] = geo_df["latitude"].apply(
        lambda value: "North" if value >= median_latitude else "South"
    )

    regional_summary_df = (
        geo_df.groupby("region_group")
        .agg(
            teams=("home_team", "nunique"),
            matches=("date", "count"),
            avg_latitude=("latitude", "mean"),
            avg_total_goals=("total_goals", "mean"),
            avg_temperature=("temperature", "mean"),
            rain_share=("rain_flag", "mean"),
        )
        .reset_index()
    )
    rain_pivot_df = (
        geo_df.groupby(["region_group", "rain_flag"])
        .agg(avg_total_goals=("total_goals", "mean"))
        .reset_index()
        .pivot(index="region_group", columns="rain_flag", values="avg_total_goals")
        .rename(columns={0: "dry_goals", 1: "rainy_goals"})
        .reset_index()
    )
    rain_pivot_df["rain_impact_on_goals"] = rain_pivot_df["rainy_goals"] - rain_pivot_df["dry_goals"]

    cold_pivot_df = (
        geo_df.groupby(["region_group", "cold_flag"])
        .agg(avg_total_goals=("total_goals", "mean"))
        .reset_index()
        .pivot(index="region_group", columns="cold_flag", values="avg_total_goals")
        .rename(columns={0: "mild_goals", 1: "cold_goals"})
        .reset_index()
    )
    cold_pivot_df["cold_impact_on_goals"] = cold_pivot_df["cold_goals"] - cold_pivot_df["mild_goals"]

    team_geo_df = (
        geo_df.groupby(["home_team", "stadium_name", "city", "latitude", "longitude", "region_group"])
        .agg(
            matches=("date", "count"),
            avg_total_goals=("total_goals", "mean"),
            rain_share=("rain_flag", "mean"),
            avg_temperature=("temperature", "mean"),
        )
        .reset_index()
    )
    sensitivity_path = OUTPUT_DIR / "phase12_climate_sensitivity_index.csv"
    if sensitivity_path.exists():
        sensitivity_df = pd.read_csv(sensitivity_path)
        team_geo_df = team_geo_df.merge(
            sensitivity_df[["home_team", "climate_sensitivity_index"]],
            on="home_team",
            how="left",
        )

    regional_output_df = regional_summary_df.merge(rain_pivot_df, on="region_group", how="left").merge(
        cold_pivot_df, on="region_group", how="left"
    )
    regional_output_df.to_csv(OUTPUT_DIR / "phase18_regional_summary.csv", index=False)
    team_geo_df.to_csv(OUTPUT_DIR / "phase18_team_geography.csv", index=False)

    regional_chart = (
        alt.Chart(regional_output_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("region_group:N", title="Region"),
            y=alt.Y("rain_impact_on_goals:Q", title="Rain Impact on Avg Goals"),
            color=alt.Color("region_group:N", legend=None),
            tooltip=[
                "region_group",
                "teams",
                "matches",
                alt.Tooltip("rain_impact_on_goals:Q", format=".2f"),
                alt.Tooltip("cold_impact_on_goals:Q", format=".2f"),
            ],
        )
        .properties(width=420, height=300, title="Phase 18 - North vs South Weather Sensitivity")
    )
    scatter = (
        alt.Chart(team_geo_df.dropna(subset=["climate_sensitivity_index"]))
        .mark_circle(size=90)
        .encode(
            x=alt.X("latitude:Q", title="Latitude"),
            y=alt.Y("climate_sensitivity_index:Q", title="Climate Sensitivity Index"),
            color=alt.Color("region_group:N", title="Region"),
            tooltip=[
                "home_team",
                "city",
                alt.Tooltip("latitude:Q", format=".2f"),
                alt.Tooltip("climate_sensitivity_index:Q", format=".2f"),
            ],
        )
        .properties(width=620, height=320, title="Phase 18 - Latitude vs Climate Sensitivity")
    )
    save_chart(regional_chart, PLOTS_DIR / "phase18_regional_weather_impact.html")
    save_chart(scatter, PLOTS_DIR / "phase18_latitude_vs_sensitivity.html")

    north_row = regional_output_df.loc[regional_output_df["region_group"] == "North"].iloc[0]
    south_row = regional_output_df.loc[regional_output_df["region_group"] == "South"].iloc[0]
    summary = "\n".join(
        [
            "Phase 18 - Geographical Analysis",
            (
                f"Using a median-latitude split at {median_latitude:.2f}, northern home teams show a "
                f"{north_row['rain_impact_on_goals']:+.2f} rain-goal delta versus "
                f"{south_row['rain_impact_on_goals']:+.2f} for southern teams."
            ),
            (
                f"Cold-weather goal impact is {north_row['cold_impact_on_goals']:+.2f} in the North and "
                f"{south_row['cold_impact_on_goals']:+.2f} in the South."
            ),
            "Geography changes the weather story: the same weather variables do not hit all stadium locations equally.",
        ]
    )
    (OUTPUT_DIR / "phase18_geographical_summary.txt").write_text(summary, encoding="utf-8")
    return {"summary": summary, "regional_output_df": regional_output_df}


def build_phase19_outputs(dataset_df: pd.DataFrame) -> dict[str, object]:
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    _, test_df = split_dataset(dataset_df)

    classifier = classification_model.named_steps["classifier"]
    class_labels = list(classification_model.classes_)
    classification_coefficients_df = pd.DataFrame({"feature": FEATURE_COLUMNS})
    classification_coefficients_df["home_win_coefficient"] = classifier.coef_[class_labels.index("win")]
    classification_coefficients_df["draw_coefficient"] = classifier.coef_[class_labels.index("draw")]
    classification_coefficients_df["away_win_coefficient"] = classifier.coef_[class_labels.index("loss")]
    classification_coefficients_df["average_absolute_coefficient"] = (
        classification_coefficients_df[
            ["home_win_coefficient", "draw_coefficient", "away_win_coefficient"]
        ]
        .abs()
        .mean(axis=1)
    )
    classification_coefficients_df["feature_group"] = classification_coefficients_df["feature"].isin(
        WEATHER_FEATURES
    ).map({True: "weather", False: "control"})
    classification_coefficients_df = classification_coefficients_df.sort_values(
        "average_absolute_coefficient", ascending=False
    )

    regression_importance_df = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "model_importance": regression_model.feature_importances_}
    )
    regression_importance_df["feature_group"] = regression_importance_df["feature"].isin(
        WEATHER_FEATURES
    ).map({True: "weather", False: "control"})
    regression_importance_df = regression_importance_df.sort_values(
        "model_importance", ascending=False
    )

    permutation = permutation_importance(
        regression_model,
        test_df[FEATURE_COLUMNS],
        test_df["total_goals"],
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    permutation_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "permutation_importance_mean": permutation.importances_mean,
            "permutation_importance_std": permutation.importances_std,
        }
    )
    permutation_df["feature_group"] = permutation_df["feature"].isin(WEATHER_FEATURES).map(
        {True: "weather", False: "control"}
    )
    permutation_df = permutation_df.sort_values("permutation_importance_mean", ascending=False)

    classification_coefficients_df.to_csv(
        OUTPUT_DIR / "phase19_classification_coefficients.csv", index=False
    )
    regression_importance_df.to_csv(
        OUTPUT_DIR / "phase19_regression_feature_importance.csv", index=False
    )
    permutation_df.to_csv(OUTPUT_DIR / "phase19_permutation_importance.csv", index=False)

    classification_chart = (
        alt.Chart(classification_coefficients_df.head(15))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("average_absolute_coefficient:Q", title="Average Absolute Coefficient"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color("feature_group:N", title="Feature Group"),
            tooltip=[
                "feature",
                "feature_group",
                alt.Tooltip("home_win_coefficient:Q", format=".4f"),
                alt.Tooltip("draw_coefficient:Q", format=".4f"),
                alt.Tooltip("away_win_coefficient:Q", format=".4f"),
            ],
        )
        .properties(width=620, height=360, title="Phase 19 - Classification Importance")
    )
    regression_chart = (
        alt.Chart(permutation_df.head(15))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("permutation_importance_mean:Q", title="Permutation Importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color("feature_group:N", title="Feature Group"),
            tooltip=[
                "feature",
                "feature_group",
                alt.Tooltip("permutation_importance_mean:Q", format=".4f"),
                alt.Tooltip("permutation_importance_std:Q", format=".4f"),
            ],
        )
        .properties(width=620, height=360, title="Phase 19 - Regression Importance")
    )
    save_chart(classification_chart, PLOTS_DIR / "phase19_classification_importance.html")
    save_chart(regression_chart, PLOTS_DIR / "phase19_regression_importance.html")

    top_classification_feature = classification_coefficients_df.iloc[0]
    top_regression_feature = permutation_df.iloc[0]
    weather_share_classification = classification_coefficients_df.loc[
        classification_coefficients_df["feature_group"] == "weather",
        "average_absolute_coefficient",
    ].sum() / classification_coefficients_df["average_absolute_coefficient"].sum()
    weather_share_regression = permutation_df.loc[
        permutation_df["feature_group"] == "weather",
        "permutation_importance_mean",
    ].clip(lower=0).sum() / permutation_df["permutation_importance_mean"].clip(lower=0).sum()

    summary = "\n".join(
        [
            "Phase 19 - Model Interpretability",
            (
                f"Top classification driver: {top_classification_feature['feature']} with average absolute "
                f"coefficient {top_classification_feature['average_absolute_coefficient']:.4f}."
            ),
            (
                f"Top regression driver by permutation importance: {top_regression_feature['feature']} at "
                f"{top_regression_feature['permutation_importance_mean']:.4f}."
            ),
            (
                f"Weather accounts for roughly {weather_share_classification:.1%} of total absolute classification "
                f"coefficient mass and {weather_share_regression:.1%} of positive regression permutation importance."
            ),
            (
                "Model behavior is still driven mostly by team-form and strength features, but weather features "
                "remain visible enough to explain small but real shifts in outcome probabilities and goal expectations."
            ),
        ]
    )
    (OUTPUT_DIR / "phase19_interpretability_summary.txt").write_text(summary, encoding="utf-8")
    return {"summary": summary}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_df = load_dataset()
    mapping_df = load_team_mapping()
    schedule_df = build_understat_schedule()
    dataset_with_ids_df = attach_understat_match_ids(dataset_df, schedule_df)
    player_df, roster_match_count = build_player_match_dataset(dataset_with_ids_df)

    phase16 = build_phase16_outputs(player_df)
    phase17 = build_phase17_outputs(dataset_df)
    phase18 = build_phase18_outputs(dataset_df, mapping_df)
    phase19 = build_phase19_outputs(dataset_df)

    print(f"Player match rows: {len(player_df)} across {roster_match_count} matches")
    print()
    print(phase16["summary"])
    print()
    print(phase17["summary"])
    print()
    print(phase18["summary"])
    print()
    print(phase19["summary"])


if __name__ == "__main__":
    main()
