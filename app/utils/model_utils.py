from __future__ import annotations

import pandas as pd

from utils.constants import (
    AWAY_FORM_COLUMNS,
    COLD_TEMP,
    FEATURE_COLUMNS,
    FREEZING_APPARENT,
    HEATWAVE_APPARENT,
    HEAVY_RAIN,
    HOME_FORM_COLUMNS,
    HOT_TEMP,
    HUMIDITY_HIGH,
    RESULT_LABELS,
    STRONG_WIND,
    WINDY_SPEED,
)
from utils.data_loader import load_models


def compute_weather_fields(row: dict) -> dict:
    apparent = row["temperature"] - min(row["wind"] * 0.03, 3.0) - min(row["rain"] * 0.08, 1.8)
    row["apparent_temperature"] = apparent
    row["rain_flag"] = int(row["rain"] > 0)
    row["cold_flag"] = int(row["temperature"] < COLD_TEMP)
    row["hot_flag"] = int(row["temperature"] >= HOT_TEMP)
    row["windy_flag"] = int(row["wind"] >= WINDY_SPEED)
    row["humidity_flag"] = int(row["humidity"] >= HUMIDITY_HIGH)
    row["apparent_cold_flag"] = int(apparent < COLD_TEMP)
    row["apparent_hot_flag"] = int(apparent >= HOT_TEMP)
    row["apparent_temperature_gap"] = apparent - row["temperature"]
    row["heavy_rain_flag"] = int(row["rain"] >= HEAVY_RAIN)
    row["strong_wind_flag"] = int(row["wind"] >= STRONG_WIND)
    row["freezing_flag"] = int(apparent <= FREEZING_APPARENT)
    row["heatwave_flag"] = int(apparent >= HEATWAVE_APPARENT)
    row["storm_flag"] = 0
    row["extreme_weather_flag"] = int(
        row["heavy_rain_flag"] or row["strong_wind_flag"] or row["freezing_flag"] or row["heatwave_flag"]
    )
    return row


def get_team_context(df: pd.DataFrame, team: str, side: str) -> dict:
    col = "home_team" if side == "home" else "away_team"
    form_cols = HOME_FORM_COLUMNS if side == "home" else AWAY_FORM_COLUMNS
    team_df = df.loc[df[col] == team].sort_values("date")
    if team_df.empty:
        return df[form_cols].median().to_dict()
    return team_df.iloc[-1][form_cols].to_dict()


def build_prediction_row(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    temperature: float,
    rain: float,
    wind: float,
    humidity: float,
    month: int,
) -> pd.DataFrame:
    row: dict = {
        "temperature": temperature,
        "rain": rain,
        "wind": wind,
        "humidity": humidity,
        "home_advantage": 1,
        "match_month": month,
        "season_year_start": int(df["season_year_start"].max()),
        "season_progress": float(df["season_progress"].median()),
    }
    row = compute_weather_fields(row)
    row["is_winter_match"] = int(month in [12, 1, 2])
    row["is_spring_match"] = int(month in [3, 4, 5])
    row["is_summer_match"] = int(month in [6, 7, 8])
    row["is_autumn_match"] = int(month in [9, 10, 11])
    row.update(get_team_context(df, home_team, "home"))
    row.update(get_team_context(df, away_team, "away"))
    row["strength_gap"] = row["home_team_strength_proxy"] - row["away_team_strength_proxy"]
    row["match_importance_proxy"] = (
        row["season_progress"] * (row["home_team_strength_proxy"] + row["away_team_strength_proxy"]) / 2
    )
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_fixture(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    temperature: float,
    rain: float,
    wind: float,
    humidity: float,
    month: int,
) -> tuple[pd.DataFrame, str, float, pd.DataFrame]:
    clf, reg = load_models()
    input_df = build_prediction_row(df, home_team, away_team, temperature, rain, wind, humidity, month)
    probs = clf.predict_proba(input_df)[0]
    predicted_result = clf.predict(input_df)[0]
    predicted_goals = float(reg.predict(input_df)[0])
    prob_df = pd.DataFrame({
        "result": [RESULT_LABELS[label] for label in clf.classes_],
        "probability": probs,
    })
    return input_df, predicted_result, predicted_goals, prob_df
