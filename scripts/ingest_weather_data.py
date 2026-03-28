from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


MATCHES_PATH = Path("data/raw/matches.csv")
TEAM_MAPPING_PATH = Path("data/raw/team_mapping.csv")
OUTPUT_PATH = Path("data/raw/weather.csv")
API_URL = "https://archive-api.open-meteo.com/v1/archive"
API_TIMEZONE = "Europe/London"
DAILY_FIELDS = [
    "temperature_2m_mean",
    "apparent_temperature_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "weather_code",
]
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60
HEAVY_RAIN_THRESHOLD = 10.0
STRONG_WIND_THRESHOLD = 50.0
FREEZING_APPARENT_TEMPERATURE_THRESHOLD = 0.0
HEATWAVE_APPARENT_TEMPERATURE_THRESHOLD = 30.0
SEVERE_WEATHER_CODES = {75, 77, 82, 85, 86, 95, 96, 99}


def load_inputs() -> pd.DataFrame:
    matches_df = pd.read_csv(MATCHES_PATH)
    mapping_df = pd.read_csv(TEAM_MAPPING_PATH)

    enriched_df = matches_df.merge(
        mapping_df,
        left_on="home_team",
        right_on="team_name",
        how="left",
        validate="many_to_one",
    )

    missing_mapping = sorted(enriched_df.loc[enriched_df["stadium_name"].isna(), "home_team"].unique())
    if missing_mapping:
        raise ValueError(f"Missing stadium mapping for teams: {missing_mapping}")

    enriched_df["date"] = pd.to_datetime(enriched_df["date"], format="%Y-%m-%d")
    return enriched_df


def request_with_retries(session: requests.Session, params: dict[str, object]) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else 65.0
                time.sleep(sleep_seconds)
                continue

            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(min(2**attempt, 30))

    raise RuntimeError("Weather API request failed after retries")


def fetch_team_weather(session: requests.Session, home_team_df: pd.DataFrame) -> pd.DataFrame:
    team_name = home_team_df["home_team"].iloc[0]
    latitude = float(home_team_df["latitude"].iloc[0])
    longitude = float(home_team_df["longitude"].iloc[0])
    start_date = home_team_df["date"].min().strftime("%Y-%m-%d")
    end_date = home_team_df["date"].max().strftime("%Y-%m-%d")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_FIELDS),
        "timezone": API_TIMEZONE,
    }

    payload = request_with_retries(session, params)
    daily_data = payload.get("daily", {})
    required_keys = {"time", *DAILY_FIELDS}
    missing_keys = sorted(required_keys - set(daily_data))
    if missing_keys:
        raise ValueError(f"Weather API response missing fields for {team_name}: {missing_keys}")

    weather_df = pd.DataFrame(
        {
            "date": daily_data["time"],
            "home_team": team_name,
            "temperature": daily_data["temperature_2m_mean"],
            "apparent_temperature": daily_data["apparent_temperature_mean"],
            "humidity": daily_data["relative_humidity_2m_mean"],
            "rain": daily_data["precipitation_sum"],
            "wind": daily_data["wind_speed_10m_max"],
            "weather_code": daily_data["weather_code"],
        }
    )
    weather_df["heavy_rain_flag"] = (weather_df["rain"] >= HEAVY_RAIN_THRESHOLD).astype(int)
    weather_df["strong_wind_flag"] = (weather_df["wind"] >= STRONG_WIND_THRESHOLD).astype(int)
    weather_df["freezing_flag"] = (
        weather_df["apparent_temperature"] <= FREEZING_APPARENT_TEMPERATURE_THRESHOLD
    ).astype(int)
    weather_df["heatwave_flag"] = (
        weather_df["apparent_temperature"] >= HEATWAVE_APPARENT_TEMPERATURE_THRESHOLD
    ).astype(int)
    weather_df["storm_flag"] = weather_df["weather_code"].isin(SEVERE_WEATHER_CODES).astype(int)
    weather_df["extreme_weather_flag"] = (
        weather_df[
            [
                "heavy_rain_flag",
                "strong_wind_flag",
                "freezing_flag",
                "heatwave_flag",
                "storm_flag",
            ]
        ].max(axis=1)
    ).astype(int)
    weather_df["date"] = pd.to_datetime(weather_df["date"], format="%Y-%m-%d")
    return weather_df


def build_weather_dataset(enriched_df: pd.DataFrame) -> pd.DataFrame:
    session = requests.Session()
    weather_frames = []

    grouped = enriched_df.groupby("home_team", sort=True)
    for _, home_team_df in grouped:
        weather_frames.append(fetch_team_weather(session, home_team_df))
        time.sleep(1)

    daily_weather_df = pd.concat(weather_frames, ignore_index=True)

    match_weather_df = enriched_df[["date", "home_team"]].merge(
        daily_weather_df,
        on=["date", "home_team"],
        how="left",
        validate="one_to_one",
    )

    required_weather_columns = [
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
    ]
    if match_weather_df[required_weather_columns].isna().any().any():
        missing_weather = match_weather_df.loc[
            match_weather_df[required_weather_columns].isna().any(axis=1),
            ["date", "home_team"],
        ]
        raise ValueError(f"Missing weather data for matches: {missing_weather.to_dict(orient='records')[:10]}")

    match_weather_df = match_weather_df.drop_duplicates(subset=["date", "home_team"])
    match_weather_df = match_weather_df.sort_values(["date", "home_team"]).reset_index(drop=True)
    match_weather_df["date"] = match_weather_df["date"].dt.strftime("%Y-%m-%d")
    return match_weather_df


def main() -> None:
    enriched_df = load_inputs()
    weather_df = build_weather_dataset(enriched_df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(weather_df)} match-level weather rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
