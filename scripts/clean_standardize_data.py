from pathlib import Path

import pandas as pd


RAW_MATCHES_PATH = Path("data/raw/matches.csv")
RAW_WEATHER_PATH = Path("data/raw/weather.csv")
PROCESSED_MATCHES_PATH = Path("data/processed/matches_clean.csv")
PROCESSED_WEATHER_PATH = Path("data/processed/weather_clean.csv")
TIMEZONE_ASSUMPTION = "Europe/London"
WEATHER_NUMERIC_COLUMNS = [
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
WEATHER_FLAG_COLUMNS = [
    "heavy_rain_flag",
    "strong_wind_flag",
    "freezing_flag",
    "heatwave_flag",
    "storm_flag",
    "extreme_weather_flag",
]


def clean_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    matches_df = matches_df.copy()
    matches_df["date"] = pd.to_datetime(matches_df["date"], errors="coerce")

    for column in ["home_team", "away_team"]:
        matches_df[column] = matches_df[column].astype(str).str.strip()

    for column in ["home_goals", "away_goals"]:
        matches_df[column] = pd.to_numeric(matches_df[column], errors="coerce")

    matches_df = matches_df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])

    for column in ["home_goals", "away_goals"]:
        if (matches_df[column] < 0).any():
            raise ValueError(f"Negative values found in {column}")
        if not (matches_df[column] % 1 == 0).all():
            raise ValueError(f"Non-integer values found in {column}")
        matches_df[column] = matches_df[column].astype(int)

    if (matches_df["home_team"] == "").any() or (matches_df["away_team"] == "").any():
        raise ValueError("Blank team names found in matches data")
    if (matches_df["home_team"] == matches_df["away_team"]).any():
        raise ValueError("Invalid matches with identical home and away teams found")

    if matches_df.duplicated(subset=["date", "home_team", "away_team"]).any():
        raise ValueError("Duplicate matches found after cleaning")

    matches_df["date"] = matches_df["date"].dt.strftime("%Y-%m-%d")
    matches_df = matches_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return matches_df


def clean_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
    weather_df["home_team"] = weather_df["home_team"].astype(str).str.strip()

    for column in WEATHER_NUMERIC_COLUMNS:
        weather_df[column] = pd.to_numeric(weather_df[column], errors="coerce")

    weather_df = weather_df.dropna(subset=["date", "home_team", *WEATHER_NUMERIC_COLUMNS])

    if (weather_df["home_team"] == "").any():
        raise ValueError("Blank home team values found in weather data")
    if not weather_df["humidity"].between(0, 100).all():
        raise ValueError("Humidity values are out of range")
    if (weather_df["rain"] < 0).any():
        raise ValueError("Negative rainfall values found")
    if (weather_df["wind"] < 0).any():
        raise ValueError("Negative wind values found")
    if (weather_df["weather_code"] < 0).any():
        raise ValueError("Negative weather codes found")
    for column in WEATHER_FLAG_COLUMNS:
        if not weather_df[column].isin([0, 1]).all():
            raise ValueError(f"Non-binary values found in {column}")
        weather_df[column] = weather_df[column].astype(int)
    weather_df["weather_code"] = weather_df["weather_code"].astype(int)
    if weather_df.duplicated(subset=["date", "home_team"]).any():
        raise ValueError("Duplicate weather rows found after cleaning")

    weather_df["date"] = weather_df["date"].dt.strftime("%Y-%m-%d")
    weather_df = weather_df.sort_values(["date", "home_team"]).reset_index(drop=True)
    return weather_df


def validate_merge_readiness(matches_df: pd.DataFrame, weather_df: pd.DataFrame) -> None:
    join_df = matches_df.merge(
        weather_df,
        on=["date", "home_team"],
        how="left",
        validate="many_to_one",
    )

    if len(join_df) != len(matches_df):
        raise ValueError("Row count changed during merge readiness validation")

    missing_weather = join_df[WEATHER_NUMERIC_COLUMNS].isna().any(axis=1)
    if missing_weather.any():
        missing_records = join_df.loc[missing_weather, ["date", "home_team", "away_team"]]
        raise ValueError(
            f"Weather is missing for cleaned matches: {missing_records.head(10).to_dict(orient='records')}"
        )


def main() -> None:
    matches_df = pd.read_csv(RAW_MATCHES_PATH)
    weather_df = pd.read_csv(RAW_WEATHER_PATH)

    cleaned_matches_df = clean_matches(matches_df)
    cleaned_weather_df = clean_weather(weather_df)
    validate_merge_readiness(cleaned_matches_df, cleaned_weather_df)

    PROCESSED_MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_matches_df.to_csv(PROCESSED_MATCHES_PATH, index=False)
    cleaned_weather_df.to_csv(PROCESSED_WEATHER_PATH, index=False)

    print(f"Saved cleaned matches to {PROCESSED_MATCHES_PATH}")
    print(f"Saved cleaned weather to {PROCESSED_WEATHER_PATH}")
    print(f"Timezone assumption aligned to {TIMEZONE_ASSUMPTION}")


if __name__ == "__main__":
    main()
