from pathlib import Path

import pandas as pd


MATCHES_PATH = Path("data/processed/matches_clean.csv")
WEATHER_PATH = Path("data/processed/weather_clean.csv")
OUTPUT_PATH = Path("data/processed/final_dataset.csv")
MAX_ALLOWED_LOSS_FRACTION = 0.05
REQUIRED_WEATHER_COLUMNS = {
    "date",
    "home_team",
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
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    matches_df = pd.read_csv(MATCHES_PATH)
    weather_df = pd.read_csv(WEATHER_PATH)
    return matches_df, weather_df


def validate_inputs(matches_df: pd.DataFrame, weather_df: pd.DataFrame) -> None:
    if matches_df.duplicated(subset=["date", "home_team", "away_team"]).any():
        raise ValueError("Duplicate match rows found in cleaned matches data")

    if weather_df.duplicated(subset=["date", "home_team"]).any():
        raise ValueError("Duplicate weather rows found in cleaned weather data")

    required_match_columns = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    required_weather_columns = REQUIRED_WEATHER_COLUMNS

    missing_match_columns = sorted(required_match_columns - set(matches_df.columns))
    missing_weather_columns = sorted(required_weather_columns - set(weather_df.columns))

    if missing_match_columns:
        raise ValueError(f"Missing columns in matches data: {missing_match_columns}")
    if missing_weather_columns:
        raise ValueError(f"Missing columns in weather data: {missing_weather_columns}")


def build_final_dataset(matches_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = matches_df.merge(
        weather_df,
        on=["date", "home_team"],
        how="left",
        validate="many_to_one",
    )

    expected_rows = len(matches_df)
    if len(merged_df) != expected_rows:
        raise ValueError(
            f"Joined row count mismatch: expected {expected_rows}, got {len(merged_df)}"
        )

    missing_weather_columns = sorted(REQUIRED_WEATHER_COLUMNS - {"date", "home_team"})
    missing_weather_mask = merged_df[missing_weather_columns].isna().any(axis=1)
    dropped_rows = int(missing_weather_mask.sum())
    loss_fraction = dropped_rows / expected_rows if expected_rows else 0

    if loss_fraction > MAX_ALLOWED_LOSS_FRACTION:
        raise ValueError(
            f"Excessive data loss after join: dropped {dropped_rows} of {expected_rows} rows"
        )

    final_df = merged_df.loc[~missing_weather_mask].copy()
    if final_df[missing_weather_columns].isna().any().any():
        raise ValueError("Weather features still missing after dropping incomplete rows")

    final_df = final_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return final_df


def main() -> None:
    matches_df, weather_df = load_inputs()
    validate_inputs(matches_df, weather_df)
    final_df = build_final_dataset(matches_df, weather_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(final_df)} rows to {OUTPUT_PATH}")
    print(f"Columns: {', '.join(final_df.columns)}")


if __name__ == "__main__":
    main()
