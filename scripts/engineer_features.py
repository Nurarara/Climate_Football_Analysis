from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/final_dataset.csv")
OUTPUT_PATH = Path("data/processed/feature_dataset.csv")
COLD_TEMPERATURE_THRESHOLD = 10.0
HOT_TEMPERATURE_THRESHOLD = 25.0
WINDY_SPEED_THRESHOLD = 30.0
HUMIDITY_THRESHOLD = 85.0
SEASON_START_MONTH = 8
ROLLING_WINDOW = 5


def determine_match_result(row: pd.Series) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "win"
    if row["home_goals"] < row["away_goals"]:
        return "loss"
    return "draw"


def add_basic_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    dataset_df = dataset_df.copy()
    dataset_df["total_goals"] = dataset_df["home_goals"] + dataset_df["away_goals"]
    dataset_df["goal_difference"] = dataset_df["home_goals"] - dataset_df["away_goals"]
    dataset_df["match_result"] = dataset_df.apply(determine_match_result, axis=1)

    dataset_df["rain_flag"] = (dataset_df["rain"] > 0).astype(int)
    dataset_df["cold_flag"] = (dataset_df["temperature"] < COLD_TEMPERATURE_THRESHOLD).astype(int)
    dataset_df["hot_flag"] = (dataset_df["temperature"] >= HOT_TEMPERATURE_THRESHOLD).astype(int)
    dataset_df["windy_flag"] = (dataset_df["wind"] >= WINDY_SPEED_THRESHOLD).astype(int)
    dataset_df["humidity_flag"] = (dataset_df["humidity"] >= HUMIDITY_THRESHOLD).astype(int)
    dataset_df["apparent_cold_flag"] = (
        dataset_df["apparent_temperature"] < COLD_TEMPERATURE_THRESHOLD
    ).astype(int)
    dataset_df["apparent_hot_flag"] = (
        dataset_df["apparent_temperature"] >= HOT_TEMPERATURE_THRESHOLD
    ).astype(int)
    dataset_df["apparent_temperature_gap"] = (
        dataset_df["apparent_temperature"] - dataset_df["temperature"]
    )
    dataset_df["home_advantage"] = 1
    return dataset_df


def add_temporal_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    dataset_df = dataset_df.copy()
    dataset_df["match_month"] = dataset_df["date"].dt.month
    dataset_df["calendar_year"] = dataset_df["date"].dt.year
    dataset_df["season_year_start"] = dataset_df["calendar_year"]
    dataset_df.loc[dataset_df["match_month"] < SEASON_START_MONTH, "season_year_start"] = (
        dataset_df["calendar_year"] - 1
    )
    dataset_df["seasonal_indicator"] = dataset_df["match_month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )
    dataset_df["is_winter_match"] = (dataset_df["seasonal_indicator"] == "winter").astype(int)
    dataset_df["is_spring_match"] = (dataset_df["seasonal_indicator"] == "spring").astype(int)
    dataset_df["is_summer_match"] = (dataset_df["seasonal_indicator"] == "summer").astype(int)
    dataset_df["is_autumn_match"] = (dataset_df["seasonal_indicator"] == "autumn").astype(int)
    dataset_df["season_match_index"] = dataset_df.groupby("season_year_start").cumcount() + 1
    season_sizes = dataset_df.groupby("season_year_start")["date"].transform("size")
    dataset_df["season_progress"] = dataset_df["season_match_index"] / season_sizes
    return dataset_df


def build_team_history(dataset_df: pd.DataFrame) -> pd.DataFrame:
    base_df = dataset_df.reset_index(drop=True).reset_index(names="match_id")

    home_history_df = base_df[
        ["match_id", "date", "home_team", "away_team", "home_goals", "away_goals"]
    ].rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_goals": "goals_for",
            "away_goals": "goals_against",
        }
    )
    home_history_df["side"] = "home"

    away_history_df = base_df[
        ["match_id", "date", "home_team", "away_team", "home_goals", "away_goals"]
    ].rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_goals": "goals_for",
            "home_goals": "goals_against",
        }
    )
    away_history_df["side"] = "away"

    team_history_df = pd.concat([home_history_df, away_history_df], ignore_index=True)
    team_history_df["date"] = pd.to_datetime(team_history_df["date"], format="%Y-%m-%d")
    team_history_df["goal_difference"] = (
        team_history_df["goals_for"] - team_history_df["goals_against"]
    )
    team_history_df["points"] = 0
    team_history_df.loc[team_history_df["goals_for"] > team_history_df["goals_against"], "points"] = 3
    team_history_df.loc[team_history_df["goals_for"] == team_history_df["goals_against"], "points"] = 1

    team_history_df = team_history_df.sort_values(["team", "date", "match_id", "side"]).reset_index(drop=True)
    return team_history_df


def add_rolling_form_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    team_history_df = build_team_history(dataset_df)
    rolling_columns = ["points", "goals_for", "goals_against", "goal_difference"]

    for column in rolling_columns:
        feature_name = f"{column}_avg_last_{ROLLING_WINDOW}"
        team_history_df[feature_name] = team_history_df.groupby("team")[column].transform(
            lambda values: values.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )

    feature_columns = [f"{column}_avg_last_{ROLLING_WINDOW}" for column in rolling_columns]

    home_form_df = team_history_df.loc[team_history_df["side"] == "home", ["match_id", *feature_columns]].rename(
        columns={column: f"home_team_{column}" for column in feature_columns}
    )
    away_form_df = team_history_df.loc[team_history_df["side"] == "away", ["match_id", *feature_columns]].rename(
        columns={column: f"away_team_{column}" for column in feature_columns}
    )

    enriched_df = dataset_df.reset_index(drop=True).reset_index(names="match_id")
    enriched_df = enriched_df.merge(home_form_df, on="match_id", how="left", validate="one_to_one")
    enriched_df = enriched_df.merge(away_form_df, on="match_id", how="left", validate="one_to_one")

    rolling_feature_columns = [column for column in enriched_df.columns if column.endswith(f"_avg_last_{ROLLING_WINDOW}")]
    enriched_df[rolling_feature_columns] = enriched_df[rolling_feature_columns].fillna(0)
    enriched_df["home_team_strength_proxy"] = (
        0.7 * enriched_df[f"home_team_points_avg_last_{ROLLING_WINDOW}"]
        + 0.3 * enriched_df[f"home_team_goals_for_avg_last_{ROLLING_WINDOW}"]
    )
    enriched_df["away_team_strength_proxy"] = (
        0.7 * enriched_df[f"away_team_points_avg_last_{ROLLING_WINDOW}"]
        + 0.3 * enriched_df[f"away_team_goals_for_avg_last_{ROLLING_WINDOW}"]
    )
    enriched_df["strength_gap"] = (
        enriched_df["home_team_strength_proxy"] - enriched_df["away_team_strength_proxy"]
    )
    enriched_df["match_importance_proxy"] = (
        enriched_df["season_progress"]
        * (enriched_df["home_team_strength_proxy"] + enriched_df["away_team_strength_proxy"])
        / 2
    )
    enriched_df = enriched_df.drop(columns=["match_id"])
    return enriched_df


def validate_output(features_df: pd.DataFrame, input_df: pd.DataFrame) -> None:
    if len(features_df) != len(input_df):
        raise ValueError("Feature engineering changed the row count")

    if features_df.duplicated(subset=["date", "home_team", "away_team"]).any():
        raise ValueError("Duplicate matches found after feature engineering")

    required_columns = [
        "total_goals",
        "goal_difference",
        "match_result",
        "rain_flag",
        "cold_flag",
        "hot_flag",
        "windy_flag",
        "humidity_flag",
        "apparent_temperature_gap",
        "home_advantage",
        "home_team_strength_proxy",
        "away_team_strength_proxy",
        "season_progress",
        "match_importance_proxy",
    ]
    missing_columns = sorted(set(required_columns) - set(features_df.columns))
    if missing_columns:
        raise ValueError(f"Missing engineered feature columns: {missing_columns}")

    if features_df[required_columns].isna().any().any():
        raise ValueError("Missing values found in engineered features")


def main() -> None:
    dataset_df = pd.read_csv(INPUT_PATH)
    dataset_df["date"] = pd.to_datetime(dataset_df["date"], format="%Y-%m-%d")
    dataset_df = dataset_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    features_df = add_basic_features(dataset_df)
    features_df = add_temporal_features(features_df)
    features_df = add_rolling_form_features(features_df)
    validate_output(features_df, dataset_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df["date"] = features_df["date"].dt.strftime("%Y-%m-%d")
    features_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(features_df)} rows to {OUTPUT_PATH}")
    print(f"Columns: {', '.join(features_df.columns)}")


if __name__ == "__main__":
    main()
