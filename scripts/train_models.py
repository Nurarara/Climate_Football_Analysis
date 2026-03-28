import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/feature_dataset.csv")
MODEL_DIR = Path("data/processed/models")
CLASSIFICATION_MODEL_PATH = MODEL_DIR / "match_outcome_model.joblib"
REGRESSION_MODEL_PATH = MODEL_DIR / "total_goals_model.joblib"
METADATA_PATH = MODEL_DIR / "training_metadata.json"
TEST_FRACTION = 0.2
RANDOM_STATE = 42

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


def load_dataset() -> pd.DataFrame:
    dataset_df = pd.read_csv(INPUT_PATH)
    dataset_df["date"] = pd.to_datetime(dataset_df["date"], format="%Y-%m-%d")
    dataset_df = dataset_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return dataset_df


def split_dataset(dataset_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(dataset_df["date"].unique())
    split_index = max(1, int(len(unique_dates) * (1 - TEST_FRACTION)))
    split_index = min(split_index, len(unique_dates) - 1)

    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train_df = dataset_df[dataset_df["date"].isin(train_dates)].copy()
    test_df = dataset_df[dataset_df["date"].isin(test_dates)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological train/test split produced an empty partition")

    return train_df, test_df


def train_classification_model(train_df: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    C=0.5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["match_result"])
    return model


def train_regression_model(train_df: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["total_goals"])
    return model


def build_metadata(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classification_model: Pipeline,
    regression_model: RandomForestRegressor,
) -> dict:
    classification_train_predictions = classification_model.predict(train_df[FEATURE_COLUMNS])
    classification_test_predictions = classification_model.predict(test_df[FEATURE_COLUMNS])
    regression_train_predictions = regression_model.predict(train_df[FEATURE_COLUMNS])
    regression_test_predictions = regression_model.predict(test_df[FEATURE_COLUMNS])

    return {
        "feature_columns": FEATURE_COLUMNS,
        "split": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_start_date": train_df["date"].min().strftime("%Y-%m-%d"),
            "train_end_date": train_df["date"].max().strftime("%Y-%m-%d"),
            "test_start_date": test_df["date"].min().strftime("%Y-%m-%d"),
            "test_end_date": test_df["date"].max().strftime("%Y-%m-%d"),
        },
        "classification": {
            "target": "match_result",
            "model": "LogisticRegression",
            "train_accuracy": round(
                accuracy_score(train_df["match_result"], classification_train_predictions), 4
            ),
            "test_accuracy": round(
                accuracy_score(test_df["match_result"], classification_test_predictions), 4
            ),
        },
        "regression": {
            "target": "total_goals",
            "model": "RandomForestRegressor",
            "train_mae": round(
                mean_absolute_error(train_df["total_goals"], regression_train_predictions), 4
            ),
            "test_mae": round(
                mean_absolute_error(test_df["total_goals"], regression_test_predictions), 4
            ),
        },
    }


def main() -> None:
    dataset_df = load_dataset()
    train_df, test_df = split_dataset(dataset_df)

    classification_model = train_classification_model(train_df)
    regression_model = train_regression_model(train_df)
    metadata = build_metadata(train_df, test_df, classification_model, regression_model)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classification_model, CLASSIFICATION_MODEL_PATH)
    joblib.dump(regression_model, REGRESSION_MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Saved classification model to {CLASSIFICATION_MODEL_PATH}")
    print(f"Saved regression model to {REGRESSION_MODEL_PATH}")
    print(f"Saved training metadata to {METADATA_PATH}")
    print(
        "Classification test accuracy: "
        f"{metadata['classification']['test_accuracy']} | Regression test MAE: {metadata['regression']['test_mae']}"
    )


if __name__ == "__main__":
    main()
