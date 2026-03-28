import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_models import (
    CLASSIFICATION_MODEL_PATH,
    FEATURE_COLUMNS,
    INPUT_PATH,
    RANDOM_STATE,
    REGRESSION_MODEL_PATH,
    load_dataset,
    split_dataset,
)


OUTPUT_METRICS_PATH = Path("data/processed/models/evaluation_metrics.json")
OUTPUT_INSIGHTS_PATH = Path("data/processed/models/insight_summary.txt")
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
NON_WEATHER_FEATURES = [feature for feature in FEATURE_COLUMNS if feature not in WEATHER_FEATURES]


def load_models() -> tuple[Pipeline, RandomForestRegressor]:
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    return classification_model, regression_model


def train_no_weather_classification_model(train_df: pd.DataFrame) -> Pipeline:
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
    model.fit(train_df[NON_WEATHER_FEATURES], train_df["match_result"])
    return model


def train_no_weather_regression_model(train_df: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_df[NON_WEATHER_FEATURES], train_df["total_goals"])
    return model


def evaluate_classification(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classification_model: Pipeline,
) -> dict:
    majority_class = train_df["match_result"].mode().iloc[0]
    baseline_predictions = np.full(len(test_df), majority_class)
    model_predictions = classification_model.predict(test_df[FEATURE_COLUMNS])

    no_weather_model = train_no_weather_classification_model(train_df)
    no_weather_predictions = no_weather_model.predict(test_df[NON_WEATHER_FEATURES])

    baseline_accuracy = accuracy_score(test_df["match_result"], baseline_predictions)
    model_accuracy = accuracy_score(test_df["match_result"], model_predictions)
    no_weather_accuracy = accuracy_score(test_df["match_result"], no_weather_predictions)

    classifier = classification_model.named_steps["classifier"]
    coefficient_strength = np.abs(classifier.coef_).mean(axis=0)
    top_index = int(np.argmax(coefficient_strength))

    return {
        "target": "match_result",
        "metric": "accuracy",
        "baseline_name": f"majority_class ({majority_class})",
        "baseline_accuracy": round(float(baseline_accuracy), 4),
        "model_accuracy": round(float(model_accuracy), 4),
        "no_weather_accuracy": round(float(no_weather_accuracy), 4),
        "model_vs_baseline_delta": round(float(model_accuracy - baseline_accuracy), 4),
        "weather_delta": round(float(model_accuracy - no_weather_accuracy), 4),
        "top_feature": FEATURE_COLUMNS[top_index],
        "top_feature_score": round(float(coefficient_strength[top_index]), 4),
    }


def evaluate_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    regression_model: RandomForestRegressor,
) -> dict:
    baseline_value = float(train_df["total_goals"].mean())
    baseline_predictions = np.full(len(test_df), baseline_value)
    model_predictions = regression_model.predict(test_df[FEATURE_COLUMNS])

    no_weather_model = train_no_weather_regression_model(train_df)
    no_weather_predictions = no_weather_model.predict(test_df[NON_WEATHER_FEATURES])

    baseline_mae = mean_absolute_error(test_df["total_goals"], baseline_predictions)
    model_mae = mean_absolute_error(test_df["total_goals"], model_predictions)
    no_weather_mae = mean_absolute_error(test_df["total_goals"], no_weather_predictions)

    importances = regression_model.feature_importances_
    top_index = int(np.argmax(importances))

    return {
        "target": "total_goals",
        "metric": "mae",
        "baseline_name": "train_mean_total_goals",
        "baseline_mae": round(float(baseline_mae), 4),
        "model_mae": round(float(model_mae), 4),
        "no_weather_mae": round(float(no_weather_mae), 4),
        "model_vs_baseline_delta": round(float(baseline_mae - model_mae), 4),
        "weather_delta": round(float(no_weather_mae - model_mae), 4),
        "top_feature": FEATURE_COLUMNS[top_index],
        "top_feature_score": round(float(importances[top_index]), 4),
    }


def build_insight_summary(classification_metrics: dict, regression_metrics: dict, split_summary: dict) -> str:
    if classification_metrics["weather_delta"] > 0:
        classification_weather_line = (
            f"Weather improves match-outcome accuracy by {classification_metrics['weather_delta']} "
            "compared with the no-weather version."
        )
    elif classification_metrics["weather_delta"] < 0:
        classification_weather_line = (
            f"Weather reduces match-outcome accuracy by {abs(classification_metrics['weather_delta'])} "
            "compared with the no-weather version."
        )
    else:
        classification_weather_line = (
            "Weather has no measurable effect on match-outcome accuracy compared with the no-weather version."
        )

    if regression_metrics["weather_delta"] > 0:
        regression_weather_line = (
            f"Weather improves total-goals MAE by {regression_metrics['weather_delta']} "
            "compared with the no-weather version."
        )
    elif regression_metrics["weather_delta"] < 0:
        regression_weather_line = (
            f"Weather worsens total-goals MAE by {abs(regression_metrics['weather_delta'])} "
            "compared with the no-weather version."
        )
    else:
        regression_weather_line = (
            "Weather has no measurable effect on total-goals MAE compared with the no-weather version."
        )

    lines = [
        "Evaluation Summary",
        f"Train/test split: {split_summary['train_rows']} train rows and {split_summary['test_rows']} test rows.",
        (
            f"Classification accuracy is {classification_metrics['model_accuracy']} versus "
            f"a baseline of {classification_metrics['baseline_accuracy']}, for a lift of "
            f"{classification_metrics['model_vs_baseline_delta']}."
        ),
        (
            f"Regression MAE is {regression_metrics['model_mae']} versus a baseline of "
            f"{regression_metrics['baseline_mae']}, improving error by {regression_metrics['model_vs_baseline_delta']}."
        ),
        classification_weather_line,
        regression_weather_line,
        (
            f"Top classification feature: {classification_metrics['top_feature']} "
            f"(score {classification_metrics['top_feature_score']})."
        ),
        (
            f"Top regression feature: {regression_metrics['top_feature']} "
            f"(importance {regression_metrics['top_feature_score']})."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing feature dataset at {INPUT_PATH}")
    if not CLASSIFICATION_MODEL_PATH.exists() or not REGRESSION_MODEL_PATH.exists():
        raise FileNotFoundError("Trained model artifacts are missing. Run Phase 7 first.")

    dataset_df = load_dataset()
    train_df, test_df = split_dataset(dataset_df)
    classification_model, regression_model = load_models()

    classification_metrics = evaluate_classification(train_df, test_df, classification_model)
    regression_metrics = evaluate_regression(train_df, test_df, regression_model)

    split_summary = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start_date": train_df["date"].min().strftime("%Y-%m-%d"),
        "train_end_date": train_df["date"].max().strftime("%Y-%m-%d"),
        "test_start_date": test_df["date"].min().strftime("%Y-%m-%d"),
        "test_end_date": test_df["date"].max().strftime("%Y-%m-%d"),
    }

    metrics = {
        "split": split_summary,
        "classification": classification_metrics,
        "regression": regression_metrics,
    }

    OUTPUT_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    OUTPUT_INSIGHTS_PATH.write_text(build_insight_summary(classification_metrics, regression_metrics, split_summary))

    print(f"Saved evaluation metrics to {OUTPUT_METRICS_PATH}")
    print(f"Saved insight summary to {OUTPUT_INSIGHTS_PATH}")
    print(
        f"Classification accuracy: {classification_metrics['model_accuracy']} | "
        f"Baseline: {classification_metrics['baseline_accuracy']}"
    )
    print(
        f"Regression MAE: {regression_metrics['model_mae']} | "
        f"Baseline: {regression_metrics['baseline_mae']}"
    )


if __name__ == "__main__":
    main()
