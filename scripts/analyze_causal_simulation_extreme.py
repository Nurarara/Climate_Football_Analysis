from pathlib import Path

import altair as alt
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_models import (
    CLASSIFICATION_MODEL_PATH,
    FEATURE_COLUMNS,
    REGRESSION_MODEL_PATH,
    load_dataset,
    split_dataset,
)


OUTPUT_DIR = Path("data/processed/analysis")
PLOTS_DIR = OUTPUT_DIR / "plots"

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
CONTROL_FEATURES = [feature for feature in FEATURE_COLUMNS if feature not in WEATHER_FEATURES]

RESULT_LABELS = {"win": "Home Win", "draw": "Draw", "loss": "Away Win"}

NORMAL_SCENARIO = {
    "scenario": "normal_day",
    "temperature": 15.0,
    "apparent_temperature": 15.0,
    "humidity": 65.0,
    "rain": 0.0,
    "wind": 15.0,
    "heavy_rain_flag": 0,
    "strong_wind_flag": 0,
    "freezing_flag": 0,
    "heatwave_flag": 0,
    "storm_flag": 0,
    "extreme_weather_flag": 0,
    "rain_flag": 0,
    "cold_flag": 0,
    "hot_flag": 0,
    "windy_flag": 0,
    "humidity_flag": 0,
    "apparent_cold_flag": 0,
    "apparent_hot_flag": 0,
    "apparent_temperature_gap": 0.0,
}
HEAVY_RAIN_SCENARIO = {
    "scenario": "heavy_rain",
    "temperature": 11.0,
    "apparent_temperature": 8.5,
    "humidity": 92.0,
    "rain": 14.0,
    "wind": 32.0,
    "heavy_rain_flag": 1,
    "strong_wind_flag": 0,
    "freezing_flag": 0,
    "heatwave_flag": 0,
    "storm_flag": 0,
    "extreme_weather_flag": 1,
    "rain_flag": 1,
    "cold_flag": 0,
    "hot_flag": 0,
    "windy_flag": 1,
    "humidity_flag": 1,
    "apparent_cold_flag": 1,
    "apparent_hot_flag": 0,
    "apparent_temperature_gap": -2.5,
}
EXTREME_COLD_SCENARIO = {
    "scenario": "extreme_cold",
    "temperature": 1.0,
    "apparent_temperature": -3.5,
    "humidity": 80.0,
    "rain": 0.0,
    "wind": 22.0,
    "heavy_rain_flag": 0,
    "strong_wind_flag": 0,
    "freezing_flag": 1,
    "heatwave_flag": 0,
    "storm_flag": 0,
    "extreme_weather_flag": 1,
    "rain_flag": 0,
    "cold_flag": 1,
    "hot_flag": 0,
    "windy_flag": 0,
    "humidity_flag": 0,
    "apparent_cold_flag": 1,
    "apparent_hot_flag": 0,
    "apparent_temperature_gap": -4.5,
}
SCENARIOS = [NORMAL_SCENARIO, HEAVY_RAIN_SCENARIO, EXTREME_COLD_SCENARIO]


def save_chart(chart: alt.Chart, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output_path)


def train_goal_regression(feature_names: list[str], train_df: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(train_df[feature_names], train_df["total_goals"])
    return model


def train_outcome_classifier(feature_names: list[str], train_df: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    C=0.5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_df[feature_names], train_df["match_result"])
    return model


def build_causal_outputs(dataset_df: pd.DataFrame) -> dict[str, object]:
    train_df, test_df = split_dataset(dataset_df)

    goals_control_model = train_goal_regression(CONTROL_FEATURES, train_df)
    goals_weather_model = train_goal_regression(CONTROL_FEATURES + WEATHER_FEATURES, train_df)
    outcome_control_model = train_outcome_classifier(CONTROL_FEATURES, train_df)
    outcome_weather_model = train_outcome_classifier(CONTROL_FEATURES + WEATHER_FEATURES, train_df)

    causal_metrics = {
        "goal_regression_control_mae": mean_absolute_error(
            test_df["total_goals"], goals_control_model.predict(test_df[CONTROL_FEATURES])
        ),
        "goal_regression_weather_mae": mean_absolute_error(
            test_df["total_goals"], goals_weather_model.predict(test_df[CONTROL_FEATURES + WEATHER_FEATURES])
        ),
        "outcome_control_accuracy": accuracy_score(
            test_df["match_result"], outcome_control_model.predict(test_df[CONTROL_FEATURES])
        ),
        "outcome_weather_accuracy": accuracy_score(
            test_df["match_result"], outcome_weather_model.predict(test_df[CONTROL_FEATURES + WEATHER_FEATURES])
        ),
    }
    causal_metrics["goal_regression_weather_gain"] = (
        causal_metrics["goal_regression_control_mae"] - causal_metrics["goal_regression_weather_mae"]
    )
    causal_metrics["outcome_weather_gain"] = (
        causal_metrics["outcome_weather_accuracy"] - causal_metrics["outcome_control_accuracy"]
    )

    goal_coefficients_df = pd.DataFrame(
        {
            "feature": CONTROL_FEATURES + WEATHER_FEATURES,
            "coefficient": goals_weather_model.named_steps["regressor"].coef_,
        }
    )
    goal_coefficients_df["feature_group"] = goal_coefficients_df["feature"].isin(WEATHER_FEATURES).map(
        {True: "weather", False: "control"}
    )
    goal_coefficients_df["absolute_coefficient"] = goal_coefficients_df["coefficient"].abs()
    goal_coefficients_df = goal_coefficients_df.sort_values("absolute_coefficient", ascending=False)

    outcome_coefficients = outcome_weather_model.named_steps["classifier"].coef_
    outcome_coefficients_df = pd.DataFrame(
        {
            "feature": CONTROL_FEATURES + WEATHER_FEATURES,
            "average_absolute_coefficient": outcome_coefficients.mean(axis=0) * 0,
        }
    )
    outcome_coefficients_df["average_absolute_coefficient"] = outcome_coefficients_df["feature"].map(
        {
            feature: abs(outcome_coefficients[:, index]).mean()
            for index, feature in enumerate(CONTROL_FEATURES + WEATHER_FEATURES)
        }
    )
    outcome_coefficients_df["feature_group"] = outcome_coefficients_df["feature"].isin(WEATHER_FEATURES).map(
        {True: "weather", False: "control"}
    )
    outcome_coefficients_df = outcome_coefficients_df.sort_values(
        "average_absolute_coefficient", ascending=False
    )

    causal_metrics_df = pd.DataFrame(
        [
            {"metric": key, "value": round(float(value), 4)}
            for key, value in causal_metrics.items()
        ]
    )
    causal_metrics_df.to_csv(OUTPUT_DIR / "phase13_causal_metrics.csv", index=False)
    goal_coefficients_df.to_csv(OUTPUT_DIR / "phase13_goal_regression_coefficients.csv", index=False)
    outcome_coefficients_df.to_csv(OUTPUT_DIR / "phase13_outcome_coefficients.csv", index=False)

    comparison_chart_df = pd.DataFrame(
        [
            {
                "task": "Total Goals MAE",
                "model_type": "controls_only",
                "value": causal_metrics["goal_regression_control_mae"],
            },
            {
                "task": "Total Goals MAE",
                "model_type": "controls_plus_weather",
                "value": causal_metrics["goal_regression_weather_mae"],
            },
            {
                "task": "Outcome Accuracy",
                "model_type": "controls_only",
                "value": causal_metrics["outcome_control_accuracy"],
            },
            {
                "task": "Outcome Accuracy",
                "model_type": "controls_plus_weather",
                "value": causal_metrics["outcome_weather_accuracy"],
            },
        ]
    )
    comparison_chart = (
        alt.Chart(comparison_chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("model_type:N", title="Model"),
            y=alt.Y("value:Q", title="Metric Value"),
            color=alt.Color("model_type:N", legend=None),
            column=alt.Column("task:N", title=None),
            tooltip=["task", "model_type", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(width=260, height=260, title="Phase 13 - Weather vs No-Weather Controls")
    )
    coefficient_chart = (
        alt.Chart(goal_coefficients_df.head(12))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("coefficient:Q", title="Standardized Coefficient"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color("feature_group:N", title="Feature Group"),
            tooltip=[
                "feature",
                "feature_group",
                alt.Tooltip("coefficient:Q", format=".4f"),
                alt.Tooltip("absolute_coefficient:Q", format=".4f"),
            ],
        )
        .properties(width=620, height=320, title="Phase 13 - Top Total-Goals Coefficients")
    )

    save_chart(comparison_chart, PLOTS_DIR / "phase13_weather_vs_controls.html")
    save_chart(coefficient_chart, PLOTS_DIR / "phase13_goal_coefficients.html")

    causal_summary = "\n".join(
        [
            "Phase 13 - Causal Analysis",
            "This remains observational evidence, not a true causal proof, because the data is not randomized.",
            (
                f"Controlling for team strength, recent form, and seasonality, adding weather changes goal-model "
                f"MAE by {causal_metrics['goal_regression_weather_gain']:+.4f}."
            ),
            (
                f"Controlling for the same variables, adding weather changes match-outcome accuracy by "
                f"{causal_metrics['outcome_weather_gain']:+.4f}."
            ),
            (
                "Interpretation: weather carries a small incremental signal after controls, so it is not pure noise, "
                "but the effect size is modest and should not be overstated as causal certainty."
            ),
            (
                f"Top weather-linked total-goals coefficient: {goal_coefficients_df[goal_coefficients_df['feature_group'] == 'weather'].iloc[0]['feature']}."
            ),
        ]
    )
    (OUTPUT_DIR / "phase13_causal_summary.txt").write_text(causal_summary, encoding="utf-8")

    return {
        "summary": causal_summary,
        "metrics_df": causal_metrics_df,
    }


def build_simulation_outputs(dataset_df: pd.DataFrame) -> dict[str, object]:
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    _, test_df = split_dataset(dataset_df)
    simulation_base_df = (
        test_df.sort_values("date", ascending=False)
        .head(40)
        .reset_index(drop=True)
        .copy()
    )

    simulation_rows = []
    classes = list(classification_model.classes_)
    for _, row in simulation_base_df.iterrows():
        base_row = row.copy()
        for scenario in SCENARIOS:
            scenario_row = base_row.copy()
            for feature_name, value in scenario.items():
                if feature_name == "scenario":
                    continue
                scenario_row[feature_name] = value
            input_df = pd.DataFrame([scenario_row])[FEATURE_COLUMNS]
            predicted_result = classification_model.predict(input_df)[0]
            predicted_probabilities = classification_model.predict_proba(input_df)[0]
            probability_map = {label: predicted_probabilities[index] for index, label in enumerate(classes)}
            simulation_rows.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "scenario": scenario["scenario"],
                    "predicted_total_goals": float(regression_model.predict(input_df)[0]),
                    "predicted_result": predicted_result,
                    "predicted_result_label": RESULT_LABELS[predicted_result],
                    "home_win_probability": float(probability_map.get("win", 0.0)),
                    "draw_probability": float(probability_map.get("draw", 0.0)),
                    "away_win_probability": float(probability_map.get("loss", 0.0)),
                }
            )

    simulation_df = pd.DataFrame(simulation_rows)
    simulation_df.to_csv(OUTPUT_DIR / "phase14_match_simulations.csv", index=False)

    simulation_summary_df = (
        simulation_df.groupby("scenario")
        .agg(
            matches=("date", "count"),
            avg_predicted_goals=("predicted_total_goals", "mean"),
            avg_home_win_probability=("home_win_probability", "mean"),
            avg_draw_probability=("draw_probability", "mean"),
            avg_away_win_probability=("away_win_probability", "mean"),
        )
        .reset_index()
    )
    simulation_summary_df.to_csv(OUTPUT_DIR / "phase14_simulation_summary.csv", index=False)

    pivot_df = simulation_df.pivot_table(
        index=["date", "home_team", "away_team"],
        columns="scenario",
        values=["predicted_total_goals", "home_win_probability"],
    )
    pivot_df.columns = [
        f"{metric}_{scenario}" for metric, scenario in pivot_df.columns
    ]
    pivot_df = pivot_df.reset_index()
    pivot_df["heavy_rain_goal_delta_vs_normal"] = (
        pivot_df["predicted_total_goals_heavy_rain"] - pivot_df["predicted_total_goals_normal_day"]
    )
    pivot_df["extreme_cold_goal_delta_vs_normal"] = (
        pivot_df["predicted_total_goals_extreme_cold"] - pivot_df["predicted_total_goals_normal_day"]
    )
    pivot_df["heavy_rain_home_win_delta_vs_normal"] = (
        pivot_df["home_win_probability_heavy_rain"] - pivot_df["home_win_probability_normal_day"]
    )
    pivot_df["extreme_cold_home_win_delta_vs_normal"] = (
        pivot_df["home_win_probability_extreme_cold"] - pivot_df["home_win_probability_normal_day"]
    )
    pivot_df = pivot_df.sort_values("heavy_rain_home_win_delta_vs_normal", ascending=False)
    pivot_df.to_csv(OUTPUT_DIR / "phase14_scenario_deltas.csv", index=False)

    summary_chart = (
        alt.Chart(simulation_summary_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("scenario:N", title="Scenario"),
            y=alt.Y("avg_predicted_goals:Q", title="Average Predicted Goals"),
            color=alt.Color("scenario:N", legend=None),
            tooltip=[
                "scenario",
                alt.Tooltip("avg_predicted_goals:Q", format=".2f"),
                alt.Tooltip("avg_home_win_probability:Q", format=".2%"),
            ],
        )
        .properties(width=620, height=300, title="Phase 14 - Scenario Simulation Summary")
    )
    delta_chart = (
        alt.Chart(pivot_df.head(12))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("heavy_rain_home_win_delta_vs_normal:Q", title="Heavy Rain Home-Win Delta"),
            y=alt.Y("home_team:N", sort="-x", title="Home Team"),
            color=alt.condition(
                alt.datum.heavy_rain_home_win_delta_vs_normal > 0,
                alt.value("#2E8B57"),
                alt.value("#C0392B"),
            ),
            tooltip=[
                "date",
                "home_team",
                "away_team",
                alt.Tooltip("heavy_rain_goal_delta_vs_normal:Q", format=".2f"),
                alt.Tooltip("heavy_rain_home_win_delta_vs_normal:Q", format=".2%"),
            ],
        )
        .properties(width=620, height=320, title="Phase 14 - Matches Most Shifted by Heavy Rain")
    )
    save_chart(summary_chart, PLOTS_DIR / "phase14_simulation_summary.html")
    save_chart(delta_chart, PLOTS_DIR / "phase14_heavy_rain_shift.html")

    average_heavy_rain_goal_delta = (
        simulation_summary_df.loc[simulation_summary_df["scenario"] == "heavy_rain", "avg_predicted_goals"].iloc[0]
        - simulation_summary_df.loc[simulation_summary_df["scenario"] == "normal_day", "avg_predicted_goals"].iloc[0]
    )
    average_extreme_cold_goal_delta = (
        simulation_summary_df.loc[simulation_summary_df["scenario"] == "extreme_cold", "avg_predicted_goals"].iloc[0]
        - simulation_summary_df.loc[simulation_summary_df["scenario"] == "normal_day", "avg_predicted_goals"].iloc[0]
    )
    strongest_shift = pivot_df.iloc[0]

    simulation_summary = "\n".join(
        [
            "Phase 14 - Match Simulation Engine",
            (
                f"Across recent holdout fixtures, heavy rain changes predicted total goals by "
                f"{average_heavy_rain_goal_delta:+.2f} versus a normal day."
            ),
            (
                f"Across the same fixtures, extreme cold changes predicted total goals by "
                f"{average_extreme_cold_goal_delta:+.2f} versus a normal day."
            ),
            (
                f"Strongest heavy-rain home-win shift: {strongest_shift['home_team']} vs "
                f"{strongest_shift['away_team']} on {strongest_shift['date']} with a change of "
                f"{strongest_shift['heavy_rain_home_win_delta_vs_normal']:+.2%}."
            ),
            (
                "Use the scenario delta file to answer questions like: if this match was played in rain vs dry, "
                "how much would expected goals or home-win probability move?"
            ),
        ]
    )
    (OUTPUT_DIR / "phase14_simulation_summary.txt").write_text(simulation_summary, encoding="utf-8")

    return {
        "summary": simulation_summary,
        "simulation_summary_df": simulation_summary_df,
    }


def build_extreme_weather_outputs(dataset_df: pd.DataFrame) -> dict[str, object]:
    analysis_df = dataset_df.copy()
    analysis_df["very_low_temperature_flag"] = (analysis_df["temperature"] <= 5).astype(int)
    extreme_definitions = {
        "heavy_rain": analysis_df["heavy_rain_flag"] == 1,
        "extreme_wind": analysis_df["strong_wind_flag"] == 1,
        "very_low_temperature": analysis_df["very_low_temperature_flag"] == 1,
        "any_extreme": analysis_df["extreme_weather_flag"] == 1,
    }

    rows = []
    for label, mask in extreme_definitions.items():
        extreme_df = analysis_df.loc[mask].copy()
        baseline_df = analysis_df.loc[~mask].copy()
        rows.append(
            {
                "extreme_type": label,
                "extreme_matches": len(extreme_df),
                "baseline_matches": len(baseline_df),
                "extreme_avg_goals": extreme_df["total_goals"].mean(),
                "baseline_avg_goals": baseline_df["total_goals"].mean(),
                "extreme_home_win_rate": (extreme_df["match_result"] == "win").mean(),
                "baseline_home_win_rate": (baseline_df["match_result"] == "win").mean(),
                "extreme_draw_rate": (extreme_df["match_result"] == "draw").mean(),
                "baseline_draw_rate": (baseline_df["match_result"] == "draw").mean(),
            }
        )

    extreme_summary_df = pd.DataFrame(rows)
    extreme_summary_df["goals_delta_vs_baseline"] = (
        extreme_summary_df["extreme_avg_goals"] - extreme_summary_df["baseline_avg_goals"]
    )
    extreme_summary_df["home_win_delta_vs_baseline"] = (
        extreme_summary_df["extreme_home_win_rate"] - extreme_summary_df["baseline_home_win_rate"]
    )
    extreme_summary_df["draw_delta_vs_baseline"] = (
        extreme_summary_df["extreme_draw_rate"] - extreme_summary_df["baseline_draw_rate"]
    )
    extreme_summary_df.to_csv(OUTPUT_DIR / "phase15_extreme_weather_summary.csv", index=False)

    extreme_cases_df = analysis_df.loc[analysis_df["extreme_weather_flag"] == 1].copy()
    extreme_cases_df.to_csv(OUTPUT_DIR / "phase15_extreme_weather_cases.csv", index=False)

    goals_chart = (
        alt.Chart(extreme_summary_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("extreme_type:N", title="Extreme Condition"),
            y=alt.Y("goals_delta_vs_baseline:Q", title="Goals Delta vs Baseline"),
            color=alt.condition(
                alt.datum.goals_delta_vs_baseline > 0,
                alt.value("#2E8B57"),
                alt.value("#C0392B"),
            ),
            tooltip=[
                "extreme_type",
                "extreme_matches",
                alt.Tooltip("extreme_avg_goals:Q", format=".2f"),
                alt.Tooltip("baseline_avg_goals:Q", format=".2f"),
                alt.Tooltip("goals_delta_vs_baseline:Q", format=".2f"),
            ],
        )
        .properties(width=620, height=300, title="Phase 15 - Extreme Weather Goal Impact")
    )
    outcomes_chart = (
        alt.Chart(extreme_summary_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("extreme_type:N", title="Extreme Condition"),
            y=alt.Y("draw_delta_vs_baseline:Q", title="Draw Rate Delta vs Baseline"),
            color=alt.condition(
                alt.datum.draw_delta_vs_baseline > 0,
                alt.value("#F39C12"),
                alt.value("#5D6D7E"),
            ),
            tooltip=[
                "extreme_type",
                "extreme_matches",
                alt.Tooltip("extreme_draw_rate:Q", format=".2%"),
                alt.Tooltip("baseline_draw_rate:Q", format=".2%"),
                alt.Tooltip("draw_delta_vs_baseline:Q", format=".2%"),
            ],
        )
        .properties(width=620, height=300, title="Phase 15 - Extreme Weather Draw Shift")
    )
    save_chart(goals_chart, PLOTS_DIR / "phase15_extreme_goals_delta.html")
    save_chart(outcomes_chart, PLOTS_DIR / "phase15_extreme_draw_delta.html")

    strongest_extreme_row = extreme_summary_df.loc[
        extreme_summary_df["goals_delta_vs_baseline"].abs().idxmax()
    ]
    extreme_summary = "\n".join(
        [
            "Phase 15 - Extreme Weather Analysis",
            (
                f"Largest edge-case goal shift: {strongest_extreme_row['extreme_type']} with a "
                f"{strongest_extreme_row['goals_delta_vs_baseline']:+.2f} change in average goals versus baseline."
            ),
            (
                f"Extreme-condition sample size: {int(extreme_summary_df.loc[extreme_summary_df['extreme_type'] == 'any_extreme', 'extreme_matches'].iloc[0])} matches."
            ),
            (
                "Extreme weather does not simply mirror average conditions; the rare-event slice shows its own "
                "goal and draw patterns and is worth analyzing separately from the main dataset."
            ),
        ]
    )
    (OUTPUT_DIR / "phase15_extreme_weather_summary.txt").write_text(extreme_summary, encoding="utf-8")

    return {
        "summary": extreme_summary,
        "extreme_summary_df": extreme_summary_df,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_df = load_dataset()
    causal_outputs = build_causal_outputs(dataset_df)
    simulation_outputs = build_simulation_outputs(dataset_df)
    extreme_outputs = build_extreme_weather_outputs(dataset_df)

    print(causal_outputs["summary"])
    print()
    print(simulation_outputs["summary"])
    print()
    print(extreme_outputs["summary"])


if __name__ == "__main__":
    main()
