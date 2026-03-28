from pathlib import Path

import altair as alt
import pandas as pd


INPUT_PATH = Path("data/processed/feature_dataset.csv")
OUTPUT_DIR = Path("data/processed/analysis")
PLOTS_DIR = OUTPUT_DIR / "plots"

MIN_GROUP_SIZE = 5


def load_dataset() -> pd.DataFrame:
    dataset_df = pd.read_csv(INPUT_PATH)
    dataset_df["date"] = pd.to_datetime(dataset_df["date"], format="%Y-%m-%d")
    dataset_df["month_name"] = dataset_df["date"].dt.strftime("%b")
    dataset_df["season_label"] = dataset_df["season_year_start"].astype(str) + "-" + (
        dataset_df["season_year_start"] + 1
    ).astype(str)
    dataset_df["points"] = dataset_df["match_result"].map({"win": 3, "draw": 1, "loss": 0})
    dataset_df["temperature_band"] = pd.cut(
        dataset_df["temperature"],
        bins=[-10, 5, 10, 15, 20, 25, 35],
        labels=["<5C", "5-10C", "10-15C", "15-20C", "20-25C", "25C+"],
        include_lowest=True,
    )
    return dataset_df


def save_chart(chart: alt.Chart, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output_path)


def build_temporal_outputs(dataset_df: pd.DataFrame) -> dict[str, object]:
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    seasonal_order = ["winter", "spring", "summer", "autumn"]

    monthly_temperature_goals_df = (
        dataset_df.dropna(subset=["temperature_band"])
        .groupby(["month_name", "temperature_band"], observed=False)
        .agg(
            matches=("date", "count"),
            avg_total_goals=("total_goals", "mean"),
            avg_temperature=("temperature", "mean"),
        )
        .reset_index()
    )
    monthly_temperature_goals_df["month_name"] = pd.Categorical(
        monthly_temperature_goals_df["month_name"], categories=month_order, ordered=True
    )
    observed_monthly_temperature_goals_df = monthly_temperature_goals_df[
        monthly_temperature_goals_df["matches"] > 0
    ].copy()

    rain_season_df = (
        dataset_df[dataset_df["seasonal_indicator"].isin(["winter", "summer"])]
        .groupby(["seasonal_indicator", "rain_flag"])
        .agg(
            matches=("date", "count"),
            avg_total_goals=("total_goals", "mean"),
            avg_goal_difference=("goal_difference", "mean"),
            avg_points=("points", "mean"),
        )
        .reset_index()
    )
    rain_season_df["condition"] = rain_season_df["rain_flag"].map({0: "Dry", 1: "Rainy"})
    rain_season_df["seasonal_indicator"] = pd.Categorical(
        rain_season_df["seasonal_indicator"], categories=seasonal_order, ordered=True
    )

    seasonal_bias_df = (
        dataset_df.groupby(["season_label", "seasonal_indicator"])
        .agg(
            matches=("date", "count"),
            avg_total_goals=("total_goals", "mean"),
            avg_temperature=("temperature", "mean"),
            rain_share=("rain_flag", "mean"),
        )
        .reset_index()
    )
    seasonal_bias_df["seasonal_indicator"] = pd.Categorical(
        seasonal_bias_df["seasonal_indicator"], categories=seasonal_order, ordered=True
    )

    season_consistency_df = (
        dataset_df.groupby(["season_label", "rain_flag"])
        .agg(
            matches=("date", "count"),
            avg_total_goals=("total_goals", "mean"),
            avg_points=("points", "mean"),
        )
        .reset_index()
    )
    season_consistency_pivot_df = (
        season_consistency_df.pivot(index="season_label", columns="rain_flag", values="avg_total_goals")
        .rename(columns={0: "dry_avg_goals", 1: "rainy_avg_goals"})
        .reset_index()
    )
    season_consistency_pivot_df["rain_impact_on_goals"] = (
        season_consistency_pivot_df["rainy_avg_goals"] - season_consistency_pivot_df["dry_avg_goals"]
    )

    monthly_chart = (
        alt.Chart(observed_monthly_temperature_goals_df)
        .mark_rect()
        .encode(
            x=alt.X("month_name:N", sort=month_order, title="Month"),
            y=alt.Y("temperature_band:N", title="Temperature Band"),
            color=alt.Color("avg_total_goals:Q", title="Avg Goals"),
            tooltip=[
                "month_name",
                "temperature_band",
                "matches",
                alt.Tooltip("avg_total_goals:Q", format=".2f"),
                alt.Tooltip("avg_temperature:Q", format=".1f"),
            ],
        )
        .properties(width=600, height=260, title="Goals vs Temperature Across Months")
    )

    rain_comparison_chart = (
        alt.Chart(rain_season_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("condition:N", title="Match Condition"),
            y=alt.Y("avg_total_goals:Q", title="Average Total Goals"),
            color=alt.Color("condition:N", legend=None),
            column=alt.Column("seasonal_indicator:N", sort=["winter", "summer"], title="Season"),
            tooltip=[
                "seasonal_indicator",
                "condition",
                "matches",
                alt.Tooltip("avg_total_goals:Q", format=".2f"),
                alt.Tooltip("avg_points:Q", format=".2f"),
            ],
        )
        .properties(width=260, height=260, title="Rain Impact in Winter vs Summer")
    )

    consistency_chart = (
        alt.Chart(season_consistency_pivot_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("season_label:N", title="Season"),
            y=alt.Y("rain_impact_on_goals:Q", title="Rain Impact on Avg Goals"),
            color=alt.condition(
                alt.datum.rain_impact_on_goals > 0,
                alt.value("#2E8B57"),
                alt.value("#C0392B"),
            ),
            tooltip=[
                "season_label",
                alt.Tooltip("dry_avg_goals:Q", format=".2f"),
                alt.Tooltip("rainy_avg_goals:Q", format=".2f"),
                alt.Tooltip("rain_impact_on_goals:Q", format=".2f"),
            ],
        )
        .properties(width=600, height=260, title="Rain Impact Consistency Across Seasons")
    )

    observed_monthly_temperature_goals_df.to_csv(
        OUTPUT_DIR / "phase11_monthly_temperature_goals.csv", index=False
    )
    rain_season_df.to_csv(OUTPUT_DIR / "phase11_rain_winter_vs_summer.csv", index=False)
    seasonal_bias_df.to_csv(OUTPUT_DIR / "phase11_seasonal_bias.csv", index=False)
    season_consistency_pivot_df.to_csv(OUTPUT_DIR / "phase11_seasonal_consistency.csv", index=False)

    save_chart(monthly_chart, PLOTS_DIR / "phase11_monthly_temperature_goals.html")
    save_chart(rain_comparison_chart, PLOTS_DIR / "phase11_rain_winter_vs_summer.html")
    save_chart(consistency_chart, PLOTS_DIR / "phase11_seasonal_consistency.html")

    top_positive_month = observed_monthly_temperature_goals_df.loc[
        observed_monthly_temperature_goals_df["avg_total_goals"].idxmax()
    ]
    top_negative_month = observed_monthly_temperature_goals_df.loc[
        observed_monthly_temperature_goals_df["avg_total_goals"].idxmin()
    ]
    winter_rain_impact = (
        rain_season_df.loc[
            (rain_season_df["seasonal_indicator"] == "winter") & (rain_season_df["condition"] == "Rainy"),
            "avg_total_goals",
        ].iloc[0]
        - rain_season_df.loc[
            (rain_season_df["seasonal_indicator"] == "winter") & (rain_season_df["condition"] == "Dry"),
            "avg_total_goals",
        ].iloc[0]
    )
    summer_rain_impact = (
        rain_season_df.loc[
            (rain_season_df["seasonal_indicator"] == "summer") & (rain_season_df["condition"] == "Rainy"),
            "avg_total_goals",
        ].iloc[0]
        - rain_season_df.loc[
            (rain_season_df["seasonal_indicator"] == "summer") & (rain_season_df["condition"] == "Dry"),
            "avg_total_goals",
        ].iloc[0]
    )
    positive_seasons = int((season_consistency_pivot_df["rain_impact_on_goals"] > 0).sum())
    negative_seasons = int((season_consistency_pivot_df["rain_impact_on_goals"] <= 0).sum())

    temporal_summary = "\n".join(
        [
            "Phase 11 - Temporal & Seasonal Analysis",
            (
                f"Highest-scoring month/temperature bucket: {top_positive_month['month_name']} at "
                f"{top_positive_month['temperature_band']} with {top_positive_month['avg_total_goals']:.2f} goals."
            ),
            (
                f"Lowest-scoring month/temperature bucket: {top_negative_month['month_name']} at "
                f"{top_negative_month['temperature_band']} with {top_negative_month['avg_total_goals']:.2f} goals."
            ),
            f"Winter rain impact on goals: {winter_rain_impact:+.2f} compared with dry winter matches.",
            f"Summer rain impact on goals: {summer_rain_impact:+.2f} compared with dry summer matches.",
            (
                f"Seasonal consistency check: rain increased average goals in {positive_seasons} seasons and "
                f"reduced or matched them in {negative_seasons} seasons."
            ),
            (
                "Weather effects are not perfectly consistent over time, which suggests a seasonal bias and "
                "context dependence rather than a single stable weather rule."
            ),
        ]
    )
    (OUTPUT_DIR / "phase11_temporal_summary.txt").write_text(temporal_summary, encoding="utf-8")

    return {
        "monthly_temperature_goals_df": observed_monthly_temperature_goals_df,
        "rain_season_df": rain_season_df,
        "season_consistency_pivot_df": season_consistency_pivot_df,
        "summary": temporal_summary,
    }


def build_team_climate_sensitivity_outputs(dataset_df: pd.DataFrame) -> dict[str, object]:
    home_df = dataset_df.copy()
    strength_baseline_df = (
        home_df.groupby("home_team")
        .agg(mean_strength_proxy=("home_team_strength_proxy", lambda values: values[values > 0].mean()))
        .reset_index()
    )
    fallback_strength_df = (
        home_df.groupby("home_team")
        .agg(fallback_strength_proxy=("home_team_strength_proxy", "mean"))
        .reset_index()
    )
    strength_baseline_df = strength_baseline_df.merge(fallback_strength_df, on="home_team", how="left")
    strength_baseline_df["mean_strength_proxy"] = strength_baseline_df["mean_strength_proxy"].fillna(
        strength_baseline_df["fallback_strength_proxy"]
    )
    strength_baseline_df["mean_strength_proxy"] = strength_baseline_df["mean_strength_proxy"].clip(lower=1.0)
    strength_baseline_df = strength_baseline_df.drop(columns=["fallback_strength_proxy"])

    team_condition_df = (
        home_df.groupby(["home_team", "rain_flag", "cold_flag"])
        .agg(
            matches=("date", "count"),
            avg_points=("points", "mean"),
            avg_goal_difference=("goal_difference", "mean"),
            avg_total_goals=("total_goals", "mean"),
            avg_strength_proxy=("home_team_strength_proxy", "mean"),
        )
        .reset_index()
    )

    rainy_vs_dry_df = (
        home_df.groupby(["home_team", "rain_flag"])
        .agg(
            matches=("date", "count"),
            avg_points=("points", "mean"),
            avg_goal_difference=("goal_difference", "mean"),
        )
        .reset_index()
        .pivot(index="home_team", columns="rain_flag", values=["matches", "avg_points", "avg_goal_difference"])
    )
    rainy_vs_dry_df = rainy_vs_dry_df.reindex(
        columns=pd.MultiIndex.from_product(
            [["matches", "avg_points", "avg_goal_difference"], [0, 1]]
        )
    )
    rainy_vs_dry_df.columns = [
        "dry_matches" if column == ("matches", 0) else
        "rainy_matches" if column == ("matches", 1) else
        "dry_points" if column == ("avg_points", 0) else
        "rainy_points" if column == ("avg_points", 1) else
        "dry_goal_difference" if column == ("avg_goal_difference", 0) else
        "rainy_goal_difference"
        for column in rainy_vs_dry_df.columns
    ]
    rainy_vs_dry_df = rainy_vs_dry_df.reset_index()

    cold_vs_normal_df = (
        home_df.groupby(["home_team", "cold_flag"])
        .agg(
            matches=("date", "count"),
            avg_points=("points", "mean"),
            avg_goal_difference=("goal_difference", "mean"),
        )
        .reset_index()
        .pivot(index="home_team", columns="cold_flag", values=["matches", "avg_points", "avg_goal_difference"])
    )
    cold_vs_normal_df = cold_vs_normal_df.reindex(
        columns=pd.MultiIndex.from_product(
            [["matches", "avg_points", "avg_goal_difference"], [0, 1]]
        )
    )
    cold_vs_normal_df.columns = [
        "normal_matches" if column == ("matches", 0) else
        "cold_matches" if column == ("matches", 1) else
        "normal_points" if column == ("avg_points", 0) else
        "cold_points" if column == ("avg_points", 1) else
        "normal_goal_difference" if column == ("avg_goal_difference", 0) else
        "cold_goal_difference"
        for column in cold_vs_normal_df.columns
    ]
    cold_vs_normal_df = cold_vs_normal_df.reset_index()

    climate_df = rainy_vs_dry_df.merge(cold_vs_normal_df, on="home_team", how="outer").merge(
        strength_baseline_df, on="home_team", how="left"
    )
    numeric_columns = [column for column in climate_df.columns if column != "home_team"]
    climate_df[numeric_columns] = climate_df[numeric_columns].fillna(0)

    climate_df["rain_points_delta"] = climate_df["rainy_points"] - climate_df["dry_points"]
    climate_df["rain_goal_difference_delta"] = (
        climate_df["rainy_goal_difference"] - climate_df["dry_goal_difference"]
    )
    climate_df["cold_points_delta"] = climate_df["cold_points"] - climate_df["normal_points"]
    climate_df["cold_goal_difference_delta"] = (
        climate_df["cold_goal_difference"] - climate_df["normal_goal_difference"]
    )

    climate_df["rain_sample_balance"] = (
        climate_df[["rainy_matches", "dry_matches"]].min(axis=1)
        / climate_df[["rainy_matches", "dry_matches"]].max(axis=1).replace(0, 1)
    )
    climate_df["cold_sample_balance"] = (
        climate_df[["cold_matches", "normal_matches"]].min(axis=1)
        / climate_df[["cold_matches", "normal_matches"]].max(axis=1).replace(0, 1)
    )

    climate_df["strength_normalizer"] = climate_df["mean_strength_proxy"].clip(lower=1.0)
    climate_df["rain_sensitivity_component"] = (
        -(0.7 * climate_df["rain_points_delta"] + 0.3 * climate_df["rain_goal_difference_delta"])
        / climate_df["strength_normalizer"]
        * climate_df["rain_sample_balance"]
    )
    climate_df["cold_sensitivity_component"] = (
        -(0.7 * climate_df["cold_points_delta"] + 0.3 * climate_df["cold_goal_difference_delta"])
        / climate_df["strength_normalizer"]
        * climate_df["cold_sample_balance"]
    )
    climate_df["climate_sensitivity_index"] = (
        100 * (0.5 * climate_df["rain_sensitivity_component"] + 0.5 * climate_df["cold_sensitivity_component"])
    )

    climate_df["climate_sensitivity_label"] = "neutral"
    climate_df.loc[climate_df["climate_sensitivity_index"] >= 5, "climate_sensitivity_label"] = "highly affected"
    climate_df.loc[
        (climate_df["climate_sensitivity_index"] >= 2) & (climate_df["climate_sensitivity_index"] < 5),
        "climate_sensitivity_label",
    ] = "moderately affected"
    climate_df.loc[climate_df["climate_sensitivity_index"] <= -2, "climate_sensitivity_label"] = "weather-resilient"

    climate_df = climate_df.sort_values("climate_sensitivity_index", ascending=False).reset_index(drop=True)
    climate_df.to_csv(OUTPUT_DIR / "phase12_climate_sensitivity_index.csv", index=False)
    team_condition_df.to_csv(OUTPUT_DIR / "phase12_team_condition_breakdown.csv", index=False)

    ranking_chart = (
        alt.Chart(climate_df.head(10))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("climate_sensitivity_index:Q", title="Climate Sensitivity Index"),
            y=alt.Y("home_team:N", sort="-x", title="Team"),
            color=alt.Color("climate_sensitivity_label:N", title="Sensitivity"),
            tooltip=[
                "home_team",
                alt.Tooltip("climate_sensitivity_index:Q", format=".2f"),
                alt.Tooltip("rain_sensitivity_component:Q", format=".2f"),
                alt.Tooltip("cold_sensitivity_component:Q", format=".2f"),
            ],
        )
        .properties(width=620, height=320, title="Most Climate-Sensitive Teams")
    )

    comparison_chart_df = climate_df.melt(
        id_vars=["home_team"],
        value_vars=["rain_sensitivity_component", "cold_sensitivity_component"],
        var_name="component",
        value_name="value",
    )
    comparison_chart = (
        alt.Chart(comparison_chart_df.head(20))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("value:Q", title="Normalized Impact"),
            y=alt.Y("home_team:N", sort="-x", title="Team"),
            color=alt.Color("component:N", title="Weather Driver"),
            tooltip=["home_team", "component", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(width=620, height=420, title="Rain vs Cold Sensitivity Components")
    )

    save_chart(ranking_chart, PLOTS_DIR / "phase12_climate_sensitivity_rankings.html")
    save_chart(comparison_chart, PLOTS_DIR / "phase12_climate_sensitivity_components.html")

    most_affected = climate_df.iloc[0]
    least_affected = climate_df.iloc[-1]
    weather_resilient_count = int((climate_df["climate_sensitivity_index"] < 0).sum())
    highly_affected_count = int((climate_df["climate_sensitivity_index"] >= 5).sum())

    sensitivity_summary = "\n".join(
        [
            "Phase 12 - Team-Specific Climate Sensitivity",
            (
                f"Most affected team: {most_affected['home_team']} with a climate sensitivity index of "
                f"{most_affected['climate_sensitivity_index']:.2f}."
            ),
            (
                f"Least affected team: {least_affected['home_team']} with a climate sensitivity index of "
                f"{least_affected['climate_sensitivity_index']:.2f}."
            ),
            (
                f"Highly affected teams (index >= 5): {highly_affected_count}. "
                f"Weather-resilient teams (index < 0): {weather_resilient_count}."
            ),
            (
                "The index is normalized by each team's rolling strength proxy so stronger teams are not "
                "automatically treated as climate-resilient."
            ),
        ]
    )
    (OUTPUT_DIR / "phase12_climate_sensitivity_summary.txt").write_text(
        sensitivity_summary, encoding="utf-8"
    )

    return {
        "climate_df": climate_df,
        "team_condition_df": team_condition_df,
        "summary": sensitivity_summary,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_df = load_dataset()
    temporal_outputs = build_temporal_outputs(dataset_df)
    climate_outputs = build_team_climate_sensitivity_outputs(dataset_df)

    print("Saved Phase 11 outputs:")
    print(f"  {OUTPUT_DIR / 'phase11_monthly_temperature_goals.csv'}")
    print(f"  {OUTPUT_DIR / 'phase11_rain_winter_vs_summer.csv'}")
    print(f"  {OUTPUT_DIR / 'phase11_seasonal_bias.csv'}")
    print(f"  {OUTPUT_DIR / 'phase11_seasonal_consistency.csv'}")
    print(f"  {OUTPUT_DIR / 'phase11_temporal_summary.txt'}")
    print("Saved Phase 12 outputs:")
    print(f"  {OUTPUT_DIR / 'phase12_climate_sensitivity_index.csv'}")
    print(f"  {OUTPUT_DIR / 'phase12_team_condition_breakdown.csv'}")
    print(f"  {OUTPUT_DIR / 'phase12_climate_sensitivity_summary.txt'}")
    print("Saved plots:")
    print(f"  {PLOTS_DIR / 'phase11_monthly_temperature_goals.html'}")
    print(f"  {PLOTS_DIR / 'phase11_rain_winter_vs_summer.html'}")
    print(f"  {PLOTS_DIR / 'phase11_seasonal_consistency.html'}")
    print(f"  {PLOTS_DIR / 'phase12_climate_sensitivity_rankings.html'}")
    print(f"  {PLOTS_DIR / 'phase12_climate_sensitivity_components.html'}")
    print()
    print(temporal_outputs["summary"])
    print()
    print(climate_outputs["summary"])


if __name__ == "__main__":
    main()
