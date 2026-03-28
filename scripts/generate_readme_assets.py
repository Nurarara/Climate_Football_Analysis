import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
FEATURE_DATA_PATH = BASE_DIR / "data" / "processed" / "feature_dataset.csv"
EVALUATION_PATH = BASE_DIR / "data" / "processed" / "models" / "evaluation_metrics.json"
CLASSIFICATION_MODEL_PATH = BASE_DIR / "data" / "processed" / "models" / "match_outcome_model.joblib"
REGRESSION_MODEL_PATH = BASE_DIR / "data" / "processed" / "models" / "total_goals_model.joblib"

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

FORM_FEATURE_COLUMNS = [
    "home_team_points_avg_last_5",
    "home_team_goals_for_avg_last_5",
    "home_team_goals_against_avg_last_5",
    "home_team_goal_difference_avg_last_5",
    "away_team_points_avg_last_5",
    "away_team_goals_for_avg_last_5",
    "away_team_goals_against_avg_last_5",
    "away_team_goal_difference_avg_last_5",
    "match_month",
    "season_year_start",
    "season_progress",
    "is_winter_match",
    "is_spring_match",
    "is_summer_match",
    "is_autumn_match",
]

PIPELINE_STEPS = [
    "1. Football Ingestion",
    "2. Team Mapping",
    "3. Weather Ingestion",
    "4. Cleaning",
    "5. Joining",
    "6. Features",
    "7. Modeling",
    "8. Evaluation",
    "9. Dashboard",
    "10. Documentation",
]

BACKGROUND = "#F7F9FC"
PANEL = "#FFFFFF"
PRIMARY = "#1F4E79"
SECONDARY = "#2E86AB"
ACCENT = "#F39C12"
TEXT = "#1F2937"
MUTED = "#6B7280"
SUCCESS = "#2E8B57"


def load_font(size: int, bold: bool = False):
    font_candidates = []
    if bold:
        font_candidates.extend(["arialbd.ttf", "DejaVuSans-Bold.ttf"])
    font_candidates.extend(["arial.ttf", "DejaVuSans.ttf"])

    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: Optional[str] = None,
):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2 if outline else 1)


def add_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str, width: int):
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(16)
    draw.text((32, 24), title, font=title_font, fill=TEXT)
    draw.text((32, 66), subtitle, font=subtitle_font, fill=MUTED)
    draw.line((32, 98, width - 32, 98), fill="#D6DEE8", width=2)


def save_pipeline_diagram() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    box_width = 180
    box_height = 68
    gap_x = 30
    gap_y = 36
    columns = 2
    rows = 5
    width = 2 * 120 + columns * box_width + (columns - 1) * gap_x
    height = 120 + rows * box_height + (rows - 1) * gap_y + 70

    positions = []
    for index, step in enumerate(PIPELINE_STEPS):
        row = index % rows
        column = index // rows
        x = 120 + column * (box_width + gap_x)
        y = 120 + row * (box_height + gap_y)
        positions.append((step, x, y))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        f'<text x="40" y="54" font-size="30" font-family="Arial, sans-serif" font-weight="700" fill="{TEXT}">Pipeline Architecture</text>',
        f'<text x="40" y="82" font-size="15" font-family="Arial, sans-serif" fill="{MUTED}">End-to-end flow from data ingestion to interactive product layer</text>',
    ]

    for index, (step, x, y) in enumerate(positions):
        svg_parts.append(
            f'<rect x="{x}" y="{y}" rx="18" ry="18" width="{box_width}" height="{box_height}" fill="{PANEL}" stroke="{SECONDARY}" stroke-width="2"/>'
        )
        label_top, label_bottom = step.split(" ", 1)
        svg_parts.append(
            f'<text x="{x + 18}" y="{y + 28}" font-size="18" font-family="Arial, sans-serif" font-weight="700" fill="{PRIMARY}">{label_top}</text>'
        )
        svg_parts.append(
            f'<text x="{x + 18}" y="{y + 48}" font-size="15" font-family="Arial, sans-serif" fill="{TEXT}">{label_bottom}</text>'
        )

        if index < len(positions) - 1:
            next_step, next_x, next_y = positions[index + 1]
            if next_x == x:
                start_x = x + box_width / 2
                start_y = y + box_height
                end_x = next_x + box_width / 2
                end_y = next_y
                svg_parts.append(
                    f'<line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" stroke="{ACCENT}" stroke-width="3" marker-end="url(#arrow)"/>'
                )
            else:
                start_x = x + box_width
                start_y = y + box_height / 2
                end_x = next_x
                end_y = next_y + box_height / 2
                svg_parts.append(
                    f'<line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" stroke="{ACCENT}" stroke-width="3" marker-end="url(#arrow)"/>'
                )

    svg_parts.insert(
        1,
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#F39C12"/></marker></defs>',
    )
    svg_parts.append("</svg>")
    (ASSETS_DIR / "pipeline_diagram.svg").write_text("\n".join(svg_parts), encoding="utf-8")


def prepare_dataset(dataset_df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = dataset_df.copy()
    prepared_df["rain_label"] = prepared_df["rain_flag"].map({1: "Rain", 0: "No Rain"})
    prepared_df["cold_label"] = prepared_df["cold_flag"].map({1: "Cold", 0: "Mild/Warm"})
    prepared_df["temperature_band"] = pd.cut(
        prepared_df["temperature"],
        bins=[-10, 5, 10, 15, 20, 25, 35],
        labels=["<5C", "5-10C", "10-15C", "15-20C", "20-25C", "25C+"],
        include_lowest=True,
    )
    return prepared_df


def create_canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1400, 820), BACKGROUND)
    draw = ImageDraw.Draw(image)
    add_title(draw, title, subtitle, image.width)
    return image, draw


def save_overview_preview(dataset_df: pd.DataFrame) -> None:
    image, draw = create_canvas("Dashboard Preview: Overview", "Average goals vs weather conditions")

    draw_rounded_panel(draw, (32, 128, 676, 780), PANEL)
    draw_rounded_panel(draw, (708, 128, 1368, 780), PANEL)

    section_font = load_font(22, bold=True)
    label_font = load_font(16)
    value_font = load_font(18, bold=True)
    draw.text((60, 150), "Average goals by temperature band", font=section_font, fill=TEXT)
    draw.text((736, 150), "Rain vs no rain comparison", font=section_font, fill=TEXT)

    temperature_df = (
        dataset_df.dropna(subset=["temperature_band"])
        .groupby("temperature_band", observed=False)["total_goals"]
        .mean()
        .reset_index()
    )
    max_goals = max(float(temperature_df["total_goals"].max()), 1.0)

    left = 90
    bar_bottom = 690
    chart_height = 420
    bar_width = 68
    spacing = 20
    for idx, row in temperature_df.iterrows():
        x1 = left + idx * (bar_width + spacing)
        x2 = x1 + bar_width
        bar_height = int((float(row["total_goals"]) / max_goals) * chart_height)
        y1 = bar_bottom - bar_height
        draw.rounded_rectangle((x1, y1, x2, bar_bottom), radius=10, fill=SECONDARY)
        draw.text((x1, y1 - 28), f"{float(row['total_goals']):.2f}", font=label_font, fill=TEXT)
        draw.text((x1 - 8, 710), str(row["temperature_band"]), font=label_font, fill=MUTED)

    rain_df = dataset_df.groupby("rain_label")["total_goals"].mean().reset_index()
    max_rain_goals = max(float(rain_df["total_goals"].max()), 1.0)
    start_x = 790
    for idx, row in rain_df.iterrows():
        x1 = start_x + idx * 220
        x2 = x1 + 120
        bar_height = int((float(row["total_goals"]) / max_rain_goals) * 360)
        y1 = 690 - bar_height
        color = ACCENT if row["rain_label"] == "Rain" else SUCCESS
        draw.rounded_rectangle((x1, y1, x2, 690), radius=12, fill=color)
        draw.text((x1 + 18, y1 - 28), f"{float(row['total_goals']):.2f}", font=value_font, fill=TEXT)
        draw.text((x1 + 12, 710), row["rain_label"], font=section_font, fill=MUTED)

    summary_box = (958, 260, 1328, 470)
    draw_rounded_panel(draw, summary_box, "#EEF6FF", outline="#B4D3F0")
    draw.text((982, 286), "Snapshot", font=section_font, fill=PRIMARY)
    draw.text((982, 330), f"Matches: {len(dataset_df):,}", font=value_font, fill=TEXT)
    draw.text((982, 368), f"Rainy share: {dataset_df['rain_flag'].mean() * 100:.1f}%", font=value_font, fill=TEXT)
    draw.text((982, 406), f"Avg goals: {dataset_df['total_goals'].mean():.2f}", font=value_font, fill=TEXT)

    image.save(ASSETS_DIR / "overview_preview.png")


def save_team_preview(dataset_df: pd.DataFrame) -> None:
    image, draw = create_canvas("Dashboard Preview: Team Analysis", "Home-team performance in rain and cold")

    draw_rounded_panel(draw, (32, 128, 760, 780), PANEL)
    draw_rounded_panel(draw, (792, 128, 1368, 780), PANEL)

    section_font = load_font(22, bold=True)
    label_font = load_font(16)
    value_font = load_font(18, bold=True)

    team_patterns = (
        dataset_df.groupby(["home_team", "rain_label"])["goal_difference"]
        .mean()
        .unstack()
        .assign(rain_impact=lambda frame: frame["Rain"] - frame["No Rain"])
        .sort_values("rain_impact")
        .head(6)
        .reset_index()
    )
    cold_patterns = (
        dataset_df.groupby(["home_team", "cold_label"])["goal_difference"]
        .mean()
        .unstack()
        .assign(cold_impact=lambda frame: frame["Cold"] - frame["Mild/Warm"])
        .sort_values("cold_impact")
        .head(6)
        .reset_index()
    )

    draw.text((60, 150), "Teams hurt most in rain", font=section_font, fill=TEXT)
    draw.text((820, 150), "Teams hurt most in cold", font=section_font, fill=TEXT)

    y = 210
    for _, row in team_patterns.iterrows():
        draw_rounded_panel(draw, (60, y, 732, y + 72), "#F9FBFD", outline="#D7E3EF")
        draw.text((84, y + 18), str(row["home_team"]), font=value_font, fill=TEXT)
        draw.text((84, y + 42), f"No rain: {row['No Rain']:.2f}  |  Rain: {row['Rain']:.2f}", font=label_font, fill=MUTED)
        draw.text((560, y + 26), f"Impact {row['rain_impact']:.2f}", font=value_font, fill=ACCENT)
        y += 86

    y = 210
    for _, row in cold_patterns.iterrows():
        draw_rounded_panel(draw, (820, y, 1340, y + 72), "#F9FBFD", outline="#D7E3EF")
        draw.text((844, y + 18), str(row["home_team"]), font=value_font, fill=TEXT)
        draw.text((844, y + 42), f"Mild: {row['Mild/Warm']:.2f}  |  Cold: {row['Cold']:.2f}", font=label_font, fill=MUTED)
        draw.text((1180, y + 26), f"Impact {row['cold_impact']:.2f}", font=value_font, fill=SECONDARY)
        y += 86

    image.save(ASSETS_DIR / "team_analysis_preview.png")


def build_prediction_input(dataset_df: pd.DataFrame, temperature: float, rain: float, wind: float) -> pd.DataFrame:
    typical_form = dataset_df[FORM_FEATURE_COLUMNS].median().to_dict()
    humidity = float(dataset_df["humidity"].median())
    apparent_temperature = float(temperature - min(wind * 0.03, 3.0) - min(rain * 0.1, 1.5))
    row = {
        "temperature": temperature,
        "apparent_temperature": apparent_temperature,
        "humidity": humidity,
        "rain": rain,
        "wind": wind,
        "heavy_rain_flag": int(rain >= 10),
        "strong_wind_flag": int(wind >= 50),
        "freezing_flag": int(apparent_temperature <= 0),
        "heatwave_flag": int(apparent_temperature >= 30),
        "storm_flag": 0,
        "extreme_weather_flag": int(rain >= 10 or wind >= 50 or apparent_temperature <= 0 or apparent_temperature >= 30),
        "rain_flag": int(rain > 0),
        "cold_flag": int(temperature < 10.0),
        "hot_flag": int(temperature >= 25.0),
        "windy_flag": int(wind >= 30.0),
        "humidity_flag": int(humidity >= 85.0),
        "apparent_cold_flag": int(apparent_temperature < 10.0),
        "apparent_hot_flag": int(apparent_temperature >= 25.0),
        "apparent_temperature_gap": apparent_temperature - temperature,
        "home_advantage": 1,
    }
    row.update(typical_form)
    row["home_team_strength_proxy"] = (
        0.7 * row["home_team_points_avg_last_5"] + 0.3 * row["home_team_goals_for_avg_last_5"]
    )
    row["away_team_strength_proxy"] = (
        0.7 * row["away_team_points_avg_last_5"] + 0.3 * row["away_team_goals_for_avg_last_5"]
    )
    row["strength_gap"] = row["home_team_strength_proxy"] - row["away_team_strength_proxy"]
    row["match_importance_proxy"] = row["season_progress"] * (
        row["home_team_strength_proxy"] + row["away_team_strength_proxy"]
    ) / 2
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def save_prediction_preview(dataset_df: pd.DataFrame) -> None:
    image, draw = create_canvas("Dashboard Preview: Prediction Tool", "Weather-driven outcome and goals estimate")
    draw_rounded_panel(draw, (32, 128, 580, 780), PANEL)
    draw_rounded_panel(draw, (612, 128, 1368, 780), PANEL)

    section_font = load_font(22, bold=True)
    label_font = load_font(18)
    value_font = load_font(26, bold=True)

    temperature = 12.0
    rain = 3.5
    wind = 24.0

    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    prediction_df = build_prediction_input(dataset_df, temperature, rain, wind)
    predicted_result = classification_model.predict(prediction_df)[0]
    predicted_probabilities = classification_model.predict_proba(prediction_df)[0]
    predicted_goals = float(regression_model.predict(prediction_df)[0])

    draw.text((60, 150), "User Inputs", font=section_font, fill=TEXT)
    input_rows = [("Temperature", f"{temperature:.1f} C"), ("Rain", f"{rain:.1f} mm"), ("Wind", f"{wind:.1f} km/h")]
    y = 230
    for label, value in input_rows:
        draw_rounded_panel(draw, (60, y, 552, y + 92), "#F9FBFD", outline="#D7E3EF")
        draw.text((88, y + 22), label, font=label_font, fill=MUTED)
        draw.text((88, y + 50), value, font=value_font, fill=TEXT)
        y += 112

    draw.text((640, 150), "Predicted Outputs", font=section_font, fill=TEXT)
    draw_rounded_panel(draw, (640, 210, 990, 356), "#EEF8F0", outline="#B7DFC1")
    draw_rounded_panel(draw, (1012, 210, 1340, 356), "#EEF6FF", outline="#B4D3F0")
    result_label = {"win": "Home Win", "draw": "Draw", "loss": "Away Win"}[predicted_result]
    draw.text((668, 242), "Predicted Match Result", font=label_font, fill=MUTED)
    draw.text((668, 286), result_label, font=value_font, fill=SUCCESS)
    draw.text((1040, 242), "Predicted Total Goals", font=label_font, fill=MUTED)
    draw.text((1040, 286), f"{predicted_goals:.2f}", font=value_font, fill=PRIMARY)

    draw.text((640, 412), "Outcome probabilities", font=section_font, fill=TEXT)
    probs = list(zip(["Home Win", "Draw", "Away Win"], predicted_probabilities))
    base_x = 680
    bar_bottom = 710
    bar_width = 120
    spacing = 70
    for idx, (label, value) in enumerate(probs):
        x1 = base_x + idx * (bar_width + spacing)
        x2 = x1 + bar_width
        bar_height = int(value * 220)
        y1 = bar_bottom - bar_height
        color = [SECONDARY, ACCENT, SUCCESS][idx]
        draw.rounded_rectangle((x1, y1, x2, bar_bottom), radius=12, fill=color)
        draw.text((x1 + 24, y1 - 28), f"{value:.0%}", font=label_font, fill=TEXT)
        draw.text((x1 + 8, 730), label, font=label_font, fill=MUTED)

    image.save(ASSETS_DIR / "prediction_tool_preview.png")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_df = prepare_dataset(pd.read_csv(FEATURE_DATA_PATH))
    with open(EVALUATION_PATH, "r", encoding="utf-8") as file:
        json.load(file)

    save_pipeline_diagram()
    save_overview_preview(dataset_df)
    save_team_preview(dataset_df)
    save_prediction_preview(dataset_df)
    print("Saved assets to assets/")


if __name__ == "__main__":
    main()
