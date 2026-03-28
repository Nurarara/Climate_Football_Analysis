from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FEATURE_DATA_PATH = DATA_DIR / "processed" / "feature_dataset.csv"
MODELS_DIR = DATA_DIR / "processed" / "models"
ANALYSIS_DIR = DATA_DIR / "processed" / "analysis"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "match_outcome_model.joblib"
REGRESSION_MODEL_PATH = MODELS_DIR / "total_goals_model.joblib"

# Weather thresholds
COLD_TEMP = 10.0
HOT_TEMP = 25.0
WINDY_SPEED = 30.0
HUMIDITY_HIGH = 85.0
HEAVY_RAIN = 10.0
STRONG_WIND = 50.0
FREEZING_APPARENT = 0.0
HEATWAVE_APPARENT = 30.0

RESULT_LABELS = {"win": "Home Win", "draw": "Draw", "loss": "Away Win"}

FEATURE_COLUMNS = [
    "temperature", "apparent_temperature", "humidity", "rain", "wind",
    "heavy_rain_flag", "strong_wind_flag", "freezing_flag", "heatwave_flag",
    "storm_flag", "extreme_weather_flag", "rain_flag", "cold_flag", "hot_flag",
    "windy_flag", "humidity_flag", "apparent_cold_flag", "apparent_hot_flag",
    "apparent_temperature_gap", "home_advantage", "match_month",
    "season_year_start", "season_progress",
    "is_winter_match", "is_spring_match", "is_summer_match", "is_autumn_match",
    "home_team_points_avg_last_5", "home_team_goals_for_avg_last_5",
    "home_team_goals_against_avg_last_5", "home_team_goal_difference_avg_last_5",
    "away_team_points_avg_last_5", "away_team_goals_for_avg_last_5",
    "away_team_goals_against_avg_last_5", "away_team_goal_difference_avg_last_5",
    "home_team_strength_proxy", "away_team_strength_proxy",
    "strength_gap", "match_importance_proxy",
]

HOME_FORM_COLUMNS = [
    "home_team_points_avg_last_5", "home_team_goals_for_avg_last_5",
    "home_team_goals_against_avg_last_5", "home_team_goal_difference_avg_last_5",
    "home_team_strength_proxy",
]

AWAY_FORM_COLUMNS = [
    "away_team_points_avg_last_5", "away_team_goals_for_avg_last_5",
    "away_team_goals_against_avg_last_5", "away_team_goal_difference_avg_last_5",
    "away_team_strength_proxy",
]

MONTH_ORDER = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
TEMP_BAND_ORDER = ["<5C", "5-10C", "10-15C", "15-20C", "20-25C", "25C+"]
