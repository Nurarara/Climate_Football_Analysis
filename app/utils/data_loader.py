from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils.constants import (
    ANALYSIS_DIR,
    CLASSIFICATION_MODEL_PATH,
    FEATURE_DATA_PATH,
    MODELS_DIR,
    REGRESSION_MODEL_PATH,
    RESULT_LABELS,
)


@st.cache_data
def load_feature_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["result_label"] = df["match_result"].map(RESULT_LABELS)
    df["rain_label"] = df["rain_flag"].map({1: "Rain", 0: "No Rain"})
    df["temperature_band"] = pd.cut(
        df["temperature"],
        bins=[-10, 5, 10, 15, 20, 25, 40],
        labels=["<5C", "5-10C", "10-15C", "15-20C", "20-25C", "25C+"],
        include_lowest=True,
    )
    return df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_models():
    return joblib.load(CLASSIFICATION_MODEL_PATH), joblib.load(REGRESSION_MODEL_PATH)


def check_artifacts() -> list[Path]:
    required = [
        FEATURE_DATA_PATH,
        CLASSIFICATION_MODEL_PATH,
        REGRESSION_MODEL_PATH,
        ANALYSIS_DIR / "phase12_climate_sensitivity_index.csv",
        ANALYSIS_DIR / "phase15_extreme_weather_summary.csv",
        ANALYSIS_DIR / "phase17_climate_change_scenarios.csv",
        ANALYSIS_DIR / "phase19_permutation_importance.csv",
    ]
    return [p for p in required if not p.exists()]
