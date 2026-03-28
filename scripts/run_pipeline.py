from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "pipeline_manifest.json"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script_path: str
    outputs: tuple[str, ...]


PIPELINE_STEPS = [
    PipelineStep("phase01_ingest_matches", "scripts/ingest_football_data.py", ("data/raw/matches.csv",)),
    PipelineStep("phase02_team_mapping", "scripts/create_team_mapping.py", ("data/raw/team_mapping.csv",)),
    PipelineStep("phase03_ingest_weather", "scripts/ingest_weather_data.py", ("data/raw/weather.csv",)),
    PipelineStep(
        "phase04_clean_standardize",
        "scripts/clean_standardize_data.py",
        ("data/processed/matches_clean.csv", "data/processed/weather_clean.csv"),
    ),
    PipelineStep("phase05_join_data", "scripts/join_data.py", ("data/processed/final_dataset.csv",)),
    PipelineStep("phase06_engineer_features", "scripts/engineer_features.py", ("data/processed/feature_dataset.csv",)),
    PipelineStep(
        "phase07_train_models",
        "scripts/train_models.py",
        (
            "data/processed/models/match_outcome_model.joblib",
            "data/processed/models/total_goals_model.joblib",
            "data/processed/models/training_metadata.json",
        ),
    ),
    PipelineStep(
        "phase08_evaluate_models",
        "scripts/evaluate_models.py",
        (
            "data/processed/models/evaluation_metrics.json",
            "data/processed/models/insight_summary.txt",
        ),
    ),
    PipelineStep(
        "phase11_12_temporal_climate",
        "scripts/analyze_temporal_and_climate_sensitivity.py",
        (
            "data/processed/analysis/phase11_temporal_summary.txt",
            "data/processed/analysis/phase12_climate_sensitivity_summary.txt",
        ),
    ),
    PipelineStep(
        "phase13_15_causal_simulation_extreme",
        "scripts/analyze_causal_simulation_extreme.py",
        (
            "data/processed/analysis/phase13_causal_summary.txt",
            "data/processed/analysis/phase14_simulation_summary.txt",
            "data/processed/analysis/phase15_extreme_weather_summary.txt",
        ),
    ),
    PipelineStep(
        "phase16_19_player_future_geo_interpretability",
        "scripts/analyze_player_future_geo_interpretability.py",
        (
            "data/processed/analysis/phase16_player_climate_summary.txt",
            "data/processed/analysis/phase17_climate_change_summary.txt",
            "data/processed/analysis/phase18_geographical_summary.txt",
            "data/processed/analysis/phase19_interpretability_summary.txt",
        ),
    ),
    PipelineStep(
        "phase10_assets",
        "scripts/generate_readme_assets.py",
        (
            "assets/pipeline_diagram.svg",
            "assets/overview_preview.png",
            "assets/team_analysis_preview.png",
            "assets/prediction_tool_preview.png",
        ),
    ),
]


def configure_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Premier League climate pipeline end to end.")
    parser.add_argument("--from-step", choices=[step.name for step in PIPELINE_STEPS], default=PIPELINE_STEPS[0].name)
    parser.add_argument("--to-step", choices=[step.name for step in PIPELINE_STEPS], default=PIPELINE_STEPS[-1].name)
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child scripts.")
    return parser.parse_args()


def validate_outputs(step: PipelineStep) -> list[str]:
    missing_outputs = []
    for relative_path in step.outputs:
        full_path = PROJECT_ROOT / relative_path
        if not full_path.exists() or full_path.stat().st_size == 0:
            missing_outputs.append(relative_path)
    return missing_outputs


def run_step(step: PipelineStep, python_executable: str) -> dict[str, object]:
    command = [python_executable, step.script_path]
    logging.info("Starting %s", step.name)
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    duration_seconds = round(time.perf_counter() - started_at, 2)

    if completed.stdout.strip():
        logging.info("[%s stdout]\n%s", step.name, completed.stdout.strip())
    if completed.stderr.strip():
        logging.warning("[%s stderr]\n%s", step.name, completed.stderr.strip())

    if completed.returncode != 0:
        raise RuntimeError(f"{step.name} failed with exit code {completed.returncode}")

    missing_outputs = validate_outputs(step)
    if missing_outputs:
        raise RuntimeError(f"{step.name} finished but missing outputs: {missing_outputs}")

    logging.info("Completed %s in %.2fs", step.name, duration_seconds)
    return {
        "name": step.name,
        "script_path": step.script_path,
        "duration_seconds": duration_seconds,
        "outputs": list(step.outputs),
    }


def write_manifest(log_path: Path, steps_run: list[dict[str, object]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "steps_run": steps_run,
        "log_path": str(log_path),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_path = configure_logging()

    step_names = [step.name for step in PIPELINE_STEPS]
    start_index = step_names.index(args.from_step)
    end_index = step_names.index(args.to_step)
    if start_index > end_index:
        raise ValueError("--from-step must come before or equal to --to-step")

    selected_steps = PIPELINE_STEPS[start_index : end_index + 1]
    logging.info("Pipeline run started with %d step(s)", len(selected_steps))
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Python executable: %s", args.python)

    completed_steps: list[dict[str, object]] = []
    overall_started_at = time.perf_counter()
    try:
        for step in selected_steps:
            completed_steps.append(run_step(step, args.python))
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        write_manifest(log_path, completed_steps)
        raise

    total_duration = round(time.perf_counter() - overall_started_at, 2)
    write_manifest(log_path, completed_steps)
    logging.info("Pipeline finished successfully in %.2fs", total_duration)
    logging.info("Manifest written to %s", MANIFEST_PATH)
    logging.info("Log written to %s", log_path)


if __name__ == "__main__":
    main()
