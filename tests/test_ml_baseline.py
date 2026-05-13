from __future__ import annotations

import pandas as pd

from src.features import build_features
from src.ml import ML_INPUT_COLUMNS, train_and_export_ml_baseline
from src.simulator.config import ScenarioConfig
from src.simulator.exporters import export_run
from src.simulator.runner import run_scenario


def test_ml_baseline_trains_and_exports(tmp_path) -> None:
    cfg = ScenarioConfig(
        duration_days=30,
        step_minutes=60,
        k_s_per_hour=0.003,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df, report = run_scenario(cfg)
    export_paths = export_run(cfg, df, report, tmp_path)
    dataset = pd.read_parquet(export_paths["dataset_parquet"])
    features = build_features(cfg, df)

    paths = train_and_export_ml_baseline(dataset, tmp_path, features)

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    predictions = pd.read_parquet(paths["ml_predictions_parquet"])
    assert {"state_pred", "RUL_pred_h", "split"}.issubset(predictions.columns)
