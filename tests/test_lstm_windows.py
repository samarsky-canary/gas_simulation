from __future__ import annotations

import numpy as np
import pandas as pd

from src.lstm import LSTM_INPUT_COLUMNS, build_lstm_windows, export_lstm_windows
from src.simulator.config import ScenarioConfig
from src.simulator.exporters import export_run
from src.simulator.runner import run_scenario


def test_lstm_windows_shape_for_10_minute_step(tmp_path) -> None:
    cfg = ScenarioConfig(
        duration_days=3,
        step_minutes=10,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
        lstm_window_hours=24,
    )
    df, report = run_scenario(cfg)
    paths = export_run(cfg, df, report, tmp_path)
    dataset = pd.read_parquet(paths["dataset_parquet"])

    windows = build_lstm_windows(cfg, dataset)

    assert windows.sequence_length == 144
    assert windows.X_state.ndim == 3
    assert windows.X_state.shape[1:] == (144, len(LSTM_INPUT_COLUMNS))
    assert windows.y_state.shape == (windows.X_state.shape[0],)


def test_export_lstm_windows_creates_npz(tmp_path) -> None:
    cfg = ScenarioConfig(duration_days=3, step_minutes=30, p_missing=0, p_spike=0, p_stuck=0)
    df, report = run_scenario(cfg)
    paths = export_run(cfg, df, report, tmp_path)
    dataset = pd.read_parquet(paths["dataset_parquet"])
    windows = build_lstm_windows(cfg, dataset)

    export_paths = export_lstm_windows(cfg, windows, tmp_path)

    assert all(path.exists() and path.stat().st_size > 0 for path in export_paths.values())
    loaded = np.load(export_paths["lstm_state_npz"])
    assert loaded["X"].shape[1] == windows.sequence_length
