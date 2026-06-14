from __future__ import annotations

import pandas as pd

from src.simulator.config import ScenarioConfig
from src.simulator.runner import run_scenario
from src.visualization import build_plots
from src.visualization.plots import _aggregate_rul_source_plot_data


def test_build_plots_creates_png_files(tmp_path) -> None:
    cfg = ScenarioConfig(duration_days=1, step_minutes=30, p_missing=0, p_spike=0, p_stuck=0)
    df, _ = run_scenario(cfg)

    paths = build_plots(cfg, df, tmp_path)

    png_paths = [path for path in paths.values() if path.suffix == ".png"]
    assert set(paths) == {
        "q",
        "pressure",
        "delta_p",
        "state",
        "rul_comparison",
        "rul_source_periods",
        "data_confidence",
        "spike_rate",
        "description",
        "diagnostics",
    }
    assert len(png_paths) == 8
    assert all(path.exists() and path.stat().st_size > 0 for path in png_paths)
    assert paths["description"].exists()
    assert paths["diagnostics"].exists()


def test_rul_source_plot_data_is_aggregated_hourly() -> None:
    decisions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=24, freq="5min"),
            "RUL_ml_h": range(24),
            "RUL_analytic_h": range(24),
            "RUL_fused_h": range(24),
            "confidence_total": [0.8] * 24,
            "confidence_consistency": [0.9] * 24,
            "rul_source": ["ml_baseline"] * 13 + ["conservative_min"] * 11,
        }
    )

    aggregated = _aggregate_rul_source_plot_data(decisions)

    assert len(aggregated) == 2
    assert aggregated["rul_source"].tolist() == [
        "ml_baseline",
        "conservative_min",
    ]
