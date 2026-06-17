from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.simulator.config import ScenarioConfig
from src.visualization.ml_quality import (
    build_current_run_quality_figure,
    build_training_scenario_figure,
    current_run_quality,
    training_quality_summary,
)


def _metrics() -> dict[str, object]:
    return {
        "rul_regressor": {
            "mae_h": 100.0,
            "rmse_h": 150.0,
            "r2": 0.8,
            "by_scenario": [
                {"scenario": "slow_clogging", "mae_h": 80.0},
                {"scenario": "missing_data", "mae_h": 60.0},
            ],
            "analytic_baseline": {
                "mae_h": 125.0,
                "rmse_h": 180.0,
                "r2": 0.7,
                "by_scenario": [
                    {"scenario": "slow_clogging", "mae_h": 110.0},
                    {"scenario": "missing_data", "mae_h": 70.0},
                ],
            },
        }
    }


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h"),
            "RUL_oracle_h": [100.0, 90.0, 80.0, 70.0],
            "RUL_ml_h": [110.0, 85.0, 75.0, 72.0],
            "RUL_analytic_h": [120.0, 100.0, 90.0, 80.0],
            "RUL_fused_h": [105.0, 92.0, 78.0, 71.0],
        }
    )


def test_training_quality_summary_compares_ml_with_analytic() -> None:
    summary = training_quality_summary(_metrics())

    assert summary["ml_mae_h"] == 100.0
    assert summary["analytic_mae_h"] == 125.0
    assert summary["improvement_percent"] == 20.0
    assert summary["ml_r2"] == 0.8


def test_current_run_quality_uses_oracle_only_for_demo_metrics() -> None:
    summary = current_run_quality(_decisions())

    assert summary["ml_mae_h"] == 5.5
    assert summary["analytic_mae_h"] == 12.5
    assert summary["hybrid_mae_h"] == 2.5
    assert summary["current_ml_error_h"] == 2.0
    assert summary["current_analytic_error_h"] == 10.0
    assert summary["current_hybrid_error_h"] == 1.0


def test_ml_quality_figures_are_plotly_figures() -> None:
    scenario_figure = build_training_scenario_figure(_metrics())
    current_figure = build_current_run_quality_figure(
        ScenarioConfig(), _decisions()
    )

    assert isinstance(scenario_figure, go.Figure)
    assert isinstance(current_figure, go.Figure)
    assert len(scenario_figure.data) == 2
    assert list(scenario_figure.data[0].x) == ["Пропуски данных", "Медленное засорение"]
    assert len(current_figure.data) == 7
