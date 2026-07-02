from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.features import build_features
from src.hybrid import build_hybrid_decisions
from src.ml import predict_ml_baseline, train_and_export_ml_baseline
from src.simulator.config import ScenarioConfig
from src.simulator.exporters import build_canonical_dataset
from src.simulator.runner import run_scenario
from src.visualization import build_interactive_plot
from src.visualization.interactive import (
    INTERACTIVE_PLOT_BUILDERS,
    _quality_issue_summary,
)
from src.visualization.plots import _aggregate_rul_source_plot_data


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


def test_all_interactive_plots_are_plotly_figures(tmp_path) -> None:
    cfg = ScenarioConfig(
        duration_days=30,
        step_minutes=60,
        k_s_per_hour=0.003,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df, _ = run_scenario(cfg)
    dataset = build_canonical_dataset(cfg, df)
    features = build_features(cfg, df)
    paths = train_and_export_ml_baseline(dataset, tmp_path, features)
    predictions = predict_ml_baseline(dataset, paths["rul_model"], features)
    decisions = build_hybrid_decisions(cfg, dataset, features, predictions)

    figures = {
        key: build_interactive_plot(key, cfg, df, decisions)
        for key in INTERACTIVE_PLOT_BUILDERS
    }

    assert set(figures) == {
        "pressure",
        "delta_p",
        "state",
        "rul_comparison",
        "rul_source_periods",
        "data_confidence",
    }
    assert all(isinstance(figure, go.Figure) for figure in figures.values())
    assert all(len(figure.data) > 0 for figure in figures.values())
    source_figure = figures["rul_source_periods"]
    assert source_figure.layout.xaxis2.matches == "x"
    assert source_figure.layout.xaxis3.matches == "x"
    assert source_figure.layout.xaxis4.matches is None
    rul_trace_names = {trace.name for trace in figures["rul_comparison"].data}
    assert "Истинный остаточный ресурс" in rul_trace_names


def test_quality_issue_summary_reports_total_percentages() -> None:
    telemetry = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="30min"),
            "p_in_mpa": [0.6, None, 0.6, 0.6],
            "p_out_mpa": [0.5, 0.5, 0.5, 0.5],
            "delta_p_kpa": [1.0, None, 1.0, 1.0],
            "q_m3h": [600.0] * 4,
            "t_c": [15.0] * 4,
            "quality_code": ["good", "missing", "good", "good"],
            "spike_event": [False, False, True, False],
        }
    )

    hourly, missing_percent, spike_percent = _quality_issue_summary(telemetry)

    assert missing_percent == 25.0
    assert spike_percent == 25.0
    assert hourly["missing_percent"].tolist() == [50.0, 0.0]
    assert hourly["spike_percent"].tolist() == [0.0, 50.0]
