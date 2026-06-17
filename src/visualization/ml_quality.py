from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.simulator.config import ScenarioConfig


SCENARIO_LABELS = {
    "normal": "Нормальный режим",
    "slow_clogging": "Медленное засорение",
    "rapid_clogging": "Быстрое засорение",
    "flow_spikes": "Скачки расхода",
    "sensor_bias": "Дрейф датчика",
    "sensor_stuck": "Залипание датчика",
    "missing_data": "Пропуски данных",
}


def training_quality_summary(metrics: dict[str, object]) -> dict[str, float | None]:
    """Возвращает основные test-метрики ML и аналитического baseline."""
    rul = metrics.get("rul_regressor", {})
    analytic = rul.get("analytic_baseline", {})
    ml_mae = _number(rul.get("mae_h"))
    analytic_mae = _number(analytic.get("mae_h"))
    improvement_percent = None
    if ml_mae is not None and analytic_mae not in (None, 0.0):
        improvement_percent = (analytic_mae - ml_mae) / analytic_mae * 100.0
    return {
        "ml_mae_h": ml_mae,
        "analytic_mae_h": analytic_mae,
        "ml_rmse_h": _number(rul.get("rmse_h")),
        "analytic_rmse_h": _number(analytic.get("rmse_h")),
        "ml_r2": _number(rul.get("r2")),
        "improvement_percent": improvement_percent,
    }


def current_run_quality(decisions: pd.DataFrame) -> dict[str, float | None]:
    """Считает демонстрационные ошибки текущего симуляционного прогона."""
    valid = decisions.dropna(
        subset=["RUL_oracle_h", "RUL_ml_h", "RUL_analytic_h", "RUL_fused_h"]
    ).sort_values("timestamp")
    if valid.empty:
        return {
            "ml_mae_h": None,
            "analytic_mae_h": None,
            "hybrid_mae_h": None,
            "current_ml_error_h": None,
            "current_analytic_error_h": None,
            "current_hybrid_error_h": None,
        }
    ml_error = (valid["RUL_ml_h"] - valid["RUL_oracle_h"]).abs()
    analytic_error = (valid["RUL_analytic_h"] - valid["RUL_oracle_h"]).abs()
    hybrid_error = (valid["RUL_fused_h"] - valid["RUL_oracle_h"]).abs()
    return {
        "ml_mae_h": float(ml_error.mean()),
        "analytic_mae_h": float(analytic_error.mean()),
        "hybrid_mae_h": float(hybrid_error.mean()),
        "current_ml_error_h": float(ml_error.iloc[-1]),
        "current_analytic_error_h": float(analytic_error.iloc[-1]),
        "current_hybrid_error_h": float(hybrid_error.iloc[-1]),
    }


def build_training_scenario_figure(metrics: dict[str, object]) -> go.Figure:
    """Сравнивает test-MAE ML и аналитики по отложенным сценариям."""
    rul = metrics.get("rul_regressor", {})
    ml_rows = {
        str(row["scenario"]): row
        for row in rul.get("by_scenario", [])
        if row.get("scenario") is not None
    }
    analytic_rows = {
        str(row["scenario"]): row
        for row in rul.get("analytic_baseline", {}).get("by_scenario", [])
        if row.get("scenario") is not None
    }
    scenarios = sorted(set(ml_rows) | set(analytic_rows))
    labels = [SCENARIO_LABELS.get(scenario, scenario) for scenario in scenarios]
    figure = go.Figure()
    figure.add_bar(
        x=labels,
        y=[ml_rows.get(scenario, {}).get("mae_h") for scenario in scenarios],
        name="ML",
        marker_color="#1f77b4",
    )
    figure.add_bar(
        x=labels,
        y=[analytic_rows.get(scenario, {}).get("mae_h") for scenario in scenarios],
        name="Аналитическая оценка",
        marker_color="#8c564b",
    )
    figure.update_layout(
        title="MAE на test-прогонах по сценариям",
        barmode="group",
        template="plotly_white",
        height=480,
        yaxis_title="MAE, ч",
        xaxis_title="Сценарий",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 60, "r": 30, "t": 90, "b": 90},
    )
    return figure


def build_current_run_quality_figure(
    cfg: ScenarioConfig, decisions: pd.DataFrame
) -> go.Figure:
    """Показывает RUL и абсолютные ошибки текущего симуляционного прогона."""
    data = decisions.sort_values("timestamp").copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    hourly = (
        data.set_index("timestamp")[
            ["RUL_oracle_h", "RUL_ml_h", "RUL_analytic_h", "RUL_fused_h"]
        ]
        .resample("1h")
        .mean()
        .dropna(how="all")
        .reset_index()
    )
    hourly["ml_error_h"] = (
        hourly["RUL_ml_h"] - hourly["RUL_oracle_h"]
    ).abs()
    hourly["analytic_error_h"] = (
        hourly["RUL_analytic_h"] - hourly["RUL_oracle_h"]
    ).abs()
    hourly["hybrid_error_h"] = (
        hourly["RUL_fused_h"] - hourly["RUL_oracle_h"]
    ).abs()

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.64, 0.36],
        subplot_titles=(
            "Остаточный ресурс",
            "Абсолютная ошибка относительно Oracle",
        ),
    )
    for column, name, color, width in (
        ("RUL_oracle_h", "Oracle RUL", "#111111", 2.2),
        ("RUL_ml_h", "ML RUL", "#1f77b4", 1.7),
        ("RUL_analytic_h", "Аналитический RUL", "#8c564b", 1.7),
        ("RUL_fused_h", "Гибридный RUL", "#d62728", 2.0),
    ):
        figure.add_trace(
            go.Scatter(
                x=hourly["timestamp"],
                y=hourly[column],
                name=name,
                mode="lines",
                line={"color": color, "width": width},
            ),
            row=1,
            col=1,
        )
    for threshold, color in (
        (cfg.planned_maintenance_rul_h, "#ffbf00"),
        (cfg.urgent_maintenance_rul_h, "#7f0000"),
    ):
        figure.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=color,
            row=1,
            col=1,
        )
    for column, name, color in (
        ("ml_error_h", "Ошибка ML", "#1f77b4"),
        ("analytic_error_h", "Ошибка аналитики", "#8c564b"),
        ("hybrid_error_h", "Ошибка гибрида", "#d62728"),
    ):
        figure.add_trace(
            go.Scatter(
                x=hourly["timestamp"],
                y=hourly[column],
                name=name,
                mode="lines",
                line={"color": color, "width": 1.5},
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text="RUL, ч", row=1, col=1)
    figure.update_yaxes(title_text="Ошибка, ч", rangemode="tozero", row=2, col=1)
    figure.update_xaxes(title_text="Время", row=2, col=1)
    figure.update_layout(
        title="Качество прогноза на текущем симуляционном прогоне",
        template="plotly_white",
        hovermode="x unified",
        height=760,
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 60, "r": 30, "t": 100, "b": 50},
    )
    return figure


def _number(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if np.isnan(number) else number
