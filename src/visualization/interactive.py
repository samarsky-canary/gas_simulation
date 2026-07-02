from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.simulator.config import ScenarioConfig
from src.visualization.plots import (
    RUL_SOURCE_COLORS,
    STATE_LABELS,
    STATE_SMOOTH_HOURS,
    STATE_TICKS,
    STATE_TO_CODE,
    _aggregate_rul_source_plot_data,
    _rolling,
    _stable_state_codes,
    _weekly_rul_source_counts,
)


InteractivePlotBuilder = Callable[
    [ScenarioConfig, pd.DataFrame, pd.DataFrame | None], go.Figure
]

INTERACTIVE_PLOT_BUILDERS: dict[str, InteractivePlotBuilder] = {
    "pressure": lambda cfg, data, decisions: _pressure_plot(data),
    "delta_p": lambda cfg, data, decisions: _delta_p_plot(cfg, data),
    "state": lambda cfg, data, decisions: _state_plot(cfg, data),
    "rul_comparison": lambda cfg, data, decisions: _rul_comparison_plot(
        cfg, data, decisions
    ),
    "rul_source_periods": lambda cfg, data, decisions: _rul_source_periods_plot(
        cfg, decisions
    ),
    "data_confidence": lambda cfg, data, decisions: _data_confidence_plot(
        data, decisions
    ),
}

RUL_SOURCE_LABELS = {
    "ml_baseline": "ML-модель",
    "conservative_min": "Консервативный минимум",
    "analytic_fallback": "Аналитическая оценка",
    "analytic_data_veto": "Аналитика из-за качества данных",
    "unavailable": "Недоступно",
}

STATE_VALUE_LABELS = {
    "unknown": "неизвестно",
    "normal": "норма",
    "warning": "предупреждение",
    "critical": "критическое",
}


def build_interactive_plot(
    plot_key: str,
    cfg: ScenarioConfig,
    df: pd.DataFrame,
    hybrid_decisions: pd.DataFrame | None = None,
) -> go.Figure:
    """Строит выбранный интерактивный Plotly-график для Streamlit."""
    try:
        builder = INTERACTIVE_PLOT_BUILDERS[plot_key]
    except KeyError as error:
        raise ValueError(f"Unsupported interactive plot: {plot_key}") from error

    data = df.sort_values("timestamp").copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    decisions = None
    if hybrid_decisions is not None and not hybrid_decisions.empty:
        decisions = hybrid_decisions.sort_values("timestamp").copy()
        decisions["timestamp"] = pd.to_datetime(decisions["timestamp"])
    return builder(cfg, data, decisions)


def _pressure_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["p_in_mpa"],
            name="Давление до фильтра",
            mode="lines",
            line={"color": "#1f77b4", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["p_out_mpa"],
            name="Давление после фильтра",
            mode="lines",
            line={"color": "#ff7f0e", "width": 1},
        )
    )
    return _style_figure(
        fig,
        title="Давление до и после фильтра",
        yaxis_title="Давление, МПа",
        rangeslider=True,
    )


def _delta_p_plot(cfg: ScenarioConfig, df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["delta_p_kpa"],
            name="Наблюдаемый перепад",
            mode="lines",
            line={"color": "#d62728", "width": 1},
            opacity=0.65,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=_rolling(df, "delta_p_kpa", cfg, hours=24),
            name="Средний перепад за 24 ч",
            mode="lines",
            line={"color": "#111111", "width": 2},
        )
    )
    _add_thresholds(fig, cfg.dp_warn_kpa, cfg.dp_crit_kpa, "кПа")
    return _style_figure(
        fig,
        title="Перепад давления во времени",
        yaxis_title="Перепад, кПа",
        rangeslider=True,
    )


def _state_plot(cfg: ScenarioConfig, df: pd.DataFrame) -> go.Figure:
    raw_codes = df["state_obs"].map(STATE_TO_CODE).fillna(-1)
    stable_codes = _stable_state_codes(cfg, df)
    raw_labels = df["state_obs"].map(STATE_VALUE_LABELS).fillna("неизвестно")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=raw_codes,
            name="Исходное состояние",
            mode="lines",
            line={"color": "#9ca3af", "width": 1, "shape": "hv"},
            opacity=0.45,
            customdata=raw_labels,
            hovertemplate="%{x}<br>Состояние: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=stable_codes,
            name=f"Сглаженное состояние, окно {STATE_SMOOTH_HOURS} ч",
            mode="lines",
            line={"color": "#111111", "width": 2, "shape": "hv"},
        )
    )
    fig.update_yaxes(tickmode="array", tickvals=STATE_TICKS, ticktext=STATE_LABELS)
    return _style_figure(
        fig,
        title="Состояние фильтра: исходное и сглаженное",
        yaxis_title="Состояние",
        rangeslider=True,
    )


def _rul_comparison_plot(
    cfg: ScenarioConfig,
    df: pd.DataFrame,
    decisions: pd.DataFrame | None,
) -> go.Figure:
    rul_data = (
        df.set_index("timestamp")[["rul_analytic_h"]]
        .resample("1h")
        .mean()
        .reset_index()
    )
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Аналитический остаточный ресурс",
            "Остаточный ресурс по ML",
            "Итоговый остаточный ресурс",
        ),
    )
    _add_line(
        fig, 1, rul_data, "rul_analytic_h", "Аналитический остаточный ресурс", "#8c564b"
    )
    if decisions is not None:
        hourly = (
            decisions.set_index("timestamp")[["RUL_ml_h", "RUL_fused_h"]]
            .resample("1h")
            .mean()
            .reset_index()
        )
        _add_line(fig, 2, hourly, "RUL_ml_h", "Остаточный ресурс по ML", "#1f77b4")
        _add_line(
            fig, 3, hourly, "RUL_fused_h", "Итоговый остаточный ресурс", "#d62728"
        )

    for row in range(1, 4):
        _add_subplot_thresholds(fig, cfg, row)
        fig.update_yaxes(title_text="Остаточный ресурс, ч", row=row, col=1)
    fig.update_xaxes(title_text="Время", row=3, col=1)
    return _style_figure(
        fig,
        title="Оценки остаточного ресурса, среднее за 1 час",
        height=760,
    )


def _rul_source_periods_plot(
    cfg: ScenarioConfig,
    decisions: pd.DataFrame | None,
) -> go.Figure:
    _require_decisions(decisions)
    plot_data = _aggregate_rul_source_plot_data(decisions)
    weekly = _weekly_rul_source_counts(decisions)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        row_heights=[0.4, 0.16, 0.2, 0.24],
        subplot_titles=(
            "Оценки остаточного ресурса",
            "Выбранный источник остаточного ресурса",
            "Доверие",
            "Количество решений по источникам за 7 дней",
        ),
    )
    _add_line(fig, 1, plot_data, "RUL_ml_h", "Остаточный ресурс по ML", "#1f77b4")
    _add_line(
        fig,
        1,
        plot_data,
        "RUL_analytic_h",
        "Аналитический остаточный ресурс",
        "#8c564b",
    )
    _add_line(
        fig,
        1,
        plot_data,
        "RUL_fused_h",
        "Итоговый остаточный ресурс",
        "#d62728",
        width=2,
    )
    _add_subplot_thresholds(fig, cfg, 1)

    sources = [
        source
        for source in RUL_SOURCE_COLORS
        if source in set(plot_data["rul_source"].dropna().astype(str))
    ]
    positions = {source: index for index, source in enumerate(sources)}
    for source in sources:
        subset = plot_data[plot_data["rul_source"].astype(str).eq(source)]
        fig.add_trace(
            go.Scatter(
                x=subset["timestamp"],
                y=[positions[source]] * len(subset),
                name=RUL_SOURCE_LABELS.get(source, source),
                mode="markers",
                marker={"color": RUL_SOURCE_COLORS[source], "size": 8, "symbol": "square"},
                hovertemplate=(
                    f"%{{x}}<br>{RUL_SOURCE_LABELS.get(source, source)}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(positions.values()),
        ticktext=[RUL_SOURCE_LABELS.get(source, source) for source in positions],
        row=2,
        col=1,
    )

    _add_line(fig, 3, plot_data, "confidence_total", "Итоговое доверие", "#111111")
    if "confidence_data" in plot_data.columns:
        _add_line(fig, 3, plot_data, "confidence_data", "Доверие к данным", "#1f77b4")
    _add_line(
        fig,
        3,
        plot_data,
        "confidence_consistency",
        "Согласованность ML и аналитической оценки",
        "#ff7f0e",
    )
    fig.add_hline(y=0.65, line_dash="dot", line_color="#1f77b4", row=3, col=1)
    fig.add_hline(y=0.30, line_dash="dot", line_color="#d62728", row=3, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=3, col=1)

    if not weekly.empty:
        week_centers = weekly.index + pd.Timedelta(days=3.5)
        for column, label, color in (
            ("analytic", "аналитика", "#d62728"),
            ("hybrid", "гибрид", "#ff7f0e"),
            ("ml", "ML", "#1f77b4"),
        ):
            fig.add_trace(
                go.Bar(
                    x=week_centers,
                    y=weekly[column],
                    name=label,
                    marker_color=color,
                ),
                row=4,
                col=1,
            )
    fig.update_yaxes(title_text="Остаточный ресурс, ч", row=1, col=1)
    fig.update_yaxes(title_text="Источник", row=2, col=1)
    fig.update_yaxes(title_text="0...1", row=3, col=1)
    fig.update_yaxes(title_text="Решений", row=4, col=1)
    return _style_figure(
        fig,
        title="",
        height=1000,
    )


def _data_confidence_plot(
    telemetry: pd.DataFrame,
    decisions: pd.DataFrame | None,
) -> go.Figure:
    _require_decisions(decisions)
    confidence_hourly = (
        decisions.set_index("timestamp")
        .resample("1h")
        .agg(
            data_confidence_mean=("confidence_data", "mean"),
            consistency_mean=("confidence_consistency", "mean"),
            total_confidence_mean=("confidence_total", "mean"),
        )
        .reset_index()
    )
    quality_hourly, missing_percent, spike_percent = _quality_issue_summary(telemetry)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.58, 0.42],
        subplot_titles=(
            "Доверие к прогнозу",
            "Пропуски и всплески телеметрии",
        ),
    )
    _add_line(
        fig,
        1,
        confidence_hourly,
        "data_confidence_mean",
        "Доверие к данным",
        "#1f77b4",
    )
    _add_line(
        fig,
        1,
        confidence_hourly,
        "consistency_mean",
        "Согласованность ML и аналитической оценки",
        "#2ca02c",
    )
    _add_line(
        fig,
        1,
        confidence_hourly,
        "total_confidence_mean",
        "Итоговое доверие к прогнозу",
        "#111111",
        width=2,
    )
    fig.add_hline(y=0.45, line_dash="dash", line_color="#ff7f0e", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=quality_hourly["timestamp"],
            y=quality_hourly["missing_percent"],
            name="Пропуски",
            mode="lines",
            line={"color": "#9467bd", "width": 1.5},
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=quality_hourly["timestamp"],
            y=quality_hourly["spike_percent"],
            name="Всплески",
            mode="lines",
            line={"color": "#d62728", "width": 1.5},
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=1,
        y=0.39,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="bottom",
        text=(
            f"<b>За весь период:</b> пропуски {missing_percent:.2f}% · "
            f"всплески {spike_percent:.2f}%"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#9ca3af",
        borderwidth=1,
    )
    fig.update_yaxes(title_text="Доверие, 0…1", range=[-0.02, 1.02], row=1, col=1)
    fig.update_yaxes(
        title_text="Доля строк, %",
        rangemode="tozero",
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Время", row=2, col=1)
    return _style_figure(
        fig,
        title="Доверие к прогнозу и качество телеметрии",
        height=760,
    )


def _quality_issue_summary(
    telemetry: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Считает часовые и общие доли строк с пропусками и всплесками."""
    sensor_columns = [
        column
        for column in ("p_in_mpa", "p_out_mpa", "delta_p_kpa", "q_m3h", "t_c")
        if column in telemetry.columns
    ]
    missing = telemetry[sensor_columns].isna().any(axis=1)
    if "quality_code" in telemetry.columns:
        missing |= telemetry["quality_code"].eq("missing")
    spike = (
        telemetry["spike_event"].fillna(False).astype(bool)
        if "spike_event" in telemetry.columns
        else pd.Series(False, index=telemetry.index)
    )
    quality = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(telemetry["timestamp"]),
            "missing": missing.astype(float),
            "spike": spike.astype(float),
        }
    )
    hourly = (
        quality.set_index("timestamp")
        .resample("1h")
        .agg(
            missing_percent=("missing", "mean"),
            spike_percent=("spike", "mean"),
        )
        .mul(100)
        .reset_index()
    )
    return hourly, float(missing.mean() * 100), float(spike.mean() * 100)


def _add_line(
    fig: go.Figure,
    row: int,
    data: pd.DataFrame,
    column: str,
    name: str,
    color: str,
    *,
    width: float = 1.5,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data[column],
            name=name,
            mode="lines",
            line={"color": color, "width": width},
        ),
        row=row,
        col=1,
    )


def _add_thresholds(
    fig: go.Figure,
    warning: float,
    critical: float,
    unit: str,
) -> None:
    fig.add_hline(
        y=warning,
        line_dash="dash",
        line_color="#ffbf00",
        annotation_text=f"Порог предупреждения: {warning:g} {unit}",
    )
    fig.add_hline(
        y=critical,
        line_dash="dash",
        line_color="#7f0000",
        annotation_text=f"Критический порог: {critical:g} {unit}",
    )


def _add_subplot_thresholds(fig: go.Figure, cfg: ScenarioConfig, row: int) -> None:
    fig.add_hline(
        y=cfg.planned_maintenance_rul_h,
        line_dash="dash",
        line_color="#ffbf00",
        row=row,
        col=1,
    )
    fig.add_hline(
        y=cfg.urgent_maintenance_rul_h,
        line_dash="dash",
        line_color="#7f0000",
        row=row,
        col=1,
    )


def _require_decisions(decisions: pd.DataFrame | None) -> None:
    if decisions is None or decisions.empty:
        raise ValueError("Для этого графика необходимы гибридные решения.")


def _style_figure(
    fig: go.Figure,
    *,
    title: str,
    yaxis_title: str | None = None,
    height: int = 560,
    rangeslider: bool = False,
) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 30, "t": 90, "b": 50},
        uirevision="gas-filter-simulation",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    if yaxis_title is not None:
        fig.update_yaxes(title_text=yaxis_title)
    if rangeslider:
        fig.update_xaxes(rangeslider={"visible": True})
    return fig
