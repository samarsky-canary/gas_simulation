from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.simulator.config import ScenarioConfig


STATE_TO_CODE = {"unknown": -1, "normal": 0, "warning": 1, "critical": 2}
STATE_TICKS = [-1, 0, 1, 2]
STATE_LABELS = ["неизвестно", "норма", "предупреждение", "критическое"]
STATE_SMOOTH_HOURS = 3
RUL_SOURCE_PLOT_FREQUENCY = "1h"
RUL_SOURCE_COLORS = {
    "ml_baseline": "#1f77b4",
    "conservative_min": "#ff7f0e",
    "analytic_fallback": "#d62728",
    "analytic_data_veto": "#9467bd",
    "unavailable": "#6b7280",
}
ACTION_COLORS = {
    "monitor": "#6b7280",
    "sensor_check": "#9467bd",
    "planned_maintenance": "#ffbf00",
    "urgent_maintenance": "#d62728",
    "shutdown_request": "#111111",
    "manual_review": "#ff7f0e",
}


def build_plots(
    cfg: ScenarioConfig,
    df: pd.DataFrame,
    output_dir: Path,
    hybrid_decisions: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Строит набор PNG-графиков для визуальной проверки синтетического прогона."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data = df.sort_values("timestamp").copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    paths = {
        "q": plot_dir / "01_rashod_q.png",
        "pressure": plot_dir / "02_davleniya_pin_pout.png",
        "delta_p": plot_dir / "03_perepad_delta_p.png",
        "state": plot_dir / "06_sostoyanie_filtra.png",
        "rul_comparison": plot_dir / "09_sravnenie_rul.png",
        "rul_source_periods": plot_dir / "11_periodi_predpochteniya_rul.png",
        "data_confidence": plot_dir / "12_doverie_k_dannym.png",
        "spike_rate": plot_dir / "13_dolya_vybrosov.png",
        "description": plot_dir / "plots_description.md",
        "diagnostics": plot_dir / "plot_diagnostics.md",
    }

    _plot_q(data, paths["q"])
    _plot_pressure(data, paths["pressure"])
    _plot_delta_p(cfg, data, paths["delta_p"])
    _plot_state(cfg, data, paths["state"])
    _plot_rul_comparison(cfg, data, hybrid_decisions, paths["rul_comparison"])
    _plot_rul_source_periods(cfg, hybrid_decisions, paths["rul_source_periods"])
    _plot_data_confidence(hybrid_decisions, paths["data_confidence"])
    _plot_spike_rate(data, paths["spike_rate"])

    # These plots are intentionally disabled because they are not displayed in the UI.
    # Keep the calls here so they can be restored without reconstructing the orchestration.
    # _plot_clog(data, plot_dir / "04_zasorenie_clog_level.png")
    # _plot_rul(data, plot_dir / "05_ostatochnyi_resurs_rul.png")
    # _plot_delta_p_norm(cfg, data, plot_dir / "07_normirovannyi_perepad.png")
    # _plot_delta_p_vs_clog(cfg, data, plot_dir / "08_delta_p_i_zasorenie.png")
    # _plot_hybrid_decision_window(
    #     cfg, data, hybrid_decisions, plot_dir / "10_gibridnoe_reshenie.png"
    # )
    # _plot_dashboard(cfg, data, plot_dir / "00_obzornyi_dashboard.png")

    paths["description"].write_text(_description(), encoding="utf-8")
    paths["diagnostics"].write_text(_diagnostics(cfg, data), encoding="utf-8")
    return paths


def _plot_spike_rate(df: pd.DataFrame, path: Path) -> None:
    """Показывает часовую долю строк с искусственно инъецированными выбросами."""
    if "spike_event" not in df.columns:
        path.write_text("Нет флага spike_event для построения графика.", encoding="utf-8")
        return

    data = df[["timestamp", "spike_event"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    hourly_percent = (
        data.set_index("timestamp")["spike_event"].astype(float).resample("1h").mean() * 100.0
    )
    total_percent = float(data["spike_event"].mean() * 100.0)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(
        hourly_percent.index,
        hourly_percent,
        color="#d62728",
        linewidth=1.1,
        label="доля строк с выбросами за час",
    )
    ax.fill_between(
        hourly_percent.index,
        0,
        hourly_percent,
        color="#fecaca",
        alpha=0.45,
    )
    ax.set_ylim(bottom=0)
    ax.set_title("Доля телеметрических строк с выбросами")
    ax.set_ylabel("Выбросы, %")
    ax.set_xlabel("Время")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.text(
        0.99,
        0.90,
        f"Среднее: {total_percent:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#d62728", "alpha": 0.9},
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, path)


def _plot_data_confidence(
    hybrid_decisions: pd.DataFrame | None,
    path: Path,
) -> None:
    """Показывает доверие к данным и долю пропусков для демонстрации качества телеметрии."""
    if hybrid_decisions is None or hybrid_decisions.empty:
        path.write_text("Нет hybrid_decisions для построения графика.", encoding="utf-8")
        return

    required = {
        "timestamp",
        "confidence_data",
        "missing_rate_1h",
        "confidence_consistency",
    }
    if not required.issubset(hybrid_decisions.columns):
        path.write_text("Нет полей качества данных для построения графика.", encoding="utf-8")
        return

    data = hybrid_decisions[list(required)].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp").sort_index()
    hourly = data.resample("1h").agg(
        confidence_mean=("confidence_data", "mean"),
        missing_rate_mean=("missing_rate_1h", "mean"),
        consistency_mean=("confidence_consistency", "mean"),
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.2]},
    )

    axes[0].plot(
        hourly.index,
        hourly["confidence_mean"],
        color="#1f77b4",
        linewidth=1.2,
        label="среднее C_data за час",
    )
    axes[0].axhline(
        0.45,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.0,
        label="граница низкого доверия",
    )
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("C_data, 0...1")
    axes[0].set_title("Доверие к данным телеметрии C_data(t)")
    axes[0].legend(loc="lower left", ncol=2)
    confidence_percent = float(hourly["confidence_mean"].mean() * 100.0)
    axes[0].text(
        0.99,
        0.90,
        f"Среднее: {confidence_percent:.2f}%",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#1f77b4", "alpha": 0.9},
    )

    axes[1].plot(
        hourly.index,
        hourly["missing_rate_mean"] * 100.0,
        color="#9467bd",
        linewidth=1.0,
        label="средняя доля пропусков за час",
    )
    missing_percent = float(hourly["missing_rate_mean"].mean() * 100.0)
    axes[1].set_ylabel("Пропуски, %")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper left")
    axes[1].text(
        0.99,
        0.90,
        f"Среднее: {missing_percent:.2f}%",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#9467bd", "alpha": 0.9},
    )

    axes[2].plot(
        hourly.index,
        hourly["consistency_mean"] * 100.0,
        color="#2ca02c",
        linewidth=1.1,
        label="согласованность ML и аналитического RUL",
    )
    consistency_percent = float(hourly["consistency_mean"].mean() * 100.0)
    axes[2].set_ylim(-2, 102)
    axes[2].set_ylabel("Согласованность, %")
    axes[2].set_xlabel("Время")
    axes[2].legend(loc="lower left")
    axes[2].text(
        0.99,
        0.90,
        f"Среднее: {consistency_percent:.2f}%",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#2ca02c", "alpha": 0.9},
    )

    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, path)


def _plot_q(df: pd.DataFrame, path: Path) -> None:
    """Рисует расход газа во времени."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp"], df["q_m3h"], color="#1f77b4", linewidth=0.8)
    _style_time_axis(ax, "Расход газа Q(t)", "Расход, м3/ч")
    _save(fig, path)


def _plot_pressure(df: pd.DataFrame, path: Path) -> None:
    """Рисует входное и выходное давление на одном графике."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp"], df["p_in_mpa"], label="P_in", color="#1f77b4", linewidth=0.8)
    ax.plot(df["timestamp"], df["p_out_mpa"], label="P_out", color="#ff7f0e", linewidth=0.8)
    ax.legend(loc="best")
    _style_time_axis(ax, "Давления до и после фильтра", "Давление, МПа")
    _save(fig, path)


def _plot_delta_p(cfg: ScenarioConfig, df: pd.DataFrame, path: Path) -> None:
    """Рисует перепад давления, пороги warning/critical и сглаженный тренд."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp"], df["delta_p_kpa"], label="deltaP наблюдаемый", color="#d62728", linewidth=0.7)
    ax.plot(
        df["timestamp"],
        _rolling(df, "delta_p_kpa", cfg, hours=24),
        label="deltaP, среднее за 24 ч",
        color="#111111",
        linewidth=1.5,
    )
    ax.axhline(cfg.dp_warn_kpa, color="#ffbf00", linestyle="--", linewidth=1.0, label="порог warning")
    ax.axhline(cfg.dp_crit_kpa, color="#7f0000", linestyle="--", linewidth=1.0, label="порог critical")
    ax.legend(loc="best")
    _style_time_axis(ax, "Перепад давления deltaP(t)", "Перепад, кПа")
    _save(fig, path)


def _plot_clog(df: pd.DataFrame, path: Path) -> None:
    """Рисует скрытый уровень засорения, который недоступен реальному датчику."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp"], df["clog_level"], color="#2ca02c", linewidth=1.0)
    ax.set_ylim(-0.02, 1.02)
    _style_time_axis(ax, "Скрытый уровень засорения clog_level(t)", "Засорение, доля")
    _save(fig, path)


def _plot_rul(df: pd.DataFrame, path: Path) -> None:
    """Сравнивает oracle-RUL и аналитическую оценку остаточного ресурса."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp"], df["rul_oracle_h"], label="RUL oracle", color="#9467bd", linewidth=0.9)
    ax.plot(df["timestamp"], df["rul_analytic_h"], label="RUL аналитический", color="#8c564b", linewidth=0.9, alpha=0.85)
    ax.legend(loc="best")
    _style_time_axis(ax, "Остаточный ресурс RUL(t)", "Остаточный ресурс, ч")
    _save(fig, path)


def _plot_rul_comparison(
    cfg: ScenarioConfig,
    df: pd.DataFrame,
    hybrid_decisions: pd.DataFrame | None,
    path: Path,
) -> None:
    """Сравнивает часовые средние oracle, аналитического, ML и гибридного RUL."""
    rul_data = (
        df.set_index("timestamp")[["rul_oracle_h", "rul_analytic_h"]]
        .resample(RUL_SOURCE_PLOT_FREQUENCY)
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(
        rul_data["timestamp"],
        rul_data["rul_oracle_h"],
        label="1. RUL oracle",
        color="#4b5563",
        linewidth=1.0,
    )
    axes[1].plot(
        rul_data["timestamp"],
        rul_data["rul_analytic_h"],
        label="2. RUL аналитический",
        color="#8c564b",
        linewidth=1.0,
    )

    if hybrid_decisions is not None and not hybrid_decisions.empty:
        decisions = hybrid_decisions.sort_values("timestamp").copy()
        decisions["timestamp"] = pd.to_datetime(decisions["timestamp"])
        decisions = (
            decisions.set_index("timestamp")[["RUL_ml_h", "RUL_fused_h"]]
            .resample(RUL_SOURCE_PLOT_FREQUENCY)
            .mean()
            .reset_index()
        )
        axes[2].plot(
            decisions["timestamp"],
            decisions["RUL_ml_h"],
            label="3. RUL ML",
            color="#1f77b4",
            linewidth=1.0,
        )
        axes[3].plot(
            decisions["timestamp"],
            decisions["RUL_fused_h"],
            label="4. RUL гибрид",
            color="#d62728",
            linewidth=1.2,
        )
    else:
        axes[2].text(0.5, 0.5, "Нет ML-прогноза", transform=axes[2].transAxes, ha="center", va="center")
        axes[3].text(0.5, 0.5, "Нет гибридного RUL", transform=axes[3].transAxes, ha="center", va="center")

    for ax in axes:
        ax.axhline(
            cfg.planned_maintenance_rul_h,
            color="#ffbf00",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
            label="порог планового ТО",
        )
        ax.axhline(
            cfg.urgent_maintenance_rul_h,
            color="#7f0000",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
            label="порог срочного ТО",
        )
        ax.set_ylabel("RUL, ч")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[0].set_title("Сравнение оценок остаточного ресурса RUL(t), среднее за 1 час")
    axes[-1].set_xlabel("Время")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, path)


def _plot_hybrid_decision_window(
    cfg: ScenarioConfig,
    df: pd.DataFrame,
    hybrid_decisions: pd.DataFrame | None,
    path: Path,
) -> None:
    """Показывает гибридность решения в окне, где появляются сервисные действия."""
    if hybrid_decisions is None or hybrid_decisions.empty:
        path.write_text("Нет hybrid_decisions для построения графика.", encoding="utf-8")
        return

    decisions = hybrid_decisions.sort_values("timestamp").copy()
    decisions["timestamp"] = pd.to_datetime(decisions["timestamp"])
    window = _hybrid_focus_window(cfg, decisions)
    view = decisions[
        decisions["timestamp"].between(window[0], window[1], inclusive="both")
    ].copy()
    truth = df[df["timestamp"].between(window[0], window[1], inclusive="both")].copy()
    if view.empty:
        view = decisions
        truth = df

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2, 1.0, 1.0]},
    )

    axes[0].plot(
        view["timestamp"],
        view["RUL_ml_h"],
        label="RUL ML",
        color="#1f77b4",
        linewidth=0.9,
        alpha=0.85,
    )
    axes[0].plot(
        view["timestamp"],
        view["RUL_analytic_h"],
        label="RUL аналитический",
        color="#8c564b",
        linewidth=0.9,
        alpha=0.85,
    )
    axes[0].plot(
        view["timestamp"],
        view["RUL_fused_h"],
        label="RUL hybrid/fused",
        color="#d62728",
        linewidth=1.8,
    )
    axes[0].axhline(cfg.planned_maintenance_rul_h, color="#ffbf00", linestyle="--", linewidth=1.0, label="плановое ТО")
    axes[0].axhline(cfg.urgent_maintenance_rul_h, color="#7f0000", linestyle="--", linewidth=1.0, label="срочное ТО")
    axes[0].set_ylabel("RUL, ч")
    axes[0].set_title("Гибридное решение: сравнение RUL, доверия, источника и действия")
    axes[0].legend(loc="best")

    axes[1].plot(view["timestamp"], view["confidence_data"], label="data", color="#2ca02c", linewidth=0.9)
    axes[1].plot(view["timestamp"], view["confidence_model"], label="model", color="#1f77b4", linewidth=0.9)
    axes[1].plot(view["timestamp"], view["confidence_consistency"], label="consistency", color="#ff7f0e", linewidth=0.9)
    axes[1].plot(view["timestamp"], view["confidence_total"], label="total", color="#111111", linewidth=1.4)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("confidence")
    axes[1].legend(loc="best", ncol=4)

    _plot_category_timeline(
        axes[2],
        view,
        column="rul_source",
        colors=RUL_SOURCE_COLORS,
        title="Источник итогового RUL",
    )
    _plot_category_timeline(
        axes[3],
        view,
        column="action",
        colors=ACTION_COLORS,
        title="Итоговое действие",
    )

    if not truth.empty:
        critical = truth["state_true"].eq("critical") if "state_true" in truth.columns else pd.Series(False, index=truth.index)
        if critical.any():
            first_critical = truth.loc[critical, "timestamp"].iloc[0]
            for ax in axes:
                ax.axvline(first_critical, color="#7f0000", linestyle=":", linewidth=1.2, alpha=0.85)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    _save(fig, path)


def _plot_rul_source_periods(
    cfg: ScenarioConfig,
    hybrid_decisions: pd.DataFrame | None,
    path: Path,
) -> None:
    """Показывает, в какие периоды итоговый RUL берется из ML, аналитики или минимума."""
    if hybrid_decisions is None or hybrid_decisions.empty:
        path.write_text("Нет hybrid_decisions для построения графика.", encoding="utf-8")
        return

    decisions = hybrid_decisions.sort_values("timestamp").copy()
    decisions["timestamp"] = pd.to_datetime(decisions["timestamp"])
    plot_decisions = _aggregate_rul_source_plot_data(decisions)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.1, 1.1, 1.4]},
    )

    axes[0].plot(
        plot_decisions["timestamp"],
        plot_decisions["RUL_ml_h"],
        label="RUL ML",
        color="#1f77b4",
        linewidth=0.8,
        alpha=0.75,
    )
    axes[0].plot(
        plot_decisions["timestamp"],
        plot_decisions["RUL_analytic_h"],
        label="RUL аналитический",
        color="#8c564b",
        linewidth=0.8,
        alpha=0.75,
    )
    axes[0].plot(
        plot_decisions["timestamp"],
        plot_decisions["RUL_fused_h"],
        label="RUL итоговый",
        color="#d62728",
        linewidth=1.5,
    )
    axes[0].axhline(cfg.planned_maintenance_rul_h, color="#ffbf00", linestyle="--", linewidth=1.0, label="плановое ТО")
    axes[0].axhline(cfg.urgent_maintenance_rul_h, color="#7f0000", linestyle="--", linewidth=1.0, label="срочное ТО")
    axes[0].set_ylabel("RUL, ч")
    axes[0].set_title("Периоды предпочтения источника RUL, агрегация по 1 часу")
    axes[0].legend(loc="best")

    _plot_source_bands(axes[1], plot_decisions)
    axes[1].set_ylabel("Источник RUL")

    axes[2].plot(
        plot_decisions["timestamp"],
        plot_decisions["confidence_total"],
        label="confidence_total",
        color="#111111",
        linewidth=1.1,
    )
    axes[2].plot(
        plot_decisions["timestamp"],
        plot_decisions["confidence_consistency"],
        label="confidence_consistency",
        color="#ff7f0e",
        linewidth=0.9,
        alpha=0.8,
    )
    axes[2].axhline(0.65, color="#1f77b4", linestyle=":", linewidth=1.0, label="порог выбора ML")
    axes[2].axhline(0.30, color="#d62728", linestyle=":", linewidth=1.0, label="порог conservative_min")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("confidence")
    axes[2].legend(loc="best", ncol=4)

    _plot_weekly_rul_source_counts(axes[3], decisions)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    _save(fig, path)


def _aggregate_rul_source_plot_data(decisions: pd.DataFrame) -> pd.DataFrame:
    """Сжимает частый ряд до часовых точек только для быстрой визуализации."""
    if decisions.empty:
        return decisions.copy()

    data = decisions.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp").sort_index()
    numeric_columns = [
        "RUL_ml_h",
        "RUL_analytic_h",
        "RUL_fused_h",
        "confidence_total",
        "confidence_consistency",
    ]
    numeric = data[numeric_columns].resample(RUL_SOURCE_PLOT_FREQUENCY).mean()
    source = data["rul_source"].resample(RUL_SOURCE_PLOT_FREQUENCY).agg(_dominant_value)
    return numeric.assign(rul_source=source).dropna(how="all").reset_index()


def _dominant_value(values: pd.Series) -> str | None:
    """Возвращает наиболее частую категорию интервала с устойчивым выбором при равенстве."""
    non_null = values.dropna().astype(str)
    if non_null.empty:
        return None
    counts = non_null.value_counts(sort=False)
    maximum = counts.max()
    winners = set(counts[counts.eq(maximum)].index)
    return next(value for value in non_null if value in winners)


def _plot_source_bands(ax: plt.Axes, decisions: pd.DataFrame) -> None:
    """Рисует непрерывные цветные интервалы выбора источника RUL."""
    sources = [source for source in RUL_SOURCE_COLORS if source in set(decisions["rul_source"].astype(str))]
    if not sources:
        sources = sorted(decisions["rul_source"].dropna().astype(str).unique())
    positions = {source: index for index, source in enumerate(sources)}
    used_labels: set[str] = set()

    for source, start, end in _source_intervals(decisions):
        position = positions.get(source)
        if position is None:
            continue
        label = source if source not in used_labels else None
        ax.fill_between(
            [start, end],
            position - 0.35,
            position + 0.35,
            color=RUL_SOURCE_COLORS.get(source, "#4b5563"),
            alpha=0.75,
            step="post",
            label=label,
        )
        used_labels.add(source)

    ax.set_yticks(list(positions.values()))
    ax.set_yticklabels(list(positions.keys()))
    ax.set_ylim(-0.75, max(positions.values(), default=0) + 0.75)
    ax.legend(loc="best", ncol=3)


def _plot_weekly_rul_source_counts(ax: plt.Axes, decisions: pd.DataFrame) -> None:
    """Показывает, как часто за неделю итоговый RUL выбирался из каждого источника."""
    weekly = _weekly_rul_source_counts(decisions)
    if weekly.empty:
        ax.text(0.5, 0.5, "Нет решений для недельной агрегации", transform=ax.transAxes, ha="center", va="center")
        return

    week_centers = weekly.index + pd.Timedelta(days=3.5)
    x = mdates.date2num(week_centers.to_pydatetime())
    width_days = 1.8
    ax.bar(
        x - width_days,
        weekly["analytic"],
        width=width_days,
        label="аналитика",
        color="#d62728",
        alpha=0.82,
    )
    ax.bar(
        x,
        weekly["hybrid"],
        width=width_days,
        label="гибрид",
        color="#ff7f0e",
        alpha=0.82,
    )
    ax.bar(
        x + width_days,
        weekly["ml"],
        width=width_days,
        label="ML",
        color="#1f77b4",
        alpha=0.82,
    )
    ax.set_ylabel("решений / 7 дней")
    ax.set_title("Количество решений по источникам RUL за 7 дней")
    ax.legend(loc="best", ncol=3)


def _weekly_rul_source_counts(decisions: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует решения по источникам RUL в семидневные интервалы."""
    if decisions.empty or "timestamp" not in decisions.columns or "rul_source" not in decisions.columns:
        return pd.DataFrame()

    data = decisions[["timestamp", "rul_source"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["rul_source"] = data["rul_source"].astype(str)
    data = data.set_index("timestamp").sort_index()

    return pd.DataFrame(
        {
            "analytic": data["rul_source"].isin(["analytic_fallback", "analytic_data_veto"]).resample("7D").sum(),
            "hybrid": data["rul_source"].eq("conservative_min").resample("7D").sum(),
            "ml": data["rul_source"].eq("ml_baseline").resample("7D").sum(),
        }
    ).fillna(0)


def _source_intervals(decisions: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Сворачивает последовательность строк в интервалы с одинаковым `rul_source`."""
    if decisions.empty:
        return []
    rows = decisions[["timestamp", "rul_source"]].reset_index(drop=True)
    intervals: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    current_source = str(rows.loc[0, "rul_source"])
    start = rows.loc[0, "timestamp"]

    for index in range(1, len(rows)):
        source = str(rows.loc[index, "rul_source"])
        if source == current_source:
            continue
        end = rows.loc[index - 1, "timestamp"]
        intervals.append((current_source, start, end))
        current_source = source
        start = rows.loc[index, "timestamp"]
    intervals.append((current_source, start, rows.loc[len(rows) - 1, "timestamp"]))
    return intervals


def _hybrid_focus_window(
    cfg: ScenarioConfig, decisions: pd.DataFrame
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Выбирает окно вокруг первых значимых сервисных решений, чтобы график не был сжат 90 днями."""
    significant = decisions[
        decisions["action"].isin(["planned_maintenance", "urgent_maintenance", "shutdown_request"])
        | decisions["RUL_fused_h"].lt(cfg.planned_maintenance_rul_h)
    ]
    if significant.empty:
        start = decisions["timestamp"].min()
        end = decisions["timestamp"].max()
        return start, end
    center = significant["timestamp"].iloc[0]
    start = max(decisions["timestamp"].min(), center - pd.Timedelta(days=3))
    end = min(decisions["timestamp"].max(), center + pd.Timedelta(days=4))
    return start, end


def _plot_category_timeline(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    column: str,
    colors: dict[str, str],
    title: str,
) -> None:
    """Рисует категориальный ряд как цветные точки, чтобы были видны переключения правил."""
    categories = [category for category in colors if category in set(data[column].astype(str))]
    if not categories:
        categories = sorted(data[column].dropna().astype(str).unique())
    positions = {category: index for index, category in enumerate(categories)}
    for category in categories:
        subset = data[data[column].astype(str).eq(category)]
        if subset.empty:
            continue
        ax.scatter(
            subset["timestamp"],
            [positions[category]] * len(subset),
            s=8,
            color=colors.get(category, "#4b5563"),
            label=category,
            alpha=0.75,
        )
    ax.set_yticks(list(positions.values()))
    ax.set_yticklabels(list(positions.keys()))
    ax.set_ylabel(title)
    ax.legend(loc="best", ncol=3, markerscale=2)


def _plot_state(
    cfg: ScenarioConfig,
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Показывает raw-состояние и устойчивое состояние по сглаженному deltaP_norm."""
    fig, ax = plt.subplots(figsize=(14, 4))
    raw_codes = df["state_obs"].map(STATE_TO_CODE).fillna(-1)
    stable_codes = _stable_state_codes(cfg, df)
    unknown_raw = raw_codes.eq(-1)

    ax.step(
        df["timestamp"],
        raw_codes,
        where="post",
        color="#9ca3af",
        linewidth=0.6,
        alpha=0.45,
        label="raw state",
    )
    ax.step(
        df["timestamp"],
        stable_codes,
        where="post",
        color="#111111",
        linewidth=1.4,
        label=f"устойчивое состояние, окно {STATE_SMOOTH_HOURS} ч",
    )
    if unknown_raw.any():
        ax.scatter(
            df.loc[unknown_raw, "timestamp"],
            raw_codes.loc[unknown_raw],
            color="#d62728",
            s=8,
            alpha=0.55,
            label="raw unknown / плохие данные",
        )
    ax.set_yticks(STATE_TICKS)
    ax.set_yticklabels(STATE_LABELS)
    ax.legend(loc="best")
    _style_time_axis(ax, "Состояние фильтра state(t): raw и сглаженное", "Состояние")
    _save(fig, path)


def _plot_delta_p_norm(cfg: ScenarioConfig, df: pd.DataFrame, path: Path) -> None:
    """Рисует перепад, нормированный на расход, как более чистый индикатор засорения."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp"], df["delta_p_norm_q2"], color="#17becf", linewidth=0.7, label="deltaP_norm")
    ax.axhline(cfg.dp_warn_kpa, color="#ffbf00", linestyle="--", linewidth=1.0, label="порог warning")
    ax.axhline(cfg.dp_crit_kpa, color="#7f0000", linestyle="--", linewidth=1.0, label="порог critical")
    ax.legend(loc="best")
    _style_time_axis(ax, "Нормированный перепад deltaP_norm(t)", "Нормированный перепад")
    _save(fig, path)


def _plot_delta_p_vs_clog(cfg: ScenarioConfig, df: pd.DataFrame, path: Path) -> None:
    """Сопоставляет засорение с перепадом давления и его нормированной версией."""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df["timestamp"], df["clog_level"], color="#2ca02c", linewidth=1.2, label="clog_level")
    ax1.set_ylabel("Засорение, доля")
    ax1.set_ylim(-0.02, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(
        df["timestamp"],
        _rolling(df, "delta_p_kpa", cfg, hours=24),
        color="#d62728",
        linewidth=1.3,
        label="deltaP, среднее за 24 ч",
    )
    ax2.plot(
        df["timestamp"],
        _rolling(df, "delta_p_norm_q2", cfg, hours=24),
        color="#17becf",
        linewidth=1.0,
        alpha=0.8,
        label="deltaP_norm, среднее за 24 ч",
    )
    ax2.set_ylabel("Перепад, кПа / нормированное значение")
    ax2.axhline(cfg.dp_warn_kpa, color="#ffbf00", linestyle="--", linewidth=1.0)
    ax2.axhline(cfg.dp_crit_kpa, color="#7f0000", linestyle="--", linewidth=1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    _style_time_axis(ax1, "Сравнение deltaP(t) и clog_level(t)", "Засорение, доля")
    _save(fig, path)


def _plot_dashboard(cfg: ScenarioConfig, df: pd.DataFrame, path: Path) -> None:
    """Собирает ключевые временные ряды на один обзорный лист."""
    fig, axes = plt.subplots(7, 1, figsize=(16, 20), sharex=True)
    axes[0].plot(df["timestamp"], df["q_m3h"], color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("Q, м3/ч")
    axes[0].set_title("Обзорный dashboard симуляции")

    axes[1].plot(df["timestamp"], df["p_in_mpa"], label="P_in", color="#1f77b4", linewidth=0.7)
    axes[1].plot(df["timestamp"], df["p_out_mpa"], label="P_out", color="#ff7f0e", linewidth=0.7)
    axes[1].set_ylabel("P, МПа")
    axes[1].legend(loc="best")

    axes[2].plot(df["timestamp"], df["delta_p_kpa"], color="#d62728", linewidth=0.6)
    axes[2].plot(df["timestamp"], _rolling(df, "delta_p_kpa", cfg, 24), color="#111111", linewidth=1.2)
    axes[2].axhline(cfg.dp_warn_kpa, color="#ffbf00", linestyle="--", linewidth=0.9)
    axes[2].axhline(cfg.dp_crit_kpa, color="#7f0000", linestyle="--", linewidth=0.9)
    axes[2].set_ylabel("deltaP, кПа")

    axes[3].plot(df["timestamp"], df["clog_level"], color="#2ca02c", linewidth=1.0)
    axes[3].set_ylabel("clog")

    axes[4].plot(df["timestamp"], df["rul_oracle_h"], label="oracle", color="#9467bd", linewidth=0.8)
    axes[4].plot(df["timestamp"], df["rul_analytic_h"], label="analytic", color="#8c564b", linewidth=0.8)
    axes[4].set_ylabel("RUL, ч")
    axes[4].legend(loc="best")

    raw_codes = df["state_obs"].map(STATE_TO_CODE).fillna(-1)
    stable_codes = _stable_state_codes(cfg, df)
    axes[5].step(df["timestamp"], raw_codes, where="post", color="#9ca3af", linewidth=0.5, alpha=0.45)
    axes[5].step(df["timestamp"], stable_codes, where="post", color="#111111", linewidth=1.1)
    axes[5].set_ylabel("state")
    axes[5].set_yticks(STATE_TICKS)
    axes[5].set_yticklabels(STATE_LABELS)

    axes[6].plot(df["timestamp"], df["delta_p_norm_q2"], color="#17becf", linewidth=0.7)
    axes[6].set_ylabel("deltaP_norm")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, path)


def _rolling(df: pd.DataFrame, column: str, cfg: ScenarioConfig, hours: int) -> pd.Series:
    """Считает скользящее среднее по заданному числу часов."""
    window = max(int(hours * 60 / cfg.step_minutes), 1)
    return df[column].rolling(window, min_periods=max(window // 4, 1)).mean()


def _stable_state_codes(cfg: ScenarioConfig, df: pd.DataFrame) -> pd.Series:
    """Строит устойчивое состояние по сглаженному нормированному перепаду для графика."""
    window = max(int(STATE_SMOOTH_HOURS * 60 / cfg.step_minutes), 1)
    min_periods = max(window // 4, 1)
    dp_smooth = df["delta_p_norm_q2"].rolling(window, min_periods=min_periods).median()
    codes = pd.Series(-1, index=df.index, dtype="float64")
    codes.loc[dp_smooth < cfg.dp_warn_kpa] = 0
    codes.loc[(dp_smooth >= cfg.dp_warn_kpa) & (dp_smooth < cfg.dp_crit_kpa)] = 1
    codes.loc[dp_smooth >= cfg.dp_crit_kpa] = 2

    bad_data_rate = df["quality_code"].ne("good").rolling(window, min_periods=1).mean()
    codes.loc[bad_data_rate > 0.5] = -1
    return codes


def _style_time_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Применяет общий стиль к графикам временных рядов."""
    ax.set_title(title)
    ax.set_xlabel("Время")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.figure.autofmt_xdate()


def _save(fig: plt.Figure, path: Path) -> None:
    """Сохраняет фигуру в PNG и закрывает ее, чтобы не держать память."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _description() -> str:
    """Создает markdown-описание графиков, включенных в текущий UI pipeline."""
    return "\n".join(
        [
            "# Описание графиков",
            "",
            "- `01_rashod_q.png` - расход газа Q(t).",
            "- `02_davleniya_pin_pout.png` - входное и выходное давление.",
            "- `03_perepad_delta_p.png` - наблюдаемый перепад давления с порогами warning/critical и 24-часовым средним.",
            f"- `06_sostoyanie_filtra.png` - raw-состояние и устойчивое состояние по сглаженному `deltaP_norm`, окно {STATE_SMOOTH_HOURS} ч.",
            "- `09_sravnenie_rul.png` - сравнение oracle, аналитического, ML и гибридного RUL с порогами обслуживания.",
            "- `11_periodi_predpochteniya_rul.png` - полный временной ряд: в какие периоды итоговый RUL берется из ML, аналитики, conservative min или fallback; нижняя панель показывает недельные количества решений по источникам.",
            "- `12_doverie_k_dannym.png` - доверие к данным `C_data`, средняя доля пропусков и согласованность ML с аналитическим RUL.",
            "- `13_dolya_vybrosov.png` - часовая доля строк с инъецированными выбросами и среднее значение за прогон.",
            "",
            "Остальные функции построения графиков сохранены в коде, но их вызовы отключены в `build_plots`, поскольку UI их не отображает.",
            "",
            "Сырой deltaP зависит не только от засорения, но и от расхода. Поэтому для оценки тренда полезнее смотреть 24-часовое среднее и `deltaP_norm`.",
            "",
            "На графике состояния исходный `state_obs` оставлен полупрозрачным, а основная линия строится по сглаженному `deltaP_norm`. Краткие `unknown` из-за пропусков показываются как индикатор качества данных, но не ломают устойчивый тренд состояния.",
            "",
        ]
    )


def _diagnostics(cfg: ScenarioConfig, df: pd.DataFrame) -> str:
    """Считает корреляции, которые показывают согласованность deltaP с засорением."""
    data = df[["clog_level", "delta_p_kpa", "delta_p_norm_q2"]].copy()
    data["delta_p_ma_24h"] = _rolling(df, "delta_p_kpa", cfg, hours=24)
    data["delta_p_norm_ma_24h"] = _rolling(df, "delta_p_norm_q2", cfg, hours=24)
    corr_raw = data["clog_level"].corr(data["delta_p_kpa"])
    corr_ma = data["clog_level"].corr(data["delta_p_ma_24h"])
    corr_norm = data["clog_level"].corr(data["delta_p_norm_ma_24h"])
    return "\n".join(
        [
            "# Диагностика связи deltaP и clog_level",
            "",
            "| Пара | Корреляция Пирсона |",
            "|---|---:|",
            f"| `clog_level` и сырой `deltaP` | {corr_raw:.3f} |",
            f"| `clog_level` и `deltaP`, среднее за 24 ч | {corr_ma:.3f} |",
            f"| `clog_level` и `deltaP_norm`, среднее за 24 ч | {corr_norm:.3f} |",
            "",
            "Интерпретация: сырой `deltaP` должен содержать режимные колебания из-за расхода, поэтому он может быть заметно шумнее скрытого засорения. Для проверки тренда надежнее использовать 24-часовое среднее и нормированный перепад.",
            "",
        ]
    )
