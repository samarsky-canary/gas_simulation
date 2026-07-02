from __future__ import annotations

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


def _aggregate_rul_source_plot_data(decisions: pd.DataFrame) -> pd.DataFrame:
    """Сжимает частый ряд до часовых точек для интерактивной визуализации."""
    if decisions.empty:
        return decisions.copy()

    data = decisions.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp").sort_index()
    numeric_columns = [
        "RUL_ml_h",
        "RUL_analytic_h",
        "RUL_fused_h",
        "confidence_data",
        "confidence_total",
        "confidence_consistency",
    ]
    numeric_columns = [column for column in numeric_columns if column in data.columns]
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


def _rolling(df: pd.DataFrame, column: str, cfg: ScenarioConfig, hours: int) -> pd.Series:
    """Считает скользящее среднее по заданному числу часов."""
    window = max(int(hours * 60 / cfg.step_minutes), 1)
    return df[column].rolling(window, min_periods=max(window // 4, 1)).mean()


def _stable_state_codes(cfg: ScenarioConfig, df: pd.DataFrame) -> pd.Series:
    """Строит устойчивое состояние по сглаженному нормированному перепаду."""
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
