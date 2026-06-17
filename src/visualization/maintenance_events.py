from __future__ import annotations

import pandas as pd

from src.simulator.config import ScenarioConfig


RUL_SOURCES = [
    ("ML", "RUL_ml_h"),
    ("Аналитика", "RUL_analytic_h"),
    ("Гибрид", "RUL_fused_h"),
]


def build_maintenance_event_table(
    cfg: ScenarioConfig, decisions: pd.DataFrame
) -> pd.DataFrame:
    """Строит таблицу первых устойчивых событий планового и срочного обслуживания."""
    return pd.DataFrame(
        [
            {
                "Название": label,
                "Плановое обслуживание": _format_event_time(
                    _stable_threshold_event_time(
                        decisions,
                        column,
                        threshold_h=cfg.planned_maintenance_rul_h,
                        stable_h=cfg.stable_degraded_rul_h,
                    )
                ),
                "Срочное обслуживание": _format_event_time(
                    _stable_threshold_event_time(
                        decisions,
                        column,
                        threshold_h=cfg.urgent_maintenance_rul_h,
                        stable_h=cfg.stable_degraded_rul_h,
                    )
                ),
            }
            for label, column in RUL_SOURCES
        ]
    )


def _stable_threshold_event_time(
    decisions: pd.DataFrame,
    rul_column: str,
    *,
    threshold_h: float,
    stable_h: float,
) -> pd.Timestamp | None:
    """Возвращает время, когда RUL устойчиво ниже threshold + stable_h заданное число часов."""
    if rul_column not in decisions.columns or decisions.empty:
        return None

    data = decisions[["timestamp", rul_column]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")
    detection_threshold = float(threshold_h) + float(stable_h)

    window_start: pd.Timestamp | None = None
    previous_timestamp: pd.Timestamp | None = None
    for row in data.itertuples(index=False):
        timestamp = row.timestamp
        value = getattr(row, rul_column)
        below = pd.notna(value) and float(value) < detection_threshold

        if not below:
            window_start = None
            previous_timestamp = timestamp
            continue

        if window_start is None:
            window_start = timestamp
        elif previous_timestamp is not None:
            expected_step = timestamp - previous_timestamp
            if expected_step > pd.Timedelta(hours=float(stable_h)):
                window_start = timestamp

        if timestamp - window_start >= pd.Timedelta(hours=float(stable_h)):
            return timestamp
        previous_timestamp = timestamp
    return None


def _format_event_time(timestamp: pd.Timestamp | None) -> str:
    if timestamp is None:
        return "не наступило"
    return timestamp.strftime("%Y-%m-%d %H:%M")
