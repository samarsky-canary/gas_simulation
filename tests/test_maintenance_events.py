from __future__ import annotations

import pandas as pd

from src.simulator.config import ScenarioConfig
from src.visualization.maintenance_events import build_maintenance_event_table


def test_maintenance_event_table_uses_stable_threshold_with_time_buffer() -> None:
    cfg = ScenarioConfig(
        planned_maintenance_rul_h=600,
        urgent_maintenance_rul_h=100,
        stable_degraded_rul_h=4,
    )
    decisions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:00", periods=8, freq="1h"),
            "RUL_ml_h": [700, 605, 603, 602, 601, 600, 599, 598],
            "RUL_analytic_h": [700, 650, 610, 603, 602, 601, 600, 599],
            "RUL_fused_h": [700, 610, 603, 602, 50, 49, 48, 47],
        }
    )

    table = build_maintenance_event_table(cfg, decisions)

    assert table.to_dict("records") == [
        {
            "Название": "ML",
            "Плановое обслуживание": "2026-01-01 06:00",
            "Срочное обслуживание": "не наступило",
        },
        {
            "Название": "Аналитика",
            "Плановое обслуживание": "2026-01-01 07:00",
            "Срочное обслуживание": "не наступило",
        },
        {
            "Название": "Гибрид",
            "Плановое обслуживание": "2026-01-01 06:00",
            "Срочное обслуживание": "не наступило",
        },
    ]


def test_maintenance_event_table_uses_hourly_rul_signal() -> None:
    cfg = ScenarioConfig(
        planned_maintenance_rul_h=720,
        urgent_maintenance_rul_h=100,
        stable_degraded_rul_h=4,
        step_minutes=5,
    )
    timestamps = pd.date_range("2026-03-12 00:00", periods=60, freq="5min")
    decisions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "RUL_ml_h": [1000.0] * len(timestamps),
            "RUL_analytic_h": (
                [700.0] * 11
                + [800.0]
                + [700.0] * 11
                + [800.0]
                + [700.0] * 36
            ),
            "RUL_fused_h": [1000.0] * len(timestamps),
        }
    )

    table = build_maintenance_event_table(cfg, decisions)

    assert table.loc[table["Название"].eq("Аналитика"), "Плановое обслуживание"].item() == (
        "2026-03-12 04:00"
    )
