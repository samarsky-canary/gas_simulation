from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules import apply_rule_baseline, export_rule_baseline
from src.simulator.config import ScenarioConfig


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["r1"] * 5,
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="1h"),
            "filter_id": ["F-001"] * 5,
            "scenario_id": ["test"] * 5,
            "p_in_mpa": [0.6] * 5,
            "p_out_mpa": [0.599, 0.595, 0.591, 0.590, np.nan],
            "delta_p_kpa": [1.0, 5.0, 9.0, 10.0, np.nan],
            "q_m3h": [600.0] * 5,
            "t_c": [15.0] * 5,
            "rul_analytic_h": [100.0, 70.0, 11.0, 50.0, 100.0],
            "quality_code": ["good", "good", "good", "good", "missing"],
            "state_obs": ["normal", "warning", "warning", "critical", "unknown"],
            "state_true": ["normal", "warning", "warning", "critical", "unknown"],
            "rul_oracle_h": [100.0, 70.0, 11.0, 0.0, np.nan],
        }
    )


def test_apply_rule_baseline_thresholds() -> None:
    cfg = ScenarioConfig(dp_warn_kpa=5.0, dp_crit_kpa=10.0)

    baseline = apply_rule_baseline(cfg, _base_df())

    assert baseline["rule_state"].tolist() == [
        "normal",
        "warning",
        "warning",
        "critical",
        "unknown",
    ]
    assert baseline["rule_recommendation"].tolist() == [
        "continue_monitoring",
        "planned_maintenance",
        "urgent_maintenance",
        "urgent_maintenance",
        "inspect_sensor_data",
    ]


def test_export_rule_baseline_creates_files(tmp_path) -> None:
    cfg = ScenarioConfig()
    baseline = apply_rule_baseline(cfg, _base_df())

    paths = export_rule_baseline(cfg, baseline, tmp_path)

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
