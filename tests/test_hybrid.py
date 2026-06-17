from __future__ import annotations

import pandas as pd

from src.hybrid import (
    build_hybrid_decisions,
    export_hybrid_decisions,
)
from src.simulator.config import ScenarioConfig


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h"),
            "filter_id": ["F-001"] * 4,
            "scenario": ["test"] * 4,
            "P_in_MPa": [0.60, 0.60, 0.60, 0.60],
            "P_out_MPa": [0.598, 0.594, 0.589, 0.596],
            "deltaP_kPa": [2.0, 6.0, 11.0, 4.0],
            "Q_m3h": [600.0] * 4,
            "T_C": [15.0] * 4,
            "clog_level": [0.1, 0.5, 0.9, 0.2],
            "deltaP_norm_kPa": [2.0, 6.0, 11.0, 4.0],
            "state": ["normal", "warning", "critical", "unknown"],
            "RUL_oracle_h": [2000.0, 1000.0, 100.0, 1800.0],
            "RUL_analytic_h": [2100.0, 1100.0, 120.0, 1900.0],
            "quality_code": ["good", "good", "good", "missing"],
        }
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h"),
            "missing_rate_1h": [0.0, 0.0, 0.0, 0.5],
            "time_above_warn": [0.0, 1.0, 2.0, 0.0],
        }
    )


def _ml_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h"),
            "RUL_pred_h": [2050.0, 1000.0, 110.0, 1700.0],
        }
    )


def test_build_hybrid_decisions_actions() -> None:
    cfg = ScenarioConfig(dp_warn_kpa=5.0, dp_crit_kpa=10.0)

    decisions = build_hybrid_decisions(cfg, _dataset(), _features(), _ml_predictions())

    assert decisions["action"].tolist() == [
        "monitor",
        "planned_maintenance",
        "urgent_maintenance",
        "sensor_check",
    ]
    assert decisions["priority"].tolist() == ["P3", "P2", "P1", "P0"]
    assert decisions["RUL_fused_h"].notna().all()
    assert decisions["confidence_total"].between(0, 1).all()
    assert decisions["rule_trace"].str.contains("R-").all()
    assert {"deltaP_roll_mean_1h", "deltaP_slope_6h", "time_above_warn"}.issubset(decisions.columns)


def test_export_hybrid_decisions_creates_files(tmp_path) -> None:
    cfg = ScenarioConfig()
    decisions = build_hybrid_decisions(cfg, _dataset(), _features(), _ml_predictions())

    paths = export_hybrid_decisions(decisions, tmp_path)

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert not any("_ru" in name for name in paths)
    packages = paths["hybrid_decision_packages_jsonl"].read_text(encoding="utf-8")
    assert '"rule_trace"' in packages
