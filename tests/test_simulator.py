from __future__ import annotations

import numpy as np

from src.simulator.config import ScenarioConfig
from src.simulator.runner import run_scenario


def test_run_scenario_generates_expected_columns() -> None:
    cfg = ScenarioConfig(duration_days=2, step_minutes=10, p_missing=0, p_spike=0, p_stuck=0)
    df, report = run_scenario(cfg)

    assert len(df) == 2 * 24 * 6
    assert report.pressure_order_ok
    assert report.nonnegative_dp_ok
    assert report.nonnegative_q_ok
    assert {"p_in_mpa", "p_out_mpa", "delta_p_kpa", "clog_level", "rul_analytic_h"}.issubset(df.columns)


def test_degradation_is_monotonic_without_maintenance() -> None:
    cfg = ScenarioConfig(duration_days=5, step_minutes=10, p_missing=0, p_spike=0, p_stuck=0)
    df, report = run_scenario(cfg)

    assert report.monotonic_clog_between_maintenance_ok
    assert np.all(np.diff(df["clog_level"]) >= -1e-12)


def test_maintenance_reset_reduces_clog_level() -> None:
    cfg = ScenarioConfig(
        scenario_name="maintenance_reset",
        duration_days=4,
        step_minutes=10,
        maintenance_day=2,
        c_reset=0.01,
        k_s_per_hour=0.003,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df, _ = run_scenario(cfg)
    event_idx = int(df.index[df["maintenance_event"]][0])

    assert df.loc[event_idx, "clog_level"] < df.loc[event_idx - 1, "clog_level"]
