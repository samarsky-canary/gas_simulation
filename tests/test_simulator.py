from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig
from src.simulator.exporters import CANONICAL_DATASET_COLUMNS, export_run
from src.simulator.runner import run_scenario


def test_run_scenario_generates_expected_columns() -> None:
    cfg = ScenarioConfig(duration_days=2, step_minutes=10, p_missing=0, p_spike=0, p_stuck=0)
    df, report = run_scenario(cfg)

    assert len(df) == 2 * 24 * 6
    assert report.pressure_order_ok
    assert report.nonnegative_dp_ok
    assert report.nonnegative_q_ok
    assert {
        "p_in_mpa",
        "p_out_mpa",
        "delta_p_kpa",
        "clog_level",
        "rul_analytic_h",
        "spike_event",
    }.issubset(df.columns)


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


def test_scheduled_maintenance_resets_clog_level_every_n_hours() -> None:
    cfg = ScenarioConfig(
        duration_days=3,
        step_minutes=60,
        maintenance_interval_h=24,
        c_reset=0.01,
        k_s_per_hour=0.01,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df, _ = run_scenario(cfg)
    event_indices = df.index[df["maintenance_event"]].tolist()

    assert event_indices == [24, 48]
    for event_idx in event_indices:
        assert df.loc[event_idx, "clog_level"] < df.loc[event_idx - 1, "clog_level"]


def test_large_scheduled_maintenance_interval_does_not_create_event() -> None:
    cfg = ScenarioConfig(
        duration_days=3,
        step_minutes=60,
        maintenance_interval_h=20000,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df, _ = run_scenario(cfg)

    assert not df["maintenance_event"].any()


def test_export_run_writes_canonical_dataset_schema(tmp_path) -> None:
    cfg = ScenarioConfig(duration_days=1, step_minutes=30, p_missing=0, p_spike=0, p_stuck=0)
    df, report = run_scenario(cfg)

    paths = export_run(cfg, df, report, tmp_path)
    dataset = pd.read_parquet(paths["dataset_parquet"])

    assert list(dataset.columns) == CANONICAL_DATASET_COLUMNS
    assert paths["dataset_csv"].exists()
    assert paths["dataset_schema"].exists()
    assert not any("_ru" in name for name in paths)
