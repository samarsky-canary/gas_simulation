from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig
from src.simulator.exporters import CANONICAL_DATASET_COLUMNS, export_run
from src.simulator.labels import label_run
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


def test_export_run_writes_canonical_dataset_schema(tmp_path) -> None:
    cfg = ScenarioConfig(duration_days=1, step_minutes=30, p_missing=0, p_spike=0, p_stuck=0)
    df, report = run_scenario(cfg)

    paths = export_run(cfg, df, report, tmp_path)
    dataset = pd.read_parquet(paths["dataset_parquet"])

    assert list(dataset.columns) == CANONICAL_DATASET_COLUMNS
    assert paths["dataset_csv"].exists()
    assert paths["dataset_schema"].exists()
    assert not any("_ru" in name for name in paths)


def test_analytic_rul_is_unknown_for_near_zero_observed_delta_p() -> None:
    cfg = ScenarioConfig(
        duration_days=1,
        step_minutes=30,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df = pd.DataFrame(
        {
            "q_m3h": [cfg.q_nominal_m3h, cfg.q_nominal_m3h, cfg.q_nominal_m3h * 0.01],
            "delta_p_kpa": [cfg.dp0_kpa, 0.0, cfg.dp0_kpa],
            "q_true_m3h": [cfg.q_nominal_m3h, cfg.q_nominal_m3h, cfg.q_nominal_m3h],
            "t_true_c": [cfg.t_nominal_c, cfg.t_nominal_c, cfg.t_nominal_c],
            "delta_p_true_kpa": [cfg.dp0_kpa, cfg.dp0_kpa, cfg.dp0_kpa],
            "quality_code": ["good", "good", "good"],
        }
    )

    labeled = label_run(cfg, df)

    assert np.isfinite(labeled["rul_analytic_h"].iloc[0])
    assert np.isnan(labeled["rul_analytic_h"].iloc[1])
    assert np.isnan(labeled["rul_analytic_h"].iloc[2])
