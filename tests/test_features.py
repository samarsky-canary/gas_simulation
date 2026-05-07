from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_features, export_features
from src.simulator.config import ScenarioConfig
from src.simulator.runner import run_scenario


def test_build_features_has_required_columns() -> None:
    cfg = ScenarioConfig(duration_days=2, step_minutes=10, p_missing=0, p_spike=0, p_stuck=0)
    df, _ = run_scenario(cfg)

    features = build_features(cfg, df)

    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert len(features) == len(df)
    assert features["deltaP_norm_kPa"].notna().all()
    assert features["missing_rate_1h"].between(0, 1).all()
    assert features["time_above_warn"].ge(0).all()


def test_delta_p_slope_6h_is_positive_for_increasing_series() -> None:
    cfg = ScenarioConfig(duration_days=1, step_minutes=60, p_missing=0, p_spike=0, p_stuck=0)
    df, _ = run_scenario(cfg)
    df = df.copy()
    df["delta_p_kpa"] = np.arange(len(df), dtype=float)
    df["q_m3h"] = cfg.q_nominal_m3h
    df["quality_code"] = "good"
    df["fault_flags"] = ""

    features = build_features(cfg, df)

    assert features["deltaP_slope_6h"].dropna().tail(1).iloc[0] > 0


def test_export_features_creates_files(tmp_path) -> None:
    cfg = ScenarioConfig(duration_days=1, step_minutes=30, p_missing=0, p_spike=0, p_stuck=0)
    df, _ = run_scenario(cfg)
    features = build_features(cfg, df)

    paths = export_features(cfg, features, tmp_path)

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    exported = pd.read_parquet(paths["features_parquet"])
    assert list(exported.columns) == list(features.columns)
