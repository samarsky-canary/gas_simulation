from __future__ import annotations

import pandas as pd

from src.simulator.config import ScenarioConfig


def make_index(cfg: ScenarioConfig) -> pd.DatetimeIndex:
    periods = int(cfg.duration_days * 24 * 60 / cfg.step_minutes)
    return pd.date_range(cfg.start_time, periods=periods, freq=f"{cfg.step_minutes}min")
