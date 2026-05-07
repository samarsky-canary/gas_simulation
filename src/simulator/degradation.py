from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


def simulate_degradation(cfg: ScenarioConfig, profile: pd.DataFrame) -> pd.DataFrame:
    q = profile["q_true_m3h"].to_numpy()
    n = len(q)
    dt_h = cfg.step_minutes / 60.0
    clog = np.empty(n)
    maintenance = np.zeros(n, dtype=bool)
    clog[0] = cfg.c0

    maintenance_idx = None
    if cfg.maintenance_day is not None:
        maintenance_idx = int(cfg.maintenance_day * 24 * 60 / cfg.step_minutes)
        maintenance_idx = min(max(maintenance_idx, 1), n - 1)

    for i in range(1, n):
        if maintenance_idx is not None and i == maintenance_idx:
            clog[i] = min(cfg.c_reset, clog[i - 1])
            maintenance[i] = True
            continue
        load = (max(q[i - 1], 0.0) / cfg.q_nominal_m3h) ** cfg.gamma_load
        clog[i] = np.clip(clog[i - 1] + cfg.k_s_per_hour * load * dt_h, 0.0, 1.0)

    return pd.DataFrame({"clog_level": clog, "maintenance_event": maintenance})
