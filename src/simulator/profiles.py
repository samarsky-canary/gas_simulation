from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    values = np.zeros(n)
    innovations = rng.normal(0.0, sigma, n)
    for i in range(1, n):
        values[i] = rho * values[i - 1] + innovations[i]
    return values


def generate_profiles(
    cfg: ScenarioConfig, idx: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    n = len(idx)
    k = np.arange(n)
    steps_per_day = max(int(24 * 60 / cfg.step_minutes), 1)
    steps_per_week = max(7 * steps_per_day, 1)

    q_base = cfg.q_nominal_m3h * (
        1
        + cfg.a_q * np.sin(2 * np.pi * k / steps_per_day)
        + cfg.q_weekly_amp * np.sin(2 * np.pi * k / steps_per_week)
    )
    q = q_base + _ar1(n, cfg.q_ar_rho, cfg.q_process_std_m3h, rng)

    if cfg.scenario_name == "flow_spikes":
        starts = rng.choice(n, size=max(n // (steps_per_day * 5), 1), replace=False)
        for start in starts:
            width = int(rng.integers(3, 18))
            end = min(start + width, n)
            q[start:end] += rng.uniform(0.20, 0.45) * cfg.q_nominal_m3h

    q = np.clip(q, cfg.q_min_m3h, cfg.q_max_m3h)

    p_in = (
        cfg.p_in_nominal_mpa
        + cfg.a_p_mpa * np.sin(2 * np.pi * k / steps_per_day + 0.7)
        + _ar1(n, 0.85, cfg.p_process_std_mpa, rng)
    )
    p_in = np.clip(p_in, cfg.p_min_mpa, cfg.p_max_mpa)

    t_c = (
        cfg.t_nominal_c
        + cfg.a_t_c * np.sin(2 * np.pi * k / steps_per_day - 0.3)
        + _ar1(n, 0.90, cfg.t_process_std_c, rng)
    )
    t_c = np.clip(t_c, cfg.t_min_c, cfg.t_max_c)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "q_true_m3h": q,
            "p_in_true_mpa": p_in,
            "t_true_c": t_c,
        }
    )
