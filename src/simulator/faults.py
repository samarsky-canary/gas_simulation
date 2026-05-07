from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


OBSERVED_CHANNELS = ["p_in_mpa", "p_out_mpa", "q_m3h", "t_c"]


def apply_sensor_model(
    cfg: ScenarioConfig,
    profile: pd.DataFrame,
    physics: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(profile)
    drift = np.arange(n) * cfg.step_minutes / (60 * 24) * cfg.bias_drift_mpa_per_day

    p_in = profile["p_in_true_mpa"].to_numpy() + drift + rng.normal(0, cfg.sigma_p_mpa, n)
    p_out = physics["p_out_true_mpa"].to_numpy() + drift + rng.normal(0, cfg.sigma_p_mpa, n)
    p_in = np.clip(p_in, cfg.p_min_mpa, cfg.p_max_mpa)
    p_out = np.clip(p_out, 0.0, p_in)

    q = profile["q_true_m3h"].to_numpy() * (1 + rng.normal(0, cfg.sigma_q_rel, n))
    q = np.clip(q, 0.0, cfg.q_max_m3h * 1.2)
    t_c = profile["t_true_c"].to_numpy() + rng.normal(0, cfg.sigma_t_abs_c, n)

    if cfg.use_dp_sensor:
        delta_p = np.maximum(
            0.0, physics["delta_p_true_kpa"].to_numpy() + rng.normal(0, cfg.sigma_dp_kpa, n)
        )
        source = "sensor"
    else:
        delta_p = np.maximum(0.0, 1000.0 * (p_in - p_out))
        source = "calc"

    return pd.DataFrame(
        {
            "p_in_mpa": p_in,
            "p_out_mpa": p_out,
            "delta_p_kpa": delta_p,
            "q_m3h": q,
            "t_c": t_c,
            "delta_p_source": source,
            "fault_flags": "",
        }
    )


def inject_faults(
    cfg: ScenarioConfig, observed: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    df = observed.copy()
    n = len(df)
    flags: list[set[str]] = [set() for _ in range(n)]

    for channel in OBSERVED_CHANNELS:
        missing = rng.random(n) < cfg.p_missing
        if missing.any():
            df.loc[missing, channel] = np.nan
            for i in np.flatnonzero(missing):
                flags[i].add(f"missing:{channel}")

        spikes = rng.random(n) < cfg.p_spike
        if spikes.any():
            scale = _spike_scale(channel, cfg)
            values = rng.choice([-1.0, 1.0], size=spikes.sum()) * rng.uniform(
                0.8 * scale, 1.5 * scale, spikes.sum()
            )
            df.loc[spikes, channel] = df.loc[spikes, channel].to_numpy() + values
            for i in np.flatnonzero(spikes):
                flags[i].add(f"spike:{channel}")

        starts = np.flatnonzero(rng.random(n) < cfg.p_stuck)
        for start in starts:
            if start == 0 or pd.isna(df.at[start - 1, channel]):
                continue
            length = int(rng.integers(cfg.stuck_min_steps, cfg.stuck_max_steps + 1))
            end = min(start + length, n)
            df.loc[start:end - 1, channel] = df.at[start - 1, channel]
            for i in range(start, end):
                flags[i].add(f"stuck:{channel}")

    df["p_in_mpa"] = df["p_in_mpa"].clip(lower=cfg.p_min_mpa, upper=cfg.p_max_mpa)
    df["q_m3h"] = df["q_m3h"].clip(lower=0, upper=cfg.q_max_m3h * 1.2)
    df["t_c"] = df["t_c"].clip(lower=cfg.t_min_c - 10, upper=cfg.t_max_c + 10)
    df["p_out_mpa"] = np.minimum(df["p_out_mpa"].clip(lower=0), df["p_in_mpa"])

    if cfg.use_dp_sensor:
        sensor_missing = df["delta_p_kpa"].isna()
        df.loc[~sensor_missing, "delta_p_kpa"] = df.loc[~sensor_missing, "delta_p_kpa"].clip(lower=0)
    else:
        df["delta_p_kpa"] = np.maximum(0.0, 1000.0 * (df["p_in_mpa"] - df["p_out_mpa"]))

    df["fault_flags"] = [";".join(sorted(item)) if item else "" for item in flags]
    return df


def _spike_scale(channel: str, cfg: ScenarioConfig) -> float:
    if channel in {"p_in_mpa", "p_out_mpa"}:
        return 0.008
    if channel == "q_m3h":
        return 0.18 * cfg.q_nominal_m3h
    return 3.0
