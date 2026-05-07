from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


def label_run(cfg: ScenarioConfig, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    true_dp = out["delta_p_true_kpa"].to_numpy()
    obs_dp = out["delta_p_kpa"].to_numpy()

    out["state_true"] = _states(true_dp, cfg)
    out["state_obs"] = _states(obs_dp, cfg)
    out.loc[out["quality_code"] != "good", "state_obs"] = "unknown"
    out["alarm_flag"] = out["state_obs"].isin(["warning", "critical"])
    out["rul_oracle_h"] = _rul_oracle(true_dp, cfg)
    out["rul_analytic_h"] = _rul_analytic(cfg, out)
    out["is_censored"] = out["rul_oracle_h"].isna()
    out["delta_p_norm_q2"] = out["delta_p_kpa"] / np.maximum(
        (out["q_m3h"] / cfg.q_nominal_m3h) ** 2, 1e-3
    )
    out["rule_health_index"] = (out["delta_p_kpa"] / cfg.dp_crit_kpa).clip(lower=0, upper=1)
    return out


def _states(values: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    return np.where(
        np.isnan(values),
        "unknown",
        np.where(values < cfg.dp_warn_kpa, "normal", np.where(values < cfg.dp_crit_kpa, "warning", "critical")),
    )


def _rul_oracle(delta_p_true: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    n = len(delta_p_true)
    dt_h = cfg.step_minutes / 60.0
    result = np.full(n, np.nan)
    next_hit = np.inf
    for i in range(n - 1, -1, -1):
        if delta_p_true[i] >= cfg.dp_crit_kpa:
            next_hit = i
        if np.isfinite(next_hit):
            result[i] = (next_hit - i) * dt_h
    return result


def _rul_analytic(cfg: ScenarioConfig, df: pd.DataFrame) -> np.ndarray:
    q = df["q_true_m3h"].to_numpy()
    t_c = df["t_true_c"].to_numpy()
    clog = df["clog_level"].to_numpy()
    flow_factor = np.maximum(q / cfg.q_nominal_m3h, 1e-6) ** cfg.alpha_flow
    temp_factor = np.exp(cfg.k_mu_per_c * (cfg.t_nominal_c - t_c))
    denom = cfg.dp0_kpa * flow_factor * temp_factor
    c_crit = ((cfg.dp_crit_kpa / denom - 1.0) / cfg.k_c).clip(min=0.0, max=1.0)
    c_crit = c_crit ** (1.0 / cfg.beta)
    load = np.maximum(q / cfg.q_nominal_m3h, 1e-6) ** cfg.gamma_load
    rate = np.maximum(cfg.k_s_per_hour * load, 1e-9)
    return np.maximum(c_crit - clog, 0.0) / rate
