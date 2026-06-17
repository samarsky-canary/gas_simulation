from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


def label_run(cfg: ScenarioConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет диагностические состояния и RUL."""
    out = df.copy()
    true_dp_norm = _true_delta_p_norm(cfg, out)
    obs_dp_norm = _observed_delta_p_norm(cfg, out)

    out["delta_p_norm_q2"] = obs_dp_norm
    out["state_obs"] = _states(obs_dp_norm, cfg)
    out.loc[out["quality_code"] != "good", "state_obs"] = "unknown"
    out["rul_oracle_h"] = _rul_oracle(true_dp_norm, cfg)
    out["rul_analytic_h"] = _rul_analytic(cfg, out)
    out["is_rul_unknown"] = out["rul_oracle_h"].isna()
    return out


def _states(values: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    """Классифицирует перепад давления в normal/warning/critical/unknown."""
    return np.where(
        np.isnan(values),
        "unknown",
        np.where(values < cfg.dp_warn_kpa, "normal", np.where(values < cfg.dp_crit_kpa, "warning", "critical")),
    )


def _rul_oracle(delta_p_true: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    """Считает истинный RUL как время до ближайшего будущего достижения критического порога."""
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
    """Оценивает RUL по наблюдаемым данным и калиброванным параметрам фильтра."""
    q = df["q_m3h"].to_numpy(dtype=float)
    delta_p_norm = _observed_delta_p_norm(cfg, df).to_numpy(dtype=float)
    c_crit_raw = (cfg.dp_crit_kpa / cfg.dp0_kpa - 1.0) / cfg.k_c
    c_crit = float(np.clip(c_crit_raw, 0.0, 1.0)) ** (1.0 / cfg.beta)
    resistance = np.maximum(delta_p_norm / cfg.dp0_kpa - 1.0, 0.0)
    clog = np.clip(resistance / cfg.k_c, 0.0, 1.0) ** (1.0 / cfg.beta)
    load = np.maximum(q / cfg.q_nominal_m3h, 1e-6) ** cfg.gamma_load
    rate = np.maximum(cfg.k_s_per_hour * load, 1e-9)
    result = np.maximum(c_crit - clog, 0.0) / rate
    invalid = np.isnan(q) | np.isnan(delta_p_norm)
    result[invalid] = np.nan
    return result


def _true_delta_p_norm(cfg: ScenarioConfig, df: pd.DataFrame) -> np.ndarray:
    """Нормирует clean-перепад, чтобы состояние отражало засорение, а не режим расхода."""
    q = df["q_true_m3h"].to_numpy()
    t_c = df["t_true_c"].to_numpy()
    flow_factor = np.maximum(q / cfg.q_nominal_m3h, 1e-6) ** cfg.alpha_flow
    temp_factor = np.exp(cfg.k_mu_per_c * (cfg.t_nominal_c - t_c))
    return df["delta_p_true_kpa"].to_numpy() / np.maximum(flow_factor * temp_factor, 1e-3)


def _observed_delta_p_norm(cfg: ScenarioConfig, df: pd.DataFrame) -> np.ndarray:
    """Нормирует наблюдаемый перепад по расходу."""
    flow_factor = (df["q_m3h"] / cfg.q_nominal_m3h) ** cfg.alpha_flow
    return df["delta_p_kpa"] / np.maximum(flow_factor, 1e-3)
