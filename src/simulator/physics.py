from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


def compute_physics(
    cfg: ScenarioConfig, profile: pd.DataFrame, state: pd.DataFrame
) -> pd.DataFrame:
    """По режимам и засорению рассчитывает истинный перепад давления и выходное давление."""
    q = profile["q_true_m3h"].to_numpy()
    p_in = profile["p_in_true_mpa"].to_numpy()
    t_c = profile["t_true_c"].to_numpy()
    clog = state["clog_level"].to_numpy()

    # Перепад растет с расходом, температурной поправкой и сопротивлением засоренного фильтра.
    flow_factor = np.maximum(q / cfg.q_nominal_m3h, 1e-6) ** cfg.alpha_flow
    temp_factor = np.exp(cfg.k_mu_per_c * (cfg.t_nominal_c - t_c))
    resistance_factor = 1.0 + cfg.k_c * (clog**cfg.beta)
    delta_p_true = cfg.dp0_kpa * flow_factor * temp_factor * resistance_factor
    delta_p_true = np.maximum(delta_p_true, 0.0)
    p_out_true = np.maximum(0.0, p_in - delta_p_true / 1000.0)

    return pd.DataFrame(
        {
            "resistance_factor": resistance_factor,
            "delta_p_true_kpa": delta_p_true,
            "p_out_true_mpa": p_out_true,
        }
    )
