from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


def compute_physics(
    cfg: ScenarioConfig, profile: pd.DataFrame, state: pd.DataFrame
) -> pd.DataFrame:
    """По режимам и засорению рассчитывает истинный перепад давления и выходное давление."""
    # Берем истинный расход из профиля режимов работы.
    q = profile["q_true_m3h"].to_numpy()

    # Берем истинное входное давление перед фильтром.
    p_in = profile["p_in_true_mpa"].to_numpy()

    # Берем истинную температуру газа.
    t_c = profile["t_true_c"].to_numpy()

    # Берем скрытый уровень засорения, рассчитанный моделью деградации.
    clog = state["clog_level"].to_numpy()

    # Считаем множитель расхода: чем выше расход относительно номинала, тем больше перепад давления.
    flow_factor = np.maximum(q / cfg.q_nominal_m3h, 1e-6) ** cfg.alpha_flow

    # Считаем температурную поправку: температура влияет на вязкость и, через нее, на перепад.
    temp_factor = np.exp(cfg.k_mu_per_c * (cfg.t_nominal_c - t_c))

    # Считаем множитель сопротивления: засорение увеличивает гидравлическое сопротивление фильтра.
    resistance_factor = 1.0 + cfg.k_c * (clog**cfg.beta)

    # Считаем истинный перепад давления как базовый перепад, умноженный на все физические факторы.
    delta_p_true = cfg.dp0_kpa * flow_factor * temp_factor * resistance_factor

    # Защищаем модель от отрицательного перепада давления.
    delta_p_true = np.maximum(delta_p_true, 0.0)

    # Считаем истинное выходное давление после фильтра: входное давление минус перепад.
    p_out_true = np.maximum(0.0, p_in - delta_p_true / 1000.0)

    # Возвращаем рассчитанные физические величины как таблицу для объединения с остальными данными.
    return pd.DataFrame(
        {
            # Полный множитель сопротивления фильтра из-за засорения.
            "resistance_factor": resistance_factor,
            # Истинный перепад давления на фильтре, кПа.
            "delta_p_true_kpa": delta_p_true,
            # Истинное давление после фильтра, МПа.
            "p_out_true_mpa": p_out_true,
        }
    )
