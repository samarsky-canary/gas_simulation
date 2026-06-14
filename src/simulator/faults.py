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
    """Преобразует истинные физические каналы в наблюдаемые измерения с шумом датчиков."""
    # Определяем количество временных точек, для которых нужно смоделировать измерения.
    n = len(profile)

    # Считаем линейный дрейф смещения датчиков давления в МПа от начала симуляции.
    drift = np.arange(n) * cfg.step_minutes / (60 * 24) * cfg.bias_drift_mpa_per_day

    # Формируем наблюдаемое входное давление: истинное значение плюс дрейф и шум датчика.
    p_in = profile["p_in_true_mpa"].to_numpy() + drift + rng.normal(0, cfg.sigma_p_mpa, n)

    # Формируем наблюдаемое выходное давление: истинное значение плюс тот же дрейф и шум датчика.
    p_out = physics["p_out_true_mpa"].to_numpy() + drift + rng.normal(0, cfg.sigma_p_mpa, n)

    # Ограничиваем входное давление допустимым физическим диапазоном.
    p_in = np.clip(p_in, cfg.p_min_mpa, cfg.p_max_mpa)

    # Ограничиваем выходное давление снизу нулем и сверху текущим входным давлением.
    p_out = np.clip(p_out, 0.0, p_in)

    # Формируем наблюдаемый расход: истинный расход умножается на относительный шум датчика.
    q = profile["q_true_m3h"].to_numpy() * (1 + rng.normal(0, cfg.sigma_q_rel, n))

    # Ограничиваем наблюдаемый расход неотрицательными значениями и расширенным верхним пределом.
    q = np.clip(q, 0.0, cfg.q_max_m3h * 1.2)

    # Формируем наблюдаемую температуру: истинная температура плюс абсолютный шум датчика.
    t_c = profile["t_true_c"].to_numpy() + rng.normal(0, cfg.sigma_t_abs_c, n)

    # Перепад вычисляем как разность наблюдаемых абсолютных давлений.
    # Переводим разность давлений из МПа в кПа и запрещаем отрицательный перепад.
    delta_p = np.maximum(0.0, 1000.0 * (p_in - p_out))

    # Возвращаем наблюдаемую телеметрию как таблицу для последующей инъекции отказов.
    return pd.DataFrame(
        {
            # Наблюдаемое входное давление, МПа.
            "p_in_mpa": p_in,
            # Наблюдаемое выходное давление, МПа.
            "p_out_mpa": p_out,
            # Рассчитанный перепад давления, кПа.
            "delta_p_kpa": delta_p,
            # Наблюдаемый расход газа, м3/ч.
            "q_m3h": q,
            # Наблюдаемая температура газа, градусы Цельсия.
            "t_c": t_c,
        }
    )


def inject_faults(
    cfg: ScenarioConfig, observed: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Добавляет пропуски, выбросы и залипания датчиков в наблюдаемую телеметрию."""
    df = observed.copy()
    n = len(df)
    spike_event = np.zeros(n, dtype=bool)

    for channel in OBSERVED_CHANNELS:
        # Пропуск имитирует потерю телеметрии по конкретному каналу.
        missing = rng.random(n) < cfg.p_missing
        if missing.any():
            df.loc[missing, channel] = np.nan

        # Выброс имитирует короткий нехарактерный скачок измерения.
        spikes = rng.random(n) < cfg.p_spike
        if spikes.any():
            spike_event |= spikes
            scale = _spike_scale(channel, cfg)
            values = rng.choice([-1.0, 1.0], size=spikes.sum()) * rng.uniform(
                0.8 * scale, 1.5 * scale, spikes.sum()
            )
            df.loc[spikes, channel] = df.loc[spikes, channel].to_numpy() + values

        starts = np.flatnonzero(rng.random(n) < cfg.p_stuck)
        for start in starts:
            # Залипание удерживает значение датчика постоянным на случайном интервале.
            if start == 0 or pd.isna(df.at[start - 1, channel]):
                continue
            length = int(rng.integers(cfg.stuck_min_steps, cfg.stuck_max_steps + 1))
            end = min(start + length, n)
            df.loc[start:end - 1, channel] = df.at[start - 1, channel]

    df["p_in_mpa"] = df["p_in_mpa"].clip(lower=cfg.p_min_mpa, upper=cfg.p_max_mpa)
    df["q_m3h"] = df["q_m3h"].clip(lower=0, upper=cfg.q_max_m3h * 1.2)
    df["t_c"] = df["t_c"].clip(lower=cfg.t_min_c - 10, upper=cfg.t_max_c + 10)
    df["p_out_mpa"] = np.minimum(df["p_out_mpa"].clip(lower=0), df["p_in_mpa"])

    df["delta_p_kpa"] = np.maximum(0.0, 1000.0 * (df["p_in_mpa"] - df["p_out_mpa"]))
    df["spike_event"] = spike_event

    return df


def _spike_scale(channel: str, cfg: ScenarioConfig) -> float:
    """Возвращает характерный масштаб выброса для каждого типа датчика."""
    if channel in {"p_in_mpa", "p_out_mpa"}:
        return 0.008
    if channel == "q_m3h":
        return 0.18 * cfg.q_nominal_m3h
    return 3.0
