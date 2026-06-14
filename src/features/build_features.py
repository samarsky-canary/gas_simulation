from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


FEATURE_COLUMNS = [
    "deltaP_norm_kPa",
    "deltaP_roll_mean_1h",
    "deltaP_roll_std_1h",
    "deltaP_slope_6h",
    "deltaP_roll_mean_24h",
    "deltaP_slope_24h",
    "deltaP_slope_72h",
    "Q_roll_mean_1h",
    "missing_rate_1h",
    "time_above_warn",
    "elapsed_hours",
    "hours_since_maintenance",
    "cumulative_load_h",
]

FEATURE_EXPORT_COLUMNS = [
    "run_id",
    "timestamp",
    "filter_id",
    "scenario_id",
    *FEATURE_COLUMNS,
    "state_obs",
    "state_true",
    "rul_oracle_h",
    "rul_analytic_h",
    "is_rul_unknown",
]

FEATURE_DESCRIPTIONS = [
    {
        "name": "deltaP_norm_kPa",
        "meaning": "Перепад давления, очищенный от влияния расхода.",
        "formula": "deltaP_kPa / max((Q_m3h / Q_nominal)^alpha_flow, eps)",
        "use": "Главный признак деградации для правил и baseline-моделей.",
    },
    {
        "name": "deltaP_roll_mean_1h",
        "meaning": "Средний наблюдаемый перепад за последний час.",
        "formula": "rolling_mean(deltaP_kPa, 1h)",
        "use": "Сглаживает режимный шум и выбросы.",
    },
    {
        "name": "deltaP_roll_std_1h",
        "meaning": "Нестабильность перепада за последний час.",
        "formula": "rolling_std(deltaP_kPa, 1h)",
        "use": "Помогает отличать устойчивый рост от нестабильного режима или сбоев датчиков.",
    },
    {
        "name": "deltaP_slope_6h",
        "meaning": "Скорость роста нормированного перепада на окне 6 часов.",
        "formula": "OLS slope(deltaP_norm_kPa ~ time_hours, 6h)",
        "use": "Трендовый индикатор роста сопротивления фильтра без влияния режима расхода.",
    },
    {
        "name": "Q_roll_mean_1h",
        "meaning": "Средний расход за последний час.",
        "formula": "rolling_mean(Q_m3h, 1h)",
        "use": "Контекст режима работы фильтра.",
    },
    {
        "name": "deltaP_roll_mean_24h",
        "meaning": "Средний нормированный перепад за последние 24 часа.",
        "formula": "rolling_mean(deltaP_norm_kPa, 24h)",
        "use": "Долгосрочный уровень сопротивления фильтра.",
    },
    {
        "name": "deltaP_slope_24h",
        "meaning": "Скорость роста нормированного перепада за 24 часа.",
        "formula": "OLS slope(deltaP_norm_kPa ~ time_hours, 24h)",
        "use": "Среднесрочный наблюдаемый темп деградации.",
    },
    {
        "name": "deltaP_slope_72h",
        "meaning": "Скорость роста нормированного перепада за 72 часа.",
        "formula": "OLS slope(deltaP_norm_kPa ~ time_hours, 72h)",
        "use": "Устойчивый долгосрочный тренд деградации.",
    },
    {
        "name": "missing_rate_1h",
        "meaning": "Доля строк с пропусками за последний час.",
        "formula": "rolling_mean(row_has_missing, 1h)",
        "use": "Признак качества данных и фильтр для обучения.",
    },
    {
        "name": "time_above_warn",
        "meaning": "Накопленное время выше warning-порога с начала участка после обслуживания.",
        "formula": "cumulative_sum(deltaP_kPa >= deltaP_warn) * step_hours, reset on maintenance",
        "use": "Интерпретация правил и накопленной нагрузки в тревожной зоне.",
    },
    {
        "name": "elapsed_hours",
        "meaning": "Время от начала текущего прогона.",
        "formula": "row_index * step_hours",
        "use": "Возраст наблюдаемой траектории без использования будущих данных.",
    },
    {
        "name": "hours_since_maintenance",
        "meaning": "Время после последнего обслуживания.",
        "formula": "cumulative hours with reset on maintenance_event",
        "use": "Возраст текущего цикла эксплуатации фильтра.",
    },
    {
        "name": "cumulative_load_h",
        "meaning": "Накопленная нормированная нагрузка по наблюдаемому расходу.",
        "formula": "cumsum(max(Q / Q_nominal, 0)) * step_hours",
        "use": "Наблюдаемый интегральный эквивалент наработки.",
    },
]


def build_features(cfg: ScenarioConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Строит табличные признаки для rule-интерпретации и baseline ML-моделей."""
    features = df.copy()
    one_hour = _window_steps(cfg, hours=1)
    six_hours = _window_steps(cfg, hours=6)
    twenty_four_hours = _window_steps(cfg, hours=24)
    seventy_two_hours = _window_steps(cfg, hours=72)
    dt_h = cfg.step_minutes / 60.0

    # Нормировка на расход отделяет рост сопротивления фильтра от режима потока.
    flow_factor = np.maximum((features["q_m3h"] / cfg.q_nominal_m3h) ** cfg.alpha_flow, 1e-3)
    features["deltaP_norm_kPa"] = features["delta_p_kpa"] / flow_factor
    features["deltaP_roll_mean_1h"] = (
        features["delta_p_kpa"].rolling(one_hour, min_periods=1).mean()
    )
    features["deltaP_roll_std_1h"] = (
        features["delta_p_kpa"].rolling(one_hour, min_periods=1).std().fillna(0.0)
    )
    features["deltaP_slope_6h"] = features["deltaP_norm_kPa"].rolling(
        six_hours, min_periods=max(3, six_hours // 3)
    ).apply(lambda values: _slope(values, dt_h), raw=True)
    features["deltaP_roll_mean_24h"] = features["deltaP_norm_kPa"].rolling(
        twenty_four_hours, min_periods=1
    ).mean()
    features["deltaP_slope_24h"] = features["deltaP_norm_kPa"].rolling(
        twenty_four_hours, min_periods=min(3, twenty_four_hours)
    ).apply(lambda values: _slope(values, dt_h), raw=True)
    features["deltaP_slope_72h"] = features["deltaP_norm_kPa"].rolling(
        seventy_two_hours, min_periods=min(3, seventy_two_hours)
    ).apply(lambda values: _slope(values, dt_h), raw=True)
    features["Q_roll_mean_1h"] = features["q_m3h"].rolling(one_hour, min_periods=1).mean()

    # Признак качества показывает, насколько надежны данные в текущем часовом окне.
    missing_row = (
        features[["p_in_mpa", "p_out_mpa", "delta_p_kpa", "q_m3h", "t_c"]]
        .isna()
        .any(axis=1)
        | features["quality_code"].eq("missing")
    )
    features["missing_rate_1h"] = missing_row.rolling(one_hour, min_periods=1).mean()

    # Накопленное время выше warning сбрасывается после обслуживания фильтра.
    above_warn = features["delta_p_kpa"].ge(cfg.dp_warn_kpa).fillna(False)
    segment = features["maintenance_event"].fillna(False).astype(bool).cumsum()
    features["time_above_warn"] = above_warn.groupby(segment).cumsum() * dt_h
    features["elapsed_hours"] = np.arange(len(features), dtype=float) * dt_h
    features["hours_since_maintenance"] = features.groupby(segment).cumcount() * dt_h
    normalized_load = (
        features["q_m3h"].ffill().bfill().clip(lower=0.0) / cfg.q_nominal_m3h
    )
    features["cumulative_load_h"] = normalized_load.groupby(segment).cumsum() * dt_h

    return features[FEATURE_EXPORT_COLUMNS]


def export_features(
    cfg: ScenarioConfig,
    features: pd.DataFrame,
    output_dir: Path,
    *,
    export_csv: bool = True,
) -> dict[str, Path]:
    """Экспортирует признаки в Parquet, опциональный CSV и описание формул."""
    paths = {
        "features_parquet": output_dir / "features.parquet",
        "feature_description": output_dir / "feature_description.md",
    }
    if export_csv:
        paths["features_csv"] = output_dir / "features.csv"
        features.to_csv(paths["features_csv"], index=False, encoding="utf-8")
    features.to_parquet(paths["features_parquet"], index=False)
    paths["feature_description"].write_text(_feature_description(cfg), encoding="utf-8")
    return paths


def _window_steps(cfg: ScenarioConfig, hours: int) -> int:
    """Переводит длительность окна в часах в количество строк временного ряда."""
    return max(int(hours * 60 / cfg.step_minutes), 1)


def _slope(values: np.ndarray, dt_h: float) -> float:
    """Считает наклон линейного тренда по окну, игнорируя пропуски."""
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[mask] * dt_h
    y = values[mask]
    x_centered = x - x.mean()
    denom = np.dot(x_centered, x_centered)
    if denom == 0:
        return 0.0
    return float(np.dot(x_centered, y - y.mean()) / denom)


def _feature_description(cfg: ScenarioConfig) -> str:
    """Генерирует markdown-описание признаков, их формул и назначения."""
    lines = [
        "# Описание feature builder",
        "",
        f"Номинальный расход для нормировки: `{cfg.q_nominal_m3h}` м3/ч.",
        f"Warning-порог перепада давления: `{cfg.dp_warn_kpa}` кПа.",
        "",
        "| Признак | Смысл | Формула | Использование |",
        "|---|---|---|---|",
    ]
    for item in FEATURE_DESCRIPTIONS:
        lines.append(
            f"| `{item['name']}` | {item['meaning']} | `{item['formula']}` | {item['use']} |"
        )
    lines.extend(
        [
            "",
            "Файлы:",
            "",
            "- `features.csv` - машинно-читаемая таблица признаков.",
            "- `features.parquet` - основной аналитический формат признаков.",
            "- `feature_description.md` - это описание признаков и формул.",
            "",
        ]
    )
    return "\n".join(lines)
