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
    "Q_roll_mean_1h",
    "missing_rate_1h",
    "time_above_warn",
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

RU_FEATURE_COLUMNS = {
    "run_id": "идентификатор_прогона",
    "timestamp": "время",
    "filter_id": "идентификатор_фильтра",
    "scenario_id": "сценарий",
    "deltaP_norm_kPa": "перепад_без_влияния_расхода_кпа",
    "deltaP_roll_mean_1h": "средний_перепад_за_1ч_кпа",
    "deltaP_roll_std_1h": "нестабильность_перепада_за_1ч_кпа",
    "deltaP_slope_6h": "скорость_роста_перепада_за_6ч_кпа_ч",
    "Q_roll_mean_1h": "средний_расход_за_1ч_м3_ч",
    "missing_rate_1h": "доля_пропусков_за_1ч",
    "time_above_warn": "время_выше_warning_ч",
    "state_obs": "наблюдаемое_состояние",
    "state_true": "истинное_состояние",
    "rul_oracle_h": "остаточный_ресурс_oracle_ч",
    "rul_analytic_h": "остаточный_ресурс_аналитический_ч",
    "is_rul_unknown": "rul_неизвестен",
}

RU_VALUE_MAPS = {
    "state_obs": {
        "normal": "норма",
        "warning": "предупреждение",
        "critical": "критическое",
        "unknown": "неизвестно",
    },
    "state_true": {
        "normal": "норма",
        "warning": "предупреждение",
        "critical": "критическое",
        "unknown": "неизвестно",
    },
    "is_rul_unknown": {True: "да", False: "нет"},
}

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
]


def build_features(cfg: ScenarioConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Строит табличные признаки для rule-интерпретации и baseline ML-моделей."""
    features = df.copy()
    one_hour = _window_steps(cfg, hours=1)
    six_hours = _window_steps(cfg, hours=6)
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

    return features[FEATURE_EXPORT_COLUMNS]


def export_features(cfg: ScenarioConfig, features: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Экспортирует признаки в CSV/Parquet, русскую CSV-версию и описание формул."""
    paths = {
        "features_csv": output_dir / "features.csv",
        "features_parquet": output_dir / "features.parquet",
        "features_ru_csv": output_dir / "features_ru.csv",
        "feature_description": output_dir / "feature_description.md",
    }
    features.to_csv(paths["features_csv"], index=False, encoding="utf-8")
    features.to_parquet(paths["features_parquet"], index=False)
    _to_russian_csv(features, paths["features_ru_csv"])
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


def _to_russian_csv(features: pd.DataFrame, path: Path) -> None:
    """Создает русифицированный CSV с признаками для просмотра и отчета."""
    localized = features.copy()
    for column, value_map in RU_VALUE_MAPS.items():
        if column in localized.columns:
            localized[column] = localized[column].replace(value_map)
    localized = localized.rename(columns=RU_FEATURE_COLUMNS)
    localized.to_csv(path, index=False, encoding="utf-8-sig")


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
            "- `features_ru.csv` - русифицированная версия для просмотра.",
            "- `feature_description.md` - это описание признаков и формул.",
            "",
        ]
    )
    return "\n".join(lines)
