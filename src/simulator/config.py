from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


ScenarioName = Literal[
    "normal",
    "slow_clogging",
    "rapid_clogging",
    "flow_spikes",
    "sensor_bias",
    "sensor_stuck",
    "missing_data",
]


class ScenarioConfig(BaseModel):
    """Единая схема параметров симулятора и порогов диагностики."""

    model_config = ConfigDict(extra="forbid")

    filter_id: str = "F-001"                                                        # Идентификатор фильтра в сгенерированном наборе данных.
    scenario_name: ScenarioName = "slow_clogging"                                   # Имя сценария, выбирающее набор переопределений параметров.
    start_time: datetime = datetime.fromisoformat("2026-01-01T00:00:00+03:00")      # Начальный момент временной сетки симуляции.
    duration_days: int = Field(default=90, gt=0)                                    # Длительность симуляции в сутках.
    step_minutes: int = Field(default=5, gt=0)                                      # Шаг дискретизации временного ряда в минутах.
    seed: int = 42                                                                  # Seed генератора случайных чисел для воспроизводимости.

    q_nominal_m3h: float = Field(default=600.0, gt=0)                               # Номинальный расход газа через фильтр, м3/ч.
    q_min_m3h: float = Field(default=150.0, ge=0)                                   # Нижняя граница допустимого истинного расхода, м3/ч.
    q_max_m3h: float = Field(default=1200.0, gt=0)                                  # Верхняя граница допустимого истинного расхода, м3/ч.
    a_q: float = Field(default=0.12, ge=0)                                          # Амплитуда суточного колебания расхода относительно номинала.
    q_ar_rho: float = Field(default=0.65, ge=0, lt=1)                               # Коэффициент памяти AR(1)-шума расхода.
    q_process_std_m3h: float = Field(default=30.0, ge=0)                            # Стандартное отклонение плавного процессного шума расхода, м3/ч.

    p_in_nominal_mpa: float = Field(default=0.60, gt=0)                             # Номинальное входное давление перед фильтром, МПа.
    p_min_mpa: float = Field(default=0.10, gt=0)                                    # Нижняя граница допустимого входного давления, МПа.
    p_max_mpa: float = Field(default=1.20, gt=0)                                    # Верхняя граница допустимого входного давления, МПа.
    a_p_mpa: float = Field(default=0.015, ge=0)                                     # Амплитуда суточного колебания входного давления, МПа.
    p_process_std_mpa: float = Field(default=0.003, ge=0)                           # Стандартное отклонение плавного процессного шума давления, МПа.

    t_nominal_c: float = 15.0                                                       # Номинальная температура газа, градусы Цельсия.
    t_min_c: float = -30.0                                                          # Нижняя граница допустимой температуры, градусы Цельсия.
    t_max_c: float = 50.0                                                           # Верхняя граница допустимой температуры, градусы Цельсия.
    a_t_c: float = Field(default=4.0, ge=0)                                         # Амплитуда суточного колебания температуры, градусы Цельсия.
    t_process_std_c: float = Field(default=0.4, ge=0)                               # Стандартное отклонение плавного процессного шума температуры, градусы Цельсия.

    dp0_kpa: float = Field(default=1.2, gt=0)                                       # Базовый перепад давления на чистом фильтре при номинальных условиях, кПа.
    dp_warn_kpa: float = Field(default=5.0, gt=0)                                   # Порог предупреждения по нормированному перепаду давления, кПа.
    dp_crit_kpa: float = Field(default=10.0, gt=0)                                  # Критический порог по нормированному перепаду давления, кПа.
    planned_maintenance_rul_h: float = Field(default=1440.0, gt=0)                  # Горизонт планового обслуживания: 60 суток, часы.
    urgent_maintenance_rul_h: float = Field(default=336.0, gt=0)                    # Горизонт срочного обслуживания: 14 суток, часы.
    stable_degraded_rul_h: float = Field(default=4.0, ge=1)                         # Минимальная длительность устойчивого снижения RUL для фиксации события, часы.
    c0: float = Field(default=0.05, ge=0, le=1)                                     # Начальный скрытый уровень засорения фильтра от 0 до 1.
    k_s_per_hour: float = Field(default=4e-4, ge=0)                                 # Базовая скорость роста засорения за час при номинальной нагрузке.
    alpha_flow: float = Field(default=2.0, gt=0)                                    # Степень влияния расхода на перепад давления.
    gamma_load: float = Field(default=1.0, gt=0)                                    # Степень влияния расхода на скорость накопления засорения.
    k_c: float = Field(default=8.0, gt=0)                                           # Коэффициент влияния засорения на сопротивление фильтра.
    beta: float = Field(default=1.0, gt=0)                                          # Нелинейность влияния засорения на сопротивление фильтра.
    k_mu_per_c: float = Field(default=0.0025, ge=0)                                 # Температурный коэффициент поправки перепада давления на вязкость.

    sigma_p_mpa: float = Field(default=0.0003, ge=0)                                # Стандартное отклонение шума датчика давления, МПа.
    sigma_q_rel: float = Field(default=0.01, ge=0)                                  # Относительное стандартное отклонение шума датчика расхода.
    sigma_t_abs_c: float = Field(default=0.3, ge=0)                                 # Абсолютное стандартное отклонение шума датчика температуры, градусы Цельсия.

    p_missing: float = Field(default=0.005, ge=0, le=1)                             # Вероятность пропуска значения датчика на шаге.
    p_spike: float = Field(default=0.001, ge=0, le=1)                               # Вероятность одиночного выброса датчика на шаге.
    p_stuck: float = Field(default=0.0005, ge=0, le=1)                              # Вероятность начала участка зависшего датчика на шаге.
    bias_drift_mpa_per_day: float = Field(default=0.0, ge=0)                        # Скорость дрейфа смещения датчика давления, МПа в сутки.
    stuck_min_steps: int = Field(default=6, ge=1)                                   # Минимальная длительность зависания датчика в шагах.
    stuck_max_steps: int = Field(default=36, ge=1)                                  # Максимальная длительность зависания датчика в шагах.

    schema_version: str = "0.1.0"                                                   # Версия схемы конфигурации и выходных метаданных.
    timezone_name: str = "Europe/Astrakhan"                                         # Название часового пояса для интерпретации временной сетки.

    @model_validator(mode="after")
    def validate_ranges(self) -> "ScenarioConfig":
        """Проверяет взаимосвязанные диапазоны, которые нельзя проверить простым Field."""
        if self.q_min_m3h >= self.q_max_m3h:
            raise ValueError("q_min_m3h must be lower than q_max_m3h")
        if self.dp_warn_kpa >= self.dp_crit_kpa:
            raise ValueError("dp_warn_kpa must be lower than dp_crit_kpa")
        if self.urgent_maintenance_rul_h >= self.planned_maintenance_rul_h:
            raise ValueError("urgent_maintenance_rul_h must be lower than planned_maintenance_rul_h")
        if self.p_min_mpa >= self.p_max_mpa:
            raise ValueError("p_min_mpa must be lower than p_max_mpa")
        if self.stuck_min_steps > self.stuck_max_steps:
            raise ValueError("stuck_min_steps must be <= stuck_max_steps")
        return self


SCENARIO_OVERRIDES: dict[str, dict[str, Any]] = {
    "normal": {"k_s_per_hour": 2e-5, "p_missing": 0.001, "p_spike": 0.0, "p_stuck": 0.0},
    "slow_clogging": {"p_missing": 0.01, "p_spike": 0.005},
    "rapid_clogging": {"k_s_per_hour": 1.1e-3, "a_q": 0.18},
    "flow_spikes": {"p_spike": 0.006, "q_process_std_m3h": 75.0},
    "sensor_bias": {"bias_drift_mpa_per_day": 0.0008},
    "sensor_stuck": {"p_stuck": 0.004},
    "missing_data": {"p_missing": 0.035},
}


def load_config(path: Path | str) -> ScenarioConfig:
    """Загружает YAML-конфиг, применяет пресет сценария и возвращает валидированный объект."""
    return load_config_with_overrides(path, {})


def load_config_with_overrides(path: Path | str, overrides: Mapping[str, Any]) -> ScenarioConfig:
    """Загружает YAML-конфиг, сценарный пресет и поверх них применяет параметры запуска."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        base_raw = yaml.safe_load(fh) or {}
    explicit_overrides = {key: value for key, value in overrides.items() if value is not None}
    scenario_name = explicit_overrides.get("scenario_name", base_raw.get("scenario_name", "slow_clogging"))
    merged = {**base_raw, **SCENARIO_OVERRIDES.get(scenario_name, {}), **explicit_overrides}
    return ScenarioConfig.model_validate(merged)
