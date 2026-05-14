from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

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
    "maintenance_reset",
]


class ScenarioConfig(BaseModel):
    """Единая схема параметров симулятора и порогов диагностики."""

    model_config = ConfigDict(extra="forbid")

    filter_id: str = "F-001"
    scenario_name: ScenarioName = "slow_clogging"
    start_time: datetime = datetime.fromisoformat("2026-01-01T00:00:00+03:00")
    duration_days: int = Field(default=90, gt=0)
    step_minutes: int = Field(default=5, gt=0)
    seed: int = 42

    q_nominal_m3h: float = Field(default=600.0, gt=0)
    q_min_m3h: float = Field(default=150.0, ge=0)
    q_max_m3h: float = Field(default=1200.0, gt=0)
    a_q: float = Field(default=0.12, ge=0)
    q_weekly_amp: float = Field(default=0.05, ge=0)
    q_ar_rho: float = Field(default=0.65, ge=0, lt=1)
    q_process_std_m3h: float = Field(default=30.0, ge=0)

    p_in_nominal_mpa: float = Field(default=0.60, gt=0)
    p_min_mpa: float = Field(default=0.10, gt=0)
    p_max_mpa: float = Field(default=1.20, gt=0)
    a_p_mpa: float = Field(default=0.015, ge=0)
    p_process_std_mpa: float = Field(default=0.003, ge=0)

    t_nominal_c: float = 15.0
    t_min_c: float = -30.0
    t_max_c: float = 50.0
    a_t_c: float = Field(default=4.0, ge=0)
    t_process_std_c: float = Field(default=0.4, ge=0)

    dp0_kpa: float = Field(default=1.2, gt=0)
    dp_warn_kpa: float = Field(default=5.0, gt=0)
    dp_crit_kpa: float = Field(default=10.0, gt=0)
    planned_maintenance_rul_h: float = Field(default=72.0, gt=0)
    urgent_maintenance_rul_h: float = Field(default=12.0, gt=0)
    c0: float = Field(default=0.05, ge=0, le=1)
    k_s_per_hour: float = Field(default=4e-4, ge=0)
    alpha_flow: float = Field(default=2.0, gt=0)
    gamma_load: float = Field(default=1.0, gt=0)
    k_c: float = Field(default=8.0, gt=0)
    beta: float = Field(default=1.0, gt=0)
    k_mu_per_c: float = Field(default=0.0025, ge=0)

    sigma_p_mpa: float = Field(default=0.0003, ge=0)
    sigma_q_rel: float = Field(default=0.01, ge=0)
    sigma_t_abs_c: float = Field(default=0.3, ge=0)
    sigma_dp_kpa: float = Field(default=0.10, ge=0)
    use_dp_sensor: bool = False

    p_missing: float = Field(default=0.005, ge=0, le=1)
    p_spike: float = Field(default=0.001, ge=0, le=1)
    p_stuck: float = Field(default=0.0005, ge=0, le=1)
    bias_drift_mpa_per_day: float = Field(default=0.0, ge=0)
    stuck_min_steps: int = Field(default=6, ge=1)
    stuck_max_steps: int = Field(default=36, ge=1)

    maintenance_day: float | None = None
    c_reset: float = Field(default=0.05, ge=0, le=1)

    schema_version: str = "0.1.0"
    timezone_name: str = "Europe/Astrakhan"

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
    "slow_clogging": {},
    "rapid_clogging": {"k_s_per_hour": 1.1e-3, "a_q": 0.18},
    "flow_spikes": {"p_spike": 0.006, "q_process_std_m3h": 75.0},
    "sensor_bias": {"bias_drift_mpa_per_day": 0.0008},
    "sensor_stuck": {"p_stuck": 0.004},
    "missing_data": {"p_missing": 0.035},
    "maintenance_reset": {"maintenance_day": 45.0, "k_s_per_hour": 7e-4, "c_reset": 0.04},
}


def load_config(path: Path | str) -> ScenarioConfig:
    """Загружает YAML-конфиг, применяет пресет сценария и возвращает валидированный объект."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    scenario_name = raw.get("scenario_name", "slow_clogging")
    merged = {**SCENARIO_OVERRIDES.get(scenario_name, {}), **raw}
    return ScenarioConfig.model_validate(merged)
