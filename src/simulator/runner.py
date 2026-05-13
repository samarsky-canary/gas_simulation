from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig
from src.simulator.degradation import simulate_degradation
from src.simulator.faults import apply_sensor_model, inject_faults
from src.simulator.labels import label_run
from src.simulator.physics import compute_physics
from src.simulator.profiles import generate_profiles
from src.simulator.timebase import make_index
from src.validation.checks import QCReport, validate_run


def run_scenario(cfg: ScenarioConfig) -> tuple[pd.DataFrame, QCReport]:
    """Собирает один полный прогон симулятора от временной сетки до меток и QC."""
    rng = np.random.default_rng(cfg.seed)
    # Порядок шагов повторяет информационный поток: режимы -> деградация -> физика -> датчики -> качество -> метки.
    idx = make_index(cfg)
    profile = generate_profiles(cfg, idx, rng)
    degradation = simulate_degradation(cfg, profile)
    physics = compute_physics(cfg, profile, degradation)
    observed = apply_sensor_model(cfg, profile, physics, rng)
    observed = inject_faults(cfg, observed, rng)

    df = pd.concat([profile, degradation, physics, observed], axis=1)
    df["run_id"] = f"{cfg.scenario_name}_{cfg.seed}"
    df["filter_id"] = cfg.filter_id
    df["scenario_id"] = cfg.scenario_name
    df, report = validate_run(cfg, df)
    df = label_run(cfg, df)
    return df, report
