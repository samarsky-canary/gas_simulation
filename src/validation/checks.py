from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.simulator.config import ScenarioConfig


class QCReport(BaseModel):
    """Краткий отчет о качестве и физической корректности прогона."""

    rows: int
    rows_with_flags: int
    flag_counts: dict[str, int]
    pressure_order_ok: bool
    nonnegative_dp_ok: bool
    nonnegative_q_ok: bool
    monotonic_clog_between_maintenance_ok: bool


def validate_run(cfg: ScenarioConfig, df: pd.DataFrame) -> tuple[pd.DataFrame, QCReport]:
    """Проверяет физические ограничения и назначает quality_code для каждой строки."""
    out = df.copy()
    flags: list[set[str]] = [
        set(str(value).split(";")) if isinstance(value, str) and value else set()
        for value in out["fault_flags"]
    ]

    # Любой пропуск в наблюдаемых каналах делает строку сомнительной для онлайн-диагностики.
    missing_mask = out[["p_in_mpa", "p_out_mpa", "delta_p_kpa", "q_m3h", "t_c"]].isna().any(axis=1)
    for i in np.flatnonzero(missing_mask.to_numpy()):
        flags[i].add("missing")

    invalid_pressure = out["p_out_mpa"] > out["p_in_mpa"] + 0.002
    for i in np.flatnonzero(invalid_pressure.fillna(False).to_numpy()):
        flags[i].add("invalid_pressure_order")

    invalid_dp = out["delta_p_kpa"] < -0.1
    for i in np.flatnonzero(invalid_dp.fillna(False).to_numpy()):
        flags[i].add("invalid_dp")

    invalid_q = out["q_m3h"] < 0
    for i in np.flatnonzero(invalid_q.fillna(False).to_numpy()):
        flags[i].add("invalid_q")

    out["quality_code"] = [_quality_code(item) for item in flags]
    out["fault_flags"] = [";".join(sorted(item)) if item else "" for item in flags]

    flag_counter: Counter[str] = Counter()
    for item in flags:
        flag_counter.update(item)

    clog = out["clog_level"].to_numpy()
    maint = out["maintenance_event"].to_numpy(dtype=bool)
    diffs = np.diff(clog)
    monotonic = bool(np.all(diffs[~maint[1:]] >= -1e-9))

    report = QCReport(
        rows=len(out),
        rows_with_flags=sum(bool(item) for item in flags),
        flag_counts=dict(flag_counter),
        pressure_order_ok=bool((out["p_out_mpa"] <= out["p_in_mpa"] + 0.002).fillna(True).all()),
        nonnegative_dp_ok=bool((out["delta_p_kpa"] >= -0.1).fillna(True).all()),
        nonnegative_q_ok=bool((out["q_m3h"] >= 0).fillna(True).all()),
        monotonic_clog_between_maintenance_ok=monotonic,
    )
    return out, report


def _quality_code(flags: set[str]) -> str:
    """Сворачивает набор детальных флагов в один основной код качества строки."""
    if not flags:
        return "good"
    if "missing" in flags or any(flag.startswith("missing:") for flag in flags):
        return "missing"
    if any(flag.startswith("stuck:") for flag in flags):
        return "stuck"
    if any(flag.startswith("spike:") for flag in flags):
        return "spike"
    if "invalid_pressure_order" in flags or "invalid_dp" in flags or "invalid_q" in flags:
        return "invalid"
    return "biased"
