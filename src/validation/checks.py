from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.simulator.config import ScenarioConfig


class QCReport(BaseModel):
    """Краткий отчет о качестве и физической корректности прогона."""

    rows: int
    rows_with_quality_issues: int
    quality_counts: dict[str, int]
    pressure_order_ok: bool
    nonnegative_dp_ok: bool
    nonnegative_q_ok: bool
    monotonic_clog_between_maintenance_ok: bool


def validate_run(cfg: ScenarioConfig, df: pd.DataFrame) -> tuple[pd.DataFrame, QCReport]:
    """Проверяет физические ограничения и назначает quality_code для каждой строки."""
    out = df.copy()

    # Любой пропуск в наблюдаемых каналах делает строку сомнительной для онлайн-диагностики.
    missing_mask = out[["p_in_mpa", "p_out_mpa", "delta_p_kpa", "q_m3h", "t_c"]].isna().any(axis=1)
    invalid_pressure = out["p_out_mpa"] > out["p_in_mpa"] + 0.002
    invalid_dp = out["delta_p_kpa"] < -0.1
    invalid_q = out["q_m3h"] < 0
    invalid_mask = invalid_pressure.fillna(False) | invalid_dp.fillna(False) | invalid_q.fillna(False)
    out["quality_code"] = np.select(
        [missing_mask, invalid_mask],
        ["missing", "invalid"],
        default="good",
    )
    quality_counter: Counter[str] = Counter(out["quality_code"])

    clog = out["clog_level"].to_numpy()
    maint = out["maintenance_event"].to_numpy(dtype=bool)
    diffs = np.diff(clog)
    monotonic = bool(np.all(diffs[~maint[1:]] >= -1e-9))

    report = QCReport(
        rows=len(out),
        rows_with_quality_issues=int(out["quality_code"].ne("good").sum()),
        quality_counts=dict(quality_counter),
        pressure_order_ok=bool((out["p_out_mpa"] <= out["p_in_mpa"] + 0.002).fillna(True).all()),
        nonnegative_dp_ok=bool((out["delta_p_kpa"] >= -0.1).fillna(True).all()),
        nonnegative_q_ok=bool((out["q_m3h"] >= 0).fillna(True).all()),
        monotonic_clog_between_maintenance_ok=monotonic,
    )
    return out, report
