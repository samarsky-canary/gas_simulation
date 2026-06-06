from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


RULE_COLUMNS = [
    "run_id",
    "timestamp",
    "filter_id",
    "scenario_id",
    "delta_p_kpa",
    "delta_p_norm_kpa",
    "rul_analytic_h",
    "quality_code",
    "rule_state",
    "rule_alarm_flag",
    "rule_recommendation",
    "rule_reason",
    "state_obs",
    "state_true",
    "rul_oracle_h",
]


def apply_rule_baseline(cfg: ScenarioConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Применяет пороговые правила состояния и рекомендации обслуживания."""
    result = df.copy()
    result["delta_p_norm_kpa"] = _delta_p_norm(cfg, result)
    # Состояние определяется по нормированному перепаду, чтобы скачок расхода не выглядел как засорение.
    result["rule_state"] = np.select(
        [
            result["delta_p_norm_kpa"].isna(),
            result["delta_p_norm_kpa"] < cfg.dp_warn_kpa,
            result["delta_p_norm_kpa"] < cfg.dp_crit_kpa,
            result["delta_p_norm_kpa"] >= cfg.dp_crit_kpa,
        ],
        ["unknown", "normal", "warning", "critical"],
        default="unknown",
    )
    result.loc[result["quality_code"].eq("missing"), "rule_state"] = "unknown"
    result["rule_alarm_flag"] = result["rule_state"].isin(["warning", "critical"])

    # Рекомендация сначала строится по RUL, затем усиливается критическим состоянием или плохими данными.
    recommendation = np.full(len(result), "continue_monitoring", dtype=object)
    rul = result["rul_analytic_h"]
    recommendation[rul < cfg.planned_maintenance_rul_h] = "planned_maintenance"
    recommendation[rul < cfg.urgent_maintenance_rul_h] = "urgent_maintenance"
    recommendation[result["rule_state"].eq("critical").to_numpy()] = "urgent_maintenance"
    recommendation[result["rule_state"].eq("unknown").to_numpy()] = "inspect_sensor_data"
    result["rule_recommendation"] = recommendation
    result["rule_reason"] = result.apply(lambda row: _reason(cfg, row), axis=1)

    return result[RULE_COLUMNS]


def export_rule_baseline(
    cfg: ScenarioConfig,
    baseline: pd.DataFrame,
    output_dir: Path,
    *,
    export_csv: bool = True,
) -> dict[str, Path]:
    """Сохраняет результат rule-based baseline и описание примененных правил."""
    paths = {
        "rule_baseline_parquet": output_dir / "rule_baseline.parquet",
        "rule_baseline_description": output_dir / "rule_baseline_description.md",
    }
    if export_csv:
        paths["rule_baseline_csv"] = output_dir / "rule_baseline.csv"
        baseline.to_csv(paths["rule_baseline_csv"], index=False, encoding="utf-8")
    baseline.to_parquet(paths["rule_baseline_parquet"], index=False)
    paths["rule_baseline_description"].write_text(_description(cfg), encoding="utf-8")
    return paths


def _reason(cfg: ScenarioConfig, row: pd.Series) -> str:
    """Формирует текстовое объяснение, почему правило выдало состояние и рекомендацию."""
    if row["rule_state"] == "unknown":
        return "quality_code=missing or delta_p_norm_kpa is NaN"
    if row["rule_state"] == "normal":
        state_rule = f"delta_p_norm_kpa < {cfg.dp_warn_kpa}"
    elif row["rule_state"] == "warning":
        state_rule = f"{cfg.dp_warn_kpa} <= delta_p_norm_kpa < {cfg.dp_crit_kpa}"
    else:
        state_rule = f"delta_p_norm_kpa >= {cfg.dp_crit_kpa}"

    recommendation = row["rule_recommendation"]
    if recommendation == "urgent_maintenance":
        rec_rule = (
            f"rul_analytic_h < {cfg.urgent_maintenance_rul_h} "
            "or rule_state=critical"
        )
    elif recommendation == "planned_maintenance":
        rec_rule = f"rul_analytic_h < {cfg.planned_maintenance_rul_h}"
    else:
        rec_rule = "no maintenance threshold reached"
    return f"{state_rule}; {rec_rule}"


def _delta_p_norm(cfg: ScenarioConfig, df: pd.DataFrame) -> pd.Series:
    """Считает нормированный перепад для rule layer по наблюдаемым каналам."""
    flow_factor = (df["q_m3h"] / cfg.q_nominal_m3h) ** cfg.alpha_flow
    return df["delta_p_kpa"] / flow_factor.clip(lower=1e-3)


def _description(cfg: ScenarioConfig) -> str:
    """Генерирует markdown-описание пороговой логики baseline-модели."""
    return "\n".join(
        [
            "# Rule-based baseline",
            "",
            "Регламентно-логическая baseline-модель использует нормированный перепад давления и аналитический RUL.",
            "",
            "## Правила состояния",
            "",
            f"- Если `delta_p_norm_kpa < {cfg.dp_warn_kpa}`, то `rule_state = normal`.",
            f"- Если `{cfg.dp_warn_kpa} <= delta_p_norm_kpa < {cfg.dp_crit_kpa}`, то `rule_state = warning`.",
            f"- Если `delta_p_norm_kpa >= {cfg.dp_crit_kpa}`, то `rule_state = critical`.",
            "- Если `delta_p_norm_kpa` отсутствует или строка помечена как missing, то `rule_state = unknown`.",
            "",
            "## Правила рекомендаций",
            "",
            f"- Если `rul_analytic_h < {cfg.planned_maintenance_rul_h}`, то `rule_recommendation = planned_maintenance`.",
            f"- Если `rul_analytic_h < {cfg.urgent_maintenance_rul_h}`, то `rule_recommendation = urgent_maintenance`.",
            "- Если `rule_state = critical`, то рекомендация повышается до `urgent_maintenance`.",
            "- Если `rule_state = unknown`, то `rule_recommendation = inspect_sensor_data`.",
            "",
            "## Выходные файлы",
            "",
            "- `rule_baseline.csv` - машинно-читаемый результат правил.",
            "- `rule_baseline.parquet` - аналитический формат результата правил.",
            "- `rule_baseline_description.md` - описание логики правил.",
            "",
        ]
    )
