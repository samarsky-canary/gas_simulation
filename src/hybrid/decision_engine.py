from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


HYBRID_COLUMNS = [
    "timestamp",
    "filter_id",
    "scenario",
    "state",
    "quality_code",
    "deltaP_norm_kPa",
    "deltaP_roll_mean_1h",
    "deltaP_slope_6h",
    "time_above_warn",
    "missing_rate_1h",
    "RUL_oracle_h",
    "RUL_analytic_h",
    "RUL_ml_h",
    "RUL_fused_h",
    "rul_source",
    "confidence_data",
    "confidence_model",
    "confidence_consistency",
    "confidence_total",
]


def build_hybrid_decisions(
    cfg: ScenarioConfig,
    dataset: pd.DataFrame,
    features: pd.DataFrame,
    ml_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Строит гибридный RUL: доверие, итоговый RUL и источник прогноза."""
    data = _prepare_inputs(dataset, features, ml_predictions)
    decisions = data.apply(lambda row: _fuse_row(cfg, row), axis=1, result_type="expand")
    result = pd.concat([data, decisions], axis=1)
    return result[HYBRID_COLUMNS]


def export_hybrid_decisions(
    decisions: pd.DataFrame,
    output_dir: Path,
    *,
    export_csv: bool = True,
) -> dict[str, Path]:
    """Сохраняет результат гибридного RUL fusion и описание логики."""
    hybrid_dir = output_dir / "hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hybrid_decisions_parquet": hybrid_dir / "hybrid_decisions.parquet",
        "hybrid_description": hybrid_dir / "hybrid_decision_logic.md",
    }
    if export_csv:
        paths["hybrid_decisions_csv"] = hybrid_dir / "hybrid_decisions.csv"
        decisions.to_csv(paths["hybrid_decisions_csv"], index=False, encoding="utf-8")
    decisions.to_parquet(paths["hybrid_decisions_parquet"], index=False)
    paths["hybrid_description"].write_text(_description(), encoding="utf-8")
    return paths


def _prepare_inputs(
    dataset: pd.DataFrame, features: pd.DataFrame, ml_predictions: pd.DataFrame
) -> pd.DataFrame:
    """Объединяет датасет, признаки качества окна и ML-прогноз RUL по timestamp."""
    data = dataset.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    feature_columns = [
        "deltaP_roll_mean_1h",
        "deltaP_slope_6h",
        "missing_rate_1h",
        "time_above_warn",
    ]
    feature_keys = ["timestamp"]
    if "run_id" in data.columns and "run_id" in features.columns:
        feature_keys = ["run_id", "timestamp"]
    existing_features = [
        *feature_keys,
        *[column for column in feature_columns if column in features.columns],
    ]
    if existing_features:
        feature_data = features[existing_features].copy()
        feature_data["timestamp"] = pd.to_datetime(feature_data["timestamp"])
        data = data.merge(feature_data, on=feature_keys, how="left")
    for column in feature_columns:
        if column not in data.columns:
            data[column] = np.nan

    prediction_keys = ["timestamp"]
    if "run_id" in data.columns and "run_id" in ml_predictions.columns:
        prediction_keys = ["run_id", "timestamp"]
    prediction_columns = [*prediction_keys, "RUL_pred_h"]
    existing_predictions = [
        column for column in prediction_columns if column in ml_predictions.columns
    ]
    predictions = ml_predictions[existing_predictions].copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    data = data.merge(predictions, on=prediction_keys, how="left")
    data = data.rename(columns={"RUL_pred_h": "RUL_ml_h"})

    if "RUL_ml_h" not in data.columns:
        data["RUL_ml_h"] = np.nan
    data["deltaP_roll_mean_1h"] = data["deltaP_roll_mean_1h"].fillna(
        data["deltaP_norm_kPa"]
    )
    data["deltaP_slope_6h"] = data["deltaP_slope_6h"].fillna(0.0)
    data["missing_rate_1h"] = data["missing_rate_1h"].fillna(0.0)
    data["time_above_warn"] = data["time_above_warn"].fillna(0.0)
    return data


def _fuse_row(cfg: ScenarioConfig, row: pd.Series) -> dict[str, object]:
    """Рассчитывает доверия и выбирает итоговый RUL для одной временной точки."""
    data_hard_veto = _has_physical_violation(row)
    confidence_data = _confidence_data(row, data_hard_veto)
    confidence_model = _confidence_model(row)
    confidence_consistency = _confidence_consistency(cfg, row)
    confidence_total = float(
        np.clip(confidence_data * confidence_model * confidence_consistency, 0.0, 1.0)
    )

    rul_fused, rul_source = _fuse_rul(
        row,
        confidence_total=confidence_total,
        confidence_consistency=confidence_consistency,
        data_hard_veto=data_hard_veto,
    )

    return {
        "RUL_fused_h": rul_fused,
        "rul_source": rul_source,
        "confidence_data": confidence_data,
        "confidence_model": confidence_model,
        "confidence_consistency": confidence_consistency,
        "confidence_total": confidence_total,
    }


def _confidence_data(row: pd.Series, hard_veto: bool) -> float:
    """Оценивает доверие к данным по quality_code и пропускам в окне."""
    if hard_veto:
        return 0.05

    quality_code = str(row.get("quality_code", "unknown"))
    base_by_quality = {
        "good": 1.0,
        "missing": 0.45,
        "invalid": 0.20,
    }
    confidence = base_by_quality.get(quality_code, 0.50)
    missing_rate = float(row.get("missing_rate_1h", 0.0) or 0.0)
    confidence -= min(missing_rate, 1.0) * 0.35
    return float(np.clip(confidence, 0.0, 1.0))


def _confidence_model(row: pd.Series) -> float:
    """Оценивает доверие к текущему ML-прогнозу RUL baseline-модели."""
    rul_ml = row.get("RUL_ml_h", np.nan)
    if pd.isna(rul_ml) or float(rul_ml) < 0:
        return 0.0
    return 0.70


def _confidence_consistency(cfg: ScenarioConfig, row: pd.Series) -> float:
    """Оценивает согласованность ML-RUL и аналитического RUL."""
    rul_ml = row.get("RUL_ml_h", np.nan)
    rul_analytic = row.get("RUL_analytic_h", np.nan)
    if pd.isna(rul_ml) or pd.isna(rul_analytic):
        return 0.35
    denominator = max(
        float(abs(rul_ml)),
        float(abs(rul_analytic)),
        cfg.planned_maintenance_rul_h,
        1.0,
    )
    relative_gap = abs(float(rul_ml) - float(rul_analytic)) / denominator
    confidence = 1.0 - relative_gap
    return float(np.clip(confidence, 0.05, 1.0))


def _fuse_rul(
    row: pd.Series,
    *,
    confidence_total: float,
    confidence_consistency: float,
    data_hard_veto: bool,
) -> tuple[float, str]:
    """Выбирает итоговый RUL: ML, консервативный минимум или аналитический fallback."""
    rul_ml = _nan_to_none(row.get("RUL_ml_h", np.nan))
    rul_analytic = _nan_to_none(row.get("RUL_analytic_h", np.nan))

    if rul_analytic is None and rul_ml is None:
        return np.nan, "unavailable"
    if data_hard_veto:
        return _fallback_value(rul_analytic, rul_ml), "analytic_data_veto"
    if rul_ml is None:
        return _fallback_value(rul_analytic, rul_ml), "analytic_fallback"
    if confidence_total >= 0.65 and confidence_consistency >= 0.55:
        return rul_ml, "ml_baseline"
    if rul_analytic is not None and confidence_total >= 0.30:
        return min(rul_ml, rul_analytic), "conservative_min"
    return _fallback_value(rul_analytic, rul_ml), "analytic_fallback"


def _has_physical_violation(row: pd.Series) -> bool:
    """Проверяет физически невозможные наблюдения, которые блокируют ML-решение."""
    p_in = row.get("P_in_MPa", np.nan)
    p_out = row.get("P_out_MPa", np.nan)
    delta_p = row.get("deltaP_kPa", np.nan)
    q = row.get("Q_m3h", np.nan)
    if pd.isna(p_in) or pd.isna(p_out) or pd.isna(delta_p) or pd.isna(q):
        return False
    return bool(float(p_out) > float(p_in) or float(delta_p) < 0 or float(q) < 0)


def _nan_to_none(value: object) -> float | None:
    """Преобразует NaN в None для удобного выбора fallback."""
    if pd.isna(value):
        return None
    return float(value)


def _fallback_value(primary: float | None, secondary: float | None) -> float:
    """Возвращает первый доступный RUL или NaN."""
    if primary is not None:
        return primary
    if secondary is not None:
        return secondary
    return np.nan


def _description() -> str:
    """Формирует описание гибридного RUL fusion layer."""
    return "\n".join(
        [
            "# Гибридный RUL fusion",
            "",
            "Слой объединяет наблюдаемую телеметрию, признаки качества, ML-прогноз RUL baseline-модели и аналитический RUL.",
            "",
            "## Контур",
            "",
            "```text",
            "dataset + features + ML predictions",
            "-> confidence_data",
            "-> confidence_model",
            "-> confidence_consistency",
            "-> confidence_total",
            "-> RUL_fused_h + rul_source",
            "```",
            "",
            "## Источники RUL",
            "",
            "- `ml_baseline` - используется ML-прогноз при достаточном доверии и согласованности.",
            "- `conservative_min` - используется минимум из ML-RUL и аналитического RUL при среднем доверии.",
            "- `analytic_fallback` - используется аналитический RUL при низком доверии или недоступном ML-прогнозе.",
            "- `analytic_data_veto` - используется аналитика при вето качества данных.",
            "- `unavailable` - ни ML, ни аналитический RUL недоступны.",
            "",
            "События планового и срочного обслуживания в UI рассчитываются отдельно по устойчивому пересечению RUL-порогов.",
            "",
        ]
    )
