from __future__ import annotations

import json
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
    "fault_flags",
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
    "action",
    "priority",
    "due_time_h",
    "rule_trace",
    "explanation",
]

RU_COLUMNS = {
    "timestamp": "время",
    "filter_id": "идентификатор_фильтра",
    "scenario": "сценарий",
    "state": "состояние",
    "quality_code": "код_качества",
    "fault_flags": "флаги_сбоев",
    "deltaP_norm_kPa": "нормированный_перепад_давления_кпа",
    "deltaP_roll_mean_1h": "средний_перепад_за_1ч_кпа",
    "deltaP_slope_6h": "скорость_роста_перепада_за_6ч",
    "time_above_warn": "время_выше_warning_ч",
    "missing_rate_1h": "доля_пропусков_за_1ч",
    "RUL_oracle_h": "остаточный_ресурс_oracle_ч",
    "RUL_analytic_h": "остаточный_ресурс_аналитический_ч",
    "RUL_ml_h": "остаточный_ресурс_ml_ч",
    "RUL_fused_h": "итоговый_остаточный_ресурс_ч",
    "rul_source": "источник_итогового_rul",
    "confidence_data": "доверие_к_данным",
    "confidence_model": "доверие_к_модели",
    "confidence_consistency": "доверие_к_согласованности",
    "confidence_total": "общее_доверие",
    "action": "действие",
    "priority": "приоритет",
    "due_time_h": "срок_ч",
    "rule_trace": "сработавшие_правила",
    "explanation": "объяснение",
}

RU_VALUES = {
    "state": {
        "normal": "норма",
        "warning": "предупреждение",
        "critical": "критическое",
        "unknown": "неизвестно",
    },
    "quality_code": {
        "good": "хорошие данные",
        "missing": "есть пропуски",
        "spike": "выброс",
        "stuck": "залипание датчика",
        "invalid": "некорректные данные",
        "biased": "смещение датчика",
    },
    "rul_source": {
        "ml_baseline": "ML baseline",
        "conservative_min": "консервативный минимум",
        "analytic_fallback": "аналитический fallback",
        "analytic_data_veto": "аналитика из-за вето качества данных",
        "unavailable": "недоступно",
    },
    "action": {
        "monitor": "продолжить мониторинг",
        "sensor_check": "проверить датчики",
        "planned_maintenance": "плановое обслуживание",
        "urgent_maintenance": "срочное обслуживание",
        "shutdown_request": "запрос на останов",
        "manual_review": "ручная проверка",
    },
}


def build_hybrid_decisions(
    cfg: ScenarioConfig,
    dataset: pd.DataFrame,
    features: pd.DataFrame,
    ml_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Строит гибридные решения: доверие, fusion RUL, действие, приоритет и объяснение."""
    data = _prepare_inputs(dataset, features, ml_predictions)
    decisions = data.apply(lambda row: _decide_row(cfg, row), axis=1, result_type="expand")
    result = pd.concat([data, decisions], axis=1)
    return result[HYBRID_COLUMNS]


def export_hybrid_decisions(decisions: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Сохраняет гибридные решения и описание логики принятия решений."""
    hybrid_dir = output_dir / "hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hybrid_decisions_csv": hybrid_dir / "hybrid_decisions.csv",
        "hybrid_decisions_parquet": hybrid_dir / "hybrid_decisions.parquet",
        "hybrid_decisions_ru_csv": hybrid_dir / "hybrid_decisions_ru.csv",
        "hybrid_description": hybrid_dir / "hybrid_decision_logic.md",
        "hybrid_decision_packages_jsonl": hybrid_dir / "decision_packages.jsonl",
        "hybrid_decision_cards_md": hybrid_dir / "decision_cards.md",
    }
    decisions.to_csv(paths["hybrid_decisions_csv"], index=False, encoding="utf-8")
    decisions.to_parquet(paths["hybrid_decisions_parquet"], index=False)
    _to_russian_csv(decisions, paths["hybrid_decisions_ru_csv"])
    paths["hybrid_description"].write_text(_description(), encoding="utf-8")
    _write_decision_packages_jsonl(decisions, paths["hybrid_decision_packages_jsonl"])
    paths["hybrid_decision_cards_md"].write_text(_decision_cards_markdown(decisions), encoding="utf-8")
    return paths


def _prepare_inputs(
    dataset: pd.DataFrame, features: pd.DataFrame, ml_predictions: pd.DataFrame
) -> pd.DataFrame:
    """Объединяет датасет, признаки качества окна и ML-прогноз RUL по timestamp."""
    data = dataset.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    feature_columns = [
        "timestamp",
        "deltaP_roll_mean_1h",
        "deltaP_slope_6h",
        "missing_rate_1h",
        "time_above_warn",
    ]
    existing_features = [column for column in feature_columns if column in features.columns]
    if existing_features:
        feature_data = features[existing_features].copy()
        feature_data["timestamp"] = pd.to_datetime(feature_data["timestamp"])
        data = data.merge(feature_data, on="timestamp", how="left")
    for column in feature_columns:
        if column != "timestamp" and column not in data.columns:
            data[column] = np.nan

    prediction_columns = ["timestamp", "RUL_pred_h", "state_pred"]
    existing_predictions = [column for column in prediction_columns if column in ml_predictions.columns]
    predictions = ml_predictions[existing_predictions].copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    data = data.merge(predictions, on="timestamp", how="left")
    data = data.rename(columns={"RUL_pred_h": "RUL_ml_h"})

    if "RUL_ml_h" not in data.columns:
        data["RUL_ml_h"] = np.nan
    if "state_pred" not in data.columns:
        data["state_pred"] = ""
    data["deltaP_roll_mean_1h"] = data["deltaP_roll_mean_1h"].fillna(data["deltaP_norm_kPa"])
    data["deltaP_slope_6h"] = data["deltaP_slope_6h"].fillna(0.0)
    data["missing_rate_1h"] = data["missing_rate_1h"].fillna(0.0)
    data["time_above_warn"] = data["time_above_warn"].fillna(0.0)
    return data


def _decide_row(cfg: ScenarioConfig, row: pd.Series) -> dict[str, object]:
    """Применяет продукционные правила к одной временной точке."""
    data_hard_veto = _has_physical_violation(row)
    confidence_data = _confidence_data(row, data_hard_veto)
    confidence_model = _confidence_model(row)
    confidence_consistency = _confidence_consistency(cfg, row)
    confidence_total = float(
        np.clip(confidence_data * confidence_model * confidence_consistency, 0.0, 1.0)
    )

    rul_fused, rul_source, fusion_rule = _fuse_rul(
        row,
        confidence_total=confidence_total,
        confidence_consistency=confidence_consistency,
        data_hard_veto=data_hard_veto,
    )
    action, priority, due_time_h, trace, explanation = _apply_rules(
        cfg,
        row,
        rul_fused=rul_fused,
        rul_source=rul_source,
        confidence_data=confidence_data,
        confidence_consistency=confidence_consistency,
        confidence_total=confidence_total,
        data_hard_veto=data_hard_veto,
        fusion_rule=fusion_rule,
    )

    return {
        "RUL_fused_h": rul_fused,
        "rul_source": rul_source,
        "confidence_data": confidence_data,
        "confidence_model": confidence_model,
        "confidence_consistency": confidence_consistency,
        "confidence_total": confidence_total,
        "action": action,
        "priority": priority,
        "due_time_h": due_time_h,
        "rule_trace": ";".join(trace),
        "explanation": explanation,
    }


def _confidence_data(row: pd.Series, hard_veto: bool) -> float:
    """Оценивает доверие к данным по quality_code, fault_flags и пропускам в окне."""
    if hard_veto:
        return 0.05

    quality_code = str(row.get("quality_code", "unknown"))
    base_by_quality = {
        "good": 1.0,
        "spike": 0.75,
        "stuck": 0.65,
        "missing": 0.45,
        "biased": 0.60,
        "invalid": 0.20,
    }
    confidence = base_by_quality.get(quality_code, 0.50)
    flags = str(row.get("fault_flags", "") or "")
    if flags:
        if "missing" in flags:
            confidence -= 0.20
        if "stuck" in flags:
            confidence -= 0.15
        if "spike" in flags:
            confidence -= 0.10
    missing_rate = float(row.get("missing_rate_1h", 0.0) or 0.0)
    confidence -= min(missing_rate, 1.0) * 0.35
    return float(np.clip(confidence, 0.0, 1.0))


def _confidence_model(row: pd.Series) -> float:
    """Оценивает доверие к текущему ML-прогнозу RUL baseline-модели."""
    rul_ml = row.get("RUL_ml_h", np.nan)
    if pd.isna(rul_ml) or float(rul_ml) < 0:
        return 0.0
    confidence = 0.70
    state_pred = str(row.get("state_pred", "") or "")
    state = str(row.get("state", "") or "")
    if state_pred and state and state_pred == state:
        confidence += 0.10
    elif state_pred and state and state_pred != state:
        confidence -= 0.15
    return float(np.clip(confidence, 0.0, 1.0))


def _confidence_consistency(cfg: ScenarioConfig, row: pd.Series) -> float:
    """Оценивает согласованность ML-RUL и аналитического RUL."""
    rul_ml = row.get("RUL_ml_h", np.nan)
    rul_analytic = row.get("RUL_analytic_h", np.nan)
    if pd.isna(rul_ml) or pd.isna(rul_analytic):
        return 0.35
    denominator = max(float(abs(rul_ml)), float(abs(rul_analytic)), cfg.planned_maintenance_rul_h, 1.0)
    relative_gap = abs(float(rul_ml) - float(rul_analytic)) / denominator
    confidence = 1.0 - relative_gap
    return float(np.clip(confidence, 0.05, 1.0))


def _fuse_rul(
    row: pd.Series,
    *,
    confidence_total: float,
    confidence_consistency: float,
    data_hard_veto: bool,
) -> tuple[float, str, str]:
    """Выбирает итоговый RUL: ML, консервативный минимум или аналитический fallback."""
    rul_ml = _nan_to_none(row.get("RUL_ml_h", np.nan))
    rul_analytic = _nan_to_none(row.get("RUL_analytic_h", np.nan))

    if rul_analytic is None and rul_ml is None:
        return np.nan, "unavailable", "R-FUSE-000"
    if data_hard_veto:
        return _fallback_value(rul_analytic, rul_ml), "analytic_data_veto", "R-FUSE-003"
    if rul_ml is None:
        return _fallback_value(rul_analytic, rul_ml), "analytic_fallback", "R-FUSE-003"
    if confidence_total >= 0.65 and confidence_consistency >= 0.55:
        return rul_ml, "ml_baseline", "R-FUSE-001"
    if rul_analytic is not None and confidence_total >= 0.30:
        return min(rul_ml, rul_analytic), "conservative_min", "R-FUSE-002"
    return _fallback_value(rul_analytic, rul_ml), "analytic_fallback", "R-FUSE-003"


def _apply_rules(
    cfg: ScenarioConfig,
    row: pd.Series,
    *,
    rul_fused: float,
    rul_source: str,
    confidence_data: float,
    confidence_consistency: float,
    confidence_total: float,
    data_hard_veto: bool,
    fusion_rule: str,
) -> tuple[str, str, float, list[str], str]:
    """Применяет группы правил DATA_QUALITY, SAFETY, CONSISTENCY и MAINTENANCE."""
    trace: list[str] = [fusion_rule]
    state = str(row.get("state", "unknown"))
    quality_code = str(row.get("quality_code", "unknown"))
    fault_flags = str(row.get("fault_flags", "") or "")

    if _is_low_outlet_pressure(cfg, row) and confidence_data >= 0.50:
        trace.extend(["R-SAFE-002", "R-EXPL-001"])
        return (
            "shutdown_request",
            "P0",
            0.0,
            trace,
            "Запрошен останов: выходное давление ниже допустимого уровня при приемлемом качестве данных.",
        )

    if data_hard_veto:
        trace.extend(["R-DQ-001", "R-EXPL-001"])
        return (
            "sensor_check",
            "P0",
            0.0,
            trace,
            "ML-прогноз заблокирован: обнаружено физически некорректное измерение. Сначала требуется проверка датчиков.",
        )

    if quality_code != "good" and confidence_data < 0.45:
        trace.extend(["R-DQ-002", "R-EXPL-001"])
        return (
            "sensor_check",
            "P0",
            0.0,
            trace,
            f"Данные ненадёжны: quality_code={quality_code}, fault_flags={fault_flags or 'нет'}. Требуется проверка датчиков.",
        )

    if state == "critical" and confidence_data >= 0.50:
        trace.extend(["R-SAFE-001", "R-EXPL-001"])
        return (
            "urgent_maintenance",
            "P1",
            _due_time(rul_fused, cfg.urgent_maintenance_rul_h),
            trace,
            "Фильтр находится в critical-состоянии при достаточном качестве данных. Требуется срочное обслуживание.",
        )

    if confidence_consistency < 0.25 and confidence_data >= 0.50:
        trace.extend(["R-CONS-001", "R-EXPL-001"])
        return (
            "manual_review",
            "P1",
            _due_time(rul_fused, cfg.planned_maintenance_rul_h),
            trace,
            "ML-RUL и аналитический RUL сильно расходятся. Итоговый RUL выбран консервативно, решение требует ручной проверки.",
        )

    if confidence_total < 0.20:
        trace.extend(["R-CONS-002", "R-EXPL-001"])
        return (
            "manual_review",
            "P1",
            _due_time(rul_fused, cfg.planned_maintenance_rul_h),
            trace,
            "Общее доверие к решению низкое. Требуется ручная проверка перед назначением обслуживания.",
        )

    if _is_rul_below(rul_fused, cfg.urgent_maintenance_rul_h):
        trace.extend(["R-MNT-003", "R-EXPL-001"])
        return (
            "urgent_maintenance",
            "P1",
            _due_time(rul_fused, cfg.urgent_maintenance_rul_h),
            trace,
            f"Итоговый RUL={rul_fused:.1f} ч ниже срочного горизонта. Источник RUL: {rul_source}.",
        )

    if state == "warning" and _is_rul_below(rul_fused, cfg.planned_maintenance_rul_h):
        trace.extend(["R-MNT-002", "R-EXPL-001"])
        return (
            "planned_maintenance",
            "P2",
            _due_time(rul_fused, cfg.planned_maintenance_rul_h),
            trace,
            f"Фильтр в warning-состоянии, итоговый RUL={rul_fused:.1f} ч ниже планового горизонта. Рекомендовано плановое ТО.",
        )

    if _is_rul_below(rul_fused, cfg.planned_maintenance_rul_h):
        trace.extend(["R-MNT-002", "R-EXPL-001"])
        return (
            "planned_maintenance",
            "P2",
            _due_time(rul_fused, cfg.planned_maintenance_rul_h),
            trace,
            f"Итоговый RUL={rul_fused:.1f} ч ниже планового горизонта. Рекомендовано плановое ТО.",
        )

    trace.extend(["R-MNT-001", "R-EXPL-001"])
    return (
        "monitor",
        "P3",
        np.nan,
        trace,
        f"Критические условия не обнаружены. Итоговый RUL выбран из источника {rul_source}; продолжается мониторинг.",
    )


def _has_physical_violation(row: pd.Series) -> bool:
    """Проверяет физически невозможные наблюдения, которые блокируют ML-решение."""
    p_in = row.get("P_in_MPa", np.nan)
    p_out = row.get("P_out_MPa", np.nan)
    delta_p = row.get("deltaP_kPa", np.nan)
    q = row.get("Q_m3h", np.nan)
    if pd.isna(p_in) or pd.isna(p_out) or pd.isna(delta_p) or pd.isna(q):
        return False
    return bool(float(p_out) > float(p_in) or float(delta_p) < 0 or float(q) < 0)


def _is_low_outlet_pressure(cfg: ScenarioConfig, row: pd.Series) -> bool:
    """Проверяет опасно низкое выходное давление."""
    p_out = row.get("P_out_MPa", np.nan)
    if pd.isna(p_out):
        return False
    return bool(float(p_out) < cfg.p_min_mpa)


def _is_rul_below(value: float, threshold: float) -> bool:
    """Безопасно сравнивает RUL с порогом."""
    return not pd.isna(value) and float(value) < threshold


def _due_time(rul_fused: float, horizon_h: float) -> float:
    """Назначает срок действия как минимум между RUL и нормативным горизонтом."""
    if pd.isna(rul_fused):
        return float(horizon_h)
    return float(max(0.0, min(float(rul_fused), horizon_h)))


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


def _to_russian_csv(decisions: pd.DataFrame, path: Path) -> None:
    """Создает русифицированный CSV с решениями гибридной модели."""
    localized = decisions.copy()
    for column, mapping in RU_VALUES.items():
        if column in localized.columns:
            localized[column] = localized[column].replace(mapping)
    localized = localized.rename(columns=RU_COLUMNS)
    localized.to_csv(path, index=False, encoding="utf-8-sig")


def _write_decision_packages_jsonl(decisions: pd.DataFrame, path: Path) -> None:
    """Сохраняет пакет объяснения для каждой временной точки в JSON Lines."""
    with path.open("w", encoding="utf-8") as fh:
        for _, row in decisions.iterrows():
            package = _decision_package(row)
            fh.write(json.dumps(package, ensure_ascii=False) + "\n")


def _decision_cards_markdown(decisions: pd.DataFrame) -> str:
    """Формирует читаемые карточки решений для последних и типовых событий."""
    sample = _sample_card_rows(decisions)
    lines = [
        "# Карточки объяснения решений",
        "",
        "Файл показывает человекочитаемый вид пакета объяснения. Полный машинно-читаемый журнал для каждой временной точки находится в `decision_packages.jsonl`.",
        "",
        "В текущей версии `RUL ML baseline` является временным заменителем будущего `RUL LSTM`.",
        "",
    ]
    for _, row in sample.iterrows():
        package = _decision_package(row)
        lines.extend(_decision_card_lines(package))
        lines.append("")
    return "\n".join(lines)


def _sample_card_rows(decisions: pd.DataFrame) -> pd.DataFrame:
    """Выбирает несколько репрезентативных карточек: последнюю строку и первые разные действия."""
    indices: list[int] = []
    if len(decisions) > 0:
        indices.append(int(decisions.index[-1]))
    action_order = [
        "sensor_check",
        "shutdown_request",
        "urgent_maintenance",
        "planned_maintenance",
        "manual_review",
        "monitor",
    ]
    for action in action_order:
        matches = decisions.index[decisions["action"].eq(action)]
        if len(matches) > 0:
            indices.append(int(matches[0]))
    unique_indices = list(dict.fromkeys(indices))
    return decisions.loc[unique_indices]


def _decision_package(row: pd.Series) -> dict[str, object]:
    """Собирает структурированный пакет объяснения решения."""
    return {
        "timestamp": _string_value(row.get("timestamp")),
        "filter_id": _string_value(row.get("filter_id")),
        "scenario": _string_value(row.get("scenario")),
        "state": _string_value(row.get("state")),
        "quality": {
            "quality_code": _string_value(row.get("quality_code")),
            "fault_flags": _string_value(row.get("fault_flags")),
        },
        "rul": {
            "ml_baseline_h": _number_or_none(row.get("RUL_ml_h")),
            "analytic_h": _number_or_none(row.get("RUL_analytic_h")),
            "fused_h": _number_or_none(row.get("RUL_fused_h")),
            "source": _string_value(row.get("rul_source")),
        },
        "confidence": {
            "data": _number_or_none(row.get("confidence_data")),
            "model": _number_or_none(row.get("confidence_model")),
            "consistency": _number_or_none(row.get("confidence_consistency")),
            "total": _number_or_none(row.get("confidence_total")),
        },
        "decision": {
            "action": _string_value(row.get("action")),
            "priority": _string_value(row.get("priority")),
            "due_time_h": _number_or_none(row.get("due_time_h")),
        },
        "rule_trace": _rule_trace_list(row.get("rule_trace")),
        "key_features": _key_features(row),
        "explanation": _string_value(row.get("explanation")),
        "what_to_do": _what_to_do(row),
    }


def _decision_card_lines(package: dict[str, object]) -> list[str]:
    """Преобразует пакет объяснения в markdown-карточку."""
    quality = package["quality"]
    rul = package["rul"]
    confidence = package["confidence"]
    decision = package["decision"]
    key_features = package["key_features"]
    rule_trace = package["rule_trace"]
    what_to_do = package["what_to_do"]

    lines = [
        f"## Фильтр {package['filter_id']} | {package['timestamp']}",
        "",
        f"Состояние: `{str(package['state']).upper()}`",
        f"Качество данных: `{str(quality['quality_code']).upper()}`",
        f"Флаги данных: `{quality['fault_flags'] or 'нет'}`",
        "",
        f"RUL ML baseline: `{_fmt_optional(rul['ml_baseline_h'])}` ч",
        f"RUL analytic: `{_fmt_optional(rul['analytic_h'])}` ч",
        f"RUL fused: `{_fmt_optional(rul['fused_h'])}` ч",
        f"Источник RUL: `{rul['source']}`",
        f"Confidence total: `{_fmt_optional(confidence['total'])}`",
        "",
        f"Действие: `{decision['action']}`",
        f"Приоритет: `{decision['priority']}`",
        f"Срок: `{_format_due(decision['due_time_h'])}`",
        "",
        "Почему:",
    ]
    lines.extend([f"- {item}" for item in _why_lines(package)])
    lines.append(f"- Сработали правила: `{'; '.join(rule_trace)}`")
    lines.append("")
    lines.append("Ключевые признаки:")
    lines.extend([f"- `{item['name']}` = `{_fmt_optional(item['value'])}`: {item['meaning']}" for item in key_features])
    lines.append("")
    lines.append("Что делать:")
    lines.extend([f"- {item}" for item in what_to_do])
    return lines


def _key_features(row: pd.Series) -> list[dict[str, object]]:
    """Выбирает основные признаки, которые объясняют правило."""
    return [
        {
            "name": "deltaP_norm_kPa",
            "value": _number_or_none(row.get("deltaP_norm_kPa")),
            "meaning": "нормированный перепад давления, главный индикатор сопротивления фильтра",
        },
        {
            "name": "deltaP_roll_mean_1h",
            "value": _number_or_none(row.get("deltaP_roll_mean_1h")),
            "meaning": "сглаженный перепад за последний час",
        },
        {
            "name": "deltaP_slope_6h",
            "value": _number_or_none(row.get("deltaP_slope_6h")),
            "meaning": "скорость изменения перепада за 6 часов",
        },
        {
            "name": "time_above_warn",
            "value": _number_or_none(row.get("time_above_warn")),
            "meaning": "сколько часов перепад находится выше warning-порога",
        },
        {
            "name": "missing_rate_1h",
            "value": _number_or_none(row.get("missing_rate_1h")),
            "meaning": "доля проблемных данных в часовом окне",
        },
    ]


def _why_lines(package: dict[str, object]) -> list[str]:
    """Формирует короткие причины решения на русском языке."""
    rul = package["rul"]
    confidence = package["confidence"]
    decision = package["decision"]
    key = {item["name"]: item["value"] for item in package["key_features"]}
    lines = [
        f"Состояние фильтра: `{package['state']}`.",
        f"Итоговый RUL выбран из источника `{rul['source']}`.",
        f"Общее доверие к решению: `{_fmt_optional(confidence['total'])}`.",
    ]
    if key.get("time_above_warn") is not None and float(key["time_above_warn"]) > 0:
        lines.append(f"`deltaP_norm` находится выше warning-порога `{_fmt_optional(key['time_above_warn'])}` ч.")
    if confidence.get("consistency") is not None:
        lines.append(f"Согласованность ML и аналитического RUL: `{_fmt_optional(confidence['consistency'])}`.")
    if decision["action"] in {"sensor_check", "manual_review"}:
        lines.append("Правила ограничили прямое использование ML-прогноза.")
    return lines


def _what_to_do(row: pd.Series) -> list[str]:
    """Возвращает список практических действий для выбранного action."""
    action = str(row.get("action", ""))
    due = _format_due(_number_or_none(row.get("due_time_h")))
    if action == "monitor":
        return ["Продолжить мониторинг.", "Повторно оценивать состояние на следующих окнах данных."]
    if action == "sensor_check":
        return ["Проверить датчики давления, расхода и температуры.", "Не использовать ML-прогноз как основание для ТО до проверки качества данных."]
    if action == "planned_maintenance":
        return [f"Назначить плановое обслуживание: {due}.", "Продолжить мониторинг до выполнения обслуживания."]
    if action == "urgent_maintenance":
        return [f"Назначить срочное обслуживание: {due}.", "Проверить состояние фильтра и подготовить замену."]
    if action == "shutdown_request":
        return ["Передать запрос на безопасный останов.", "Проверить выходное давление и состояние технологической линии."]
    if action == "manual_review":
        return ["Передать случай ответственному инженеру.", "Сравнить ML-прогноз, аналитический RUL и качество данных."]
    return ["Проверить строку решения вручную."]


def _rule_trace_list(value: object) -> list[str]:
    """Преобразует строковый rule_trace в список правил."""
    if pd.isna(value) or not str(value):
        return []
    return [item for item in str(value).split(";") if item]


def _format_due(value: object) -> str:
    """Форматирует срок выполнения действия."""
    number = _number_or_none(value)
    if number is None:
        return "срок не задан"
    if number <= 0:
        return "немедленно"
    return f"в течение {number:.1f} ч"


def _fmt_optional(value: object) -> str:
    """Форматирует число или пустое значение для карточки."""
    number = _number_or_none(value)
    if number is None:
        return "н/д"
    return f"{number:.3f}"


def _number_or_none(value: object) -> float | None:
    """Преобразует числовое значение в float или None."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _string_value(value: object) -> str:
    """Безопасно преобразует значение в строку для JSON/Markdown."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _description() -> str:
    """Формирует описание гибридного decision layer."""
    return "\n".join(
        [
            "# Гибридная логика принятия решений",
            "",
            "Слой объединяет наблюдаемую телеметрию, признаки качества, ML-прогноз RUL baseline-модели и аналитический RUL.",
            "",
            "В текущей версии `RUL_ml_h` используется как временный заменитель будущего `RUL_LSTM_h`. После обучения LSTM можно заменить источник прогноза, сохранив правила fusion и decision layer.",
            "",
            "## Контур",
            "",
            "```text",
            "dataset + features + ML predictions",
            "-> confidence_data",
            "-> confidence_model",
            "-> confidence_consistency",
            "-> confidence_total",
            "-> RUL_fused_h",
            "-> rule engine",
            "-> action + priority + due_time_h + explanation",
            "```",
            "",
            "## Источники RUL",
            "",
            "- `ml_baseline` - используется ML-прогноз при достаточном доверии и согласованности.",
            "- `conservative_min` - используется минимум из ML-RUL и аналитического RUL при среднем доверии.",
            "- `analytic_fallback` - используется аналитический RUL при низком доверии или недоступном ML-прогнозе.",
            "- `analytic_data_veto` - используется аналитика при вето качества данных.",
            "",
            "## Основные действия",
            "",
            "- `monitor` - продолжать мониторинг.",
            "- `sensor_check` - проверить датчики и качество данных.",
            "- `planned_maintenance` - назначить плановое обслуживание.",
            "- `urgent_maintenance` - назначить срочное обслуживание.",
            "- `shutdown_request` - запросить останов из-за опасного давления.",
            "- `manual_review` - передать случай на ручную проверку.",
            "",
            "## Группы правил",
            "",
            "- `R-DQ-*` - правила качества данных.",
            "- `R-SAFE-*` - правила безопасности.",
            "- `R-CONS-*` - правила согласованности прогнозов.",
            "- `R-FUSE-*` - правила выбора итогового RUL.",
            "- `R-MNT-*` - правила обслуживания.",
            "- `R-EXPL-*` - формирование объяснения.",
            "",
            "## Пакеты объяснения",
            "",
            "- `decision_packages.jsonl` - полный структурированный пакет объяснения для каждой временной точки.",
            "- `decision_cards.md` - человекочитаемые карточки для последней точки и типовых действий.",
            "",
        ]
    )
