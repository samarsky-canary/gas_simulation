from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.simulator.config import ScenarioConfig
from src.validation.checks import QCReport


OBSERVED_COLUMNS = [
    "run_id",
    "timestamp",
    "filter_id",
    "scenario_id",
    "p_in_mpa",
    "p_out_mpa",
    "delta_p_kpa",
    "q_m3h",
    "t_c",
    "delta_p_source",
    "quality_code",
    "state_obs",
    "alarm_flag",
    "delta_p_norm_q2",
    "rule_health_index",
]

TRUTH_COLUMNS = [
    "run_id",
    "timestamp",
    "filter_id",
    "scenario_id",
    "q_true_m3h",
    "p_in_true_mpa",
    "t_true_c",
    "p_out_true_mpa",
    "delta_p_true_kpa",
    "resistance_factor",
    "clog_level",
    "maintenance_event",
    "state_true",
    "rul_oracle_h",
    "rul_analytic_h",
    "is_censored",
]

CANONICAL_DATASET_COLUMNS = [
    "timestamp",
    "filter_id",
    "scenario",
    "P_in_MPa",
    "P_out_MPa",
    "deltaP_kPa",
    "Q_m3h",
    "T_C",
    "rho_rel",
    "clog_level",
    "deltaP_norm_kPa",
    "state",
    "RUL_oracle_h",
    "RUL_analytic_h",
    "quality_code",
]

ML_INPUT_COLUMNS = [
    "P_in_MPa",
    "P_out_MPa",
    "deltaP_kPa",
    "Q_m3h",
    "T_C",
    "rho_rel",
    "deltaP_norm_kPa",
]

ML_FORBIDDEN_INPUT_COLUMNS = [
    "clog_level",
    "RUL_oracle_h",
    "state",
]

RU_COLUMN_NAMES = {
    "run_id": "идентификатор_прогона",
    "timestamp": "время",
    "filter_id": "идентификатор_фильтра",
    "scenario_id": "сценарий",
    "q_true_m3h": "истинный_расход_м3_ч",
    "p_in_true_mpa": "истинное_входное_давление_мпа",
    "t_true_c": "истинная_температура_с",
    "p_out_true_mpa": "истинное_выходное_давление_мпа",
    "delta_p_true_kpa": "истинный_перепад_давления_кпа",
    "resistance_factor": "коэффициент_сопротивления",
    "clog_level": "уровень_засорения",
    "maintenance_event": "событие_обслуживания",
    "p_in_mpa": "входное_давление_мпа",
    "p_out_mpa": "выходное_давление_мпа",
    "delta_p_kpa": "перепад_давления_кпа",
    "q_m3h": "расход_м3_ч",
    "t_c": "температура_с",
    "delta_p_source": "источник_перепада_давления",
    "quality_code": "код_качества",
    "state_true": "истинное_состояние",
    "state_obs": "наблюдаемое_состояние",
    "alarm_flag": "флаг_тревоги",
    "rul_oracle_h": "остаточный_ресурс_oracle_ч",
    "rul_analytic_h": "остаточный_ресурс_аналитический_ч",
    "is_censored": "цензурировано",
    "delta_p_norm_q2": "перепад_нормированный_по_расходу",
    "rule_health_index": "индекс_состояния_по_правилу",
}

RU_VALUE_MAPS = {
    "delta_p_source": {"calc": "расчет по давлениям", "sensor": "датчик перепада"},
    "quality_code": {
        "good": "хорошие данные",
        "missing": "есть пропуски",
        "invalid": "некорректные данные",
    },
    "state_true": {
        "normal": "норма",
        "warning": "предупреждение",
        "critical": "критическое",
        "unknown": "неизвестно",
    },
    "state_obs": {
        "normal": "норма",
        "warning": "предупреждение",
        "critical": "критическое",
        "unknown": "неизвестно",
    },
    "alarm_flag": {True: "да", False: "нет"},
    "maintenance_event": {True: "да", False: "нет"},
    "is_censored": {True: "да", False: "нет"},
}

OPERATION_DESCRIPTIONS = [
    {
        "step": "1. Загрузка конфигурации",
        "description": "Чтение YAML-файла, применение сценарного пресета и проверка диапазонов параметров.",
    },
    {
        "step": "2. Построение временной сетки",
        "description": "Формирование равномерного ряда timestamp с заданными датой начала, горизонтом и шагом дискретизации.",
    },
    {
        "step": "3. Генерация режимов",
        "description": "Расчет истинных профилей расхода, входного давления и температуры с суточной/недельной вариацией и AR(1)-шумом.",
    },
    {
        "step": "4. Модель деградации",
        "description": "Рекурсивный расчет скрытого уровня засорения фильтра с учетом нагрузки по расходу и возможного обслуживания.",
    },
    {
        "step": "5. Физическая модель",
        "description": "Расчет истинного перепада давления, коэффициента сопротивления и выходного давления после фильтра.",
    },
    {
        "step": "6. Модель датчиков",
        "description": "Добавление измерительного шума, дрейфа давления и расчет наблюдаемого перепада по давлениям или DP-датчику.",
    },
    {
        "step": "7. Инжекция сбоев",
        "description": "Имитация пропусков, кратковременных выбросов и залипания датчиков с последующей оценкой качества строки.",
    },
    {
        "step": "8. Контроль качества",
        "description": "Проверка порядка давлений, неотрицательности расхода и перепада, монотонности засорения между обслуживаниями.",
    },
    {
        "step": "9. Диагностика и прогноз",
        "description": "Назначение состояний normal/warning/critical, флага тревоги, oracle-RUL и аналитической оценки остаточного ресурса.",
    },
    {
        "step": "10. Экспорт",
        "description": "Сохранение наблюдаемых данных, скрытых меток, полного debug-набора, русских CSV и metadata.json.",
    },
]


def export_run(
    cfg: ScenarioConfig, df: pd.DataFrame, report: QCReport, output_dir: Path
) -> dict[str, Path]:
    """Сохраняет наблюдаемые данные, истинные метки, debug-набор и метаданные прогона."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # raw используется как вход онлайн-контура, truth - как скрытая истина для обучения и оценки.
    raw = df[OBSERVED_COLUMNS]
    truth = df[TRUTH_COLUMNS]
    paths = {
        "raw_observed_csv": output_dir / "raw_observed.csv",
        "raw_observed_parquet": output_dir / "raw_observed.parquet",
        "truth_labels_csv": output_dir / "truth_labels.csv",
        "truth_labels_parquet": output_dir / "truth_labels.parquet",
        "wide_debug_csv": output_dir / "wide_debug.csv",
        "wide_debug_parquet": output_dir / "wide_debug.parquet",
        "raw_observed_ru_csv": output_dir / "raw_observed_ru.csv",
        "truth_labels_ru_csv": output_dir / "truth_labels_ru.csv",
        "wide_debug_ru_csv": output_dir / "wide_debug_ru.csv",
        "operations_description": output_dir / "operations_description.md",
        "metadata": output_dir / "metadata.json",
        "dataset_csv": output_dir / "dataset.csv",
        "dataset_parquet": output_dir / "dataset.parquet",
        "dataset_schema": output_dir / "dataset_schema.md",
    }

    dataset = _canonical_dataset(cfg, df)
    raw.to_csv(paths["raw_observed_csv"], index=False, encoding="utf-8")
    raw.to_parquet(paths["raw_observed_parquet"], index=False)
    truth.to_csv(paths["truth_labels_csv"], index=False, encoding="utf-8")
    truth.to_parquet(paths["truth_labels_parquet"], index=False)
    df.to_csv(paths["wide_debug_csv"], index=False, encoding="utf-8")
    df.to_parquet(paths["wide_debug_parquet"], index=False)
    _to_russian_csv(raw, paths["raw_observed_ru_csv"])
    _to_russian_csv(truth, paths["truth_labels_ru_csv"])
    _to_russian_csv(df, paths["wide_debug_ru_csv"])
    paths["operations_description"].write_text(_operations_markdown(), encoding="utf-8")
    dataset.to_csv(paths["dataset_csv"], index=False, encoding="utf-8")
    dataset.to_parquet(paths["dataset_parquet"], index=False)
    paths["dataset_schema"].write_text(_dataset_schema_markdown(), encoding="utf-8")

    metadata = {
        "schema_version": cfg.schema_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone_name": cfg.timezone_name,
        "rows": len(df),
        "columns": list(df.columns),
        "canonical_dataset_columns": CANONICAL_DATASET_COLUMNS,
        "ml_input_columns": ML_INPUT_COLUMNS,
        "ml_forbidden_input_columns": ML_FORBIDDEN_INPUT_COLUMNS,
        "russian_column_names": RU_COLUMN_NAMES,
        "operation_descriptions": OPERATION_DESCRIPTIONS,
        "config": cfg.model_dump(mode="json"),
        "config_hash_sha256": _config_hash(cfg),
        "qc_report": report.model_dump(),
        "units": {
            "p_in_mpa": "MPa",
            "p_out_mpa": "MPa",
            "delta_p_kpa": "kPa",
            "q_m3h": "m3/h",
            "t_c": "degC",
            "rul_oracle_h": "h",
            "rul_analytic_h": "h",
        },
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def _canonical_dataset(cfg: ScenarioConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Собирает зафиксированный датасет с внешними именами колонок для CSV/Parquet."""
    # rho_rel и deltaP_norm считаются из наблюдаемых каналов, поэтому их можно подавать на вход модели.
    rho_rel = (df["p_in_mpa"] / cfg.p_in_nominal_mpa) * (
        (cfg.t_nominal_c + 273.15) / (df["t_c"] + 273.15)
    )
    delta_p_norm = df["delta_p_kpa"] / (
        ((df["q_m3h"] / cfg.q_nominal_m3h) ** cfg.alpha_flow * rho_rel).clip(lower=1e-3)
    )
    dataset = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "filter_id": df["filter_id"],
            "scenario": df["scenario_id"],
            "P_in_MPa": df["p_in_mpa"],
            "P_out_MPa": df["p_out_mpa"],
            "deltaP_kPa": df["delta_p_kpa"],
            "Q_m3h": df["q_m3h"],
            "T_C": df["t_c"],
            "rho_rel": rho_rel,
            "clog_level": df["clog_level"],
            "deltaP_norm_kPa": delta_p_norm,
            "state": df["state_obs"],
            "RUL_oracle_h": df["rul_oracle_h"],
            "RUL_analytic_h": df["rul_analytic_h"],
            "quality_code": df["quality_code"],
        }
    )
    return dataset[CANONICAL_DATASET_COLUMNS]


def _to_russian_csv(df: pd.DataFrame, path: Path) -> None:
    """Создает CSV для чтения человеком: русские заголовки и русские значения категорий."""
    localized = df.copy()
    for column, value_map in RU_VALUE_MAPS.items():
        if column in localized.columns:
            localized[column] = localized[column].replace(value_map)
    localized = localized.rename(columns={key: value for key, value in RU_COLUMN_NAMES.items() if key in localized.columns})
    localized.to_csv(path, index=False, encoding="utf-8-sig")


def _operations_markdown() -> str:
    """Формирует текстовое описание операций симулятора для отчетных артефактов."""
    lines = ["# Описание операций симулятора", ""]
    for item in OPERATION_DESCRIPTIONS:
        lines.append(f"## {item['step']}")
        lines.append("")
        lines.append(item["description"])
        lines.append("")
    lines.append("## Выходные файлы")
    lines.append("")
    lines.append("- `raw_observed.csv` - машинно-читаемая наблюдаемая телеметрия с англоязычными колонками.")
    lines.append("- `raw_observed_ru.csv` - та же наблюдаемая телеметрия с русскими заголовками и значениями состояний.")
    lines.append("- `truth_labels.csv` - скрытые состояния и целевые метки для обучения/оценки моделей.")
    lines.append("- `truth_labels_ru.csv` - русифицированная версия скрытых состояний и меток.")
    lines.append("- `wide_debug.csv` - полный отладочный набор с наблюдаемыми и истинными каналами.")
    lines.append("- `wide_debug_ru.csv` - русифицированная версия полного отладочного набора.")
    lines.append("- `metadata.json` - конфигурация, seed, QC-отчет, словарь русских колонок и описание операций.")
    lines.append("")
    return "\n".join(lines)


def _dataset_schema_markdown() -> str:
    """Формирует документ, который фиксирует контракт датасета и ML-входы."""
    rows = [
        ("timestamp", "datetime", "-", "Временная метка наблюдения."),
        ("filter_id", "string", "-", "Идентификатор фильтра."),
        ("scenario", "string", "-", "Имя сценария генерации."),
        ("P_in_MPa", "float", "МПа", "Наблюдаемое входное давление."),
        ("P_out_MPa", "float", "МПа", "Наблюдаемое выходное давление."),
        ("deltaP_kPa", "float", "кПа", "Наблюдаемый перепад давления."),
        ("Q_m3h", "float", "м3/ч", "Наблюдаемый расход газа."),
        ("T_C", "float", "°C", "Наблюдаемая температура газа."),
        ("rho_rel", "float", "отн. ед.", "Относительная плотность, рассчитанная из наблюдаемых P и T."),
        ("clog_level", "float", "0..1", "Скрытый уровень засорения симулятора."),
        ("deltaP_norm_kPa", "float", "кПа", "Перепад, нормированный на расход и относительную плотность."),
        ("state", "category", "-", "Состояние по наблюдаемому перепаду: normal/warning/critical/unknown."),
        ("RUL_oracle_h", "float", "ч", "Истинный RUL до критического порога, доступен только в синтетике."),
        ("RUL_analytic_h", "float", "ч", "Аналитическая оценка остаточного ресурса."),
        ("quality_code", "category", "-", "Код качества строки."),
    ]
    lines = [
        "# Зафиксированный формат датасета",
        "",
        "Основной контракт для обмена и последующего обучения фиксируется файлами `dataset.csv` и `dataset.parquet`.",
        "",
        "## Колонки",
        "",
        "| Колонка | Тип | Единицы | Описание |",
        "|---|---|---:|---|",
    ]
    for name, dtype, unit, description in rows:
        lines.append(f"| `{name}` | `{dtype}` | {unit} | {description} |")
    lines.extend(
        [
            "",
            "## Можно подавать на вход ML-модели",
            "",
            *[f"- `{column}`" for column in ML_INPUT_COLUMNS],
            "",
            "## Нельзя подавать на вход ML-модели",
            "",
            *[f"- `{column}`" for column in ML_FORBIDDEN_INPUT_COLUMNS],
            "",
            "`clog_level`, `RUL_oracle_h` и `state` являются скрытыми/целевыми полями симулятора. Их можно использовать как target или для оценки качества, но нельзя включать в признаки входной последовательности.",
            "",
        ]
    )
    return "\n".join(lines)


def _config_hash(cfg: ScenarioConfig) -> str:
    """Считает хэш конфигурации, чтобы можно было проверить воспроизводимость набора."""
    payload = json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
