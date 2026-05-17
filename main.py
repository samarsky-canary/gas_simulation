from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import build_features, export_features
from src.hybrid import (
    build_hybrid_decisions,
    export_hybrid_decisions,
    format_console_decision_summary,
)
from src.ml import train_and_export_ml_baseline
from src.rules import apply_rule_baseline, export_rule_baseline
from src.simulator.config import SCENARIO_OVERRIDES, ScenarioConfig, load_config
from src.simulator.exporters import build_canonical_dataset, export_run
from src.simulator.runner import run_scenario
from src.visualization import build_plots


OUTPUT_LABELS = {
    "raw_observed_csv": "наблюдаемая телеметрия CSV",
    "raw_observed_parquet": "наблюдаемая телеметрия Parquet",
    "truth_labels_csv": "истинные метки CSV",
    "truth_labels_parquet": "истинные метки Parquet",
    "wide_debug_csv": "полный отладочный набор CSV",
    "wide_debug_parquet": "полный отладочный набор Parquet",
    "raw_observed_ru_csv": "наблюдаемая телеметрия CSV на русском",
    "truth_labels_ru_csv": "истинные метки CSV на русском",
    "wide_debug_ru_csv": "полный отладочный набор CSV на русском",
    "operations_description": "описание операций",
    "metadata": "метаданные",
    "dataset_csv": "зафиксированный датасет CSV",
    "dataset_parquet": "зафиксированный датасет Parquet",
    "dataset_schema": "описание схемы датасета",
    "features_csv": "признаки CSV",
    "features_parquet": "признаки Parquet",
    "features_ru_csv": "признаки CSV на русском",
    "feature_description": "описание признаков",
    "rule_baseline_csv": "rule-based baseline CSV",
    "rule_baseline_parquet": "rule-based baseline Parquet",
    "rule_baseline_ru_csv": "rule-based baseline CSV на русском",
    "rule_baseline_description": "описание rule-based baseline",
    "ml_predictions_csv": "ML baseline предсказания CSV",
    "ml_predictions_parquet": "ML baseline предсказания Parquet",
    "ml_metrics_json": "ML baseline метрики JSON",
    "ml_report": "ML baseline отчет",
    "rul_model": "модель RandomForest для RUL",
    "hybrid_decisions_csv": "гибридные решения CSV",
    "hybrid_decisions_parquet": "гибридные решения Parquet",
    "hybrid_decisions_ru_csv": "гибридные решения CSV на русском",
    "hybrid_description": "описание гибридной логики",
    "hybrid_decision_packages_jsonl": "пакеты объяснения решений JSONL",
    "hybrid_decision_cards_md": "карточки объяснения решений Markdown",
}

ML_TRAIN_SCENARIOS = ("slow_clogging", "rapid_clogging", "flow_spikes", "maintenance_reset")
ML_TEST_SCENARIOS = ("slow_clogging", "rapid_clogging", "flow_spikes", "maintenance_reset")
ML_STRESS_TEST_SCENARIOS = ("sensor_bias", "sensor_stuck", "missing_data")
ML_TRAIN_SEEDS = (7, 13, 21)
ML_TEST_SEEDS = (42, 101)
ML_STRESS_TEST_SEEDS = (42,)


def main() -> None:
    """Запускает полный конвейер: симуляция, экспорт, признаки, правила и графики."""
    cfg = load_config(Path("configs/base.yaml"))
    df, report = run_scenario(cfg)
    output_dir = Path("outputs") / cfg.scenario_name
    paths = export_run(cfg, df, report, output_dir)
    dataset = pd.read_parquet(paths["dataset_parquet"])
    features = build_features(cfg, df)
    feature_paths = export_features(cfg, features, output_dir)
    rule_baseline = apply_rule_baseline(cfg, df)
    rule_paths = export_rule_baseline(cfg, rule_baseline, output_dir)
    ml_dataset, ml_features, test_run_ids = _build_ml_training_corpus(cfg, df)
    ml_paths = train_and_export_ml_baseline(
        ml_dataset, output_dir, ml_features, test_run_ids=test_run_ids
    )
    ml_predictions = pd.read_parquet(ml_paths["ml_predictions_parquet"])
    hybrid_decisions = build_hybrid_decisions(cfg, dataset, features, ml_predictions)
    hybrid_paths = export_hybrid_decisions(hybrid_decisions, output_dir)
    plot_paths = build_plots(cfg, df, output_dir, hybrid_decisions)

    print(f"Сгенерировано строк: {len(df)}")
    print(f"Сценарий: {cfg.scenario_name}")
    print(f"Строк с проблемами качества: {report.rows_with_quality_issues}")
    print("Созданные файлы:")
    for name, path in paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Созданные признаки:")
    for name, path in feature_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Создан rule-based baseline:")
    for name, path in rule_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Обучен ML baseline:")
    for name, path in ml_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Создана гибридная логика решений:")
    for name, path in hybrid_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Созданные графики:")
    for path in plot_paths.values():
        print(f"- {path}")
    print("")
    print(format_console_decision_summary(hybrid_decisions))


def _build_ml_training_corpus(
    base_cfg: ScenarioConfig, current_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Генерирует train/test корпус ML из независимых прогонов с split по run_id."""
    current_run_id = str(current_df["run_id"].iloc[0])
    datasets = [build_canonical_dataset(base_cfg, current_df)]
    features = [build_features(base_cfg, current_df)]
    test_run_ids = {current_run_id}
    generated_run_ids = {current_run_id}

    ordinal = 1
    for scenario_name, seed, is_test in _ml_corpus_plan():
        run_id = f"{scenario_name}_{seed}"
        if run_id in generated_run_ids:
            if is_test:
                test_run_ids.add(run_id)
            continue
        run_cfg = _ml_variant_config(base_cfg, scenario_name, seed, ordinal)
        run_df, _ = run_scenario(run_cfg)
        datasets.append(build_canonical_dataset(run_cfg, run_df))
        features.append(build_features(run_cfg, run_df))
        generated_run_ids.add(run_id)
        if is_test:
            test_run_ids.add(run_id)
        ordinal += 1

    return (
        pd.concat(datasets, ignore_index=True),
        pd.concat(features, ignore_index=True),
        test_run_ids,
    )


def _ml_corpus_plan() -> list[tuple[str, int, bool]]:
    """Возвращает план независимых прогонов: основные train, основные test и stress-test."""
    train = [
        (scenario_name, seed, False)
        for scenario_name in ML_TRAIN_SCENARIOS
        for seed in ML_TRAIN_SEEDS
    ]
    test = [
        (scenario_name, seed, True)
        for scenario_name in ML_TEST_SCENARIOS
        for seed in ML_TEST_SEEDS
    ]
    stress_test = [
        (scenario_name, seed, True)
        for scenario_name in ML_STRESS_TEST_SCENARIOS
        for seed in ML_STRESS_TEST_SEEDS
    ]
    return [*train, *test, *stress_test]


def _ml_variant_config(
    base_cfg: ScenarioConfig, scenario_name: str, seed: int, ordinal: int
) -> ScenarioConfig:
    """Создает конфигурацию обучающего прогона с пресетом сценария и отдельным seed."""
    raw = base_cfg.model_dump()
    raw.update(SCENARIO_OVERRIDES.get(scenario_name, {}))
    raw.update(
        {
            "filter_id": f"F-ML-{ordinal:03d}",
            "scenario_name": scenario_name,
            "seed": seed,
        }
    )
    return ScenarioConfig.model_validate(raw)


if __name__ == "__main__":
    main()
