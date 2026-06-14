from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features import build_features, export_features
from src.hybrid import (
    build_hybrid_decisions,
    export_hybrid_decisions,
    format_console_decision_summary,
)
from src.ml import predict_and_export_ml_baseline, train_ml_baseline
from src.rules import apply_rule_baseline, export_rule_baseline
from src.simulator.config import SCENARIO_OVERRIDES, ScenarioConfig, load_config
from src.simulator.exporters import build_canonical_dataset, export_run
from src.simulator.runner import run_scenario
from src.storage import StoredMLTraining, TrainingRepository
from src.visualization import build_plots


OUTPUT_LABELS = {
    "raw_observed_csv": "наблюдаемая телеметрия CSV",
    "raw_observed_parquet": "наблюдаемая телеметрия Parquet",
    "truth_labels_csv": "истинные метки CSV",
    "truth_labels_parquet": "истинные метки Parquet",
    "wide_debug_csv": "полный отладочный набор CSV",
    "wide_debug_parquet": "полный отладочный набор Parquet",
    "operations_description": "описание операций",
    "metadata": "метаданные",
    "dataset_csv": "зафиксированный датасет CSV",
    "dataset_parquet": "зафиксированный датасет Parquet",
    "dataset_schema": "описание схемы датасета",
    "features_csv": "признаки CSV",
    "features_parquet": "признаки Parquet",
    "feature_description": "описание признаков",
    "rule_baseline_csv": "rule-based baseline CSV",
    "rule_baseline_parquet": "rule-based baseline Parquet",
    "rule_baseline_description": "описание rule-based baseline",
    "ml_predictions_csv": "ML baseline предсказания CSV",
    "ml_predictions_parquet": "ML baseline предсказания Parquet",
    "ml_metrics_json": "ML baseline метрики JSON",
    "ml_report": "ML baseline отчет",
    "rul_model": "модель RandomForest для RUL",
    "hybrid_decisions_csv": "гибридные решения CSV",
    "hybrid_decisions_parquet": "гибридные решения Parquet",
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
ML_CORPUS_MIN_DURATION_DAYS = 90
@dataclass(frozen=True)
class PipelineResult:
    """Результаты полного запуска симуляционного конвейера."""

    cfg: ScenarioConfig
    output_dir: Path
    row_count: int
    quality_issue_rows: int
    export_paths: dict[str, Path]
    feature_paths: dict[str, Path]
    rule_paths: dict[str, Path]
    ml_paths: dict[str, Path]
    hybrid_paths: dict[str, Path]
    plot_paths: dict[str, Path]
    decision_summary: str


def run_pipeline(
    cfg: ScenarioConfig,
    output_dir: Path | None = None,
    training_repository: TrainingRepository | None = None,
) -> PipelineResult:
    """Запускает симуляцию, экспорт, признаки, baseline-модели, гибридную логику и графики."""
    df, report = run_scenario(cfg)
    output_dir = output_dir or Path("outputs") / cfg.scenario_name
    paths = export_run(cfg, df, report, output_dir, export_csv=False)
    dataset = build_canonical_dataset(cfg, df)
    features = build_features(cfg, df)
    feature_paths = export_features(cfg, features, output_dir, export_csv=False)
    rule_baseline = apply_rule_baseline(cfg, df)
    rule_paths = export_rule_baseline(cfg, rule_baseline, output_dir, export_csv=False)
    repository = training_repository or TrainingRepository.from_env()
    repository.initialize()
    stored_training = repository.load_latest()
    ml_predictions, ml_paths = predict_and_export_ml_baseline(
        dataset,
        output_dir,
        None,
        features,
        export_csv=False,
        model=stored_training.model,
        metrics=stored_training.metrics,
        report=stored_training.report,
        model_reference=f"postgresql:ml_training_runs/{stored_training.training_id}",
    )
    hybrid_decisions = build_hybrid_decisions(cfg, dataset, features, ml_predictions)
    hybrid_paths = export_hybrid_decisions(
        hybrid_decisions, output_dir, export_csv=False
    )
    plot_paths = build_plots(cfg, df, output_dir, hybrid_decisions)
    return PipelineResult(
        cfg=cfg,
        output_dir=output_dir,
        row_count=len(df),
        quality_issue_rows=report.rows_with_quality_issues,
        export_paths=paths,
        feature_paths=feature_paths,
        rule_paths=rule_paths,
        ml_paths=ml_paths,
        hybrid_paths=hybrid_paths,
        plot_paths=plot_paths,
        decision_summary=format_console_decision_summary(hybrid_decisions),
    )


def main() -> None:
    """Запускает полный конвейер: симуляция, экспорт, признаки, правила и графики."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train and replace the shared ML cache, then exit.",
    )
    args = parser.parse_args()
    cfg = load_config(Path("configs/base.yaml"))
    if args.train_ml:
        training = train_ml_baseline_database(cfg)
        print("ML baseline trained and stored in PostgreSQL:")
        print(f"- training_id: {training.training_id}")
        print(f"- model_id: {training.model_id}")
        print(f"- artifact_sha256: {training.artifact_sha256}")
        return
    result = run_pipeline(cfg)

    print(f"Сгенерировано строк: {result.row_count}")
    print(f"Сценарий: {result.cfg.scenario_name}")
    print(f"Строк с проблемами качества: {result.quality_issue_rows}")
    print("Созданные файлы:")
    for name, path in result.export_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Созданные признаки:")
    for name, path in result.feature_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Создан rule-based baseline:")
    for name, path in result.rule_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Обучен ML baseline:")
    for name, path in result.ml_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Создана гибридная логика решений:")
    for name, path in result.hybrid_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Созданные графики:")
    for path in result.plot_paths.values():
        print(f"- {path}")
    print("")
    print(result.decision_summary)


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


def train_ml_baseline_database(
    base_cfg: ScenarioConfig,
    training_repository: TrainingRepository | None = None,
) -> StoredMLTraining:
    """Обучает общий ML baseline и регистрирует новую версию в PostgreSQL."""
    ml_dataset, ml_features, test_run_ids = _build_cached_ml_training_corpus(base_cfg)
    trained = train_ml_baseline(
        ml_dataset,
        ml_features,
        test_run_ids=test_run_ids,
    )
    repository = training_repository or TrainingRepository.from_env()
    repository.initialize()
    return repository.save(trained)


def _build_cached_ml_training_corpus(
    base_cfg: ScenarioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Генерирует стабильный ML-корпус для кэша без привязки к текущему UI-прогону."""
    datasets = []
    features = []
    test_run_ids = set()
    generated_run_ids = set()

    for ordinal, (scenario_name, seed, is_test) in enumerate(_ml_corpus_plan(), start=1):
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
            "duration_days": max(base_cfg.duration_days, ML_CORPUS_MIN_DURATION_DAYS),
            "seed": seed,
        }
    )
    return ScenarioConfig.model_validate(raw)


if __name__ == "__main__":
    main()
