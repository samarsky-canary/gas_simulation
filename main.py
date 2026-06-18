from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_features, export_features
from src.hybrid import (
    build_hybrid_decisions,
    export_hybrid_decisions,
)
from src.ml import predict_and_export_ml_baseline, train_ml_baseline
from src.simulator.config import SCENARIO_OVERRIDES, ScenarioConfig, load_config
from src.simulator.exporters import build_canonical_dataset, export_run
from src.simulator.runner import run_scenario
from src.storage import StoredMLTraining, TrainingRepository


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
    "ml_predictions_csv": "ML baseline предсказания CSV",
    "ml_predictions_parquet": "ML baseline предсказания Parquet",
    "ml_metrics_json": "ML baseline метрики JSON",
    "ml_report": "ML baseline отчет",
    "rul_model": "модель RandomForest для RUL",
    "hybrid_decisions_csv": "гибридные решения CSV",
    "hybrid_decisions_parquet": "гибридные решения Parquet",
    "hybrid_description": "описание гибридной логики",
}

ML_TRAIN_SCENARIOS = ("slow_clogging", "rapid_clogging", "flow_spikes")
ML_TEST_SCENARIOS = ("slow_clogging", "rapid_clogging", "flow_spikes")
ML_STRESS_TEST_SCENARIOS = ("sensor_bias", "sensor_stuck", "missing_data")
ML_TRAIN_SEEDS = (7, 13, 21)
ML_TEST_SEEDS = (42, 101)
ML_STRESS_TEST_SEEDS = (42,)
ML_CORPUS_MIN_DURATION_DAYS = 90
ML_RANDOMIZED_SCENARIOS = tuple(SCENARIO_OVERRIDES)
ML_DEFAULT_DATASET_COUNT = 100
ML_DEFAULT_TEST_SHARE = 0.2
ML_DEFAULT_CORPUS_SEED = 20260614
ML_DEFAULT_STEP_MINUTES = 30


@dataclass(frozen=True)
class PipelineResult:
    """Результаты полного запуска симуляционного конвейера."""

    cfg: ScenarioConfig
    output_dir: Path
    row_count: int
    quality_issue_rows: int
    export_paths: dict[str, Path]
    feature_paths: dict[str, Path]
    ml_paths: dict[str, Path]
    hybrid_paths: dict[str, Path]


def run_pipeline(
    cfg: ScenarioConfig,
    output_dir: Path | None = None,
    training_repository: TrainingRepository | None = None,
) -> PipelineResult:
    """Запускает симуляцию, экспорт, признаки, ML и гибридную логику."""
    df, report = run_scenario(cfg)
    output_dir = output_dir or Path("outputs") / cfg.scenario_name
    paths = export_run(cfg, df, report, output_dir, export_csv=False)
    dataset = build_canonical_dataset(cfg, df)
    features = build_features(cfg, df)
    feature_paths = export_features(cfg, features, output_dir, export_csv=False)
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
    return PipelineResult(
        cfg=cfg,
        output_dir=output_dir,
        row_count=len(df),
        quality_issue_rows=report.rows_with_quality_issues,
        export_paths=paths,
        feature_paths=feature_paths,
        ml_paths=ml_paths,
        hybrid_paths=hybrid_paths,
    )


def main() -> None:
    """Запускает полный конвейер: симуляция, экспорт, признаки, ML и гибридный RUL."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train a new ML model version, store it in PostgreSQL, then exit.",
    )
    parser.add_argument(
        "--train-datasets",
        type=int,
        default=ML_DEFAULT_DATASET_COUNT,
        help="Number of randomized simulation datasets used for ML training.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=ML_DEFAULT_CORPUS_SEED,
        help="Seed controlling randomized training configurations.",
    )
    parser.add_argument(
        "--test-share",
        type=float,
        default=ML_DEFAULT_TEST_SHARE,
        help="Share of complete simulation runs reserved for test.",
    )
    parser.add_argument(
        "--training-step-minutes",
        type=int,
        default=ML_DEFAULT_STEP_MINUTES,
        help="Sampling step for generated training datasets.",
    )
    args = parser.parse_args()
    cfg = load_config(Path("configs/base.yaml"))
    if args.train_ml:
        if args.train_datasets < 2:
            parser.error("--train-datasets must be at least 2")
        if not 0 < args.test_share < 1:
            parser.error("--test-share must be between 0 and 1")
        if not 1 <= args.training_step_minutes <= 120:
            parser.error("--training-step-minutes must be between 1 and 120")
        training = train_ml_baseline_database(
            cfg,
            dataset_count=args.train_datasets,
            corpus_seed=args.training_seed,
            test_share=args.test_share,
            step_minutes=args.training_step_minutes,
        )
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
    print("Обучен ML baseline:")
    for name, path in result.ml_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Создана гибридная логика решений:")
    for name, path in result.hybrid_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")


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
    *,
    dataset_count: int = ML_DEFAULT_DATASET_COUNT,
    corpus_seed: int = ML_DEFAULT_CORPUS_SEED,
    test_share: float = ML_DEFAULT_TEST_SHARE,
    step_minutes: int = ML_DEFAULT_STEP_MINUTES,
) -> StoredMLTraining:
    """Обучает общий ML baseline и регистрирует новую версию в PostgreSQL."""
    ml_dataset, ml_features, test_run_ids, corpus_metadata = (
        _build_randomized_ml_training_corpus(
            base_cfg,
            dataset_count=dataset_count,
            corpus_seed=corpus_seed,
            test_share=test_share,
            step_minutes=step_minutes,
        )
    )
    trained = train_ml_baseline(
        ml_dataset,
        ml_features,
        test_run_ids=test_run_ids,
        corpus_metadata=corpus_metadata,
    )
    repository = training_repository or TrainingRepository.from_env()
    repository.initialize()
    return repository.save(trained)


def _build_randomized_ml_training_corpus(
    base_cfg: ScenarioConfig,
    *,
    dataset_count: int,
    corpus_seed: int,
    test_share: float,
    step_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], dict[str, object]]:
    """Генерирует воспроизводимый корпус с вариацией режимов, физики и дефектов."""
    if dataset_count < 2:
        raise ValueError("dataset_count must be at least 2")
    if not 0 < test_share < 1:
        raise ValueError("test_share must be between 0 and 1")
    if not 1 <= step_minutes <= 120:
        raise ValueError("step_minutes must be between 1 and 120")

    rng = np.random.default_rng(corpus_seed)
    configs = [
        _randomized_ml_config(
            base_cfg, rng, ordinal, step_minutes, corpus_seed=corpus_seed
        )
        for ordinal in range(1, dataset_count + 1)
    ]
    test_count = min(dataset_count - 1, max(1, round(dataset_count * test_share)))
    test_ordinals = set(
        int(value)
        for value in rng.choice(
            np.arange(1, dataset_count + 1), size=test_count, replace=False
        )
    )

    datasets: list[pd.DataFrame] = []
    features: list[pd.DataFrame] = []
    test_run_ids: set[str] = set()
    run_metadata: list[dict[str, object]] = []
    for ordinal, run_cfg in enumerate(configs, start=1):
        print(
            f"[{ordinal}/{dataset_count}] generating "
            f"{run_cfg.scenario_name}, seed={run_cfg.seed}"
        )
        run_df, _ = run_scenario(run_cfg)
        run_id = str(run_df["run_id"].iloc[0])
        datasets.append(build_canonical_dataset(run_cfg, run_df))
        features.append(build_features(run_cfg, run_df))
        split = "test" if ordinal in test_ordinals else "train"
        if split == "test":
            test_run_ids.add(run_id)
        run_metadata.append(
            {
                "run_id": run_id,
                "split": split,
                "config": run_cfg.model_dump(mode="json"),
            }
        )

    metadata = {
        "dataset_count": dataset_count,
        "corpus_seed": corpus_seed,
        "test_share": test_share,
        "step_minutes": step_minutes,
        "runs": run_metadata,
    }
    return (
        pd.concat(datasets, ignore_index=True),
        pd.concat(features, ignore_index=True),
        test_run_ids,
        metadata,
    )


def _randomized_ml_config(
    base_cfg: ScenarioConfig,
    rng: np.random.Generator,
    ordinal: int,
    step_minutes: int,
    *,
    corpus_seed: int,
) -> ScenarioConfig:
    """Создаёт валидную конфигурацию одного разнообразного обучающего прогона."""
    scenario_name = ML_RANDOMIZED_SCENARIOS[(ordinal - 1) % len(ML_RANDOMIZED_SCENARIOS)]
    raw = base_cfg.model_dump()
    raw.update(SCENARIO_OVERRIDES[scenario_name])

    q_nominal = float(rng.uniform(450.0, 900.0))
    p_nominal = float(rng.uniform(0.45, 0.8))
    k_s_per_hour = _training_k_s_per_hour(
        scenario_name, float(raw["k_s_per_hour"]), rng
    )
    raw.update(
        {
            "filter_id": f"F-ML-{ordinal:04d}",
            "scenario_name": scenario_name,
            "step_minutes": step_minutes,
            "seed": corpus_seed_for_run(corpus_seed, ordinal),
            "a_q": float(rng.uniform(0.06, 0.25)),
            "q_nominal_m3h": q_nominal,
            "q_min_m3h": q_nominal * 0.5,
            "q_max_m3h": q_nominal * 1.5,
            "q_process_std_m3h": float(rng.uniform(15.0, 90.0)),
            "p_in_nominal_mpa": p_nominal,
            "p_min_mpa": p_nominal * 0.5,
            "p_max_mpa": p_nominal * 1.5,
            "a_p_mpa": float(rng.uniform(0.005, 0.03)),
            "t_nominal_c": float(rng.uniform(5.0, 25.0)),
            "a_t_c": float(rng.uniform(2.0, 10.0)),
            "dp0_kpa": float(rng.uniform(0.8, 1.8)),
            "c0": float(rng.uniform(0.01, 0.2)),
            "k_c": float(rng.uniform(5.5, 10.5)),
            "k_s_per_hour": k_s_per_hour,
            "sigma_p_mpa": float(rng.uniform(0.0001, 0.0008)),
            "sigma_q_rel": float(rng.uniform(0.003, 0.03)),
            "sigma_t_abs_c": float(rng.uniform(0.1, 0.8)),
            "p_missing": min(0.08, float(raw["p_missing"]) * float(rng.uniform(0.5, 2.0))),
            "p_spike": min(0.02, float(raw["p_spike"]) * float(rng.uniform(0.5, 2.0))),
            "p_stuck": min(0.01, float(raw["p_stuck"]) * float(rng.uniform(0.5, 2.0))),
            "bias_drift_mpa_per_day": float(raw["bias_drift_mpa_per_day"])
            * float(rng.uniform(0.7, 1.5)),
        }
    )
    raw["duration_days"] = _training_duration_days(raw, rng)
    if scenario_name == "normal":
        raw["duration_days"] = max(
            int(raw["duration_days"]),
            int(rng.integers(300, 366)),
        )
    return ScenarioConfig.model_validate(raw)


def _training_k_s_per_hour(
    scenario_name: str,
    base_rate: float,
    rng: np.random.Generator,
) -> float:
    """Подбирает скорость деградации для обучающих прогонов с достижимым RUL target."""
    if scenario_name == "normal":
        return float(rng.uniform(1.0e-4, 1.8e-4))
    if scenario_name == "rapid_clogging":
        return float(base_rate * rng.uniform(0.7, 1.7))
    return float(base_rate * rng.uniform(0.55, 1.6))


def _training_duration_days(
    raw: dict[str, object],
    rng: np.random.Generator,
) -> int:
    """Подбирает горизонт, чтобы синтетическая траектория достигала critical."""
    dp0 = float(raw["dp0_kpa"])
    dp_critical = float(raw["dp_crit_kpa"])
    k_c = float(raw["k_c"])
    beta = float(raw["beta"])
    c0 = float(raw["c0"])
    rate = max(float(raw["k_s_per_hour"]), 1e-9)
    critical_raw = max((dp_critical / dp0 - 1.0) / k_c, 0.0)
    critical_clog = min(critical_raw, 1.0) ** (1.0 / beta)
    required_hours = max(critical_clog - c0, 0.0) / rate
    buffer = float(rng.uniform(1.15, 1.35))
    return max(60, min(365, int(np.ceil(required_hours * buffer / 24.0))))


def corpus_seed_for_run(corpus_seed: int, ordinal: int) -> int:
    """Возвращает уникальный воспроизводимый seed для run_id корпуса."""
    return (abs(corpus_seed) + ordinal) % 2_000_000_000 + 1


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
