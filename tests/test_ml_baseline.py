from __future__ import annotations

import pandas as pd

from main import _build_randomized_ml_training_corpus
from src.features import build_features
from src.ml import (
    ML_INPUT_COLUMNS,
    predict_and_export_ml_baseline,
    train_and_export_ml_baseline,
    train_ml_baseline,
)
from src.simulator.config import ScenarioConfig
from src.simulator.exporters import build_canonical_dataset, export_run
from src.simulator.runner import run_scenario


def test_ml_baseline_trains_and_exports(tmp_path) -> None:
    cfg = ScenarioConfig(
        duration_days=30,
        step_minutes=60,
        k_s_per_hour=0.003,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df, report = run_scenario(cfg)
    export_paths = export_run(cfg, df, report, tmp_path)
    dataset = pd.read_parquet(export_paths["dataset_parquet"])
    features = build_features(cfg, df)

    paths = train_and_export_ml_baseline(dataset, tmp_path, features)

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    predictions = pd.read_parquet(paths["ml_predictions_parquet"])
    assert {"RUL_pred_h", "split"}.issubset(predictions.columns)
    assert "state_pred" not in predictions.columns


def test_ml_baseline_predicts_from_exported_model(tmp_path) -> None:
    cfg = ScenarioConfig(
        duration_days=30,
        step_minutes=60,
        k_s_per_hour=0.003,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df, report = run_scenario(cfg)
    export_paths = export_run(cfg, df, report, tmp_path / "source")
    dataset = pd.read_parquet(export_paths["dataset_parquet"])
    features = build_features(cfg, df)

    train_paths = train_and_export_ml_baseline(dataset, tmp_path / "train", features)
    predictions, predict_paths = predict_and_export_ml_baseline(
        dataset,
        tmp_path / "predict",
        train_paths["rul_model"],
        features,
        cached_metrics_path=train_paths["ml_metrics_json"],
        cached_report_path=train_paths["ml_report"],
    )

    assert all(path.exists() and path.stat().st_size > 0 for path in predict_paths.values())
    assert set(predictions["split"]) == {"inference"}
    assert len(predictions) == len(dataset)


def test_ml_baseline_predicts_from_in_memory_model(tmp_path) -> None:
    cfg = ScenarioConfig(
        duration_days=30,
        step_minutes=60,
        k_s_per_hour=0.003,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df, report = run_scenario(cfg)
    export_paths = export_run(cfg, df, report, tmp_path / "source")
    dataset = pd.read_parquet(export_paths["dataset_parquet"])
    features = build_features(cfg, df)
    trained = train_ml_baseline(dataset, features)

    predictions, paths = predict_and_export_ml_baseline(
        dataset,
        tmp_path / "predict",
        None,
        features,
        model=trained.model,
        metrics=trained.metrics,
        report=trained.report,
        model_reference="postgresql:ml_training_runs/test",
    )

    assert set(predictions["split"]) == {"inference"}
    assert "rul_model" not in paths
    assert paths["ml_predictions_parquet"].exists()


def test_ml_baseline_corrects_analytic_rul_instead_of_replacing_it(tmp_path) -> None:
    cfg = ScenarioConfig(
        duration_days=90,
        step_minutes=60,
        k_s_per_hour=0.00045,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )
    df, _ = run_scenario(cfg)
    dataset = build_canonical_dataset(cfg, df)
    features = build_features(cfg, df)
    trained = train_ml_baseline(dataset, features)

    predictions, _ = predict_and_export_ml_baseline(
        dataset,
        tmp_path,
        None,
        features,
        model=trained.model,
        metrics=trained.metrics,
        report=trained.report,
    )

    first_analytic = float(dataset["RUL_analytic_h"].iloc[0])
    first_prediction = float(predictions["RUL_pred_h"].iloc[0])
    assert trained.metrics["strategy"] == "analytic_residual_correction"
    assert abs(first_prediction - first_analytic) < 0.25 * first_analytic


def test_randomized_training_corpus_has_independent_runs() -> None:
    cfg = ScenarioConfig(
        duration_days=2,
        step_minutes=60,
        p_missing=0,
        p_spike=0,
        p_stuck=0,
    )

    dataset, features, test_run_ids, metadata = _build_randomized_ml_training_corpus(
        cfg,
        dataset_count=8,
        corpus_seed=123,
        test_share=0.25,
        step_minutes=120,
    )

    assert dataset["run_id"].nunique() == 8
    assert features["run_id"].nunique() == 8
    assert len(test_run_ids) == 2
    assert metadata["dataset_count"] == 8
    assert len(metadata["runs"]) == 8
    assert dataset["scenario"].nunique() == 8
