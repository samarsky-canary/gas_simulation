from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ML_INPUT_COLUMNS = [
    "P_in_MPa",
    "P_out_MPa",
    "deltaP_kPa",
    "Q_m3h",
    "T_C",
    "deltaP_norm_kPa",
    "deltaP_roll_mean_1h",
    "deltaP_roll_std_1h",
    "deltaP_slope_6h",
    "Q_roll_mean_1h",
    "missing_rate_1h",
    "time_above_warn",
]

ML_OUTPUT_COLUMNS = [
    "timestamp",
    "filter_id",
    "scenario",
    "split",
    "state_true",
    "state_pred",
    "RUL_oracle_h",
    "RUL_pred_h",
]


def train_and_export_ml_baseline(
    dataset: pd.DataFrame, output_dir: Path, features: pd.DataFrame | None = None
) -> dict[str, Path]:
    """Обучает RandomForest-бейзлайны для RUL и state и сохраняет метрики/предсказания."""
    ml_dir = output_dir / "ml_baseline"
    ml_dir.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_dataset(_attach_features(dataset, features))
    train_mask = _time_split_mask(prepared, train_share=0.70)

    regressor, reg_metrics = _train_rul_regressor(prepared, train_mask)
    classifier, cls_metrics = _train_state_classifier(prepared, train_mask)
    predictions = _build_predictions(prepared, train_mask, regressor, classifier)

    paths = {
        "ml_predictions_csv": ml_dir / "ml_predictions.csv",
        "ml_predictions_parquet": ml_dir / "ml_predictions.parquet",
        "ml_metrics_json": ml_dir / "ml_metrics.json",
        "ml_report": ml_dir / "ml_baseline_report.md",
        "rul_model": ml_dir / "random_forest_rul.joblib",
        "state_model": ml_dir / "random_forest_state.joblib",
    }
    predictions.to_csv(paths["ml_predictions_csv"], index=False, encoding="utf-8")
    predictions.to_parquet(paths["ml_predictions_parquet"], index=False)
    joblib.dump(regressor, paths["rul_model"])
    joblib.dump(classifier, paths["state_model"])

    metrics = {
        "input_columns": ML_INPUT_COLUMNS,
        "split": {
            "type": "time_ordered",
            "train_share": 0.70,
            "train_rows": int(train_mask.sum()),
            "test_rows": int((~train_mask).sum()),
        },
        "rul_regressor": reg_metrics,
        "state_classifier": cls_metrics,
    }
    paths["ml_metrics_json"].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["ml_report"].write_text(_report_markdown(metrics), encoding="utf-8")
    return paths


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только строки с валидными ML-входами и сортирует ряд по времени."""
    data = dataset.sort_values("timestamp").copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data[ML_INPUT_COLUMNS] = data[ML_INPUT_COLUMNS].ffill().bfill()
    return data.dropna(subset=ML_INPUT_COLUMNS)


def _attach_features(dataset: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
    """Добавляет к каноническому датасету инженерные признаки feature builder без target leakage."""
    if features is None:
        missing = [column for column in ML_INPUT_COLUMNS if column not in dataset.columns]
        if missing:
            raise ValueError(f"Missing ML input columns: {missing}")
        return dataset

    feature_columns = [
        "timestamp",
        "deltaP_roll_mean_1h",
        "deltaP_roll_std_1h",
        "deltaP_slope_6h",
        "Q_roll_mean_1h",
        "missing_rate_1h",
        "time_above_warn",
    ]
    existing = [column for column in feature_columns if column in features.columns]
    merged = dataset.merge(features[existing], on="timestamp", how="left")
    return merged


def _time_split_mask(data: pd.DataFrame, train_share: float) -> pd.Series:
    """Делит временной ряд на train/test без перемешивания будущих точек в прошлое."""
    split_index = int(len(data) * train_share)
    mask = np.zeros(len(data), dtype=bool)
    mask[:split_index] = True
    return pd.Series(mask, index=data.index)


def _train_rul_regressor(
    data: pd.DataFrame, train_mask: pd.Series
) -> tuple[RandomForestRegressor, dict[str, float | int]]:
    """Обучает RandomForestRegressor предсказывать oracle-RUL по наблюдаемым признакам."""
    target = "RUL_oracle_h"
    valid = data[target].notna()
    train = data[train_mask & valid]
    test = data[(~train_mask) & valid]
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[ML_INPUT_COLUMNS], train[target])
    pred = model.predict(test[ML_INPUT_COLUMNS])
    rmse = float(np.sqrt(mean_squared_error(test[target], pred)))
    return model, {
        "target": target,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae_h": float(mean_absolute_error(test[target], pred)),
        "rmse_h": rmse,
        "r2": float(r2_score(test[target], pred)),
    }


def _train_state_classifier(
    data: pd.DataFrame, train_mask: pd.Series
) -> tuple[RandomForestClassifier, dict[str, object]]:
    """Обучает RandomForestClassifier определять состояние фильтра по наблюдаемым признакам."""
    target = "state"
    valid = data[target].isin(["normal", "warning", "critical"])
    train = data[train_mask & valid]
    test = data[(~train_mask) & valid]
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[ML_INPUT_COLUMNS], train[target])
    pred = model.predict(test[ML_INPUT_COLUMNS])
    return model, {
        "target": target,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "accuracy": float(accuracy_score(test[target], pred)),
        "classification_report": classification_report(
            test[target], pred, output_dict=True, zero_division=0
        ),
    }


def _build_predictions(
    data: pd.DataFrame,
    train_mask: pd.Series,
    regressor: RandomForestRegressor,
    classifier: RandomForestClassifier,
) -> pd.DataFrame:
    """Формирует таблицу предсказаний ML-бейзлайна на train и test участках."""
    result = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "filter_id": data["filter_id"],
            "scenario": data["scenario"],
            "split": np.where(train_mask, "train", "test"),
            "state_true": data["state"],
            "RUL_oracle_h": data["RUL_oracle_h"],
        }
    )
    result["state_pred"] = classifier.predict(data[ML_INPUT_COLUMNS])
    result["RUL_pred_h"] = regressor.predict(data[ML_INPUT_COLUMNS])
    return result[ML_OUTPUT_COLUMNS]


def _report_markdown(metrics: dict[str, object]) -> str:
    """Генерирует краткий отчет для магистерской работы: входы, split и метрики."""
    rul = metrics["rul_regressor"]
    cls = metrics["state_classifier"]
    return "\n".join(
        [
            "# ML baseline",
            "",
            "Классический baseline обучает две модели RandomForest:",
            "",
            "- `RandomForestRegressor` для прогноза `RUL_oracle_h`.",
            "- `RandomForestClassifier` для классификации `state`.",
            "",
            "## Входные признаки",
            "",
            *[f"- `{column}`" for column in metrics["input_columns"]],
            "",
            "Скрытые и целевые поля `clog_level`, `RUL_oracle_h`, `state` не используются как входы.",
            "",
            "## Split",
            "",
            f"- Тип: `{metrics['split']['type']}`.",
            f"- Train: `{metrics['split']['train_rows']}` строк.",
            f"- Test: `{metrics['split']['test_rows']}` строк.",
            "",
            "## RUL regression",
            "",
            f"- MAE: `{rul['mae_h']:.3f}` ч.",
            f"- RMSE: `{rul['rmse_h']:.3f}` ч.",
            f"- R2: `{rul['r2']:.3f}`.",
            "",
            "## State classification",
            "",
            f"- Accuracy: `{cls['accuracy']:.3f}`.",
            "",
        ]
    )
