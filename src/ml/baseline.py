from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
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
    "run_id",
    "timestamp",
    "filter_id",
    "scenario",
    "split",
    "RUL_oracle_h",
    "RUL_pred_h",
]


def train_and_export_ml_baseline(
    dataset: pd.DataFrame,
    output_dir: Path,
    features: pd.DataFrame | None = None,
    test_run_ids: set[str] | None = None,
) -> dict[str, Path]:
    """Обучает RandomForest-бейзлайн для RUL и сохраняет метрики/предсказания."""
    ml_dir = output_dir / "ml_baseline"
    ml_dir.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_dataset(_attach_features(dataset, features))
    train_mask, split_info = _split_mask(prepared, train_share=0.70, test_run_ids=test_run_ids)

    regressor, reg_metrics = _train_rul_regressor(prepared, train_mask)
    predictions = _build_predictions(prepared, train_mask, regressor)

    paths = {
        "ml_predictions_csv": ml_dir / "ml_predictions.csv",
        "ml_predictions_parquet": ml_dir / "ml_predictions.parquet",
        "ml_metrics_json": ml_dir / "ml_metrics.json",
        "ml_report": ml_dir / "ml_baseline_report.md",
        "rul_model": ml_dir / "random_forest_rul.joblib",
    }
    predictions.to_csv(paths["ml_predictions_csv"], index=False, encoding="utf-8")
    predictions.to_parquet(paths["ml_predictions_parquet"], index=False)
    joblib.dump(regressor, paths["rul_model"])

    metrics = {
        "input_columns": ML_INPUT_COLUMNS,
        "split": split_info,
        "rul_regressor": reg_metrics,
    }
    paths["ml_metrics_json"].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["ml_report"].write_text(_report_markdown(metrics), encoding="utf-8")
    return paths


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только строки с валидными ML-входами и сортирует ряд по времени."""
    sort_columns = ["run_id", "timestamp"] if "run_id" in dataset.columns else ["timestamp"]
    data = dataset.sort_values(sort_columns).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if "run_id" in data.columns:
        data[ML_INPUT_COLUMNS] = data.groupby("run_id")[ML_INPUT_COLUMNS].transform(
            lambda column: column.ffill().bfill()
        )
    else:
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
        "deltaP_roll_mean_1h",
        "deltaP_roll_std_1h",
        "deltaP_slope_6h",
        "Q_roll_mean_1h",
        "missing_rate_1h",
        "time_above_warn",
    ]
    merge_keys = ["timestamp"]
    if "run_id" in dataset.columns and "run_id" in features.columns:
        merge_keys = ["run_id", "timestamp"]
    existing = [*merge_keys, *[column for column in feature_columns if column in features.columns]]
    merged = dataset.merge(features[existing], on=merge_keys, how="left")
    return merged


def _split_mask(
    data: pd.DataFrame, train_share: float, test_run_ids: set[str] | None = None
) -> tuple[pd.Series, dict[str, object]]:
    """Выбирает train/test split: по run_id для корпуса прогонов, иначе по времени."""
    if "run_id" in data.columns and data["run_id"].nunique() > 1:
        return _run_id_split_mask(data, train_share=train_share, test_run_ids=test_run_ids)
    mask = _time_split_mask(data, train_share=train_share)
    return mask, {
        "type": "time_ordered",
        "train_share": train_share,
        "train_rows": int(mask.sum()),
        "test_rows": int((~mask).sum()),
    }


def _time_split_mask(data: pd.DataFrame, train_share: float) -> pd.Series:
    """Делит временной ряд на train/test без перемешивания будущих точек в прошлое."""
    split_index = int(len(data) * train_share)
    mask = np.zeros(len(data), dtype=bool)
    mask[:split_index] = True
    return pd.Series(mask, index=data.index)


def _run_id_split_mask(
    data: pd.DataFrame, train_share: float, test_run_ids: set[str] | None = None
) -> tuple[pd.Series, dict[str, object]]:
    """Делит датасет по целым независимым прогонам, чтобы не смешивать соседние точки."""
    all_run_ids = sorted(str(run_id) for run_id in data["run_id"].dropna().unique())
    if test_run_ids:
        selected_test = sorted(set(test_run_ids) & set(all_run_ids))
    else:
        test_count = max(1, int(round(len(all_run_ids) * (1.0 - train_share))))
        selected_test = all_run_ids[-test_count:]
    selected_train = [run_id for run_id in all_run_ids if run_id not in selected_test]
    if not selected_train or not selected_test:
        mask = _time_split_mask(data, train_share=train_share)
        return mask, {
            "type": "time_ordered_fallback",
            "train_share": train_share,
            "train_rows": int(mask.sum()),
            "test_rows": int((~mask).sum()),
        }
    mask = ~data["run_id"].astype(str).isin(selected_test)
    return mask, {
        "type": "group_by_run_id",
        "train_run_ids": selected_train,
        "test_run_ids": selected_test,
        "train_rows": int(mask.sum()),
        "test_rows": int((~mask).sum()),
    }


def _train_rul_regressor(
    data: pd.DataFrame, train_mask: pd.Series
) -> tuple[RandomForestRegressor, dict[str, object]]:
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
    metrics = {
        "target": target,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        **_regression_metrics(test[target], pred),
    }
    metrics["by_run_id"] = _regression_group_metrics(test, pred, target, "run_id")
    metrics["by_scenario"] = _regression_group_metrics(test, pred, target, "scenario")
    return model, metrics


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | None]:
    """Считает основные метрики регрессии RUL и безопасно обрабатывает маленькие группы."""
    if len(y_true) == 0:
        return {"mae_h": None, "rmse_h": None, "r2": None}
    result: dict[str, float | None] = {
        "mae_h": float(mean_absolute_error(y_true, y_pred)),
        "rmse_h": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": None,
    }
    if len(y_true) >= 2:
        result["r2"] = float(r2_score(y_true, y_pred))
    return result


def _regression_group_metrics(
    test: pd.DataFrame, y_pred: np.ndarray, target: str, group_column: str
) -> list[dict[str, object]]:
    """Считает RUL-метрики отдельно по run_id или scenario."""
    if group_column not in test.columns or len(test) == 0:
        return []
    scored = test[[group_column, target]].copy()
    scored["prediction"] = y_pred
    rows: list[dict[str, object]] = []
    for group_value, group in scored.groupby(group_column, sort=True):
        rows.append(
            {
                group_column: str(group_value),
                "rows": int(len(group)),
                **_regression_metrics(group[target], group["prediction"].to_numpy()),
            }
        )
    return rows


def _build_predictions(
    data: pd.DataFrame,
    train_mask: pd.Series,
    regressor: RandomForestRegressor,
) -> pd.DataFrame:
    """Формирует таблицу предсказаний ML-бейзлайна на train и test участках."""
    result = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "run_id": data["run_id"] if "run_id" in data.columns else "",
            "filter_id": data["filter_id"],
            "scenario": data["scenario"],
            "split": np.where(train_mask, "train", "test"),
            "RUL_oracle_h": data["RUL_oracle_h"],
        }
    )
    result["RUL_pred_h"] = regressor.predict(data[ML_INPUT_COLUMNS])
    return result[ML_OUTPUT_COLUMNS]


def _report_markdown(metrics: dict[str, object]) -> str:
    """Генерирует краткий отчет для магистерской работы: входы, split и метрики."""
    rul = metrics["rul_regressor"]
    split = metrics["split"]
    split_lines = [
        "## Split",
        "",
        f"- Тип: `{split['type']}`.",
    ]
    if "train_run_ids" in split:
        split_lines.append(f"- Train run_id: `{', '.join(split['train_run_ids'])}`.")
    if "test_run_ids" in split:
        split_lines.append(f"- Test run_id: `{', '.join(split['test_run_ids'])}`.")
    split_lines.extend(
        [
            f"- Train: `{split['train_rows']}` строк.",
            f"- Test: `{split['test_rows']}` строк.",
            "",
        ]
    )
    return "\n".join(
        [
            "# ML baseline",
            "",
            "Классический baseline обучает одну модель RandomForest:",
            "",
            "- `RandomForestRegressor` для прогноза `RUL_oracle_h`.",
            "",
            "## Входные признаки",
            "",
            *[f"- `{column}`" for column in metrics["input_columns"]],
            "",
            "Скрытые и целевые поля `clog_level`, `RUL_oracle_h`, `state` не используются как входы.",
            "",
            *split_lines,
            "## RUL regression",
            "",
            f"- MAE: `{_fmt_metric(rul['mae_h'])}` ч.",
            f"- RMSE: `{_fmt_metric(rul['rmse_h'])}` ч.",
            f"- R2: `{_fmt_metric(rul['r2'])}`.",
            "",
            "### RUL по сценариям",
            "",
            *_regression_markdown_rows(rul.get("by_scenario", []), "scenario"),
            "",
        ]
    )


def _regression_markdown_rows(rows: object, key: str) -> list[str]:
    """Форматирует групповые RUL-метрики для markdown-отчета."""
    if not rows:
        return ["- Нет групповых метрик."]
    return [
        (
            f"- `{row[key]}`: rows=`{row['rows']}`, "
            f"MAE=`{_fmt_metric(row['mae_h'])}` ч, "
            f"RMSE=`{_fmt_metric(row['rmse_h'])}` ч, "
            f"R2=`{_fmt_metric(row['r2'])}`."
        )
        for row in rows
    ]


def _fmt_metric(value: object) -> str:
    """Форматирует числовую метрику или пустое значение."""
    if value is None or pd.isna(value):
        return "н/д"
    return f"{float(value):.3f}"
