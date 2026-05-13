from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


LSTM_INPUT_COLUMNS = [
    "P_in_MPa",
    "P_out_MPa",
    "deltaP_kPa",
    "Q_m3h",
    "T_C",
    "rho_rel",
    "deltaP_norm_kPa",
]

STATE_CLASSES = ["normal", "warning", "critical", "unknown"]
STATE_TO_ID = {state: idx for idx, state in enumerate(STATE_CLASSES)}


@dataclass(frozen=True)
class LSTMWindowDataset:
    """Контейнер с окнами временного ряда для RUL и state-задач."""

    X_rul: np.ndarray
    y_rul: np.ndarray
    timestamps_rul: np.ndarray
    X_state: np.ndarray
    y_state: np.ndarray
    timestamps_state: np.ndarray
    sequence_length: int
    feature_names: list[str]
    state_classes: list[str]


def build_lstm_windows(cfg: ScenarioConfig, dataset: pd.DataFrame) -> LSTMWindowDataset:
    """Преобразует временной ряд в окна вида X=(samples, sequence_length, features)."""
    data = dataset.sort_values("timestamp").copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    sequence_length = _sequence_length(cfg)

    # Входы LSTM состоят только из наблюдаемых и производных признаков без скрытых target-полей.
    inputs = data[LSTM_INPUT_COLUMNS].ffill().bfill()
    windows: list[np.ndarray] = []
    end_indices: list[int] = []
    for end in range(sequence_length - 1, len(data), cfg.lstm_stride_steps):
        start = end - sequence_length + 1
        window = inputs.iloc[start : end + 1].to_numpy(dtype=np.float32)
        if np.isnan(window).any():
            continue
        windows.append(window)
        end_indices.append(end)

    if not windows:
        empty_X = np.empty((0, sequence_length, len(LSTM_INPUT_COLUMNS)), dtype=np.float32)
        empty_t = np.array([], dtype="datetime64[ns]")
        return LSTMWindowDataset(
            X_rul=empty_X,
            y_rul=np.array([], dtype=np.float32),
            timestamps_rul=empty_t,
            X_state=empty_X,
            y_state=np.array([], dtype=np.int64),
            timestamps_state=empty_t,
            sequence_length=sequence_length,
            feature_names=LSTM_INPUT_COLUMNS,
            state_classes=STATE_CLASSES,
        )

    X_all = np.stack(windows).astype(np.float32)
    indexed = data.iloc[end_indices].reset_index(drop=True)

    # Для RUL-регрессии отбрасываем окна, где oracle-RUL цензурирован и target отсутствует.
    rul_target = indexed["RUL_oracle_h"].to_numpy(dtype=np.float32)
    rul_mask = ~np.isnan(rul_target)
    X_rul = X_all[rul_mask]
    y_rul = rul_target[rul_mask]
    timestamps_rul = indexed.loc[rul_mask, "timestamp"].to_numpy()

    # Для state-классификации кодируем состояния в стабильные целочисленные классы.
    state_ids = indexed["state"].map(STATE_TO_ID)
    state_mask = state_ids.notna().to_numpy()
    X_state = X_all[state_mask]
    y_state = state_ids[state_mask].to_numpy(dtype=np.int64)
    timestamps_state = indexed.loc[state_mask, "timestamp"].to_numpy()

    return LSTMWindowDataset(
        X_rul=X_rul,
        y_rul=y_rul,
        timestamps_rul=timestamps_rul,
        X_state=X_state,
        y_state=y_state,
        timestamps_state=timestamps_state,
        sequence_length=sequence_length,
        feature_names=LSTM_INPUT_COLUMNS,
        state_classes=STATE_CLASSES,
    )


def export_lstm_windows(
    cfg: ScenarioConfig, windows: LSTMWindowDataset, output_dir: Path
) -> dict[str, Path]:
    """Сохраняет LSTM-окна, target-массивы и описание формата."""
    lstm_dir = output_dir / "lstm_windows"
    lstm_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "lstm_rul_npz": lstm_dir / "lstm_rul_windows.npz",
        "lstm_state_npz": lstm_dir / "lstm_state_windows.npz",
        "lstm_metadata": lstm_dir / "lstm_windows_metadata.json",
        "lstm_description": lstm_dir / "lstm_windows_description.md",
    }
    np.savez_compressed(
        paths["lstm_rul_npz"],
        X=windows.X_rul,
        y=windows.y_rul,
        timestamps=_timestamp_strings(windows.timestamps_rul),
        feature_names=np.array(windows.feature_names),
    )
    np.savez_compressed(
        paths["lstm_state_npz"],
        X=windows.X_state,
        y=windows.y_state,
        timestamps=_timestamp_strings(windows.timestamps_state),
        feature_names=np.array(windows.feature_names),
        state_classes=np.array(windows.state_classes),
    )
    metadata = _metadata(cfg, windows)
    paths["lstm_metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["lstm_description"].write_text(_description(metadata), encoding="utf-8")
    return paths


def _sequence_length(cfg: ScenarioConfig) -> int:
    """Переводит окно в часах в число точек ряда с учетом шага дискретизации."""
    return int(round(cfg.lstm_window_hours * 60 / cfg.step_minutes))


def _timestamp_strings(values: np.ndarray) -> np.ndarray:
    """Сохраняет временные метки как строки ISO, чтобы не терять timezone в NumPy."""
    return np.array([pd.Timestamp(value).isoformat() for value in values])


def _metadata(cfg: ScenarioConfig, windows: LSTMWindowDataset) -> dict[str, object]:
    """Собирает машинно-читаемое описание формы массивов и используемых колонок."""
    return {
        "window_hours": cfg.lstm_window_hours,
        "step_minutes": cfg.step_minutes,
        "stride_steps": cfg.lstm_stride_steps,
        "sequence_length": windows.sequence_length,
        "input_features": windows.feature_names,
        "forbidden_input_columns": ["clog_level", "RUL_oracle_h", "state"],
        "rul_task": {
            "target": "RUL_oracle_h",
            "X_shape": list(windows.X_rul.shape),
            "y_shape": list(windows.y_rul.shape),
        },
        "state_task": {
            "target": "state",
            "classes": windows.state_classes,
            "X_shape": list(windows.X_state.shape),
            "y_shape": list(windows.y_state.shape),
        },
    }


def _description(metadata: dict[str, object]) -> str:
    """Генерирует markdown-описание LSTM-окон для отчета и проверки."""
    return "\n".join(
        [
            "# LSTM windows",
            "",
            "Временной ряд преобразован в окна фиксированной длины.",
            "",
            f"- Окно: `{metadata['window_hours']}` ч.",
            f"- Шаг данных: `{metadata['step_minutes']}` мин.",
            f"- Длина последовательности: `{metadata['sequence_length']}` точек.",
            f"- Stride: `{metadata['stride_steps']}` шаг.",
            "",
            "## Входные признаки",
            "",
            *[f"- `{column}`" for column in metadata["input_features"]],
            "",
            "## Нельзя подавать на вход",
            "",
            *[f"- `{column}`" for column in metadata["forbidden_input_columns"]],
            "",
            "## RUL task",
            "",
            f"- Target: `{metadata['rul_task']['target']}`.",
            f"- `X.shape = {tuple(metadata['rul_task']['X_shape'])}`.",
            f"- `y.shape = {tuple(metadata['rul_task']['y_shape'])}`.",
            "",
            "## State task",
            "",
            f"- Target: `{metadata['state_task']['target']}`.",
            f"- Classes: `{metadata['state_task']['classes']}`.",
            f"- `X.shape = {tuple(metadata['state_task']['X_shape'])}`.",
            f"- `y.shape = {tuple(metadata['state_task']['y_shape'])}`.",
            "",
        ]
    )
