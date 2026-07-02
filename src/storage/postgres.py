from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from src.ml import TrainedMLBaseline


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ml_models (
    model_id uuid PRIMARY KEY,
    model_name varchar(100) NOT NULL UNIQUE,
    model_type varchar(100) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ml_training_runs (
    training_id uuid PRIMARY KEY,
    model_id uuid NOT NULL REFERENCES ml_models(model_id),
    status varchar(20) NOT NULL CHECK (status IN ('completed', 'failed')),
    trained_at timestamptz NOT NULL DEFAULT now(),
    split_type varchar(50) NOT NULL,
    train_rows integer NOT NULL CHECK (train_rows >= 0),
    test_rows integer NOT NULL CHECK (test_rows >= 0),
    input_columns jsonb NOT NULL,
    metrics jsonb NOT NULL,
    report_markdown text NOT NULL,
    artifact_format varchar(50) NOT NULL,
    artifact_sha256 char(64) NOT NULL,
    artifact bytea NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_ml_training_runs_latest
    ON ml_training_runs (model_id, trained_at DESC)
    WHERE status = 'completed';

CREATE TABLE IF NOT EXISTS latest_run_metadata (
    singleton_id boolean PRIMARY KEY DEFAULT true CHECK (singleton_id),
    run_id text NOT NULL,
    scenario_name text NOT NULL,
    filter_id text NOT NULL,
    saved_at timestamptz NOT NULL DEFAULT now(),
    row_count integer NOT NULL CHECK (row_count >= 0),
    feature_row_count integer NOT NULL CHECK (feature_row_count >= 0),
    config jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS latest_run_raw_metrics (
    row_index integer PRIMARY KEY,
    run_id text NOT NULL,
    timestamp timestamptz NOT NULL,
    payload jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_latest_run_raw_metrics_timestamp
    ON latest_run_raw_metrics (timestamp);

CREATE TABLE IF NOT EXISTS latest_run_features (
    row_index integer PRIMARY KEY,
    run_id text NOT NULL,
    timestamp timestamptz NOT NULL,
    payload jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_latest_run_features_timestamp
    ON latest_run_features (timestamp);
"""


@dataclass(frozen=True)
class StoredMLTraining:
    training_id: UUID
    model_id: UUID
    model_name: str
    model_type: str
    trained_at: datetime
    model: Any
    metrics: dict[str, object]
    report: str
    artifact_sha256: str


class TrainingRepository:
    """Хранилище версий обученной ML-модели в PostgreSQL."""

    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("DATABASE_URL is required.")
        self.database_url = database_url

    @classmethod
    def from_env(cls) -> "TrainingRepository":
        return cls(os.environ.get("DATABASE_URL", ""))

    def initialize(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(SCHEMA_SQL)

    def save(
        self,
        trained: TrainedMLBaseline,
        *,
        model_name: str = "random_forest_rul",
        model_type: str = "RandomForestRegressor",
    ) -> StoredMLTraining:
        artifact = _serialize_model(trained.model)
        artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        split = trained.metrics["split"]
        rul = trained.metrics["rul_regressor"]

        with psycopg.connect(self.database_url) as connection:
            row = connection.execute(
                """
                INSERT INTO ml_models (model_id, model_name, model_type)
                VALUES (gen_random_uuid(), %s, %s)
                ON CONFLICT (model_name) DO UPDATE
                SET model_type = EXCLUDED.model_type
                RETURNING model_id
                """,
                (model_name, model_type),
            ).fetchone()
            model_id = row[0]
            training_row = connection.execute(
                """
                INSERT INTO ml_training_runs (
                    training_id, model_id, status, split_type, train_rows, test_rows,
                    input_columns, metrics, report_markdown, artifact_format,
                    artifact_sha256, artifact
                )
                VALUES (
                    gen_random_uuid(), %s, 'completed', %s, %s, %s,
                    %s, %s, %s, 'joblib', %s, %s
                )
                RETURNING training_id, trained_at
                """,
                (
                    model_id,
                    split["type"],
                    rul["train_rows"],
                    rul["test_rows"],
                    Jsonb(trained.metrics["input_columns"]),
                    Jsonb(trained.metrics),
                    trained.report,
                    artifact_sha256,
                    artifact,
                ),
            ).fetchone()

        return StoredMLTraining(
            training_id=training_row[0],
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            trained_at=training_row[1],
            model=trained.model,
            metrics=trained.metrics,
            report=trained.report,
            artifact_sha256=artifact_sha256,
        )

    def load_latest(self, model_name: str = "random_forest_rul") -> StoredMLTraining:
        with psycopg.connect(self.database_url) as connection:
            row = connection.execute(
                """
                SELECT
                    t.training_id, m.model_id, m.model_name, m.model_type, t.trained_at,
                    t.artifact, t.metrics, t.report_markdown, t.artifact_sha256
                FROM ml_training_runs t
                JOIN ml_models m ON m.model_id = t.model_id
                WHERE m.model_name = %s AND t.status = 'completed'
                ORDER BY t.trained_at DESC
                LIMIT 1
                """,
                (model_name,),
            ).fetchone()
        if row is None:
            raise RuntimeError(
                f"No completed training found for model {model_name!r}. "
                "Run `python main.py --train-ml` first."
            )

        artifact = bytes(row[5])
        actual_sha256 = hashlib.sha256(artifact).hexdigest()
        if actual_sha256 != row[8]:
            raise RuntimeError(f"Stored artifact checksum mismatch for training {row[0]}.")
        return StoredMLTraining(
            training_id=row[0],
            model_id=row[1],
            model_name=row[2],
            model_type=row[3],
            trained_at=row[4],
            model=_deserialize_model(artifact),
            metrics=row[6],
            report=row[7],
            artifact_sha256=row[8],
        )


class LatestRunRepository:
    """Хранилище сырых метрик и признаков последнего пользовательского прогона."""

    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("DATABASE_URL is required.")
        self.database_url = database_url

    @classmethod
    def from_env(cls) -> "LatestRunRepository":
        return cls(os.environ.get("DATABASE_URL", ""))

    def initialize(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(SCHEMA_SQL)

    def replace(
        self,
        *,
        cfg: Any,
        raw_metrics: pd.DataFrame,
        features: pd.DataFrame,
    ) -> None:
        """Заменяет содержимое таблиц последнего прогона свежими данными."""
        run_id = _first_value(raw_metrics, "run_id")
        filter_id = _first_value(raw_metrics, "filter_id")
        scenario_name = getattr(cfg, "scenario_name", _first_value(raw_metrics, "scenario_id"))
        config = (
            cfg.model_dump(mode="json")
            if hasattr(cfg, "model_dump")
            else _to_jsonable(cfg)
        )
        raw_rows = _dataframe_payload_rows(raw_metrics)
        feature_rows = _dataframe_payload_rows(features)

        with psycopg.connect(self.database_url) as connection:
            connection.execute(SCHEMA_SQL)
            connection.execute("TRUNCATE latest_run_features, latest_run_raw_metrics")
            connection.execute("DELETE FROM latest_run_metadata")
            connection.execute(
                """
                INSERT INTO latest_run_metadata (
                    singleton_id, run_id, scenario_name, filter_id,
                    row_count, feature_row_count, config
                )
                VALUES (true, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    scenario_name,
                    filter_id,
                    len(raw_metrics),
                    len(features),
                    Jsonb(config),
                ),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO latest_run_raw_metrics (
                        row_index, run_id, timestamp, payload
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    raw_rows,
                )
                cursor.executemany(
                    """
                    INSERT INTO latest_run_features (
                        row_index, run_id, timestamp, payload
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    feature_rows,
                )


def _serialize_model(model: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def _deserialize_model(artifact: bytes) -> Any:
    return joblib.load(io.BytesIO(artifact))


def _dataframe_payload_rows(
    data: pd.DataFrame,
) -> list[tuple[int, str, datetime, Jsonb]]:
    rows: list[tuple[int, str, datetime, Jsonb]] = []
    for row_index, record in enumerate(data.to_dict(orient="records")):
        payload = {key: _to_jsonable(value) for key, value in record.items()}
        run_id = str(payload.get("run_id") or "")
        timestamp = _timestamp_value(record.get("timestamp"))
        if not run_id:
            raise ValueError("DataFrame row has empty run_id.")
        if timestamp is None:
            raise ValueError("DataFrame row has empty timestamp.")
        rows.append((row_index, run_id, timestamp, Jsonb(payload)))
    return rows


def _timestamp_value(value: object) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp.to_pydatetime()


def _first_value(data: pd.DataFrame, column: str) -> str:
    if column not in data.columns or data.empty:
        return ""
    value = data[column].iloc[0]
    if pd.isna(value):
        return ""
    return str(value)


def _to_jsonable(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
