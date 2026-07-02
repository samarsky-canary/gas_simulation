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
