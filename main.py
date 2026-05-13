from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import build_features, export_features
from src.hybrid import build_hybrid_decisions, export_hybrid_decisions
from src.lstm import build_lstm_windows, export_lstm_windows
from src.ml import train_and_export_ml_baseline
from src.rules import apply_rule_baseline, export_rule_baseline
from src.simulator.config import load_config
from src.simulator.exporters import export_run
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
    "state_model": "модель RandomForest для state",
    "hybrid_decisions_csv": "гибридные решения CSV",
    "hybrid_decisions_parquet": "гибридные решения Parquet",
    "hybrid_decisions_ru_csv": "гибридные решения CSV на русском",
    "hybrid_description": "описание гибридной логики",
    "hybrid_decision_packages_jsonl": "пакеты объяснения решений JSONL",
    "hybrid_decision_cards_md": "карточки объяснения решений Markdown",
    "lstm_rul_npz": "LSTM окна для RUL",
    "lstm_state_npz": "LSTM окна для state",
    "lstm_metadata": "метаданные LSTM-окон",
    "lstm_description": "описание LSTM-окон",
}


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
    ml_paths = train_and_export_ml_baseline(dataset, output_dir, features)
    ml_predictions = pd.read_parquet(ml_paths["ml_predictions_parquet"])
    hybrid_decisions = build_hybrid_decisions(cfg, dataset, features, ml_predictions)
    hybrid_paths = export_hybrid_decisions(hybrid_decisions, output_dir)
    lstm_windows = build_lstm_windows(cfg, dataset)
    lstm_paths = export_lstm_windows(cfg, lstm_windows, output_dir)
    plot_paths = build_plots(cfg, df, output_dir)

    print(f"Сгенерировано строк: {len(df)}")
    print(f"Сценарий: {cfg.scenario_name}")
    print(f"Строк с флагами качества: {report.rows_with_flags}")
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
    print("Созданы LSTM окна:")
    for name, path in lstm_paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Созданные графики:")
    for path in plot_paths.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
