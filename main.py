from __future__ import annotations

from pathlib import Path

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
}


def main() -> None:
    cfg = load_config(Path("configs/base.yaml"))
    df, report = run_scenario(cfg)
    output_dir = Path("outputs") / cfg.scenario_name
    paths = export_run(cfg, df, report, output_dir)
    plot_paths = build_plots(cfg, df, output_dir)

    print(f"Сгенерировано строк: {len(df)}")
    print(f"Сценарий: {cfg.scenario_name}")
    print(f"Строк с флагами качества: {report.rows_with_flags}")
    print("Созданные файлы:")
    for name, path in paths.items():
        label = OUTPUT_LABELS.get(name, name)
        print(f"- {label}: {path}")
    print("Созданные графики:")
    for path in plot_paths.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
