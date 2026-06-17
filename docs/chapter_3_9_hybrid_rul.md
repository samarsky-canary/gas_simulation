# 3.9. Реализация гибридного RUL

Гибридный слой реализован в модуле `src/hybrid/decision_engine.py`. Его задача - объединить три источника информации: ML-прогноз `RUL_ml_h`, аналитическую оценку `RUL_analytic_h` и качество входных данных.

На вход `build_hybrid_decisions(...)` подаются канонический датасет, таблица инженерных признаков и таблица ML-предсказаний. Слой объединяет их по `timestamp`, а при наличии нескольких прогонов - по `run_id` и `timestamp`.

Для каждой временной точки рассчитываются:

- `confidence_data`;
- `confidence_model`;
- `confidence_consistency`;
- `confidence_total`;
- `RUL_fused_h`;
- `rul_source`.

Итоговый `RUL_fused_h` выбирается по правилам: использовать ML при высоком доверии, взять консервативный минимум при среднем доверии или перейти к аналитическому fallback при низком доверии, пропусках ML либо вето качества данных.

Слой больше не назначает действия обслуживания. События планового и срочного обслуживания в UI рассчитываются отдельно по устойчивому пересечению RUL-порогов.

| Функция | Назначение |
|---|---|
| `build_hybrid_decisions(cfg, dataset, features, ml_predictions)` | Основная функция гибридного слоя. Объединяет датасет, признаки и ML-прогнозы, затем для каждой строки рассчитывает итоговый RUL и доверие. |
| `_prepare_inputs(dataset, features, ml_predictions)` | Добавляет признаки качества окна, тренд перепада, время выше warning-порога и ML-прогноз `RUL_ml_h`. |
| `_fuse_row(cfg, row)` | Считает confidence и вызывает выбор итогового RUL для одной временной точки. |
| `_confidence_data(row, hard_veto)` | Оценивает доверие к данным по `quality_code`, физическим нарушениям и `missing_rate_1h`. |
| `_confidence_model(row)` | Оценивает применимость ML-прогноза. |
| `_confidence_consistency(cfg, row)` | Сравнивает `RUL_ml_h` и `RUL_analytic_h`. |
| `_fuse_rul(row, confidence_total, confidence_consistency, data_hard_veto)` | Выбирает `RUL_fused_h` и `rul_source`. Возможные источники: `ml_baseline`, `conservative_min`, `analytic_fallback`, `analytic_data_veto`, `unavailable`. |
| `_has_physical_violation(row)` | Проверяет физически невозможные наблюдения: `P_out > P_in`, отрицательный `deltaP_kPa` или отрицательный расход. |
| `export_hybrid_decisions(decisions, output_dir)` | Сохраняет результаты гибридного слоя в Parquet/CSV и Markdown-описание логики. |
