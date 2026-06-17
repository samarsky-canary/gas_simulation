# Процесс работы программы

Документ описывает текущую версию прототипа без нейросетевых этапов. Основной контур строится вокруг классического ML baseline, аналитического RUL и гибридного RUL fusion.

## 1. Запуск

Основной запуск:

```powershell
.\.venv\Scripts\python.exe main.py
```

Программа читает `configs/base.yaml`, применяет сценарные параметры, выполняет полный расчет и пишет результаты в:

```text
outputs/<scenario_name>/
```

Например:

```text
outputs/slow_clogging/
```

## 2. Общий конвейер

```text
конфиг
-> симулятор
-> экспорт датасета
-> feature builder
-> RandomForest ML baseline
-> гибридные решения
-> графики
-> UI-таблицы событий обслуживания
```

## 3. Симуляция

Симулятор строит временной ряд работы одного фильтра:

- расход газа `Q_m3h`;
- входное давление `P_in_MPa`;
- температуру `T_C`;
- скрытый уровень засорения `clog_level`;
- перепад давления `deltaP_kPa`;
- выходное давление `P_out_MPa`;
- качество данных.

Шумы, пропуски, выбросы, залипания и дрейф датчиков задаются в YAML-конфиге.

## 4. Фиксированный датасет

Главные файлы:

```text
dataset.csv
dataset.parquet
dataset_schema.md
```

Минимальный контракт колонок:

- `run_id`;
- `timestamp`;
- `filter_id`;
- `scenario`;
- `P_in_MPa`;
- `P_out_MPa`;
- `deltaP_kPa`;
- `Q_m3h`;
- `T_C`;
- `clog_level`;
- `deltaP_norm_kPa`;
- `state`;
- `RUL_oracle_h`;
- `RUL_analytic_h`;
- `quality_code`.

Для входа ML-моделей можно использовать наблюдаемые и производные признаки. Нельзя использовать скрытые или целевые поля: `clog_level`, `RUL_oracle_h`, `state`.

## 5. Признаки

Feature builder создает признаки для анализа, ML baseline и гибридного слоя:

- `deltaP_norm_kPa`;
- `deltaP_roll_mean_1h`;
- `deltaP_roll_std_1h`;
- `deltaP_slope_6h`;
- `Q_roll_mean_1h`;
- `missing_rate_1h`;
- `time_above_warn`.

`deltaP_norm_kPa` важен, потому что сырой перепад зависит от расхода газа. Нормированный перепад лучше отражает именно рост сопротивления фильтра.

## 6. ML Baseline

ML baseline обучает одну модель RandomForest:

- `RandomForestRegressor` прогнозирует `RUL_oracle_h`.

Для обучения и оценки используется корпус из нескольких независимых синтетических прогонов. Каждый прогон имеет свой `run_id`, сценарий и seed. Train/test split выполняется по целым `run_id`: train содержит одни seed основных сценариев, test содержит другие seed тех же сценариев и отдельные стресс-сценарии `sensor_bias`, `sensor_stuck`, `missing_data`. Так модель видит полный диапазон RUL в train и проверяется на независимых траекториях.

Артефакты:

```text
ml_baseline/random_forest_rul.joblib
ml_baseline/ml_predictions.csv
ml_baseline/ml_predictions.parquet
ml_baseline/ml_metrics.json
ml_baseline/ml_baseline_report.md
```

ML-прогноз RUL записывается как `RUL_ml_h`.

## 7. Гибридные решения

Гибридный слой объединяет:

- качество данных;
- ML-прогноз RUL;
- аналитический RUL;
- согласованность прогнозов.

На выходе формируются:

- `RUL_fused_h`;
- `rul_source`;
- `confidence_data`;
- `confidence_model`;
- `confidence_consistency`;
- `confidence_total`.

Файлы:

```text
hybrid/hybrid_decisions.csv
hybrid/hybrid_decisions.parquet
hybrid/hybrid_decision_logic.md
```

## 8. Как читать решение

`rul_source` показывает, откуда взят итоговый RUL:

- `ml_baseline` - использован ML-прогноз;
- `conservative_min` - взят минимум из ML и аналитического RUL;
- `analytic_fallback` - использован аналитический RUL;
- `analytic_data_veto` - ML заблокирован из-за качества данных.

`confidence_total` показывает общее доверие к прогнозу. Низкое значение обычно ведет к аналитическому fallback.

## 9. События обслуживания

UI рассчитывает события планового и срочного обслуживания отдельно по `RUL_ml_h`, `RUL_analytic_h` и `RUL_fused_h`. Событие фиксируется, когда RUL устойчиво ниже порога обслуживания с учетом `stable_degraded_rul_h`.

## 10. Графики

Графики находятся в:

Статические PNG-графики больше не формируются в `outputs`. Визуализация выполняется в Streamlit через интерактивные Plotly-графики: давление, перепад, состояние фильтра, сравнение RUL, периоды предпочтения источника RUL и доверие к прогнозу.

## 11. Итог

Текущий результат - проверяемый ML-only стенд:

```text
симулятор + признаки + RandomForest ML + гибридный RUL fusion + интерактивный вывод
```

Эта версия самодостаточна для анализа качества признаков, сравнения ML с аналитическим RUL, проверки гибридного RUL fusion и подготовки разделов магистерской работы.
