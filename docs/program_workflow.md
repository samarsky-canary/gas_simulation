# Процесс работы программы

Документ описывает текущую версию прототипа без нейросетевых этапов. Основной контур теперь строится вокруг классического ML baseline, аналитического RUL и rule-based decision layer.

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
-> rule-based baseline
-> RandomForest ML baseline
-> гибридные решения
-> графики
-> консольная сводка и карточки решений
```

## 3. Симуляция

Симулятор строит временной ряд работы одного фильтра:

- расход газа `Q_m3h`;
- входное давление `P_in_MPa`;
- температуру `T_C`;
- скрытый уровень засорения `clog_level`;
- перепад давления `deltaP_kPa`;
- выходное давление `P_out_MPa`;
- качество данных и флаги дефектов.

Шумы, пропуски, выбросы, залипания и дрейф датчиков задаются в YAML-конфиге.

## 4. Фиксированный датасет

Главные файлы:

```text
dataset.csv
dataset.parquet
dataset_schema.md
```

Минимальный контракт колонок:

- `timestamp`;
- `filter_id`;
- `scenario`;
- `P_in_MPa`;
- `P_out_MPa`;
- `deltaP_kPa`;
- `Q_m3h`;
- `T_C`;
- `rho_rel`;
- `clog_level`;
- `deltaP_norm_kPa`;
- `state`;
- `RUL_oracle_h`;
- `RUL_analytic_h`;
- `quality_code`;
- `fault_flags`.

Для входа ML-моделей можно использовать наблюдаемые и производные признаки. Нельзя использовать скрытые или целевые поля: `clog_level`, `RUL_oracle_h`, `state`.

## 5. Признаки

Feature builder создает признаки для анализа, ML baseline и правил:

- `deltaP_norm_kPa`;
- `deltaP_roll_mean_1h`;
- `deltaP_roll_std_1h`;
- `deltaP_slope_6h`;
- `Q_roll_mean_1h`;
- `missing_rate_1h`;
- `time_above_warn`.

`deltaP_norm_kPa` важен, потому что сырой перепад зависит от расхода газа. Нормированный перепад лучше отражает именно рост сопротивления фильтра.

## 6. Rule-Based Baseline

Rule-based baseline определяет состояние и рекомендацию по простым правилам:

- ниже warning-порога - `normal`;
- между warning и critical - `warning`;
- выше critical - `critical`;
- плохие или неполные данные - `unknown`;
- малый `RUL_analytic_h` - плановое или срочное обслуживание.

Это регламентно-логическая точка сравнения для ML.

## 7. ML Baseline

ML baseline обучает две модели RandomForest:

- `RandomForestRegressor` прогнозирует `RUL_oracle_h`;
- `RandomForestClassifier` прогнозирует `state`.

Артефакты:

```text
ml_baseline/random_forest_rul.joblib
ml_baseline/random_forest_state.joblib
ml_baseline/ml_predictions.csv
ml_baseline/ml_predictions.parquet
ml_baseline/ml_metrics.json
ml_baseline/ml_baseline_report.md
```

ML-прогноз RUL записывается как `RUL_ml_h`.

## 8. Гибридные решения

Гибридный слой объединяет:

- качество данных;
- ML-прогноз RUL;
- аналитический RUL;
- согласованность прогнозов;
- правила безопасности и обслуживания.

На выходе формируются:

- `RUL_fused_h`;
- `rul_source`;
- `confidence_data`;
- `confidence_model`;
- `confidence_consistency`;
- `confidence_total`;
- `action`;
- `priority`;
- `due_time_h`;
- `rule_trace`;
- `explanation`.

Файлы:

```text
hybrid/hybrid_decisions.csv
hybrid/hybrid_decisions.parquet
hybrid/hybrid_decisions_ru.csv
hybrid/hybrid_decision_logic.md
hybrid/decision_packages.jsonl
hybrid/decision_cards.md
```

## 9. Как читать решение

Основные действия:

- `monitor` - продолжать мониторинг;
- `sensor_check` - проверить датчики;
- `planned_maintenance` - назначить плановое обслуживание;
- `urgent_maintenance` - назначить срочное обслуживание;
- `shutdown_request` - запросить безопасный останов;
- `manual_review` - передать случай инженеру.

`rul_source` показывает, откуда взят итоговый RUL:

- `ml_baseline` - использован ML-прогноз;
- `conservative_min` - взят минимум из ML и аналитического RUL;
- `analytic_fallback` - использован аналитический RUL;
- `analytic_data_veto` - ML заблокирован из-за качества данных.

`confidence_total` показывает общее доверие к решению. Низкое значение обычно ведет к fallback или ручной проверке.

## 10. Карточки решений

В конце запуска программа печатает краткие карточки в консоль. Полный человекочитаемый файл:

```text
hybrid/decision_cards.md
```

Карточка показывает:

- фильтр и время;
- состояние;
- качество данных;
- ML RUL;
- аналитический RUL;
- итоговый RUL;
- источник RUL;
- confidence;
- действие;
- приоритет;
- срок;
- сработавшие правила;
- объяснение;
- что делать.

## 11. Графики

Графики находятся в:

```text
plots/
```

Ключевые графики:

- `01_rashod_q.png` - расход газа;
- `02_davleniya_pin_pout.png` - входное и выходное давление;
- `03_perepad_delta_p.png` - сырой перепад давления;
- `04_zasorenie_clog_level.png` - скрытое засорение;
- `05_ostatochnyi_resurs_rul.png` - остаточный ресурс;
- `06_sostoyanie_filtra.png` - состояние фильтра и отметки карточек решений;
- `07_normirovannyi_perepad.png` - нормированный перепад с warning/critical порогами;
- `08_delta_p_i_zasorenie.png` - сравнение перепада и засорения.

## 12. Итог

Текущий результат - проверяемый ML-only стенд:

```text
симулятор + признаки + rule baseline + RandomForest ML + гибридные правила + объяснимый вывод
```

Эта версия самодостаточна для анализа качества признаков, сравнения правил и ML, проверки гибридного decision layer и подготовки разделов магистерской работы.
