# Gas Filter Simulator

Прототип полуфизического симулятора одного газового фильтра для генерации синтетической телеметрии, диагностических меток и RUL-бейзлайнов.

## Запуск

```powershell
.\.venv\Scripts\python.exe main.py
```

Результаты пишутся в `outputs/<scenario_name>/`:

- `raw_observed.csv` / `raw_observed.parquet` - наблюдаемая телеметрия и QC.
- `raw_observed_ru.csv` - наблюдаемая телеметрия с русскими заголовками и русифицированными состояниями.
- `truth_labels.csv` / `truth_labels.parquet` - скрытые состояния и целевые метки.
- `truth_labels_ru.csv` - скрытые состояния и метки с русскими заголовками.
- `wide_debug.csv` / `wide_debug.parquet` - полный набор для отладки.
- `wide_debug_ru.csv` - полный отладочный набор с русскими заголовками.
- `operations_description.md` - описание операций симулятора от загрузки конфига до экспорта.
- `metadata.json` - конфигурация, seed, версия схемы, QC-отчет, словарь русских колонок и описание операций.
- `plots/` - PNG-графики для визуальной проверки временных рядов.
- `features.csv` / `features.parquet` - таблица признаков для baseline-моделей и интерпретации правил.
- `features_ru.csv` - русифицированная таблица признаков.
- `feature_description.md` - описание признаков, формул и назначения.
- `rule_baseline.csv` / `rule_baseline.parquet` - результат регламентно-логической baseline-модели.
- `rule_baseline_ru.csv` - русифицированный результат rule-based baseline.
- `rule_baseline_description.md` - описание правил состояния и рекомендаций.

Англоязычные CSV/Parquet оставлены как стабильная машинная схема для кода, ML и последующей обработки. Русские CSV предназначены для просмотра, отчета и ручной проверки.

## Операции симулятора

1. Загрузка YAML-конфигурации и применение сценарных параметров.
2. Построение временной сетки с заданным шагом.
3. Генерация профилей расхода, входного давления и температуры.
4. Расчет скрытого уровня засорения фильтра.
5. Расчет истинного перепада давления и выходного давления.
6. Наложение шумов датчиков и возможного дрейфа.
7. Инжекция пропусков, выбросов и залипания датчиков.
8. Проверка качества данных и физических ограничений.
9. Расчет диагностических состояний, тревоги и RUL.
10. Расчет признаков для baseline-моделей и интерпретации правил.
11. Расчет rule-based baseline: состояние фильтра и рекомендация обслуживания.
12. Экспорт CSV, Parquet, metadata и русифицированных отчетных файлов.
13. Построение графиков по расходу, давлениям, перепаду, засорению, RUL, состоянию и нормированному перепаду.

## Признаки

Feature builder создает минимальный набор признаков:

- `deltaP_norm_kPa` - перепад, очищенный от влияния расхода.
- `deltaP_roll_mean_1h` - средний перепад за 1 час.
- `deltaP_roll_std_1h` - нестабильность перепада за 1 час.
- `deltaP_slope_6h` - скорость роста перепада на окне 6 часов.
- `Q_roll_mean_1h` - средний расход за 1 час.
- `missing_rate_1h` - доля строк с пропусками за 1 час.
- `time_above_warn` - накопленное время выше warning-порога после последнего обслуживания.

В таблицу признаков также добавлены `state_obs`, `state_true`, `rul_oracle_h`, `rul_analytic_h` и `is_censored`, чтобы один файл можно было использовать для быстрых baseline-экспериментов.

## Rule-Based Baseline

Правила состояния:

- `delta_p_kpa < dp_warn_kpa` -> `normal`.
- `dp_warn_kpa <= delta_p_kpa < dp_crit_kpa` -> `warning`.
- `delta_p_kpa >= dp_crit_kpa` -> `critical`.
- пропуск `delta_p_kpa` или `quality_code = missing` -> `unknown`.

Правила рекомендаций:

- `rul_analytic_h < 72` -> `planned_maintenance`.
- `rul_analytic_h < 12` -> `urgent_maintenance`.
- `rule_state = critical` повышает рекомендацию до `urgent_maintenance`.
- `rule_state = unknown` -> `inspect_sensor_data`.

## Графики

После запуска в `outputs/<scenario_name>/plots/` создаются:

- `00_obzornyi_dashboard.png` - все ключевые каналы на одном листе.
- `01_rashod_q.png` - расход газа `Q(t)`.
- `02_davleniya_pin_pout.png` - `P_in(t)` и `P_out(t)`.
- `03_perepad_delta_p.png` - `deltaP(t)` с порогами и 24-часовым средним.
- `04_zasorenie_clog_level.png` - скрытое засорение `clog_level(t)`.
- `05_ostatochnyi_resurs_rul.png` - `RUL(t)`.
- `06_sostoyanie_filtra.png` - состояние фильтра `state(t)`.
- `07_normirovannyi_perepad.png` - нормированный перепад `deltaP_norm(t)`.
- `08_delta_p_i_zasorenie.png` - сравнение `deltaP(t)` и `clog_level(t)`.
- `plot_diagnostics.md` - численная проверка связи `deltaP` и `clog_level`.

Для оценки модели важно смотреть не только сырой `deltaP`, но и его 24-часовое среднее и `deltaP_norm`: сырой перепад реагирует на расход, поэтому может быть шумным даже при корректном росте засорения.

## Структура

```text
src/simulator/
  config.py       # Pydantic-конфиг и сценарные пресеты
  timebase.py     # временная сетка
  profiles.py     # Q(t), P_in(t), T(t)
  degradation.py  # скрытый clog_level и обслуживание
  physics.py      # deltaP_true и P_out_true
  faults.py       # шумы, пропуски, выбросы, залипания
  labels.py       # state, alarm, RUL_oracle, RUL_analytic
  exporters.py    # CSV/Parquet/metadata
  runner.py       # сборка одного прогона
```

## Проверка

```powershell
.\.venv\Scripts\python.exe -m pytest
```
