# Gas Filter Simulator

Прототип полуфизического симулятора одного газового фильтра для генерации синтетической телеметрии, диагностических меток и RUL-бейзлайнов.

## Запуск

```powershell
.\.venv\Scripts\python.exe main.py
```

Streamlit-интерфейс для демонстрации:

```powershell
streamlit run app.py
```

ML-модель обучается отдельно и переиспользуется всеми запусками:

```powershell
$env:DATABASE_URL="postgresql://gas_simulation:gas_simulation@localhost:5432/gas_simulation"
.\.venv\Scripts\python.exe main.py --train-ml --train-datasets 100
```

Сведения об обучении, метрики, Markdown-отчет и бинарный `joblib`-артефакт хранятся
в PostgreSQL. Обычный запуск и Streamlit выполняют только inference по последнему
успешному обучению. Если записи нет, система попросит выполнить команду обучения.

Для корпуса из 100 наборов по умолчанию:

- 80 полных прогонов используются для train, 20 для test.
- Сценарии равномерно чередуются между всеми 8 типами.
- Физические параметры, режимы, шумы и дефекты варьируются воспроизводимо.
- Длительность каждого прогона составляет 60–120 суток, шаг — 30 минут.
- Точные конфигурации всех прогонов сохраняются в `metrics.corpus` PostgreSQL.

Дополнительные параметры:

```powershell
.\.venv\Scripts\python.exe main.py --train-ml `
  --train-datasets 100 `
  --test-share 0.2 `
  --training-seed 20260614 `
  --training-step-minutes 30
```

## Запуск в Docker

Сборка и запуск контейнера:

```powershell
docker build -t gas-simulation:latest .
docker run --rm -p 8501:8501 -v ${PWD}/outputs:/app/outputs `
  -e DATABASE_URL="postgresql://user:password@host.docker.internal:5432/gas_simulation" `
  gas-simulation:latest
```

После запуска приложение доступно по адресу `http://localhost:8501`.
Результаты UI-прогонов сохраняются в локальный каталог `outputs/`.
Для одиночного контейнера PostgreSQL должен быть запущен отдельно.

Альтернативно можно запустить через Docker Compose:

```powershell
docker compose up --build
```

Compose поднимает PostgreSQL и приложение. Перед первым запуском inference нужно
однократно зарегистрировать модель:

```powershell
docker compose run --rm gas-simulation python main.py --train-ml --train-datasets 100
docker compose up
```

Результаты обычного pipeline пишутся в `outputs/<scenario_name>/`. Для больших таблиц
pipeline сохраняет только Parquet, чтобы не дублировать сериализацию в CSV:

- `dataset.parquet` - зафиксированный основной датасет с финальным контрактом колонок.
- `dataset_schema.md` - описание колонок, разрешенных входов ML-моделей и запрещенных скрытых/целевых полей.
- `raw_observed.parquet` - наблюдаемая телеметрия и QC.
- `truth_labels.parquet` - скрытые состояния и целевые метки.
- `wide_debug.parquet` - полный набор для отладки.
- `operations_description.md` - описание операций симулятора от загрузки конфига до экспорта.
- `metadata.json` - конфигурация, seed, версия схемы, QC-отчет и описание операций.
- `features.parquet` - таблица признаков для baseline-моделей и интерпретации правил.
- `feature_description.md` - описание признаков, формул и назначения.
- `rule_baseline.parquet` - результат регламентно-логической baseline-модели.
- `rule_baseline_description.md` - описание правил состояния и рекомендаций.
- `ml_baseline/` - предсказания и копия метрик/отчета использованного обучения.

## Хранение обучений

PostgreSQL использует две таблицы:

- `ml_models` - логические модели и их типы.
- `ml_training_runs` - версии обучений, split, размеры выборок, JSON-метрики,
  отчет, SHA-256 и сериализованный артефакт модели.

Схема создается автоматически приложением и также зафиксирована в `db/init.sql`.
Подключение задается обязательной переменной `DATABASE_URL`.
Пример локальных переменных находится в `.env.example`.

## Формат Датасета

Основной фиксированный формат находится в `dataset.parquet`:

- `run_id`
- `timestamp`
- `filter_id`
- `scenario`
- `P_in_MPa`
- `P_out_MPa`
- `deltaP_kPa`
- `Q_m3h`
- `T_C`
- `clog_level`
- `deltaP_norm_kPa`
- `state`
- `RUL_oracle_h`
- `RUL_analytic_h`
- `quality_code`

Для входа ML-моделей можно использовать только наблюдаемые и производные поля: `P_in_MPa`, `P_out_MPa`, `deltaP_kPa`, `Q_m3h`, `T_C`, `deltaP_norm_kPa`.

Нельзя подавать на вход ML-моделей: `clog_level`, `RUL_oracle_h`, `state`. Это скрытые или целевые поля симулятора.

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
12. Inference готового ML-baseline RandomForestRegressor для RUL.
13. Построение гибридных решений на основе ML-прогноза, аналитического RUL и правил.
14. Экспорт Parquet и metadata без дублирующих CSV.
15. Интерактивная визуализация доступна в Streamlit UI через Plotly.

## Признаки

Feature builder создает минимальный набор признаков:

- `deltaP_norm_kPa` - перепад, очищенный от влияния расхода.
- `deltaP_roll_mean_1h` - средний перепад за 1 час.
- `deltaP_roll_std_1h` - нестабильность перепада за 1 час.
- `deltaP_slope_6h` - скорость роста нормированного перепада на окне 6 часов.
- `Q_roll_mean_1h` - средний расход за 1 час.
- `missing_rate_1h` - доля строк с пропусками за 1 час.
- `time_above_warn` - накопленное время выше warning-порога от начала прогона.

Симулятор больше не моделирует сброс засорения обслуживанием: `clog_level` монотонно накапливается в пределах одного прогона. События планового и срочного обслуживания в UI рассчитываются отдельно по RUL-порогам.

В таблицу признаков также добавлены `state_obs`, `rul_oracle_h`, `rul_analytic_h` и `is_rul_unknown`, чтобы один файл можно было использовать для быстрых baseline-экспериментов.

Подробные функции расчета каждого признака вынесены в `docs/feature_calculation_functions.md`.
Функции расчета сырых наблюдаемых метрик описаны в `docs/raw_metric_calculation_functions.md`.
Подробная аналитическая оценка параметров описана в `docs/analytic_estimation.md`.

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

## ML Baseline

В текущей версии ML baseline обучается отдельной командой как независимая
оценка остаточного ресурса:

- `RandomForestRegressor` напрямую прогнозирует `RUL_oracle_h`.
- Аналитический RUL не передается модели как вход и используется только как
  отдельный baseline для сравнения качества.
- Для ML генерируется корпус из нескольких независимых `run_id` с разными `scenario_name` и `seed`.
- Split выполняется по `run_id`: train содержит одни seed, test содержит другие seed тех же сценариев и стресс-сценарии `sensor_bias`, `sensor_stuck`, `missing_data`.
- Строки обучения балансируются по прогонам и диапазонам RUL.
- В отчете сохраняются общие метрики, разрезы по сценариям и прямое сравнение
  MAE/RMSE с аналитической оценкой.
- Входы: наблюдаемая телеметрия, тренды перепада за 6/24/72 часа, время работы
  и накопленная нагрузка.
- Скрытые и целевые поля `clog_level`, `RUL_oracle_h`, `state` не используются как входы.

Артефакт модели и история обучений лежат в PostgreSQL. Для каждого inference-запуска
в `outputs/<scenario_name>/ml_baseline/` сохраняются:

- `ml_predictions.parquet`
- `ml_metrics.json`
- `ml_baseline_report.md`

Подробное описание ML-слоя: `docs/ml_baseline_detailed.md`.

## Графики

Streamlit отображает через Plotly шесть интерактивных графиков с масштабированием,
наведением, включением/отключением рядов и выбором временного диапазона:

- давление до и после фильтра;
- перепад `deltaP`;
- состояние фильтра;
- сравнение RUL;
- периоды предпочтения источника RUL;
- доверие к данным, согласованность ML с аналитической оценкой, итоговое доверие,
  часовые доли пропусков и всплесков, а также их общие проценты за запуск.

Статические PNG больше не генерируются: визуализация выполняется только интерактивно в Streamlit через Plotly.

Для оценки модели важно смотреть не только сырой `deltaP`, но и его 24-часовое среднее и `deltaP_norm`: сырой перепад реагирует на расход, поэтому может быть шумным даже при корректном росте засорения.

## Структура

```text
src/simulator/
  config.py       # Pydantic-конфиг и сценарные пресеты
  timebase.py     # временная сетка
  profiles.py     # Q(t), P_in(t), T(t)
  degradation.py  # скрытый clog_level
  physics.py      # deltaP_true и P_out_true
  faults.py       # шумы, пропуски, выбросы, залипания
  labels.py       # state, alarm, RUL_oracle, RUL_analytic
  exporters.py    # Parquet/metadata и опциональный CSV
  runner.py       # сборка одного прогона
```

## Проверка

```powershell
.\.venv\Scripts\python.exe -m pytest
```
