# Текущая ML-система симулятора газового фильтра

Документ описывает текущую версию проекта как есть: без нейросетевых моделей, без explainable AI и без подробного разбора карточек решений. Основной контур системы:

```text
configs/base.yaml
-> симулятор
-> таблицы CSV/Parquet
-> feature builder
-> rule-based baseline
-> RandomForest ML baseline
-> reasoner / recommendation engine
-> графики и консольная сводка
```

Отдельный слой, который читает ML-результат и выдаёт рекомендации, описан в файле `docs/recommendation_engine.md`.

## 1. Точка запуска программы

Программа запускается командой:

```powershell
.\.venv\Scripts\python.exe main.py
```

Основной сценарий запуска находится в `main.py`. Он выполняет шаги строго последовательно:

1. Загружает конфигурацию из `configs/base.yaml`.
2. Запускает симулятор и получает общий DataFrame `df`.
3. Экспортирует наблюдаемые данные, скрытые метки, debug-таблицу и фиксированный датасет.
4. Строит признаки feature builder.
5. Строит rule-based baseline.
6. Обучает классические ML baseline-модели RandomForest.
7. Передаёт датасет, признаки и ML-предсказания в рекомендательный слой.
8. Строит графики.
9. Печатает в консоль список созданных файлов и краткую сводку решений.

## 2. Как описывается объект и датчики в `base.yaml`

Файл `configs/base.yaml` описывает один прогон симулятора. Его можно воспринимать как паспорт эксперимента: какой фильтр моделируется, на каком горизонте, с какими режимами, шумами, дефектами и порогами.

### 2.1. Идентификатор фильтра и сценарий

| Параметр | Пример | Смысл |
|---|---:|---|
| `filter_id` | `F-001` | Идентификатор фильтра. Попадает во все выходные таблицы. |
| `scenario_name` | `slow_clogging` | Имя сценария генерации данных. |
| `start_time` | `2026-01-01T00:00:00+03:00` | Начало временного ряда. |
| `duration_days` | `90` | Длительность моделирования в сутках. |
| `step_minutes` | `5` | Шаг дискретизации временного ряда. |
| `seed` | `42` | Seed генератора случайных чисел для воспроизводимости. |

Количество строк рассчитывается так:

```text
rows = duration_days * 24 * 60 / step_minutes
```

Для текущих значений:

```text
90 * 24 * 60 / 5 = 25920 строк
```

### 2.2. Расход газа `Q`

| Параметр | Смысл |
|---|---|
| `q_nominal_m3h` | Номинальный расход газа. Используется для нормировки перепада и расчёта нагрузки. |
| `q_min_m3h` | Нижняя граница расхода. |
| `q_max_m3h` | Верхняя граница расхода. |
| `a_q` | Амплитуда суточной сезонности расхода. |
| `q_weekly_amp` | Амплитуда недельной сезонности расхода. |
| `q_ar_rho` | Коэффициент корреляции AR(1)-шума расхода. |
| `q_process_std_m3h` | Масштаб случайных режимных колебаний расхода. |

Смысл: расход не является постоянным. Он колеблется в течение суток и недели, а также имеет плавный случайный шум. Это важно, потому что сырой перепад давления сильно зависит от расхода.

### 2.3. Входное давление `P_in`

| Параметр | Смысл |
|---|---|
| `p_in_nominal_mpa` | Номинальное входное давление. |
| `p_min_mpa` | Минимальное допустимое давление. |
| `p_max_mpa` | Максимальное допустимое давление. |
| `a_p_mpa` | Амплитуда суточных колебаний входного давления. |
| `p_process_std_mpa` | Случайные плавные колебания входного давления. |

Входное давление используется как канал телеметрии и для расчета наблюдаемого перепада давления.

### 2.4. Температура газа `T`

| Параметр | Смысл |
|---|---|
| `t_nominal_c` | Номинальная температура. |
| `t_min_c` | Нижняя граница температуры. |
| `t_max_c` | Верхняя граница температуры. |
| `a_t_c` | Амплитуда суточных температурных колебаний. |
| `t_process_std_c` | Случайные плавные колебания температуры. |

Температура влияет на физическую модель перепада через температурный коэффициент.

### 2.5. Перепад давления, состояния и горизонты ТО

| Параметр | Смысл |
|---|---|
| `dp0_kpa` | Базовый перепад на чистом фильтре при номинальном расходе. |
| `dp_warn_kpa` | Порог предупредительного состояния по нормированному перепаду. |
| `dp_crit_kpa` | Порог критического состояния по нормированному перепаду. |
| `planned_maintenance_rul_h` | Горизонт планового обслуживания. |
| `urgent_maintenance_rul_h` | Горизонт срочного обслуживания. |

Важно: состояния фильтра считаются по нормированному перепаду, а не по сырому `deltaP_kPa`.

### 2.6. Деградация фильтра

| Параметр | Смысл |
|---|---|
| `c0` | Начальный скрытый уровень засорения. |
| `k_s_per_hour` | Базовая скорость роста засорения в час. |
| `alpha_flow` | Степень влияния расхода на перепад давления. |
| `gamma_load` | Степень влияния расхода на скорость деградации. |
| `k_c` | Коэффициент влияния засорения на сопротивление фильтра. |
| `beta` | Нелинейность влияния засорения. |
| `k_mu_per_c` | Температурный коэффициент физической модели перепада. |

Скрытый уровень засорения обозначается `clog_level`. Он находится в диапазоне от `0` до `1`.

### 2.7. Модель датчиков

| Параметр | Смысл |
|---|---|
| `sigma_p_mpa` | Шум датчиков давления. |
| `sigma_q_rel` | Относительный шум датчика расхода. |
| `sigma_t_abs_c` | Абсолютный шум датчика температуры. |

Наблюдаемый перепад строится из наблюдаемых абсолютных давлений:

```text
deltaP_obs = max(0, 1000 * (P_in_obs - P_out_obs))
```

### 2.8. Дефекты и качество данных

| Параметр | Смысл |
|---|---|
| `p_missing` | Вероятность пропуска по каждому наблюдаемому каналу. |
| `p_spike` | Вероятность кратковременного выброса. |
| `p_stuck` | Вероятность начала залипания датчика. |
| `bias_drift_mpa_per_day` | Дрейф давления в МПа за сутки. |
| `stuck_min_steps` | Минимальная длительность залипания. |
| `stuck_max_steps` | Максимальная длительность залипания. |

Детальные дефекты не сохраняются отдельной колонкой. В итоговых таблицах остаётся только общий `quality_code`: `good`, `missing` или `invalid`.

## 3. Сценарии

Допустимые сценарии задаются в `ScenarioName`:

```python
normal
slow_clogging
rapid_clogging
flow_spikes
sensor_bias
sensor_stuck
missing_data
maintenance_reset
```

В `src/simulator/config.py` есть `SCENARIO_OVERRIDES`. Это встроенные переопределения параметров:

| Сценарий | Что меняется |
|---|---|
| `normal` | Очень малая скорость деградации, почти нет дефектов данных. |
| `slow_clogging` | Базовый сценарий из `base.yaml`. |
| `rapid_clogging` | Увеличена скорость деградации и амплитуда расхода. |
| `flow_spikes` | Увеличены скачки и шум расхода. |
| `sensor_bias` | Включён дрейф давления. |
| `sensor_stuck` | Увеличена вероятность залипания датчиков. |
| `missing_data` | Увеличена вероятность пропусков. |
| `maintenance_reset` | Добавлено обслуживание в середине ряда и сброс засорения. |

## 4. Как работает симулятор

Основная функция симуляции:

```python
run_scenario(cfg)
```

Она выполняет цепочку:

```text
timebase
-> profiles
-> degradation
-> physics
-> sensor model
-> fault injection
-> validation
-> labels and RUL
```

### 4.1. Временная сетка

Функция `make_index(cfg)` создаёт равномерный временной ряд:

```text
periods = duration_days * 24 * 60 / step_minutes
freq = step_minutes
```

Результат: колонка `timestamp`.

### 4.2. Генерация истинных режимов

Функция `generate_profiles(...)` создаёт истинные профили:

- `q_true_m3h`;
- `p_in_true_mpa`;
- `t_true_c`.

Расход:

```text
Q_base(t) = Q_nominal * (1
  + a_q * sin(2*pi*k/steps_per_day)
  + q_weekly_amp * sin(2*pi*k/steps_per_week))

Q_true(t) = Q_base(t) + AR(1)-noise
```

После этого расход ограничивается:

```text
q_min_m3h <= Q_true <= q_max_m3h
```

Входное давление:

```text
P_in_true(t) = P_in_nominal
  + a_p * sin(2*pi*k/steps_per_day + 0.7)
  + AR(1)-noise
```

Температура:

```text
T_true(t) = T_nominal
  + a_t * sin(2*pi*k/steps_per_day - 0.3)
  + AR(1)-noise
```

### 4.3. Скрытая деградация

Функция `simulate_degradation(...)` создаёт:

- `clog_level`;
- `maintenance_event`.

Формула шага:

```text
dt_h = step_minutes / 60
load(t) = (max(Q_true(t), 0) / Q_nominal) ^ gamma_load

clog(t+1) = clip(
  clog(t) + k_s_per_hour * load(t) * dt_h,
  0,
  1
)
```

Здесь `load(t)` уже включает степень `gamma_load`.

Событие обслуживания может появиться двумя способами:

- `maintenance_day` задаёт разовое обслуживание в конкретный день от начала симуляции;
- `maintenance_interval_h` задаёт обслуживание по графику каждые N часов.

Если обслуживание наступило, то в соответствующей точке:

```text
clog(t) = min(c_reset, clog(t-1))
maintenance_event = true
```

Например, если `maintenance_interval_h = 20000`, событие будет создано через 20000 часов. Если длительность симуляции меньше 20000 часов, `maintenance_event` не появится.

### 4.4. Физическая модель перепада давления

Функция `compute_physics(...)` создаёт:

- `resistance_factor`;
- `delta_p_true_kpa`;
- `p_out_true_mpa`.

Формулы:

```text
flow_factor = max(Q_true / Q_nominal, 1e-6) ^ alpha_flow

temp_factor = exp(k_mu_per_c * (T_nominal - T_true))

resistance_factor = 1 + k_c * clog_level ^ beta

deltaP_true = dp0_kpa * flow_factor * temp_factor * resistance_factor

P_out_true = max(0, P_in_true - deltaP_true / 1000)
```

Смысл: перепад давления растёт из-за трёх факторов:

- расхода;
- температурной поправки;
- увеличения сопротивления из-за засорения.

### 4.5. Модель датчиков

Функция `apply_sensor_model(...)` превращает истинные каналы в наблюдаемые:

- `p_in_mpa`;
- `p_out_mpa`;
- `q_m3h`;
- `t_c`;
- `delta_p_kpa`;

Давления:

```text
drift(t) = t_days * bias_drift_mpa_per_day

P_in_obs = P_in_true + drift + N(0, sigma_p_mpa)
P_out_obs = P_out_true + drift + N(0, sigma_p_mpa)
```

Расход:

```text
Q_obs = Q_true * (1 + N(0, sigma_q_rel))
```

Температура:

```text
T_obs = T_true + N(0, sigma_t_abs_c)
```

Перепад давления:

```text
deltaP_obs = max(0, 1000 * (P_in_obs - P_out_obs))
```

### 4.6. Инжекция дефектов датчиков

Функция `inject_faults(...)` добавляет дефекты в наблюдаемые каналы:

- пропуски;
- выбросы;
- залипания.

Пропуск:

```text
random < p_missing
=> значение канала становится NaN
```

Выброс:

```text
random < p_spike
=> к значению добавляется случайный скачок
```

Залипание:

```text
random < p_stuck
=> значение канала удерживается постоянным на интервале
```

После дефектов значения снова ограничиваются физическими диапазонами.

### 4.7. Контроль качества

Функция `validate_run(...)` проверяет:

- есть ли пропуски в наблюдаемых каналах;
- не нарушен ли порядок давлений;
- не отрицателен ли перепад;
- не отрицателен ли расход;
- монотонно ли растёт `clog_level` между обслуживаниями.

Формируются:

- `quality_code`;
- объект `QCReport`.

Логика `quality_code`:

| Условие | `quality_code` |
|---|---|
| Нет проблем качества | `good` |
| Есть пропуски | `missing` |
| Есть физическое нарушение | `invalid` |

### 4.8. Состояния и RUL

Функция `label_run(...)` добавляет:

- `delta_p_norm_q2`;
- `state_true`;
- `state_obs`;
- `alarm_flag`;
- `rul_oracle_h`;
- `rul_analytic_h`;
- `is_rul_unknown`;
- `rule_health_index`.

Наблюдаемый нормированный перепад:

```text
flow_factor = (Q_obs / Q_nominal) ^ alpha_flow

deltaP_norm_obs = deltaP_obs / max(flow_factor, 1e-3)
```

Истинный нормированный перепад:

```text
flow_factor = max(Q_true / Q_nominal, 1e-6) ^ alpha_flow
temp_factor = exp(k_mu_per_c * (T_nominal - T_true))

deltaP_norm_true = deltaP_true / max(flow_factor * temp_factor, 1e-3)
```

Состояние:

```text
если deltaP_norm < dp_warn_kpa:
    state = normal
если dp_warn_kpa <= deltaP_norm < dp_crit_kpa:
    state = warning
если deltaP_norm >= dp_crit_kpa:
    state = critical
если значение отсутствует:
    state = unknown
```

Для `state_obs` дополнительно:

```text
если quality_code != good:
    state_obs = unknown
```

## 5. Как формируются таблицы

После симуляции общий DataFrame `df` содержит все каналы: истинные, наблюдаемые, качество, состояния и RUL. Далее `export_run(...)` нарезает его на несколько таблиц.

### 5.1. `raw_observed.csv` / `raw_observed.parquet`

Это наблюдаемая телеметрия. Её смысл: данные, которые могли бы прийти из датчиков и онлайн-контура.

| Колонка | Как создаётся | Смысл |
|---|---|---|
| `run_id` | `scenario_name_seed` | Идентификатор прогона. |
| `timestamp` | `make_index` | Временная метка. |
| `filter_id` | `base.yaml` | Идентификатор фильтра. |
| `scenario_id` | `scenario_name` | Сценарий. |
| `p_in_mpa` | sensor model | Наблюдаемое входное давление. |
| `p_out_mpa` | sensor model | Наблюдаемое выходное давление. |
| `delta_p_kpa` | sensor model | Наблюдаемый перепад. |
| `q_m3h` | sensor model | Наблюдаемый расход. |
| `t_c` | sensor model | Наблюдаемая температура. |
| `quality_code` | validation | Код качества строки. |
| `state_obs` | labels | Наблюдаемое состояние. |
| `alarm_flag` | labels | Тревога warning/critical. |
| `delta_p_norm_q2` | labels | Нормированный наблюдаемый перепад. |
| `rule_health_index` | labels | Индекс состояния от 0 до 1. |

Русская версия: `raw_observed_ru.csv`.

### 5.2. `truth_labels.csv` / `truth_labels.parquet`

Это скрытая истина симулятора. Её смысл: обучение, отладка и оценка качества моделей.

| Колонка | Как создаётся | Смысл |
|---|---|---|
| `run_id` | runner | Идентификатор прогона. |
| `timestamp` | timebase | Время. |
| `filter_id` | config | Идентификатор фильтра. |
| `scenario_id` | config | Сценарий. |
| `q_true_m3h` | profiles | Истинный расход. |
| `p_in_true_mpa` | profiles | Истинное входное давление. |
| `t_true_c` | profiles | Истинная температура. |
| `p_out_true_mpa` | physics | Истинное выходное давление. |
| `delta_p_true_kpa` | physics | Истинный перепад. |
| `resistance_factor` | physics | Коэффициент сопротивления фильтра. |
| `clog_level` | degradation | Скрытый уровень засорения. |
| `maintenance_event` | degradation | Событие обслуживания. |
| `state_true` | labels | Истинное состояние по clean-перепаду. |
| `rul_oracle_h` | labels | Истинный RUL до critical. |
| `rul_analytic_h` | labels | Аналитический RUL. |
| `is_rul_unknown` | labels | Нет critical в будущем горизонте, поэтому точный oracle-RUL неизвестен. |

Русская версия: `truth_labels_ru.csv`.

### 5.3. `wide_debug.csv` / `wide_debug.parquet`

Это полный отладочный набор. Он содержит все колонки общего `df`: и истинные каналы, и наблюдаемые, и признаки качества, и состояния.

Назначение:

- проверять корректность симулятора;
- сравнивать clean и observed каналы;
- отлаживать формулы;
- строить диагностические графики.

Русская версия: `wide_debug_ru.csv`.

### 5.4. `dataset.csv` / `dataset.parquet`

Это главный фиксированный датасет для ML и дальнейшего анализа.

| Колонка | Источник | Можно ли подавать в ML | Смысл |
|---|---|---:|---|
| `run_id` | runner | нет, служебная | Идентификатор синтетического прогона. |
| `timestamp` | timebase | нет, служебная | Время наблюдения. |
| `filter_id` | config | нет, служебная | Фильтр. |
| `scenario` | config | нет, служебная | Сценарий. |
| `P_in_MPa` | `p_in_mpa` | да | Наблюдаемое входное давление. |
| `P_out_MPa` | `p_out_mpa` | да | Наблюдаемое выходное давление. |
| `deltaP_kPa` | `delta_p_kpa` | да | Наблюдаемый перепад. |
| `Q_m3h` | `q_m3h` | да | Наблюдаемый расход. |
| `T_C` | `t_c` | да | Наблюдаемая температура. |
| `clog_level` | degradation | нет | Скрытая переменная симулятора. |
| `deltaP_norm_kPa` | расчёт | да | Перепад, нормированный по расходу. |
| `state` | `state_obs` | target | Наблюдаемое состояние. |
| `RUL_oracle_h` | labels | target | Истинный RUL для обучения. |
| `RUL_analytic_h` | labels | baseline/fallback | Аналитический RUL. |
| `quality_code` | validation | можно для фильтрации | Код качества строки. |

Формула `deltaP_norm_kPa`:

```text
deltaP_norm_kPa = deltaP_kPa
  / max((Q_m3h / Q_nominal) ^ alpha_flow, 1e-3)
```

### 5.5. `metadata.json`

Содержит:

- версию схемы;
- время генерации;
- число строк;
- список колонок;
- фиксированный список колонок датасета;
- разрешённые ML-входы;
- запрещённые ML-входы;
- русские названия колонок;
- описание операций;
- полный конфиг;
- SHA-256 хэш конфига;
- QC-отчёт;
- единицы измерения.

## 6. Feature Builder

Feature builder вызывается так:

```python
features = build_features(cfg, df)
```

Он создаёт:

```text
features.csv
features.parquet
features_ru.csv
feature_description.md
```

### 6.1. Колонки `features`

| Колонка | Смысл |
|---|---|
| `run_id` | Идентификатор прогона. |
| `timestamp` | Время. |
| `filter_id` | Идентификатор фильтра. |
| `scenario_id` | Сценарий. |
| `deltaP_norm_kPa` | Нормированный перепад. |
| `deltaP_roll_mean_1h` | Средний перепад за 1 час. |
| `deltaP_roll_std_1h` | Стандартное отклонение перепада за 1 час. |
| `deltaP_slope_6h` | Наклон нормированного перепада за 6 часов. |
| `Q_roll_mean_1h` | Средний расход за 1 час. |
| `missing_rate_1h` | Доля строк с пропусками за 1 час. |
| `time_above_warn` | Накопленное время выше warning-порога. |
| `state_obs` | Наблюдаемое состояние. |
| `state_true` | Истинное состояние. |
| `rul_oracle_h` | Истинный RUL. |
| `rul_analytic_h` | Аналитический RUL. |
| `is_rul_unknown` | Флаг неизвестного oracle-RUL. |

### 6.2. Формулы признаков

#### `deltaP_norm_kPa`

```text
flow_factor = max((q_m3h / q_nominal_m3h) ^ alpha_flow, 1e-3)

deltaP_norm_kPa = delta_p_kpa / flow_factor
```

Смысл: очищает перепад от влияния расхода.

#### `deltaP_roll_mean_1h`

```text
window_1h = 60 / step_minutes

deltaP_roll_mean_1h = rolling_mean(delta_p_kpa, window_1h)
```

Смысл: сглаживает шум и короткие выбросы.

#### `deltaP_roll_std_1h`

```text
deltaP_roll_std_1h = rolling_std(delta_p_kpa, window_1h)
```

Смысл: показывает нестабильность перепада.

#### `deltaP_slope_6h`

```text
window_6h = 6 * 60 / step_minutes

deltaP_slope_6h = OLS_slope(deltaP_norm_kPa ~ time_hours, window_6h)
```

Смысл: показывает скорость изменения нормированного перепада. Если значение положительное, сопротивление фильтра растёт без учёта кратковременного изменения расхода.

#### `Q_roll_mean_1h`

```text
Q_roll_mean_1h = rolling_mean(q_m3h, window_1h)
```

Смысл: контекст режима работы.

#### `missing_rate_1h`

Сначала строится флаг строки:

```text
row_has_missing =
  any NaN in [p_in_mpa, p_out_mpa, delta_p_kpa, q_m3h, t_c]
  OR quality_code == missing
```

Затем:

```text
missing_rate_1h = rolling_mean(row_has_missing, window_1h)
```

Смысл: показывает качество данных на ближайшем часовом окне.

#### `time_above_warn`

```text
above_warn = delta_p_kpa >= dp_warn_kpa
time_above_warn = cumulative_sum(above_warn) * dt_h
```

Сброс происходит после `maintenance_event`.

## 7. Rule-Based Baseline

Rule-based baseline строится отдельно от ML:

```python
rule_baseline = apply_rule_baseline(cfg, df)
```

Выходные файлы:

```text
rule_baseline.csv
rule_baseline.parquet
rule_baseline_ru.csv
rule_baseline_description.md
```

### 7.1. Колонки rule baseline

| Колонка | Смысл |
|---|---|
| `run_id` | Идентификатор прогона. |
| `timestamp` | Время. |
| `filter_id` | Фильтр. |
| `scenario_id` | Сценарий. |
| `delta_p_kpa` | Сырой наблюдаемый перепад. |
| `delta_p_norm_kpa` | Нормированный перепад для правил. |
| `rul_analytic_h` | Аналитический RUL. |
| `quality_code` | Качество данных. |
| `rule_state` | Состояние по правилам. |
| `rule_alarm_flag` | Тревога по правилам. |
| `rule_recommendation` | Рекомендация по правилам. |
| `rule_reason` | Текстовое основание правила. |
| `state_obs` | Состояние из симулятора. |
| `state_true` | Истинное состояние. |
| `rul_oracle_h` | Истинный RUL. |

### 7.2. Правила состояния

```text
если delta_p_norm_kpa отсутствует:
    rule_state = unknown

если delta_p_norm_kpa < dp_warn_kpa:
    rule_state = normal

если dp_warn_kpa <= delta_p_norm_kpa < dp_crit_kpa:
    rule_state = warning

если delta_p_norm_kpa >= dp_crit_kpa:
    rule_state = critical

если quality_code == missing:
    rule_state = unknown
```

### 7.3. Правила рекомендации

```text
по умолчанию:
    rule_recommendation = continue_monitoring

если rul_analytic_h < planned_maintenance_rul_h:
    rule_recommendation = planned_maintenance

если rul_analytic_h < urgent_maintenance_rul_h:
    rule_recommendation = urgent_maintenance

если rule_state == critical:
    rule_recommendation = urgent_maintenance

если rule_state == unknown:
    rule_recommendation = inspect_sensor_data
```

## 8. RUL: oracle, analytic и ML

В системе есть три разных RUL-поля.

### 8.1. `RUL_oracle_h`

Это истинный остаточный ресурс, доступный только в симуляторе.

Он считается по clean-траектории:

```text
RUL_oracle_h(t) = время от текущей точки t
                  до ближайшей будущей точки,
                  где deltaP_norm_true >= dp_crit_kpa
```

Если в будущем critical не наступает, значение остаётся `NaN`. Такая строка помечается:

```text
is_rul_unknown = true
```

`RUL_oracle_h` используется как target для обучения ML-регрессора.

### 8.2. `RUL_analytic_h`

Это аналитическая оценка остаточного ресурса через скрытую формулу деградации.

Сначала считается критический уровень засорения:

```text
c_crit_raw = (dp_crit_kpa / dp0_kpa - 1) / k_c
c_crit = clip(c_crit_raw, 0, 1) ^ (1 / beta)
```

Затем текущая скорость деградации:

```text
load = max(Q_true / Q_nominal, 1e-6) ^ gamma_load
rate = max(k_s_per_hour * load, 1e-9)
```

И остаточный ресурс:

```text
RUL_analytic_h = max(c_crit - clog_level, 0) / rate
```

Смысл: аналитическая модель отвечает на вопрос “если текущий темп засорения сохранится, сколько часов осталось до критического уровня”.

### 8.3. `RUL_ml_h`

ML-прогноз создаётся RandomForestRegressor.

Target:

```text
RUL_oracle_h
```

Входные признаки:

- `P_in_MPa`;
- `P_out_MPa`;
- `deltaP_kPa`;
- `Q_m3h`;
- `T_C`;
- `deltaP_norm_kPa`;
- `deltaP_roll_mean_1h`;
- `deltaP_roll_std_1h`;
- `deltaP_slope_6h`;
- `Q_roll_mean_1h`;
- `missing_rate_1h`;
- `time_above_warn`.

Запрещённые входы:

- `clog_level`;
- `RUL_oracle_h`;
- `state`.

## 9. ML Baseline

ML baseline запускается так:

```python
ml_paths = train_and_export_ml_baseline(
    ml_dataset,
    output_dir,
    ml_features,
    test_run_ids={holdout_run_id},
)
```

### 9.1. Подготовка данных

1. Для ML создаётся корпус из нескольких независимых синтетических прогонов.
2. Каждый прогон получает свой `run_id`, сценарий и seed.
3. Канонический `dataset` объединяется с признаками feature builder по `run_id` и `timestamp`.
4. Данные сортируются по `run_id` и времени.
5. Входные признаки заполняются внутри каждого `run_id`:

```text
ffill().bfill()
```

6. Строки без входных признаков удаляются.

### 9.2. Train/test split

Основное разбиение выполняется по целым независимым прогонам:

```text
train = основные сценарии с train-seed
test = основные сценарии с другими seed + stress-test сценарии
```

В текущей конфигурации train строится по сценариям `slow_clogging`, `rapid_clogging`, `flow_spikes`, `maintenance_reset` с seed `7`, `13`, `21`. Test строится по тем же основным сценариям с seed `42`, `101`, а также по стресс-сценариям `sensor_bias`, `sensor_stuck`, `missing_data` с seed `42`.

Это нужно, чтобы модель видела полный диапазон RUL на обучающих траекториях и проверялась на независимых траекториях, а не на соседних точках того же ряда. Если в данных только один `run_id`, код использует временной split как fallback.

### 9.3. RUL-регрессор

Модель:

```text
RandomForestRegressor
```

Параметры:

```text
n_estimators = 120
max_depth = 14
min_samples_leaf = 5
random_state = 42
n_jobs = -1
```

Обучается только на строках, где `RUL_oracle_h` не `NaN`.

Метрики:

- MAE в часах;
- RMSE в часах;
- R2.

### 9.4. State-классификатор

Модель:

```text
RandomForestClassifier
```

Параметры:

```text
n_estimators = 120
max_depth = 12
min_samples_leaf = 5
class_weight = balanced
random_state = 42
n_jobs = -1
```

Target:

```text
state
```

Обучается только на состояниях:

```text
normal
warning
critical
```

`unknown` не используется как класс обучения.

Метрики:

- accuracy;
- classification report.

### 9.5. Таблица `ml_predictions`

Файлы:

```text
ml_baseline/ml_predictions.csv
ml_baseline/ml_predictions.parquet
```

Колонки:

| Колонка | Смысл |
|---|---|
| `run_id` | Идентификатор прогона. |
| `timestamp` | Время. |
| `filter_id` | Фильтр. |
| `scenario` | Сценарий. |
| `split` | `train` или `test`. |
| `state_true` | Истинное состояние из датасета. |
| `state_pred` | Предсказанное состояние RandomForestClassifier. |
| `RUL_oracle_h` | Истинный RUL. |
| `RUL_pred_h` | Предсказанный RUL RandomForestRegressor. |

Именно `RUL_pred_h` дальше переименуется в `RUL_ml_h` внутри рекомендательного слоя.

### 9.6. Артефакты ML baseline

```text
ml_baseline/random_forest_rul.joblib
ml_baseline/random_forest_state.joblib
ml_baseline/ml_metrics.json
ml_baseline/ml_baseline_report.md
```

`joblib`-файлы нужны для повторного использования моделей без переобучения.

## 10. Графики

Функция:

```python
build_plots(cfg, df, output_dir, hybrid_decisions)
```

Создаёт:

| Файл | Смысл |
|---|---|
| `00_obzornyi_dashboard.png` | Общий обзор ключевых каналов. |
| `01_rashod_q.png` | Расход газа. |
| `02_davleniya_pin_pout.png` | Входное и выходное давление. |
| `03_perepad_delta_p.png` | Сырой перепад давления. |
| `04_zasorenie_clog_level.png` | Скрытое засорение. |
| `05_ostatochnyi_resurs_rul.png` | Oracle и аналитический RUL. |
| `06_sostoyanie_filtra.png` | Состояние фильтра, сглаженное состояние и отметки решений. |
| `07_normirovannyi_perepad.png` | Нормированный перепад с warning/critical порогами. |
| `08_delta_p_i_zasorenie.png` | Сравнение перепада и засорения. |
| `plots_description.md` | Описание графиков. |
| `plot_diagnostics.md` | Численная диагностика связи перепада и засорения. |

## 11. Что выводится в консоль

После запуска программа печатает:

1. Количество строк.
2. Имя сценария.
3. Количество строк с проблемами качества.
4. Список созданных файлов.
5. Список созданных признаков.
6. Список файлов rule-based baseline.
7. Список файлов ML baseline.
8. Список файлов гибридных решений.
9. Список графиков.
10. Краткую сводку рекомендательного слоя.

Сводка рекомендательного слоя включает:

- число временных точек;
- распределение действий;
- распределение источников RUL;
- среднее и минимальное `confidence_total`;
- несколько кратких карточек решений.

Подробное устройство рекомендательного слоя описано отдельно в `docs/recommendation_engine.md`.

## 12. Итоговая логика текущей версии

Текущая система не просто генерирует синтетический ряд. Она строит полный экспериментальный контур:

```text
1. Задаём объект, датчики, шумы и сценарий в base.yaml.
2. Генерируем истинную физическую траекторию фильтра.
3. Превращаем её в наблюдаемую телеметрию датчиков.
4. Добавляем дефекты данных.
5. Проверяем качество и физическую корректность.
6. Считаем состояния и RUL.
7. Фиксируем датасет.
8. Строим признаки.
9. Обучаем rule-based baseline и RandomForest ML baseline.
10. Передаём ML-прогноз в рекомендательный слой.
11. Получаем действие по обслуживанию и диагностические графики.
```
