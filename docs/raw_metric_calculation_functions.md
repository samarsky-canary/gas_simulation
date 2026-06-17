# Функции расчета сырых наблюдаемых метрик

Документ описывает расчет сырых метрик, то есть наблюдаемой телеметрии после модели датчиков, шума, пропусков, выбросов, залипания и контроля качества. Эти поля ближе всего к тому, что в реальной системе пришло бы от датчиков.

Основные реализации:

- `src/simulator/faults.py`
- `src/validation/checks.py`
- `src/simulator/labels.py`
- `src/simulator/runner.py`

## 1. Место в конвейере

Сырые метрики появляются после расчета clean-физики:

```text
истинные профили
-> скрытая деградация
-> clean-физика
-> модель датчиков
-> инжекция дефектов
-> контроль качества
-> наблюдаемые состояния и тревоги
```

В `runner.py` это соответствует шагам:

```python
observed = apply_sensor_model(cfg, profile, physics, rng)
observed = inject_faults(cfg, observed, rng)
df = concat(profile, degradation, physics, observed)
df, report = validate_run(cfg, df)
df = label_run(cfg, df)
```

## 2. Модель датчиков

Функция:

```python
apply_sensor_model(cfg, profile, physics, rng)
```

Она преобразует clean/true каналы в наблюдаемые измерения.

### 2.1. Дрейф давления

Функция расчета:

```python
n = len(profile)
drift = arange(n) * step_minutes / (60 * 24) * bias_drift_mpa_per_day
```

Смысл:

```text
Линейное смещение датчиков давления во времени.
```

Если `bias_drift_mpa_per_day = 0`, дрейфа нет.

Единицы:

```text
МПа
```

### 2.2. `p_in_mpa`

Функция расчета:

```python
p_in_mpa =
    p_in_true_mpa
    + drift
    + Normal(0, sigma_p_mpa)

p_in_mpa = clip(p_in_mpa, p_min_mpa, p_max_mpa)
```

Смысл:

```text
Наблюдаемое входное давление до фильтра.
```

Отличается от `p_in_true_mpa`, потому что содержит шум датчика и возможный дрейф.

### 2.3. `p_out_mpa`

Функция расчета:

```python
p_out_mpa =
    p_out_true_mpa
    + drift
    + Normal(0, sigma_p_mpa)

p_out_mpa = clip(p_out_mpa, 0.0, p_in_mpa)
```

Смысл:

```text
Наблюдаемое выходное давление после фильтра.
```

В модели выходное давление ограничивается сверху текущим `p_in_mpa`, чтобы после датчиков не возникал отрицательный перепад.

### 2.4. `q_m3h`

Функция расчета:

```python
q_m3h = q_true_m3h * (1 + Normal(0, sigma_q_rel))
q_m3h = clip(q_m3h, 0.0, q_max_m3h * 1.2)
```

Смысл:

```text
Наблюдаемый расход газа.
```

Шум расхода относительный: если `sigma_q_rel = 0.01`, шум примерно 1% от текущего расхода.

### 2.5. `t_c`

Функция расчета:

```python
t_c = t_true_c + Normal(0, sigma_t_abs_c)
```

Смысл:

```text
Наблюдаемая температура газа.
```

Шум температуры абсолютный, в градусах Цельсия.

### 2.6. `delta_p_kpa`

Функция расчета:

```python
delta_p_kpa = max(0.0, 1000.0 * (p_in_mpa - p_out_mpa))
```

Смысл:

```text
Наблюдаемый перепад давления на фильтре.
```

Важно: перепад давления считается из двух наблюдаемых абсолютных давлений:

```text
P_in - P_out
```

Множитель `1000` переводит МПа в кПа.

## 3. Инжекция дефектов датчиков

Функция:

```python
inject_faults(cfg, observed, rng)
```

Дефекты добавляются к каналам:

```text
p_in_mpa
p_out_mpa
q_m3h
t_c
```

После дефектов `delta_p_kpa` пересчитывается заново по `p_in_mpa - p_out_mpa`.

### 3.1. Пропуски

Функция расчета:

```python
missing = random(0, 1) < p_missing

if missing:
    channel = NaN
```

Смысл:

```text
Имитация потери телеметрии по конкретному датчику.
```

Пропуск может возникнуть независимо в каждом наблюдаемом канале.

### 3.2. Выбросы

Функция расчета:

```python
spike = random(0, 1) < p_spike

if spike:
    channel = channel + sign * Uniform(0.8 * scale, 1.5 * scale)
```

Где:

```text
sign = -1 или +1
```

Масштаб выброса:

```python
if channel in {"p_in_mpa", "p_out_mpa"}:
    scale = 0.008
elif channel == "q_m3h":
    scale = 0.18 * q_nominal_m3h
else:
    scale = 3.0
```

Смысл:

```text
Имитация кратковременного скачка датчика.
```

### 3.3. Залипание датчика

Функция расчета:

```python
start = random(0, 1) < p_stuck

if start and previous_value is not NaN:
    length = randint(stuck_min_steps, stuck_max_steps)
    channel[start : start + length] = previous_value
```

Смысл:

```text
Датчик на несколько шагов удерживает старое значение.
```

Такой дефект опасен тем, что данные выглядят численно валидными, но перестают отражать реальное изменение процесса.

### 3.4. Ограничения после дефектов

После пропусков, выбросов и залипаний значения снова ограничиваются:

```python
p_in_mpa = clip(p_in_mpa, p_min_mpa, p_max_mpa)
q_m3h = clip(q_m3h, 0, q_max_m3h * 1.2)
t_c = clip(t_c, t_min_c - 10, t_max_c + 10)
p_out_mpa = min(clip(p_out_mpa, lower=0), p_in_mpa)
```

Затем перепад пересчитывается:

```python
delta_p_kpa = max(0.0, 1000.0 * (p_in_mpa - p_out_mpa))
```

## 4. Контроль качества сырых данных

Функция:

```python
validate_run(cfg, df)
```

Она назначает `quality_code` каждой строке и формирует общий `QCReport`.

### 4.1. `missing_mask`

Функция расчета:

```python
missing_mask =
    any_nan([p_in_mpa, p_out_mpa, delta_p_kpa, q_m3h, t_c])
```

Смысл:

```text
В строке есть хотя бы один пропуск в наблюдаемой телеметрии.
```

### 4.2. `invalid_pressure`

Функция расчета:

```python
invalid_pressure = p_out_mpa > p_in_mpa + 0.002
```

Смысл:

```text
Выходное давление стало физически выше входного с учетом допуска 0.002 МПа.
```

### 4.3. `invalid_dp`

Функция расчета:

```python
invalid_dp = delta_p_kpa < -0.1
```

Смысл:

```text
Перепад давления стал отрицательным с учетом допуска -0.1 кПа.
```

### 4.4. `invalid_q`

Функция расчета:

```python
invalid_q = q_m3h < 0
```

Смысл:

```text
Расход газа стал отрицательным.
```

### 4.5. `quality_code`

Функция расчета:

```python
if missing_mask:
    quality_code = "missing"
elif invalid_pressure or invalid_dp or invalid_q:
    quality_code = "invalid"
else:
    quality_code = "good"
```

Смысл:

```text
Итоговая оценка качества строки наблюдаемой телеметрии.
```

Приоритет:

```text
missing выше invalid
```

То есть если в строке одновременно есть пропуск и физическое нарушение, строка будет помечена как `missing`.

## 5. Наблюдаемые диагностические метрики

После `quality_code` вызывается `label_run(...)`, где часть диагностических метрик считается именно по сырым наблюдаемым каналам.

### 5.1. `delta_p_norm_q2`

Функция расчета:

```python
flow_factor = (q_m3h / q_nominal_m3h) ** alpha_flow
delta_p_norm_q2 = delta_p_kpa / max(flow_factor, 1e-3)
```

Смысл:

```text
Наблюдаемый перепад, нормированный по расходу.
```

Это сырая диагностическая метрика, потому что использует `delta_p_kpa` и `q_m3h` после модели датчиков и дефектов.

### 5.2. `state_obs`

Сначала состояние считается по `delta_p_norm_q2`:

```python
if delta_p_norm_q2 is NaN:
    state_obs = unknown
elif delta_p_norm_q2 < dp_warn_kpa:
    state_obs = normal
elif delta_p_norm_q2 < dp_crit_kpa:
    state_obs = warning
else:
    state_obs = critical
```

Потом применяется качество данных:

```python
if quality_code != "good":
    state_obs = unknown
```

Смысл:

```text
Наблюдаемое состояние фильтра по сырым датчикам.
```

Именно это поле попадает в `dataset.csv` как `state`.

## 6. Raw observed таблица

Файл:

```text
raw_observed.csv
raw_observed.parquet
```

Колонки:

| Колонка | Расчет / источник |
|---|---|
| `run_id` | `scenario_name_seed` |
| `timestamp` | временная сетка |
| `filter_id` | config |
| `scenario_id` | config |
| `p_in_mpa` | sensor model + faults |
| `p_out_mpa` | sensor model + faults |
| `delta_p_kpa` | `max(0, 1000 * (p_in_mpa - p_out_mpa))` |
| `q_m3h` | sensor model + faults |
| `t_c` | sensor model + faults |
| `quality_code` | `good`, `missing`, `invalid` |
| `state_obs` | состояние по `delta_p_norm_q2` и `quality_code` |
| `delta_p_norm_q2` | нормированный наблюдаемый перепад |

## 7. Отличие сырых метрик от чистых

Сырые метрики:

```text
p_in_mpa
p_out_mpa
delta_p_kpa
q_m3h
t_c
quality_code
state_obs
delta_p_norm_q2
```

содержат:

- шум датчиков;
- дрейф;
- пропуски;
- выбросы;
- залипания;
- ограничения физического диапазона.

Чистые метрики:

```text
p_in_true_mpa
p_out_true_mpa
delta_p_true_kpa
q_true_m3h
t_true_c
```

используются как скрытая истина для обучения, проверки и диагностики, но в реальной онлайн-системе напрямую недоступны.
