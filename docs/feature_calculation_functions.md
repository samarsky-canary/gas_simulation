# Функции расчета признаков симулятора

Документ описывает расчетные признаки, которые формируются после симуляции телеметрии. Основные реализации находятся в:

- `src/features/build_features.py`
- `src/simulator/labels.py`
- `src/simulator/exporters.py`

## 1. Базовые обозначения

| Обозначение | Смысл |
|---|---|
| `step_minutes` | Шаг временного ряда в минутах. |
| `dt_h` | Шаг временного ряда в часах: `step_minutes / 60`. |
| `Q_nominal` | Номинальный расход `cfg.q_nominal_m3h`. |
| `alpha_flow` | Степень влияния расхода на перепад давления. |
| `gamma_load` | Степень влияния расхода на скорость деградации. |
| `dp_warn_kpa` | Warning-порог нормированного перепада. |
| `dp_crit_kpa` | Critical-порог нормированного перепада. |
| `eps` | Малое число для защиты от деления на ноль. В коде используется `1e-3` или `1e-6` в зависимости от расчета. |

Окно в строках считается так:

```text
window_steps(hours) = max(int(hours * 60 / step_minutes), 1)
```

Для текущего шага `5` минут:

```text
1 час = 12 строк
6 часов = 72 строки
```

## 2. Признаки Feature Builder

Эти признаки экспортируются в `features.csv` и используются в ML baseline, rule layer и гибридной логике.

### 2.1. `deltaP_norm_kPa`

Функция расчета:

```python
flow_factor = max((q_m3h / q_nominal_m3h) ** alpha_flow, 1e-3)
deltaP_norm_kPa = delta_p_kpa / flow_factor
```

Смысл:

```text
Нормированный перепад давления.
```

Сырой `delta_p_kpa` зависит от расхода: при большем расходе перепад растет даже без дополнительного засорения. Нормировка убирает основную часть влияния расхода, чтобы признак лучше отражал сопротивление фильтра.

Используется:

- как вход ML-модели;
- в rule-based baseline;
- в графиках;
- как основной диагностический признак деградации.

### 2.2. `deltaP_roll_mean_1h`

Функция расчета:

```python
one_hour = window_steps(1)
deltaP_roll_mean_1h = rolling_mean(delta_p_kpa, window=one_hour, min_periods=1)
```

Смысл:

```text
Средний наблюдаемый перепад давления за последний час.
```

Этот признак сглаживает шум, короткие выбросы и режимные колебания. Если сырой `delta_p_kpa` скачет, среднее за час показывает более устойчивый уровень перепада.

Используется:

- как вход ML-модели;
- в гибридных карточках как объясняющий признак;
- для визуальной интерпретации тренда.

### 2.3. `deltaP_roll_std_1h`

Функция расчета:

```python
one_hour = window_steps(1)
deltaP_roll_std_1h = rolling_std(delta_p_kpa, window=one_hour, min_periods=1)
deltaP_roll_std_1h = fillna(deltaP_roll_std_1h, 0.0)
```

Смысл:

```text
Нестабильность перепада давления за последний час.
```

Если значение большое, перепад в часовом окне сильно колебался. Это может означать нестабильный режим расхода, шум, выбросы или проблемы с датчиком.

Используется:

- как вход ML-модели;
- чтобы модель отличала устойчивый рост перепада от кратковременной нестабильности.

### 2.4. `deltaP_slope_6h`

Функция расчета:

```python
six_hours = window_steps(6)
deltaP_slope_6h = rolling_apply(
    deltaP_norm_kPa,
    window=six_hours,
    min_periods=max(3, six_hours // 3),
    function=OLS_slope
)
```

Внутри `OLS_slope`:

```python
mask = not_nan(values)

if count(mask) < 2:
    return NaN

x = [0, 1, 2, ...] * dt_h
y = deltaP_norm_kPa values

x_centered = x - mean(x)
denom = dot(x_centered, x_centered)

if denom == 0:
    return 0

slope = dot(x_centered, y - mean(y)) / denom
```

Смысл:

```text
Скорость изменения нормированного перепада давления за 6 часов.
```

Если `deltaP_slope_6h > 0`, нормированный перепад растет. Если меньше нуля, нормированный перепад снижается. Это не уровень засорения, а локальный тренд сопротивления фильтра.

Важно: тренд считается по `deltaP_norm_kPa`, а не по сырому `delta_p_kpa`, чтобы рост расхода сам по себе не выглядел как рост засорения.

Единицы:

```text
кПа / час
```

Используется:

- как вход ML-модели;
- для оценки краткосрочного тренда;
- в карточках решений как объясняющий признак.

### 2.5. `Q_roll_mean_1h`

Функция расчета:

```python
one_hour = window_steps(1)
Q_roll_mean_1h = rolling_mean(q_m3h, window=one_hour, min_periods=1)
```

Смысл:

```text
Средний расход газа за последний час.
```

Это контекст режима работы. Один и тот же перепад давления при низком и высоком расходе имеет разный смысл, поэтому модели полезно знать не только текущий расход, но и сглаженный режим.

Используется:

- как вход ML-модели;
- для учета режима эксплуатации.

### 2.6. `missing_rate_1h`

Сначала строится флаг проблемной строки:

```python
row_has_missing =
    any_nan([p_in_mpa, p_out_mpa, delta_p_kpa, q_m3h, t_c])
    OR quality_code == "missing"
```

Затем:

```python
one_hour = window_steps(1)
missing_rate_1h = rolling_mean(row_has_missing, window=one_hour, min_periods=1)
```

Смысл:

```text
Доля строк с пропусками за последний час.
```

Пример:

```text
missing_rate_1h = 0.25
```

Значит, за последний час примерно 25% строк были с пропусками или помечены как `missing`.

Используется:

- как вход ML-модели;
- в гибридной логике для расчета `confidence_data`;
- для решения `sensor_check`.

### 2.7. `time_above_warn`

Функция расчета:

```python
above_warn = delta_p_kpa >= dp_warn_kpa
time_above_warn = cumulative_sum(above_warn) * dt_h
```

Смысл:

```text
Сколько часов фильтр провел выше warning-порога от начала прогона.
```

Важно: в текущем коде используется сырой `delta_p_kpa`, а не `deltaP_norm_kPa`.

Используется:

- как вход ML-модели;
- как объясняющий признак накопленной работы в тревожной зоне.

## 3. Диагностические признаки из `labels.py`

Эти поля формируются в `label_run(...)` и попадают в `raw_observed`, `truth_labels`, `dataset` или используются в правилах.

### 3.1. `delta_p_norm_q2`

Функция расчета:

```python
flow_factor = (q_m3h / q_nominal_m3h) ** alpha_flow
delta_p_norm_q2 = delta_p_kpa / max(flow_factor, 1e-3)
```

Смысл:

```text
Наблюдаемый перепад, нормированный по расходу.
```

Это аналог `deltaP_norm_kPa`, но в внутреннем snake_case формате симулятора.

Используется:

- для `state_obs`;
- для `rule_health_index`;
- в raw observed таблице;
- в графиках нормированного перепада.

### 3.2. `true_delta_p_norm`

Функция расчета:

```python
flow_factor = max(q_true_m3h / q_nominal_m3h, 1e-6) ** alpha_flow
temp_factor = exp(k_mu_per_c * (t_nominal_c - t_true_c))

true_delta_p_norm =
    delta_p_true_kpa / max(flow_factor * temp_factor, 1e-3)
```

Смысл:

```text
Истинный clean-перепад, очищенный от влияния расхода и температуры.
```

Используется:

- для `state_true`;
- для `rul_oracle_h`.

В реальные данные это поле напрямую не попадает как наблюдаемый признак, потому что использует hidden/true значения.

### 3.3. `state_true`

Функция расчета:

```python
if true_delta_p_norm is NaN:
    state_true = unknown
elif true_delta_p_norm < dp_warn_kpa:
    state_true = normal
elif true_delta_p_norm < dp_crit_kpa:
    state_true = warning
else:
    state_true = critical
```

Смысл:

```text
Истинное состояние фильтра по clean-траектории.
```

Используется:

- как скрытая истина для проверки качества;
- в `truth_labels.csv`;
- для анализа расхождения наблюдаемого и истинного состояния.

### 3.4. `state_obs`

Сначала состояние считается по наблюдаемому нормированному перепаду:

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

Затем применяется правило качества:

```python
if quality_code != "good":
    state_obs = unknown
```

Смысл:

```text
Состояние, которое было бы доступно по наблюдаемым датчикам.
```

Используется:

- в `dataset` как поле `state`;
- как target для классификатора состояния;
- в rule-based и hybrid logic.

### 3.5. `alarm_flag`

Функция расчета:

```python
alarm_flag = state_obs in {"warning", "critical"}
```

Смысл:

```text
Бинарный флаг тревожного состояния.
```

Используется:

- в raw observed таблице;
- для быстрой фильтрации тревожных точек.

### 3.6. `rul_oracle_h`

Функция расчета:

```python
result = NaN for all rows
next_hit = infinity

for i from last row to first row:
    if true_delta_p_norm[i] >= dp_crit_kpa:
        next_hit = i

    if next_hit is finite:
        result[i] = (next_hit - i) * dt_h
```

Смысл:

```text
Истинное время до ближайшего будущего достижения critical-порога.
```

Если critical в будущем не наступает:

```text
rul_oracle_h = NaN
```

Используется:

- как target для ML-регрессора;
- как oracle-метка для оценки качества;
- не должен использоваться как входной признак ML.

### 3.7. `rul_analytic_h`

Сначала считается критический уровень засорения:

```python
c_crit_raw = (dp_crit_kpa / dp0_kpa - 1.0) / k_c
c_crit = clip(c_crit_raw, 0.0, 1.0) ** (1.0 / beta)
```

Затем текущая нагрузка и скорость деградации:

```python
load = max(q_true_m3h / q_nominal_m3h, 1e-6) ** gamma_load
rate = max(k_s_per_hour * load, 1e-9)
```

И аналитический RUL:

```python
rul_analytic_h = max(c_crit - clog_level, 0.0) / rate
```

Смысл:

```text
Физически интерпретируемая оценка остаточного ресурса.
```

Она отвечает на вопрос: если текущий темп засорения сохранится, сколько часов осталось до критического уровня.

Используется:

- как аналитический baseline;
- в гибридном выборе `RUL_fused_h`;
- как fallback, если ML недоступен или данные плохие.

### 3.8. `is_rul_unknown`

Функция расчета:

```python
is_rul_unknown = is_nan(rul_oracle_h)
```

Смысл:

```text
Флаг, что точный oracle-RUL неизвестен, потому что в будущем горизонте critical не наступил.
```

Используется:

- для интерпретации target;
- для понимания, какие строки нельзя использовать как полноценную RUL-метку.

### 3.9. `rule_health_index`

Функция расчета:

```python
rule_health_index = clip(delta_p_norm_q2 / dp_crit_kpa, 0, 1)
```

Смысл:

```text
Индекс близости к критическому состоянию.
```

Интерпретация:

```text
0.0  - перепад около нуля
0.5  - половина critical-порога
1.0  - critical-порог достигнут или превышен
```

Используется:

- как простой rule-based индекс состояния;
- в raw observed таблице.

## 4. Признаки канонического `dataset`

В `dataset.csv` часть признаков просто переименовывается из внутренних колонок, а часть пересчитывается.

| Колонка dataset | Источник / расчет |
|---|---|
| `run_id` | `df["run_id"]` |
| `timestamp` | `df["timestamp"]` |
| `filter_id` | `df["filter_id"]` |
| `scenario` | `df["scenario_id"]` |
| `P_in_MPa` | `df["p_in_mpa"]` |
| `P_out_MPa` | `df["p_out_mpa"]` |
| `deltaP_kPa` | `df["delta_p_kpa"]` |
| `Q_m3h` | `df["q_m3h"]` |
| `T_C` | `df["t_c"]` |
| `clog_level` | `df["clog_level"]`, скрытая переменная |
| `state` | `df["state_obs"]` |
| `RUL_oracle_h` | `df["rul_oracle_h"]`, target |
| `RUL_analytic_h` | `df["rul_analytic_h"]`, analytical baseline |
| `quality_code` | `df["quality_code"]` |

### 4.1. `deltaP_norm_kPa` в `dataset`

Функция расчета:

```python
deltaP_norm_kPa =
    delta_p_kpa / max((q_m3h / q_nominal_m3h) ** alpha_flow, 1e-3)
```

Смысл:

```text
Нормированный перепад, который можно использовать как наблюдаемый вход ML.
```

## 5. Какие признаки можно подавать в ML

Разрешенные входы:

```text
P_in_MPa
P_out_MPa
deltaP_kPa
Q_m3h
T_C
deltaP_norm_kPa
deltaP_roll_mean_1h
deltaP_roll_std_1h
deltaP_slope_6h
Q_roll_mean_1h
missing_rate_1h
time_above_warn
```

Запрещенные входы:

```text
clog_level
RUL_oracle_h
state
state_true
rul_oracle_h
```

Причина:

```text
Это скрытые или целевые поля. Их можно использовать для обучения как target или для оценки качества, но нельзя подавать как признаки модели.
```
