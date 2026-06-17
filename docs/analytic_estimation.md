# Аналитическая оценка параметров

Документ описывает аналитические расчеты, которые используются в симуляторе и диагностическом контуре без ML. Основные реализации находятся в:

- `src/simulator/physics.py`;
- `src/simulator/labels.py`;
- `src/simulator/exporters.py`;
- `src/features/build_features.py`.

## 1. Общая идея

Аналитическая часть использует физические и rule-based формулы:

```text
режим работы + засорение -> перепад давления -> состояние -> RUL
```

ML здесь не участвует. Все величины считаются напрямую по формулам и параметрам `ScenarioConfig`.

## 2. Входные параметры

Основные параметры из `configs/base.yaml`:

| Параметр | Смысл |
|---|---|
| `q_nominal_m3h` | Номинальный расход газа. |
| `dp0_kpa` | Базовый перепад давления на чистом фильтре при номинальных условиях. |
| `dp_warn_kpa` | Warning-порог по нормированному перепаду. |
| `dp_crit_kpa` | Critical-порог по нормированному перепаду. |
| `alpha_flow` | Степень влияния расхода на перепад давления. |
| `gamma_load` | Степень влияния расхода на скорость засорения. |
| `k_c` | Коэффициент влияния засорения на сопротивление фильтра. |
| `beta` | Нелинейность влияния засорения на сопротивление. |
| `k_mu_per_c` | Температурный коэффициент поправки перепада. |
| `k_s_per_hour` | Базовая скорость накопления засорения за час. |
| `c0` | Начальный уровень засорения. |

## 3. Засорение фильтра `clog_level`

Реализация: `src/simulator/degradation.py`.

`clog_level` - скрытый уровень засорения фильтра в диапазоне:

```text
0 <= clog_level <= 1
```

На каждом шаге считается нагрузка по расходу:

```text
load(t) = (max(Q_true(t), 0) / Q_nominal) ^ gamma_load
```

Затем засорение увеличивается:

```text
clog(t+1) = clip(
    clog(t) + k_s_per_hour * load(t) * dt_h,
    0,
    1
)
```

где:

```text
dt_h = step_minutes / 60
```

В текущей версии симулятора сброс засорения не моделируется: внутри одного прогона `clog_level` монотонно накапливается до верхней границы.

## 4. Коэффициент расхода `flow_factor`

Реализация: `src/simulator/physics.py`.

Расход влияет на перепад давления. Чем выше расход относительно номинального, тем выше перепад.

Формула:

```text
flow_factor(t) =
max(Q_true(t) / Q_nominal, 1e-6) ^ alpha_flow
```

Где:

- `Q_true(t)` - истинный расход в момент времени `t`;
- `Q_nominal` - номинальный расход;
- `alpha_flow` - степень влияния расхода.

Если `alpha_flow = 2`, перепад примерно масштабируется как квадрат относительного расхода.

## 5. Температурный коэффициент `temp_factor`

Реализация: `src/simulator/physics.py`.

Температура влияет на вязкостную поправку перепада давления.

Формула:

```text
temp_factor(t) =
exp(k_mu_per_c * (T_nominal - T_true(t)))
```

Где:

- `T_nominal` - номинальная температура;
- `T_true(t)` - истинная температура в момент времени;
- `k_mu_per_c` - температурный коэффициент.

Если температура ниже номинальной, множитель становится больше, и расчетный перепад увеличивается.

## 6. Коэффициент сопротивления `resistance_factor`

Реализация: `src/simulator/physics.py`.

`resistance_factor` показывает, насколько сопротивление фильтра выросло из-за засорения.

Формула:

```text
resistance_factor(t) =
1 + k_c * clog_level(t) ^ beta
```

Где:

- `1` - сопротивление чистого фильтра;
- `k_c` - сила влияния засорения;
- `beta` - нелинейность влияния засорения.

Если `clog_level = 0`, то:

```text
resistance_factor = 1
```

Если `clog_level` растет, сопротивление увеличивается.

## 7. Истинный перепад давления `delta_p_true_kpa`

Реализация: `src/simulator/physics.py`.

Истинный перепад давления считается как базовый перепад на чистом фильтре, умноженный на факторы расхода, температуры и засорения.

Формула:

```text
delta_p_true_kpa(t) =
dp0_kpa
* flow_factor(t)
* temp_factor(t)
* resistance_factor(t)
```

После расчета значение ограничивается снизу:

```text
delta_p_true_kpa(t) = max(delta_p_true_kpa(t), 0)
```

Смысл:

- `dp0_kpa` задает перепад чистого фильтра;
- `flow_factor` учитывает текущий расход;
- `temp_factor` учитывает температуру;
- `resistance_factor` учитывает засорение.

## 8. Истинное выходное давление `p_out_true_mpa`

Реализация: `src/simulator/physics.py`.

Выходное давление равно входному давлению минус перепад давления на фильтре.

Так как `delta_p_true_kpa` задан в кПа, а давление в МПа, используется деление на `1000`:

```text
p_out_true_mpa(t) =
max(0, p_in_true_mpa(t) - delta_p_true_kpa(t) / 1000)
```

Ограничение `max(0, ...)` не дает получить отрицательное давление.

## 9. Нормированный истинный перепад `true_delta_p_norm`

Реализация: `src/simulator/labels.py`, функция `_true_delta_p_norm(...)`.

Истинный нормированный перепад нужен, чтобы классифицировать состояние фильтра по засорению, а не по текущему режиму расхода или температуры.

Формула:

```text
true_delta_p_norm(t) =
delta_p_true_kpa(t) / max(flow_factor(t) * temp_factor(t), 1e-3)
```

Так как:

```text
delta_p_true_kpa =
dp0_kpa * flow_factor * temp_factor * resistance_factor
```

после нормировки остается величина, связанная в основном с сопротивлением фильтра:

```text
true_delta_p_norm ≈ dp0_kpa * resistance_factor
```

## 10. Нормированный наблюдаемый перепад `deltaP_norm_kPa`

Реализации:

- `src/simulator/labels.py`, функция `_observed_delta_p_norm(...)`;
- `src/simulator/exporters.py`, функция `build_canonical_dataset(...)`;
- `src/features/build_features.py`, функция `build_features(...)`.

Наблюдаемый нормированный перепад считается по измеренному перепаду и измеренному расходу.

Формула:

```text
deltaP_norm_kPa(t) =
deltaP_kPa(t) / max((Q_m3h(t) / Q_nominal) ^ alpha_flow, 1e-3)
```

Где:

- `deltaP_kPa(t)` - наблюдаемый сырой перепад давления;
- `Q_m3h(t)` - наблюдаемый расход;
- `Q_nominal` - номинальный расход.

Смысл:

```text
убрать основное влияние расхода из перепада давления
```

В текущей упрощенной версии наблюдаемый `deltaP_norm_kPa` нормируется только по расходу. Температурная поправка для наблюдаемой нормировки не применяется.

## 11. Состояние фильтра `state_obs`

Реализация: `src/simulator/labels.py`, функция `_states(...)`.

Состояния:

```text
normal
warning
critical
unknown
```

Правила классификации:

```text
если deltaP_norm is NaN:
    state = unknown
если deltaP_norm < dp_warn_kpa:
    state = normal
если dp_warn_kpa <= deltaP_norm < dp_crit_kpa:
    state = warning
если deltaP_norm >= dp_crit_kpa:
    state = critical
```

`state_obs` считается по наблюдаемому `deltaP_norm_kPa`.

Если строка наблюдений имеет плохое качество:

```text
quality_code != good
```

то:

```text
state_obs = unknown
```

## 12. Oracle RUL `RUL_oracle_h`

Реализация: `src/simulator/labels.py`, функция `_rul_oracle(...)`.

`RUL_oracle_h` - это истинный остаточный ресурс в синтетике. Он считается по будущему достижению critical-порога на истинной нормированной траектории.

Условие отказа:

```text
true_delta_p_norm(t) >= dp_crit_kpa
```

Для каждого момента времени алгоритм ищет ближайший будущий индекс `next_hit`, где это условие выполнено.

Формула:

```text
RUL_oracle_h(t) =
(next_hit_index - current_index) * dt_h
```

Если в будущем critical-порог не достигается:

```text
RUL_oracle_h(t) = NaN
```

Смысл `NaN`:

```text
истинный RUL неизвестен в пределах горизонта симуляции
```

Это не означает, что ресурс равен нулю. Это означает, что отказ не наступил в наблюдаемом будущем.

## 14. Критический уровень засорения `c_crit`

Реализация: `src/simulator/labels.py`, функция `_rul_analytic(...)`.

Аналитический RUL сначала вычисляет, при каком `clog_level` нормированный перепад достигнет critical-порога.

При нормировке по расходу и температуре:

```text
deltaP_norm ≈ dp0_kpa * (1 + k_c * clog_level ^ beta)
```

Критическое условие:

```text
dp_crit_kpa =
dp0_kpa * (1 + k_c * c_crit ^ beta)
```

Отсюда:

```text
c_crit_raw =
(dp_crit_kpa / dp0_kpa - 1) / k_c
```

Затем значение ограничивается диапазоном:

```text
c_crit_limited = clip(c_crit_raw, 0, 1)
```

И учитывается нелинейность `beta`:

```text
c_crit = c_crit_limited ^ (1 / beta)
```

Смысл:

```text
c_crit - это уровень засорения, при котором фильтр аналитически достигает critical
```

## 15. Текущая скорость деградации `rate`

Реализация: `src/simulator/labels.py`, функция `_rul_analytic(...)`.

Сначала считается нагрузка по расходу:

```text
load(t) =
max(Q_true(t) / Q_nominal, 1e-6) ^ gamma_load
```

Затем скорость роста засорения:

```text
rate(t) =
max(k_s_per_hour * load(t), 1e-9)
```

Единицы:

```text
доля clog_level в час
```

Ограничение `1e-9` защищает от деления на ноль.

## 16. Аналитический RUL `RUL_analytic_h`

Реализация: `src/simulator/labels.py`, функция `_rul_analytic(...)`.

Аналитический RUL оценивает, сколько часов осталось до достижения `c_crit`, если текущая скорость деградации сохранится.

Формула:

```text
RUL_analytic_h(t) =
max(c_crit - clog_level(t), 0) / rate(t)
```

Где:

- `c_crit` - критический уровень засорения;
- `clog_level(t)` - текущий скрытый уровень засорения;
- `rate(t)` - текущая скорость деградации.

Если фильтр уже достиг или превысил критический уровень засорения:

```text
c_crit - clog_level(t) <= 0
```

то:

```text
RUL_analytic_h(t) = 0
```

Важное отличие от `RUL_oracle_h`:

- `RUL_oracle_h` смотрит в будущее по фактической синтетической траектории;
- `RUL_analytic_h` считает ресурс по текущему `clog_level` и текущей скорости деградации.

Поэтому они могут различаться.

## 17. Итоговая таблица аналитических величин

| Поле | Где считается | Формула / логика |
|---|---|---|
| `clog_level` | `degradation.py` | Монотонное накопление `k_s_per_hour * load * dt_h`. |
| `flow_factor` | `physics.py` | `(Q_true / Q_nominal) ^ alpha_flow`. |
| `temp_factor` | `physics.py` | `exp(k_mu_per_c * (T_nominal - T_true))`. |
| `resistance_factor` | `physics.py` | `1 + k_c * clog_level ^ beta`. |
| `delta_p_true_kpa` | `physics.py` | `dp0_kpa * flow_factor * temp_factor * resistance_factor`. |
| `p_out_true_mpa` | `physics.py` | `max(0, p_in_true_mpa - delta_p_true_kpa / 1000)`. |
| `deltaP_norm_kPa` | `labels.py`, `exporters.py`, `features.py` | `deltaP_kPa / max((Q_m3h / Q_nominal)^alpha_flow, 1e-3)`. |
| `state_obs` | `labels.py` | Пороговая классификация наблюдаемого `deltaP_norm_kPa`, плохие данные -> `unknown`. |
| `RUL_oracle_h` | `labels.py` | Время до будущего достижения `true_delta_p_norm >= dp_crit_kpa`. |
| `c_crit` | `labels.py` | `clip((dp_crit / dp0 - 1) / k_c, 0, 1) ^ (1 / beta)`. |
| `rate` | `labels.py` | `max(k_s_per_hour * load, 1e-9)`. |
| `RUL_analytic_h` | `labels.py` | `max(c_crit - clog_level, 0) / rate`. |

## 18. Что используется в гибридной логике

Гибридный слой использует аналитические параметры как независимую опору рядом с ML:

```text
RUL_analytic_h
deltaP_norm_kPa
quality_code
missing_rate_1h
state
```

`RUL_analytic_h` сравнивается с `RUL_ml_h`.

Если ML и аналитика согласованы, доверие выше. Если они сильно расходятся, гибридная логика может выбрать:

- консервативный минимум;
- аналитический fallback;
- ручную проверку.

Таким образом, аналитическая оценка нужна не только для baseline, но и как физически интерпретируемая проверка ML-прогноза.
