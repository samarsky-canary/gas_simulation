# Продукционные правила системы

Документ фиксирует продукционные правила текущей версии: пороговый `rule-based baseline` и гибридный `decision layer`. Правила применяются к каждой временной точке телеметрии.

Актуальные реализации находятся в `src/rules/baseline.py` и `src/hybrid/decision_engine.py`. Пороговые значения соответствуют текущему `configs/base.yaml`.

## 1. Обозначения и пороги

| Обозначение | Поле / параметр | Текущий порог | Смысл |
|---|---|---:|---|
| `delta_p_norm_kpa` | нормированный перепад | - | Перепад давления, очищенный от влияния расхода. |
| `dp_warn_kpa` | config | `5.0` кПа | Порог предупредительного состояния. |
| `dp_crit_kpa` | config | `10.0` кПа | Порог критического состояния. |
| `planned_maintenance_rul_h` | config | `72.0` ч | Горизонт планового обслуживания. |
| `urgent_maintenance_rul_h` | config | `12.0` ч | Горизонт срочного обслуживания. |
| `p_min_mpa` | config | `0.10` МПа | Минимально допустимое выходное давление. |
| `confidence_data` | расчет | `0..1` | Доверие к данным. |
| `confidence_model` | расчет | `0..1` | Доверие к ML-прогнозу. |
| `confidence_consistency` | расчет | `0..1` | Согласованность ML-RUL и аналитического RUL. |
| `confidence_total` | расчет | `0..1` | Итоговое доверие к решению. |

## 2. Rule-Based Baseline

Эти правила формируют `rule_state`, `rule_alarm_flag` и `rule_recommendation`.

Порядок применения:

1. Сначала определяется дискретное состояние фильтра.
2. Затем рассчитывается флаг тревоги.
3. После этого выбирается рекомендация по обслуживанию.
4. В конце рассчитывается доверие к rule-based выводу.

### RB-STATE-001. Неизвестное состояние из-за отсутствующего перепада

ЕСЛИ:

```text
delta_p_norm_kpa is NaN
```

ТО:

```text
rule_state = unknown
```

### RB-STATE-002. Нормальное состояние

ЕСЛИ:

```text
delta_p_norm_kpa < dp_warn_kpa
```

ТО:

```text
rule_state = normal
```

### RB-STATE-003. Предупредительное состояние

ЕСЛИ:

```text
dp_warn_kpa <= delta_p_norm_kpa < dp_crit_kpa
```

ТО:

```text
rule_state = warning
```

### RB-STATE-004. Критическое состояние

ЕСЛИ:

```text
delta_p_norm_kpa >= dp_crit_kpa
```

ТО:

```text
rule_state = critical
```

### RB-STATE-005. Неизвестное состояние из-за пропусков

ЕСЛИ:

```text
quality_code == missing
```

ТО:

```text
rule_state = unknown
```

Это правило применяется после пороговой классификации и может перезаписать `normal`, `warning` или `critical`.

### RB-ALARM-001. Флаг тревоги

ЕСЛИ:

```text
rule_state in {warning, critical}
```

ТО:

```text
rule_alarm_flag = true
```

ИНАЧЕ:

```text
rule_alarm_flag = false
```

### RB-MNT-001. Мониторинг по умолчанию

ЕСЛИ не сработали правила обслуживания:

```text
no maintenance threshold reached
```

ТО:

```text
rule_recommendation = continue_monitoring
```

### RB-MNT-002. Плановое обслуживание

ЕСЛИ:

```text
rul_analytic_h < planned_maintenance_rul_h
```

При текущем конфиге:

```text
rul_analytic_h < 72 ч
```

ТО:

```text
rule_recommendation = planned_maintenance
```

### RB-MNT-003. Срочное обслуживание по RUL

ЕСЛИ:

```text
rul_analytic_h < urgent_maintenance_rul_h
```

При текущем конфиге:

```text
rul_analytic_h < 12 ч
```

ТО:

```text
rule_recommendation = urgent_maintenance
```

Это правило сильнее `RB-MNT-002`.

### RB-MNT-004. Срочное обслуживание по критическому состоянию

ЕСЛИ:

```text
rule_state == critical
```

ТО:

```text
rule_recommendation = urgent_maintenance
```

### RB-DQ-001. Проверка данных вместо обслуживания

ЕСЛИ:

```text
rule_state == unknown
```

ТО:

```text
rule_recommendation = inspect_sensor_data
```

Это правило применяется последним и может заменить рекомендацию обслуживания на проверку данных.

## 3. Расчет Confidence

Эти правила не назначают действие напрямую, но задают доверие, которое дальше используется гибридным слоем.

### CONF-DATA-001. Жесткое вето данных

ЕСЛИ есть физически невозможное наблюдение:

```text
P_out_MPa > P_in_MPa
OR deltaP_kPa < 0
OR Q_m3h < 0
```

ТО:

```text
data_hard_veto = true
confidence_data = 0.05
```

Если одно из проверяемых физических значений отсутствует, само по себе это не включает жесткое вето.

### CONF-DATA-002. Базовое доверие по quality_code

ЕСЛИ нет жесткого вето данных, ТО:

```text
quality_code == good    -> confidence_data_base = 1.00
quality_code == missing -> confidence_data_base = 0.45
quality_code == invalid -> confidence_data_base = 0.20
otherwise               -> confidence_data_base = 0.50
```

### CONF-DATA-003. Штраф за пропуски в часовом окне

ЕСЛИ нет жесткого вето данных, ТО:

```text
confidence_data = clip(
  confidence_data_base - min(missing_rate_1h, 1.0) * 0.35,
  0,
  1
)
```

### CONF-MODEL-001. Недоступный ML-прогноз

ЕСЛИ:

```text
RUL_ml_h is NaN
OR RUL_ml_h < 0
```

ТО:

```text
confidence_model = 0.0
```

### CONF-MODEL-002. Базовое доверие к ML

ЕСЛИ ML-прогноз доступен:

```text
confidence_model = 0.70
```

### CONF-MODEL-003. Бонус за совпадение состояния

ЕСЛИ:

```text
state_pred == state
```

ТО:

```text
confidence_model = confidence_model + 0.10
```

Правило применяется только когда доступны оба состояния: `state_pred` и `state`.

### CONF-MODEL-004. Штраф за расхождение состояния

ЕСЛИ:

```text
state_pred != state
```

ТО:

```text
confidence_model = confidence_model - 0.15
```

Правило применяется только когда доступны оба состояния: `state_pred` и `state`.

После бонуса или штрафа:

```text
confidence_model = clip(confidence_model, 0, 1)
```

### CONF-CONS-001. Неполная пара RUL

ЕСЛИ:

```text
RUL_ml_h is NaN
OR RUL_analytic_h is NaN
```

ТО:

```text
confidence_consistency = 0.35
```

### CONF-CONS-002. Согласованность ML и аналитики

ЕСЛИ оба RUL доступны:

```text
denominator = max(
  abs(RUL_ml_h),
  abs(RUL_analytic_h),
  planned_maintenance_rul_h,
  1.0
)

relative_gap = abs(RUL_ml_h - RUL_analytic_h) / denominator

confidence_consistency = clip(1.0 - relative_gap, 0.05, 1.0)
```

### CONF-TOTAL-001. Итоговое доверие

ВСЕГДА:

```text
confidence_total = clip(
  confidence_data * confidence_model * confidence_consistency,
  0,
  1
)
```

## 4. Правила Выбора Итогового RUL

Эти правила формируют `RUL_fused_h` и `rul_source`.

Правила проверяются сверху вниз. Первое подходящее правило выбирает итоговый `RUL_fused_h`, задает `rul_source` и добавляет код fusion-правила в `rule_trace`.

### R-FUSE-000. RUL недоступен

ЕСЛИ:

```text
RUL_ml_h is NaN
AND RUL_analytic_h is NaN
```

ТО:

```text
RUL_fused_h = NaN
rul_source = unavailable
```

### R-FUSE-003. Вето данных

ЕСЛИ:

```text
data_hard_veto == true
```

ТО:

```text
RUL_fused_h = first_available(RUL_analytic_h, RUL_ml_h)
rul_source = analytic_data_veto
```

В `rule_trace` этот вариант записывается как `R-FUSE-003`.

### R-FUSE-003. Аналитический fallback при недоступном ML

ЕСЛИ:

```text
RUL_ml_h is NaN
```

ТО:

```text
RUL_fused_h = first_available(RUL_analytic_h, RUL_ml_h)
rul_source = analytic_fallback
```

В `rule_trace` этот вариант записывается как `R-FUSE-003`.

### R-FUSE-001. Использование ML-прогноза

ЕСЛИ:

```text
confidence_total >= 0.65
AND confidence_consistency >= 0.55
```

ТО:

```text
RUL_fused_h = RUL_ml_h
rul_source = ml_baseline
```

### R-FUSE-002. Консервативный минимум

ЕСЛИ:

```text
RUL_analytic_h is not NaN
AND confidence_total >= 0.30
AND rule R-FUSE-001 did not fire
```

ТО:

```text
RUL_fused_h = min(RUL_ml_h, RUL_analytic_h)
rul_source = conservative_min
```

### R-FUSE-003. Аналитический fallback при низком доверии

ЕСЛИ не сработали предыдущие правила выбора RUL:

```text
otherwise
```

ТО:

```text
RUL_fused_h = first_available(RUL_analytic_h, RUL_ml_h)
rul_source = analytic_fallback
```

В `rule_trace` этот вариант записывается как `R-FUSE-003`.

## 5. Продукционные Правила Действий

Эти правила применяются сверху вниз. Первое сработавшее правило возвращает итоговые `action`, `priority`, `due_time_h` и `explanation`.

### R-SAFE-002. Запрос на останов по низкому выходному давлению

ЕСЛИ:

```text
P_out_MPa < p_min_mpa
AND confidence_data >= 0.50
```

ТО:

```text
action = shutdown_request
priority = P0
due_time_h = 0.0
```

Объяснение: выходное давление ниже допустимого уровня при приемлемом качестве данных.

### R-DQ-001. Проверка датчиков при физическом вето

ЕСЛИ:

```text
data_hard_veto == true
```

ТО:

```text
action = sensor_check
priority = P0
due_time_h = 0.0
```

Объяснение: ML-прогноз заблокирован из-за физически некорректного измерения.

### R-DQ-002. Проверка датчиков при ненадежных данных

ЕСЛИ:

```text
quality_code != good
AND confidence_data < 0.45
```

ТО:

```text
action = sensor_check
priority = P0
due_time_h = 0.0
```

Объяснение: данные ненадежны, требуется проверка датчиков.

### R-SAFE-001. Срочное обслуживание при критическом состоянии

ЕСЛИ:

```text
state == critical
AND confidence_data >= 0.50
```

ТО:

```text
action = urgent_maintenance
priority = P1
due_time_h = min(max(RUL_fused_h, 0), urgent_maintenance_rul_h)
```

Если `RUL_fused_h` недоступен:

```text
due_time_h = urgent_maintenance_rul_h
```

При текущем конфиге fallback-срок равен `12` часам.

### R-CONS-001. Ручная проверка при сильном расхождении RUL

ЕСЛИ:

```text
confidence_consistency < 0.25
AND confidence_data >= 0.50
```

ТО:

```text
action = manual_review
priority = P1
due_time_h = min(max(RUL_fused_h, 0), planned_maintenance_rul_h)
```

Если `RUL_fused_h` недоступен, срок берется равным `planned_maintenance_rul_h`, то есть `72` часам.

Объяснение: ML-RUL и аналитический RUL сильно расходятся.

### R-CONS-002. Ручная проверка при низком общем доверии

ЕСЛИ:

```text
confidence_total < 0.20
```

ТО:

```text
action = manual_review
priority = P1
due_time_h = min(max(RUL_fused_h, 0), planned_maintenance_rul_h)
```

Если `RUL_fused_h` недоступен, срок берется равным `planned_maintenance_rul_h`, то есть `72` часам.

Объяснение: общее доверие к решению низкое.

### R-MNT-003. Срочное обслуживание по итоговому RUL

ЕСЛИ:

```text
RUL_fused_h < urgent_maintenance_rul_h
```

ТО:

```text
action = urgent_maintenance
priority = P1
due_time_h = min(max(RUL_fused_h, 0), urgent_maintenance_rul_h)
```

При текущем конфиге верхняя граница срока равна `12` часам.

### R-MNT-002. Плановое обслуживание при warning и малом RUL

ЕСЛИ:

```text
state == warning
AND RUL_fused_h < planned_maintenance_rul_h
```

ТО:

```text
action = planned_maintenance
priority = P2
due_time_h = min(max(RUL_fused_h, 0), planned_maintenance_rul_h)
```

При текущем конфиге верхняя граница срока равна `72` часам.

### R-MNT-002. Плановое обслуживание по итоговому RUL

ЕСЛИ:

```text
RUL_fused_h < planned_maintenance_rul_h
```

ТО:

```text
action = planned_maintenance
priority = P2
due_time_h = min(max(RUL_fused_h, 0), planned_maintenance_rul_h)
```

При текущем конфиге верхняя граница срока равна `72` часам.

### R-MNT-001. Продолжение мониторинга

ЕСЛИ не сработали предыдущие правила действий:

```text
otherwise
```

ТО:

```text
action = monitor
priority = P3
due_time_h = NaN
```

Объяснение: критические условия не обнаружены, мониторинг продолжается.

## 6. Правило Объяснения

### R-EXPL-001. Формирование объяснения

При каждом финальном действии в `rule_trace` добавляется:

```text
R-EXPL-001
```

Затем формируется текстовое объяснение, включающее:

- выбранный источник RUL;
- итоговый `RUL_fused_h`;
- уровень доверия;
- сработавшие правила;
- практическое действие для оператора.

`rule_trace` показывает правила, которые определили итоговое решение: одно правило выбора RUL, одно правило действия и `R-EXPL-001`. Он не хранит полный список всех проверенных условий.

## 7. Приоритеты действий

| Приоритет | Действия | Смысл |
|---|---|---|
| `P0` | `shutdown_request`, `sensor_check` при плохих данных | Немедленная реакция: безопасность или проверка датчиков. |
| `P1` | `urgent_maintenance`, `manual_review` | Высокий приоритет: срочное ТО или инженерная проверка. |
| `P2` | `planned_maintenance` | Плановое обслуживание в пределах расчетного срока. |
| `P3` | `monitor` | Продолжать штатный мониторинг. |
