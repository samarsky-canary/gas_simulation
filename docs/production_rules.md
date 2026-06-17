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

Эти правила формируют `rule_state` и `rule_recommendation`.

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

Тревожное состояние определяется напрямую по `rule_state in {warning, critical}`.

ИНАЧЕ:

Для `normal` и `unknown` отдельный флаг тревоги не формируется.

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

Классификация состояния ML сейчас отключена, поэтому дополнительных бонусов или штрафов за `state_pred` нет.

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

Правила проверяются сверху вниз. Первое подходящее правило выбирает итоговый `RUL_fused_h` и задает `rul_source`.

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

## 5. События Обслуживания В UI

Гибридный слой больше не назначает действия обслуживания. Он только рассчитывает `RUL_fused_h`, `rul_source` и доверия.

Таблица событий обслуживания в UI рассчитывается отдельно по трем RUL-рядам:

```text
RUL_ml_h
RUL_analytic_h
RUL_fused_h
```

Для планового обслуживания фиксируется первая временная точка, где выбранный RUL устойчиво ниже:

```text
planned_maintenance_rul_h + stable_degraded_rul_h
```

Для срочного обслуживания используется:

```text
urgent_maintenance_rul_h + stable_degraded_rul_h
```

Условие должно сохраняться в течение `stable_degraded_rul_h` часов.
