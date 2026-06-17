# Гибридный RUL Fusion

Текущий гибридный слой не назначает действия обслуживания. Он выбирает итоговый остаточный ресурс и рассчитывает доверие к прогнозу.

## 1. Входы

На вход функции `build_hybrid_decisions(cfg, dataset, features, ml_predictions)` подаются:

- канонический датасет;
- таблица признаков;
- ML-предсказания RUL.

Основные входные колонки:

| Поле | Смысл |
|---|---|
| `RUL_analytic_h` | Аналитическая оценка RUL. |
| `RUL_pred_h` | ML-прогноз RUL, после merge переименуется в `RUL_ml_h`. |
| `quality_code` | Качество текущей строки данных. |
| `missing_rate_1h` | Доля пропусков в часовом окне. |
| `P_in_MPa`, `P_out_MPa`, `deltaP_kPa`, `Q_m3h` | Используются для физического veto. |

## 2. Выходы

Выходная таблица сохраняется в:

```text
hybrid/hybrid_decisions.parquet
hybrid/hybrid_decisions.csv
```

Основные поля:

| Поле | Смысл |
|---|---|
| `RUL_ml_h` | ML-прогноз RUL. |
| `RUL_analytic_h` | Аналитический RUL. |
| `RUL_fused_h` | Итоговый выбранный RUL. |
| `rul_source` | Источник итогового RUL. |
| `confidence_data` | Доверие к данным. |
| `confidence_model` | Базовое доверие к ML-прогнозу. |
| `confidence_consistency` | Согласованность ML и аналитики. |
| `confidence_total` | Итоговое доверие к прогнозу. |

## 3. Доверие к данным

Если обнаружено физически невозможное измерение, например `P_out_MPa > P_in_MPa`, отрицательный перепад или отрицательный расход, включается hard veto:

```text
confidence_data = 0.05
```

Иначе используется базовое доверие по `quality_code`:

| `quality_code` | База |
|---|---:|
| `good` | 1.00 |
| `missing` | 0.45 |
| `invalid` | 0.20 |
| другое | 0.50 |

Затем доверие уменьшается на долю пропусков:

```text
confidence_data -= min(missing_rate_1h, 1.0) * 0.35
```

## 4. Доверие к ML

Если `RUL_ml_h` отсутствует или отрицательный:

```text
confidence_model = 0.0
```

Иначе:

```text
confidence_model = 0.70
```

Это простая эвристика текущего прототипа, а не калиброванная вероятность.

## 5. Согласованность ML и аналитики

Если один из прогнозов отсутствует:

```text
confidence_consistency = 0.35
```

Иначе:

```text
relative_gap = abs(RUL_ml_h - RUL_analytic_h)
             / max(abs(RUL_ml_h), abs(RUL_analytic_h), planned_maintenance_rul_h, 1)

confidence_consistency = clip(1 - relative_gap, 0.05, 1.0)
```

## 6. Итоговое доверие

```text
confidence_total = clip(
  confidence_data * confidence_model * confidence_consistency,
  0,
  1
)
```

## 7. Выбор итогового RUL

Правила применяются сверху вниз.

### RUL недоступен

```text
если RUL_analytic_h отсутствует и RUL_ml_h отсутствует:
    RUL_fused_h = NaN
    rul_source = unavailable
```

### Вето качества данных

```text
если data_hard_veto:
    RUL_fused_h = first_available(RUL_analytic_h, RUL_ml_h)
    rul_source = analytic_data_veto
```

### ML отсутствует

```text
если RUL_ml_h отсутствует:
    RUL_fused_h = first_available(RUL_analytic_h, RUL_ml_h)
    rul_source = analytic_fallback
```

### ML при высоком доверии

```text
если confidence_total >= 0.65
и confidence_consistency >= 0.55:
    RUL_fused_h = RUL_ml_h
    rul_source = ml_baseline
```

### Консервативный минимум

```text
если RUL_analytic_h есть
и confidence_total >= 0.30:
    RUL_fused_h = min(RUL_ml_h, RUL_analytic_h)
    rul_source = conservative_min
```

### Аналитический fallback

```text
иначе:
    RUL_fused_h = first_available(RUL_analytic_h, RUL_ml_h)
    rul_source = analytic_fallback
```

## 8. События обслуживания

Плановое и срочное обслуживание не назначаются внутри гибридного слоя. UI рассчитывает события отдельно по трем рядам:

- `RUL_ml_h`;
- `RUL_analytic_h`;
- `RUL_fused_h`.

Событие фиксируется, если RUL устойчиво ниже `threshold + stable_degraded_rul_h` в течение `stable_degraded_rul_h` часов.
