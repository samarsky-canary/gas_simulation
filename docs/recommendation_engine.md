# Рекомендательный слой: как ML-прогноз превращается в решение

Этот документ описывает слой `src/hybrid/decision_engine.py`. В текущей версии он играет роль reasoner / recommendation engine: читает результат ML baseline, аналитический RUL, качество данных и признаки, а затем выдаёт управленческую рекомендацию.

Explainable AI и подробные карточки решений здесь не рассматриваются. Фокус документа: какие данные входят, как они объединяются, как считается доверие, как выбирается итоговый RUL и как назначается действие.

## 1. Назначение слоя

ML-модель сама по себе выдаёт только прогноз:

```text
RUL_pred_h
```

Но техническое решение не должно приниматься только по одному числу. Поэтому после ML работает отдельный слой правил:

```text
dataset + features + ml_predictions
-> confidence
-> RUL_fused_h
-> rule engine
-> action + priority + due_time_h + explanation
```

## 2. Входы рекомендательного слоя

Функция:

```python
build_hybrid_decisions(cfg, dataset, features, ml_predictions)
```

Принимает три таблицы.

### 2.1. `dataset`

Главный фиксированный датасет:

- телеметрия;
- состояние;
- `RUL_oracle_h`;
- `RUL_analytic_h`;
- качество данных;

### 2.2. `features`

Таблица признаков feature builder. Из неё reasoner использует:

- `deltaP_roll_mean_1h`;
- `deltaP_slope_6h`;
- `missing_rate_1h`;
- `time_above_warn`.

### 2.3. `ml_predictions`

Таблица ML baseline. Используются:

- `RUL_pred_h`.

Внутри reasoner:

```text
RUL_pred_h -> RUL_ml_h
```

## 3. Выходная таблица `hybrid_decisions`

Файлы:

```text
hybrid/hybrid_decisions.csv
hybrid/hybrid_decisions.parquet
hybrid/hybrid_decisions_ru.csv
hybrid/hybrid_decision_logic.md
hybrid/decision_packages.jsonl
hybrid/decision_cards.md
```

Основная таблица содержит колонки:

| Колонка | Смысл |
|---|---|
| `timestamp` | Временная точка решения. |
| `filter_id` | Идентификатор фильтра. |
| `scenario` | Сценарий. |
| `state` | Текущее состояние фильтра из датасета. |
| `quality_code` | Код качества данных. |
| `deltaP_norm_kPa` | Нормированный перепад. |
| `deltaP_roll_mean_1h` | Средний перепад за 1 час. |
| `deltaP_slope_6h` | Наклон нормированного перепада за 6 часов. |
| `time_above_warn` | Время выше warning-порога. |
| `missing_rate_1h` | Доля пропусков за 1 час. |
| `RUL_oracle_h` | Истинный RUL, только для оценки. |
| `RUL_analytic_h` | Аналитический RUL. |
| `RUL_ml_h` | ML-прогноз RUL. |
| `RUL_fused_h` | Итоговый выбранный RUL. |
| `rul_source` | Источник итогового RUL. |
| `confidence_data` | Доверие к данным. |
| `confidence_model` | Доверие к ML-прогнозу. |
| `confidence_consistency` | Согласованность ML и аналитики. |
| `confidence_total` | Итоговое доверие. |
| `action` | Итоговое действие. |
| `priority` | Приоритет действия. |
| `due_time_h` | Срок выполнения в часах. |
| `rule_trace` | Какие правила сработали. |
| `explanation` | Краткое текстовое объяснение. |

## 4. Подготовка входов

Внутри `_prepare_inputs(...)`:

1. `dataset.timestamp` приводится к datetime.
2. Из `features` берутся нужные признаки и присоединяются по `timestamp`.
3. Из `ml_predictions` берется `RUL_pred_h`.
4. `RUL_pred_h` переименовывается в `RUL_ml_h`.
5. Если признаки отсутствуют, ставятся безопасные значения:

```text
deltaP_roll_mean_1h = deltaP_norm_kPa
deltaP_slope_6h = 0
missing_rate_1h = 0
time_above_warn = 0
```

## 5. Контроль физического вето

Перед использованием ML-прогноза проверяется физическая корректность:

```text
если P_out_MPa > P_in_MPa:
    hard_veto = true

если deltaP_kPa < 0:
    hard_veto = true

если Q_m3h < 0:
    hard_veto = true
```

Если hard veto сработал, ML-прогноз блокируется, а действие становится `sensor_check`.

## 6. Доверие к данным `confidence_data`

Базовое доверие зависит от `quality_code`:

| `quality_code` | База |
|---|---:|
| `good` | `1.00` |
| `missing` | `0.45` |
| `invalid` | `0.20` |
| другое | `0.50` |

Если есть hard veto:

```text
confidence_data = 0.05
```

Дальше применяется штраф за долю пропусков:

```text
confidence -= min(missing_rate_1h, 1.0) * 0.35
```

Итог ограничивается:

```text
confidence_data = clip(confidence, 0, 1)
```

## 7. Доверие к ML-модели `confidence_model`

Если `RUL_ml_h` отсутствует или отрицателен:

```text
confidence_model = 0
```

Иначе стартовое значение:
Иначе:

```text
confidence_model = 0.70
```

Классификация состояния сейчас отключена, поэтому `state_pred` не влияет на доверие.

## 8. Согласованность прогнозов `confidence_consistency`

Сравниваются:

- `RUL_ml_h`;
- `RUL_analytic_h`.

Если один из них отсутствует:

```text
confidence_consistency = 0.35
```

Иначе:

```text
denominator = max(abs(RUL_ml_h),
                  abs(RUL_analytic_h),
                  planned_maintenance_rul_h,
                  1)

relative_gap = abs(RUL_ml_h - RUL_analytic_h) / denominator

confidence_consistency = clip(1 - relative_gap, 0.05, 1)
```

Смысл: если ML и аналитика близки, доверие высокое. Если сильно расходятся, доверие падает.

## 9. Итоговое доверие `confidence_total`

```text
confidence_total =
  confidence_data
  * confidence_model
  * confidence_consistency
```

Затем:

```text
confidence_total = clip(confidence_total, 0, 1)
```

Это инженерная оценка доверия ко всему решению, а не только к ML.

## 10. Fusion RUL

Итоговый RUL выбирается функцией `_fuse_rul(...)`.

### 10.1. Нет RUL вообще

```text
если RUL_analytic_h отсутствует и RUL_ml_h отсутствует:
    RUL_fused_h = NaN
    rul_source = unavailable
    rule = R-FUSE-000
```

### 10.2. Жёсткое вето данных

```text
если hard_veto:
    RUL_fused_h = RUL_analytic_h, если он есть
                  иначе RUL_ml_h
    rul_source = analytic_data_veto
    rule = R-FUSE-003
```

### 10.3. ML недоступен

```text
если RUL_ml_h отсутствует:
    RUL_fused_h = RUL_analytic_h
    rul_source = analytic_fallback
    rule = R-FUSE-003
```

### 10.4. Высокое доверие

```text
если confidence_total >= 0.65
и confidence_consistency >= 0.55:
    RUL_fused_h = RUL_ml_h
    rul_source = ml_baseline
    rule = R-FUSE-001
```

### 10.5. Среднее доверие

```text
если RUL_analytic_h есть
и confidence_total >= 0.30:
    RUL_fused_h = min(RUL_ml_h, RUL_analytic_h)
    rul_source = conservative_min
    rule = R-FUSE-002
```

### 10.6. Низкое доверие

```text
иначе:
    RUL_fused_h = RUL_analytic_h, если он есть
                  иначе RUL_ml_h
    rul_source = analytic_fallback
    rule = R-FUSE-003
```

## 11. Правила действий

После выбора `RUL_fused_h` применяются правила в порядке приоритета.

### 11.1. `R-SAFE-002`: опасно низкое выходное давление

```text
если P_out_MPa < p_min_mpa
и confidence_data >= 0.50:
    action = shutdown_request
    priority = P0
    due_time_h = 0
```

### 11.2. `R-DQ-001`: физически невозможные данные

```text
если hard_veto:
    action = sensor_check
    priority = P0
    due_time_h = 0
```

### 11.3. `R-DQ-002`: плохое качество данных

```text
если quality_code != good
и confidence_data < 0.45:
    action = sensor_check
    priority = P0
    due_time_h = 0
```

### 11.4. `R-SAFE-001`: critical-состояние

```text
если state == critical
и confidence_data >= 0.50:
    action = urgent_maintenance
    priority = P1
    due_time_h = min(RUL_fused_h, urgent_maintenance_rul_h)
```

### 11.5. `R-CONS-001`: сильное расхождение ML и аналитики

```text
если confidence_consistency < 0.25
и confidence_data >= 0.50:
    action = manual_review
    priority = P1
```

### 11.6. `R-CONS-002`: низкое общее доверие

```text
если confidence_total < 0.20:
    action = manual_review
    priority = P1
```

### 11.7. `R-MNT-003`: срочный RUL

```text
если RUL_fused_h < urgent_maintenance_rul_h:
    action = urgent_maintenance
    priority = P1
```

### 11.8. `R-MNT-002`: плановый RUL

```text
если state == warning
и RUL_fused_h < planned_maintenance_rul_h:
    action = planned_maintenance
    priority = P2
```

Или:

```text
если RUL_fused_h < planned_maintenance_rul_h:
    action = planned_maintenance
    priority = P2
```

### 11.9. `R-MNT-001`: мониторинг

Если ничего критичного не найдено:

```text
action = monitor
priority = P3
due_time_h = NaN
```

## 12. Как рассчитывается срок `due_time_h`

Функция `_due_time(...)` выбирает минимальный срок:

```text
due_time_h = min(RUL_fused_h, horizon_h)
```

Если RUL отсутствует, используется сам горизонт:

```text
due_time_h = horizon_h
```

Если срок равен `0`, в консоли он показывается как:

```text
немедленно
```

## 13. Что означают действия

| `action` | Смысл |
|---|---|
| `monitor` | Продолжать наблюдение. |
| `sensor_check` | Проверить датчики и качество данных. |
| `planned_maintenance` | Назначить плановое обслуживание. |
| `urgent_maintenance` | Назначить срочное обслуживание. |
| `shutdown_request` | Запросить безопасный останов. |
| `manual_review` | Передать случай ответственному инженеру. |

## 14. Что означают приоритеты

| Приоритет | Смысл |
|---|---|
| `P0` | Самый высокий приоритет: безопасность или качество данных блокирует решение. |
| `P1` | Срочное обслуживание или ручная проверка. |
| `P2` | Плановое обслуживание. |
| `P3` | Мониторинг. |

## 15. Как читать `rule_trace`

`rule_trace` хранит список сработавших правил через `;`.

Пример:

```text
R-FUSE-002;R-SAFE-001;R-EXPL-001
```

Это значит:

1. Итоговый RUL выбран через conservative fusion.
2. Затем сработало safety-правило critical-состояния.
3. Было сформировано объяснение.

## 16. Консольная сводка

В конце `main.py` вызывает:

```python
format_console_decision_summary(hybrid_decisions)
```

Сводка содержит:

- общее число временных точек;
- распределение `action`;
- распределение `rul_source`;
- среднее и минимальное `confidence_total`;
- несколько кратких примеров решений.

## 17. Итоговая роль reasoner

Reasoner не обучается. Это не ML-модель, а инженерный слой правил.

Его задача:

```text
1. Принять ML-прогноз.
2. Проверить качество данных.
3. Сравнить ML с аналитическим RUL.
4. Выбрать итоговый RUL.
5. Применить правила безопасности и обслуживания.
6. Выдать действие, приоритет, срок и объяснение.
```

Именно этот слой делает систему не просто прогнозной, а рекомендательной.
