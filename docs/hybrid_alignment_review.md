# Проверка соответствия новому анализу гибридной модели

Дата проверки: 2026-05-13.

## Короткий вывод

Текущий прототип корректно закрывает этапы симуляции, фиксации датасета, построения признаков, rule-based baseline, классического ML baseline, гибридного decision layer, LSTM-окон и диагностических графиков.

После проверки по новому анализу исправлены два методически важных места:

- `deltaP_norm_kPa` теперь нормируется не только на расход, но и на относительную плотность газа.
- `state`, `RUL_oracle_h`, `RUL_analytic_h` и rule-based baseline теперь опираются на нормированный перепад, а не на сырой `deltaP_kPa`.

Это важно, потому что сырой перепад давления зависит от расхода и может имитировать засорение при режимных скачках.

## Что уже соответствует концепции

1. Симулятор генерирует телеметрию, скрытый уровень засорения, шумы и дефекты качества данных.
2. Формат датасета зафиксирован в CSV/Parquet:
   `timestamp`, `filter_id`, `scenario`, `P_in_MPa`, `P_out_MPa`, `deltaP_kPa`, `Q_m3h`, `T_C`, `rho_rel`, `clog_level`, `deltaP_norm_kPa`, `state`, `RUL_oracle_h`, `RUL_analytic_h`, `quality_code`, `fault_flags`.
3. Для ML/LSTM входов используются только наблюдаемые признаки:
   `P_in_MPa`, `P_out_MPa`, `deltaP_kPa`, `Q_m3h`, `T_C`, `rho_rel`, `deltaP_norm_kPa`.
4. Скрытые/целевые переменные не подаются на вход моделей:
   `clog_level`, `RUL_oracle_h`, `state`.
5. Feature builder формирует диагностические признаки:
   `deltaP_norm_kPa`, `deltaP_roll_mean_1h`, `deltaP_roll_std_1h`, `deltaP_slope_6h`, `Q_roll_mean_1h`, `missing_rate_1h`, `time_above_warn`.
6. Rule-based baseline реализует пороговую классификацию состояния и рекомендации по `RUL_analytic_h`.
7. ML baseline обучает:
   - `RandomForestRegressor` для `RUL_oracle_h`;
   - `RandomForestClassifier` для `state`.
8. LSTM window builder формирует окна:
   - `X.shape = (samples, sequence_length, features)`;
   - `y.shape = (samples,)`;
   - основная цель: `RUL_oracle_h`.
9. Гибридный decision layer формирует итоговые решения:
   - `confidence_data`;
   - `confidence_model`;
   - `confidence_consistency`;
   - `confidence_total`;
   - `RUL_fused_h`;
   - `action`;
   - `priority`;
   - `rule_trace`;
   - `explanation`.

## Что исправлено в коде

1. `src/simulator/labels.py`
   - `state_true` считается по clean `deltaP_norm`, а не по raw `deltaP`.
   - `state_obs` считается по observed `deltaP_norm`.
   - `RUL_oracle_h` считается как время до critical по clean `deltaP_norm`.
   - `RUL_analytic_h` приведён к той же нормированной critical-логике.

2. `src/features/build_features.py`
   - `deltaP_norm_kPa = deltaP_kPa / max((Q / Q_nominal)^alpha_flow * rho_rel, eps)`.

3. `src/simulator/exporters.py`
   - канонический датасет экспортирует `deltaP_norm_kPa` в той же логике, что feature builder.

4. `src/rules/baseline.py`
   - правила состояния используют `delta_p_norm_kpa`.
   - русифицированный CSV baseline содержит колонку нормированного перепада.
   - описание правил обновлено под нормированный перепад.

5. `configs/base.yaml`
   - базовая скорость деградации увеличена до `k_s_per_hour: 0.00045`, чтобы clean-траектория доходила до critical в пределах горизонта симуляции и могла формировать `RUL_oracle_h`.

6. `src/hybrid/decision_engine.py`
   - добавлен слой гибридного принятия решений;
   - `RUL_ml_h` используется как временный заменитель будущего `RUL_LSTM_h`;
   - реализованы confidence, fusion RUL, правила качества данных, безопасности, согласованности и обслуживания;
   - формируются `action`, `priority`, `due_time_h`, `rule_trace` и `explanation`.

## Контрольные результаты

Тесты:

- `13 passed`.

Полный pipeline:

- успешно пересозданы CSV/Parquet;
- пересозданы русифицированные CSV;
- пересозданы признаки;
- пересоздан rule-based baseline;
- обучен ML baseline;
- построены гибридные решения;
- построены LSTM-окна;
- построены графики.

Диагностика связи засорения и перепада:

- корреляция `clog_level` и сырого `deltaP`: `0.832`;
- корреляция `clog_level` и 24-часового среднего `deltaP`: `0.983`;
- корреляция `clog_level` и 24-часового среднего `deltaP_norm`: `0.998`.

Это подтверждает, что нормированный перепад лучше отражает деградацию фильтра.

LSTM-окна:

- `sequence_length = 288`;
- шаг данных: 5 минут;
- окно: 24 часа;
- `X_shape = [25633, 288, 7]`;
- `y_shape = [25633]`.

Примечание: ранее обсуждался шаг 10 минут и 144 точки. Текущая конфигурация использует 5 минут и 288 точек. Это корректно для прототипа, но в тексте работы нужно явно указать выбранную дискретизацию.

## Что уже стало гибридной моделью

Теперь есть отдельный продукционный слой `src/hybrid`, который превращает прогноз и правила в управленческое решение.

Реализовано:

- `confidence_data`;
- `confidence_model`;
- `confidence_consistency`;
- `confidence_total`;
- `RUL_fused_h`;
- выбор источника итогового RUL;
- вето для физически некорректных данных;
- правила `DATA_QUALITY`, `SAFETY`, `CONSISTENCY`, `MAINTENANCE`, `EXPLANATION`;
- `rule_trace`;
- итоговые поля `action`, `priority`, `due_time_h`, `explanation`;
- журнал решений `outputs/slow_clogging/hybrid/hybrid_decisions.csv`.
- пакеты объяснения `outputs/slow_clogging/hybrid/decision_packages.jsonl`.
- человекочитаемые карточки `outputs/slow_clogging/hybrid/decision_cards.md`.

Ограничение: вместо настоящего `RUL_LSTM` пока используется `RUL_ml_h` от RandomForest baseline. Архитектура уже готова к замене этого источника на LSTM-прогноз.

## Что ещё не реализовано

Не реализованы:

- `RUL_LSTM`;
- ансамбль LSTM и интервалы `p10/median`;
- отдельная группа `ESCALATION`;
- post-maintenance правила;
- отдельная UI-карточка решения;
- SHAP или другой механизм объяснения вклада признаков.

## Правильная следующая архитектурная цель

Следующим этапом нужно заменить временный ML-прогноз на LSTM-прогноз:

```text
телеметрия
-> quality/fault checks
-> feature builder
-> RUL_LSTM вместо RUL_ml_h
-> RUL_analytic
-> confidence
-> RUL_fused
-> rule engine
-> action + explanation + rule_trace
```

Минимальный выход уже реализован в `hybrid_decisions.csv`:

- `action`;
- `priority`;
- `due_time_h`;
- `RUL_ml_h`, позднее `RUL_LSTM_h`;
- `rul_analytic_h`;
- `rul_fused_h`;
- `rul_source`;
- `confidence_total`;
- `quality_code`;
- `fault_flags`;
- `rule_trace`;
- `explanation`.

## Итоговая формулировка для работы

LSTM-модель должна прогнозировать остаточный ресурс газового фильтра по окну телеметрических данных. Аналитическая модель используется как интерпретируемый baseline и fallback. Продукционная система правил проверяет качество данных, согласованность прогнозов, выбирает итоговый RUL и формирует объяснимую рекомендацию по техническому обслуживанию.

Иначе говоря:

- LSTM отвечает за прогноз.
- Правила отвечают за доверие, безопасность и действие.
- Гибридная модель объединяет прогноз и правила в объяснимую рекомендацию.
