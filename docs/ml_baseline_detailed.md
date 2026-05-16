# ML baseline: обучение, прогнозирование и использование

Документ описывает ML-часть текущей системы: какие модели используются, какие данные подаются на вход, как формируется train/test split, какие метрики считаются и как ML-прогноз используется в гибридной логике.

Актуальная реализация находится в `src/ml/baseline.py`. Вызов из основного конвейера находится в `main.py`.

## 1. Назначение ML-слоя

ML-слой нужен не для прямого принятия решения об обслуживании, а для получения двух прогнозов:

- `RUL_pred_h` - прогноз остаточного ресурса фильтра в часах;
- `state_pred` - прогноз состояния фильтра: `normal`, `warning` или `critical`.

Дальше эти прогнозы передаются в гибридный слой `src/hybrid/decision_engine.py`, где они сравниваются с аналитическим RUL, качеством данных и продукционными правилами.

Итоговое решение принимает не ML сам по себе, а гибридная логика.

## 2. Используемые модели

В текущей версии используются две классические модели Random Forest из `scikit-learn`.

### 2.1. RUL-регрессор

Модель:

```python
RandomForestRegressor(
    n_estimators=120,
    max_depth=14,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
```

Задача:

```text
наблюдаемые признаки -> RUL_oracle_h
```

То есть модель учится предсказывать истинный остаточный ресурс `RUL_oracle_h`, который доступен только в синтетических данных.

Выход:

```text
RUL_pred_h
```

В гибридном слое это поле переименовывается в:

```text
RUL_ml_h
```

### 2.2. Классификатор состояния

Модель:

```python
RandomForestClassifier(
    n_estimators=120,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

Задача:

```text
наблюдаемые признаки -> state
```

Модель классифицирует состояние фильтра:

```text
normal / warning / critical
```

Выход:

```text
state_pred
```

`class_weight="balanced"` используется, потому что классы могут быть несбалансированы: нормальных точек обычно больше, чем warning/critical.

## 3. Входные признаки ML

Список входов зафиксирован в `ML_INPUT_COLUMNS`:

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

### 3.1. Сырые наблюдаемые признаки

```text
P_in_MPa
P_out_MPa
deltaP_kPa
Q_m3h
T_C
```

Это телеметрия после модели датчиков и дефектов. Она имитирует то, что реально было бы доступно системе мониторинга.

### 3.2. Инженерные признаки

```text
deltaP_norm_kPa
deltaP_roll_mean_1h
deltaP_roll_std_1h
deltaP_slope_6h
Q_roll_mean_1h
missing_rate_1h
time_above_warn
```

Эти признаки строятся feature builder-ом:

- `deltaP_norm_kPa` очищает перепад от влияния расхода;
- `deltaP_roll_mean_1h` сглаживает перепад за последний час;
- `deltaP_roll_std_1h` показывает нестабильность перепада за последний час;
- `deltaP_slope_6h` показывает тренд нормированного перепада за 6 часов;
- `Q_roll_mean_1h` показывает средний режим расхода за час;
- `missing_rate_1h` показывает долю проблемных строк за час;
- `time_above_warn` показывает накопленное время выше warning-порога после последнего обслуживания.

## 4. Что не подается на вход ML

В ML-входы намеренно не включаются скрытые и целевые поля:

```text
clog_level
RUL_oracle_h
state
```

Причина:

- `clog_level` - скрытая переменная симулятора, в реальной системе она неизвестна;
- `RUL_oracle_h` - целевая переменная для обучения RUL-регрессора;
- `state` - целевая переменная для классификатора состояния.

Если подать эти поля на вход, получится утечка целевой информации.

## 5. Подготовка данных

Перед обучением выполняется `_attach_features(...)`.

Если feature builder передан отдельно, к каноническому датасету присоединяются:

```text
deltaP_roll_mean_1h
deltaP_roll_std_1h
deltaP_slope_6h
Q_roll_mean_1h
missing_rate_1h
time_above_warn
```

Слияние выполняется:

```text
по run_id + timestamp
```

Если `run_id` отсутствует, fallback:

```text
по timestamp
```

Затем `_prepare_dataset(...)`:

1. сортирует строки по `run_id` и `timestamp`;
2. приводит `timestamp` к datetime;
3. заполняет пропуски во входных признаках:

```text
ffill + bfill внутри каждого run_id
```

4. удаляет строки, где после заполнения всё ещё есть пропуски в `ML_INPUT_COLUMNS`.

## 6. Корпус для обучения

В `main.py` ML обучается не только на текущем одном прогоне, а на корпусе из нескольких независимых синтетических прогонов.

Основные train-сценарии:

```text
slow_clogging
rapid_clogging
flow_spikes
maintenance_reset
```

Train seed:

```text
7, 13, 21
```

Основные test-сценарии:

```text
slow_clogging
rapid_clogging
flow_spikes
maintenance_reset
```

Test seed:

```text
42, 101
```

Stress-test сценарии:

```text
sensor_bias
sensor_stuck
missing_data
```

Stress-test seed:

```text
42
```

Каждый прогон получает свой `run_id`, например:

```text
slow_clogging_42
rapid_clogging_13
sensor_stuck_42
```

## 7. Train/test split

Если в данных есть несколько `run_id`, используется split по целым прогонам:

```text
train = одни run_id
test = другие run_id
```

Это важно, потому что соседние точки одного временного ряда очень похожи. Если случайно перемешать строки одного и того же прогона между train и test, качество будет завышено.

В текущем основном конвейере `test_run_ids` формируется явно в `main.py`. В test попадают:

- текущий прогон;
- test seed основных сценариев;
- stress-test сценарии.

Если `run_id` нет или split по прогонам невозможен, используется fallback:

```text
первые 70% временного ряда -> train
последние 30% -> test
```

Без перемешивания.

## 8. Обучение RUL-регрессора

Целевая переменная:

```text
RUL_oracle_h
```

В обучение попадают только строки, где `RUL_oracle_h` не `NaN`:

```python
valid = data["RUL_oracle_h"].notna()
```

Это значит: если в прогоне фильтр не достигает critical в будущем, oracle-RUL может быть неизвестен, и такие строки не используются как target для RUL-регрессии.

Метрики:

```text
MAE, часы
RMSE, часы
R2
```

Также считаются групповые метрики:

```text
по run_id
по scenario
```

## 9. Обучение классификатора состояния

Целевая переменная:

```text
state
```

В обучение попадают только строки с состояниями:

```text
normal
warning
critical
```

Строки с `unknown` не используются для обучения классификатора.

Метрики:

```text
accuracy
classification_report
```

Также считаются групповые метрики:

```text
accuracy по run_id
accuracy по scenario
```

## 10. Предсказания

После обучения строится таблица `ml_predictions`.

Колонки:

```text
run_id
timestamp
filter_id
scenario
split
state_true
state_pred
RUL_oracle_h
RUL_pred_h
```

Важно:

- предсказания строятся и для train, и для test;
- колонка `split` показывает, к какой части относится строка;
- в гибридный слой передаются `RUL_pred_h` и `state_pred`.

## 11. Артефакты ML

После обучения сохраняются:

```text
outputs/<scenario>/ml_baseline/random_forest_rul.joblib
outputs/<scenario>/ml_baseline/random_forest_state.joblib
outputs/<scenario>/ml_baseline/ml_predictions.csv
outputs/<scenario>/ml_baseline/ml_predictions.parquet
outputs/<scenario>/ml_baseline/ml_metrics.json
outputs/<scenario>/ml_baseline/ml_baseline_report.md
```

Назначение:

- `random_forest_rul.joblib` - обученная модель регрессии RUL;
- `random_forest_state.joblib` - обученная модель классификации состояния;
- `ml_predictions.*` - предсказания моделей;
- `ml_metrics.json` - машинно-читаемые метрики;
- `ml_baseline_report.md` - текстовый отчет по split, признакам и качеству.

## 12. Использование ML в гибридной логике

Гибридный слой принимает:

```text
dataset + features + ml_predictions
```

Из `ml_predictions` берутся:

```text
RUL_pred_h
state_pred
```

Затем:

```text
RUL_pred_h -> RUL_ml_h
```

Далее ML влияет на три вещи.

### 12.1. Доверие к модели

Если `RUL_ml_h` отсутствует или отрицателен:

```text
confidence_model = 0.0
```

Если ML-прогноз есть:

```text
confidence_model = 0.70
```

Если `state_pred` совпадает с текущим `state`:

```text
confidence_model += 0.10
```

Если не совпадает:

```text
confidence_model -= 0.15
```

### 12.2. Согласованность с аналитикой

Сравниваются:

```text
RUL_ml_h
RUL_analytic_h
```

Чем ближе они друг к другу, тем выше:

```text
confidence_consistency
```

Если они сильно расходятся, гибридный слой снижает доверие к итоговому решению.

### 12.3. Выбор итогового RUL

ML может стать итоговым источником RUL только если доверие достаточно высокое:

```text
confidence_total >= 0.65
AND confidence_consistency >= 0.55
```

Тогда:

```text
RUL_fused_h = RUL_ml_h
rul_source = ml_baseline
```

Если доверие среднее и аналитический RUL доступен:

```text
RUL_fused_h = min(RUL_ml_h, RUL_analytic_h)
rul_source = conservative_min
```

Если доверие низкое или данные плохие, система предпочитает аналитический fallback.

## 13. Как читать качество ML

Для RUL-регрессии:

- `MAE` - средняя абсолютная ошибка в часах;
- `RMSE` - среднеквадратичная ошибка в часах, сильнее штрафует большие ошибки;
- `R2` - доля объясненной дисперсии, чем ближе к `1`, тем лучше.

Для классификации:

- `accuracy` - доля правильно классифицированных состояний;
- `classification_report` - точность, полнота и F1 по каждому классу.

Групповые метрики по `scenario` важны, потому что среднее качество может выглядеть приемлемо, но отдельный сценарий, например `sensor_stuck` или `missing_data`, может быть сложным.

## 14. Ограничения текущего ML baseline

Текущий ML - это baseline, а не финальная промышленная модель.

Ограничения:

- модель обучается только на синтетических данных;
- RandomForest не моделирует временную последовательность напрямую, а видит только текущую строку и заранее рассчитанные оконные признаки;
- качество RUL зависит от наличия строк с известным `RUL_oracle_h`;
- если oracle-RUL не определён в части прогонов, такие строки не обучают RUL-регрессор;
- ML-прогноз не используется напрямую как команда на обслуживание, а проходит через confidence и правила.

Этого достаточно для магистерского прототипа: есть воспроизводимый ML baseline, понятные признаки, независимый test по прогонам, метрики качества и объяснимая гибридная надстройка.
