from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator.config import ScenarioConfig


def simulate_degradation(cfg: ScenarioConfig, profile: pd.DataFrame) -> pd.DataFrame:
    """Считает скрытое накопление засорения фильтра по профилю расхода.

    Функция моделирует внутреннее состояние фильтра, которое напрямую не измеряется
    датчиками, но затем используется физической моделью перепада давления. На каждом
    временном шаге засорение растет с базовой скоростью ``cfg.k_s_per_hour`` и
    масштабируется нагрузкой по расходу:

    ``load = (max(q_true_m3h, 0) / cfg.q_nominal_m3h) ** cfg.gamma_load``

    ``clog[i] = clog[i - 1] + cfg.k_s_per_hour * load * dt_h``

    где ``dt_h`` — длительность шага в часах. Значение ``cfg.gamma_load`` задает
    нелинейность влияния расхода: при ``1.0`` влияние линейное, при значениях выше
    ``1.0`` повышенный расход сильнее ускоряет деградацию.

    Если в конфигурации задан ``cfg.maintenance_day``, на соответствующем шаге
    выполняется обслуживание: ``clog_level`` сбрасывается до ``cfg.c_reset``, но не
    увеличивается относительно предыдущего значения.

    Args:
        cfg: Конфигурация сценария. Используются параметры ``step_minutes``,
            ``c0``, ``maintenance_day``, ``c_reset``, ``q_nominal_m3h``,
            ``gamma_load`` и ``k_s_per_hour``.
        profile: Таблица режимов работы с колонкой ``q_true_m3h``. Длина таблицы
            задает длину траектории деградации.

    Returns:
        Таблица с двумя колонками:
        ``clog_level`` — скрытый уровень засорения в диапазоне ``[0, 1]``;
        ``maintenance_event`` — булев флаг обслуживания на текущем шаге.
    """
    q = profile["q_true_m3h"].to_numpy()  # Берем истинный расход из профиля и переводим в массив NumPy.
    n = len(q)  # Запоминаем количество временных точек в симуляции.
    dt_h = cfg.step_minutes / 60.0  # Переводим шаг дискретизации из минут в часы.
    clog = np.empty(n)  # Создаем массив под скрытый уровень засорения на каждом шаге.
    maintenance = np.zeros(n, dtype=bool)  # Создаем булев массив флагов обслуживания, изначально везде False.
    clog[0] = cfg.c0  # Задаем начальное засорение фильтра из конфигурации.

    maintenance_idx = None  # По умолчанию считаем, что обслуживание в сценарии не задано.
    if cfg.maintenance_day is not None:  # Проверяем, задан ли день обслуживания в конфигурации.
        maintenance_idx = int(  # Переводим день обслуживания в номер шага временной сетки.
            cfg.maintenance_day * 24 * 60 / cfg.step_minutes
        )
        maintenance_idx = min(max(maintenance_idx, 1), n - 1)  # Ограничиваем индекс допустимым диапазоном.

    for i in range(1, n):  # Идем по всем шагам, начиная со второго, потому что первый уже инициализирован.
        if maintenance_idx is not None and i == maintenance_idx:  # Проверяем, наступил ли шаг обслуживания.
            clog[i] = min(  # Сбрасываем засорение после обслуживания до остаточного уровня.
                cfg.c_reset,
                clog[i - 1],
            )
            maintenance[i] = True  # Отмечаем, что на этом шаге было обслуживание.
            continue  # Переходим к следующему шагу без обычного прироста засорения.
        load = (  # Считаем нагрузку по расходу за прошедший интервал.
            max(q[i - 1], 0.0) / cfg.q_nominal_m3h
        ) ** cfg.gamma_load
        clog[i] = np.clip(  # Увеличиваем засорение и удерживаем его в диапазоне от 0 до 1.
            clog[i - 1] + cfg.k_s_per_hour * load * dt_h,
            0.0,
            1.0,
        )

    return pd.DataFrame(  # Возвращаем результат как таблицу, совместимую с остальным пайплайном.
        {"clog_level": clog, "maintenance_event": maintenance}
    )
