from __future__ import annotations

import json
import secrets
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from main import run_pipeline
from src.simulator.config import SCENARIO_OVERRIDES, load_config, load_config_with_overrides
from src.visualization import (
    build_current_run_quality_figure,
    build_interactive_plot,
    build_maintenance_event_table,
    build_training_scenario_figure,
    current_run_quality,
    training_quality_summary,
)


BASE_CONFIG_PATH = Path("configs/base.yaml")
OUTPUT_ROOT = Path("outputs/ui_runs")
PRG_SCHEME_IMAGE = Path("docs/prg_scheme2.png")
PRG_SCHEME_WIDTH_PX = 641
DEFAULT_UI_START_DATE = date(2026, 7, 1)
FIRST_UI_RUN_SEED = 42
MAX_RANDOM_SEED = 2**32 - 1

GRAPH_CHOICES = {
    "Давление до и после фильтра": "pressure",
    "Перепад deltaP": "delta_p",
    "Состояние фильтра": "state",
    "Оценка остаточного ресурса": "rul_comparison",
    "Источник остаточного ресурса": "rul_source_periods",
    "Доверие и качество данных": "data_confidence",
}

SCENARIO_LABELS = {
    "normal": "Нормальный режим",
    "slow_clogging": "Медленное засорение",
    "rapid_clogging": "Быстрое засорение",
    "flow_spikes": "Скачки расхода",
    "sensor_bias": "Дрейф показаний датчика",
    "sensor_stuck": "Залипание датчика",
    "missing_data": "Пропуски данных",
}
SCENARIO_CODES = {label: code for code, label in SCENARIO_LABELS.items()}


@st.cache_data(show_spinner=False)
def _read_parquet(path: str) -> pd.DataFrame:
    """Кэширует таблицы завершенного запуска между переключениями графиков."""
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> dict[str, object]:
    """Кэширует метрики использованной версии ML-модели."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_value(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "н/д"
    return f"{value:,.1f}{suffix}".replace(",", " ")


def _next_ui_seed() -> int:
    run_count = int(st.session_state.get("ui_run_count", 0))
    st.session_state["ui_run_count"] = run_count + 1
    if run_count == 0:
        return FIRST_UI_RUN_SEED
    return secrets.randbelow(MAX_RANDOM_SEED + 1)


def main() -> None:
    st.set_page_config(
        page_title="Расчёт остаточного ресурса ФГ на узле ПРГ",
        layout="wide",
    )

    base_cfg = load_config(BASE_CONFIG_PATH)

    st.title("Расчёт остаточного ресурса ФГ на узле ПРГ")

    with st.sidebar:
        st.header("Параметры запуска")
        scenario_codes = list(SCENARIO_OVERRIDES)
        scenario_labels = [SCENARIO_LABELS[code] for code in scenario_codes]
        selected_scenario = st.selectbox(
            "Сценарий",
            scenario_labels,
            index=scenario_codes.index(base_cfg.scenario_name),
            key="scenario_label",
        )
        scenario_name = SCENARIO_CODES[selected_scenario]
        scenario_p_missing = float(
            SCENARIO_OVERRIDES[scenario_name].get("p_missing", base_cfg.p_missing)
        )
        scenario_p_spike = float(
            SCENARIO_OVERRIDES[scenario_name].get("p_spike", base_cfg.p_spike)
        )
        scenario_k_s_per_hour = float(
            SCENARIO_OVERRIDES[scenario_name].get(
                "k_s_per_hour", base_cfg.k_s_per_hour
            )
        )
        scenario_a_q = float(SCENARIO_OVERRIDES[scenario_name].get("a_q", base_cfg.a_q))

        with st.form("simulation_form"):
            start_date = st.date_input(
                "Дата начала",
                value=DEFAULT_UI_START_DATE,
            )
            start_time = st.time_input(
                "Время начала",
                value=base_cfg.start_time.timetz().replace(tzinfo=None),
                step=3600,
            )
            duration_days = st.number_input(
                "Длительность, суток",
                min_value=1,
                max_value=3650,
                value=base_cfg.duration_days,
                step=1,
            )
            step_minutes = st.number_input(
                "Шаг дискретизации, минут",
                min_value=1,
                max_value=1440,
                value=base_cfg.step_minutes,
                step=1,
            )
            st.subheader("Параметры фильтра")
            q_nominal_m3h = st.number_input(
                "Паспортный / номинальный расчетный расход, м³/ч",
                min_value=1.0,
                value=float(base_cfg.q_nominal_m3h),
                step=10.0,
                format="%.1f",
            )
            q_min_m3h = q_nominal_m3h * 0.5
            q_max_m3h = q_nominal_m3h * 1.5
            st.caption(
                "Рабочий диапазон расхода рассчитывается автоматически: "
                f"{q_min_m3h:.1f}...{q_max_m3h:.1f} м³/ч."
            )
            p_in_nominal_mpa = st.number_input(
                "Номинальное входное давление, МПа",
                min_value=0.01,
                value=float(base_cfg.p_in_nominal_mpa),
                step=0.01,
                format="%.3f",
            )
            p_min_mpa = p_in_nominal_mpa * 0.5
            p_max_mpa = p_in_nominal_mpa * 1.5
            st.caption(
                "Рабочий диапазон давления рассчитывается автоматически: "
                f"{p_min_mpa:.3f}...{p_max_mpa:.3f} МПа."
            )
            dp0_kpa = st.number_input(
                "Перепад давления на чистом фильтре, кПа",
                min_value=0.01,
                value=float(base_cfg.dp0_kpa),
                step=0.1,
                format="%.2f",
            )
            dp_warn_kpa = st.number_input(
                "Предупредительный порог перепада, кПа",
                min_value=0.01,
                value=float(base_cfg.dp_warn_kpa),
                step=0.1,
                format="%.2f",
                help="Порог warning-состояния фильтра по нормированному перепаду давления.",
            )
            dp_crit_kpa = st.number_input(
                "Критический порог перепада, кПа",
                min_value=float(dp_warn_kpa) + 0.01,
                value=max(float(base_cfg.dp_crit_kpa), float(dp_warn_kpa) + 0.01),
                step=0.1,
                format="%.2f",
                help="Порог critical-состояния фильтра по нормированному перепаду давления.",
            )
            st.subheader("Расход газа")
            a_q_percent = st.number_input(
                "Амплитуда суточных колебаний расхода, %",
                min_value=0.0,
                max_value=100.0,
                value=scenario_a_q * 100.0,
                step=1.0,
                format="%.1f",
                key=f"a_q_percent_{scenario_name}",
            )
            k_s_per_hour = st.number_input(
                "Базовая скорость роста засорения, 1/ч",
                min_value=0.0,
                value=scenario_k_s_per_hour,
                step=0.00001,
                format="%.8f",
                key=f"k_s_per_hour_{scenario_name}",
            )
            st.subheader("Пороги обслуживания")
            planned_maintenance_rul_h = st.number_input(
                "Плановое обслуживание при остаточном ресурсе, ч",
                min_value=1.0,
                value=float(base_cfg.planned_maintenance_rul_h),
                step=24.0,
                format="%.1f",
            )
            urgent_maintenance_rul_h = st.number_input(
                "Срочное обслуживание при остаточном ресурсе, ч",
                min_value=1.0,
                value=float(base_cfg.urgent_maintenance_rul_h),
                step=24.0,
                format="%.1f",
            )
            stable_degraded_rul_h = st.number_input(
                "Устойчивое состояние в течение N часов",
                min_value=1.0,
                value=float(base_cfg.stable_degraded_rul_h),
                step=1.0,
                format="%.1f",
                help=(
                    "Событие фиксируется, если RUL устойчиво ниже "
                    "`порог + N часов` в течение N часов."
                ),
            )
            p_missing_percent = st.number_input(
                "Вероятность пропуска по каждому датчику, %",
                min_value=0.0,
                max_value=100.0,
                value=scenario_p_missing * 100.0,
                step=0.1,
                format="%.2f",
                key=f"p_missing_percent_{scenario_name}",
                help=(
                    "Вероятность пропуска отдельно для P_in, P_out, Q и T "
                    "на каждом временном шаге."
                ),
            )
            p_spike_percent = st.number_input(
                "Вероятность выброса по каждому датчику, %",
                min_value=0.0,
                max_value=100.0,
                value=scenario_p_spike * 100.0,
                step=0.1,
                format="%.2f",
                key=f"p_spike_percent_{scenario_name}",
                help=(
                    "Вероятность кратковременного импульсного отклонения отдельно "
                    "для P_in, P_out, Q и T на каждом временном шаге."
                ),
            )
            submitted = st.form_submit_button("Запустить симуляцию", type="primary")

    if submitted:
        start_datetime = datetime.combine(
            start_date,
            start_time,
            tzinfo=base_cfg.start_time.tzinfo,
        )
        overrides = {
            "scenario_name": scenario_name,
            "start_time": start_datetime.isoformat(),
            "seed": _next_ui_seed(),
            "duration_days": int(duration_days),
            "step_minutes": int(step_minutes),
            "q_nominal_m3h": float(q_nominal_m3h),
            "q_min_m3h": float(q_min_m3h),
            "q_max_m3h": float(q_max_m3h),
            "a_q": float(a_q_percent) / 100.0,
            "p_in_nominal_mpa": float(p_in_nominal_mpa),
            "p_min_mpa": float(p_min_mpa),
            "p_max_mpa": float(p_max_mpa),
            "dp0_kpa": float(dp0_kpa),
            "dp_warn_kpa": float(dp_warn_kpa),
            "dp_crit_kpa": float(dp_crit_kpa),
            "k_s_per_hour": float(k_s_per_hour),
            "planned_maintenance_rul_h": float(planned_maintenance_rul_h),
            "urgent_maintenance_rul_h": float(urgent_maintenance_rul_h),
            "stable_degraded_rul_h": float(stable_degraded_rul_h),
            "p_missing": float(p_missing_percent) / 100.0,
            "p_spike": float(p_spike_percent) / 100.0,
        }
        cfg = load_config_with_overrides(BASE_CONFIG_PATH, overrides)
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_ROOT / f"{run_stamp}_{cfg.scenario_name}"

        with st.spinner("Выполняется симуляция и построение графиков"):
            st.session_state["pipeline_result"] = run_pipeline(cfg, output_dir)

    result = st.session_state.get("pipeline_result")
    if result is None:
        st.subheader("Упрощенная схема ПРГ")
        if PRG_SCHEME_IMAGE.exists():
            st.image(
                str(PRG_SCHEME_IMAGE),
                caption="Схема оборудования и газопроводов ПРГ",
                width=PRG_SCHEME_WIDTH_PX,
            )
        else:
            st.warning(f"Файл схемы не найден: {PRG_SCHEME_IMAGE}")
        st.info("Задайте конфигурацию фильтра и запустите симуляцию.")
        return

    telemetry = _read_parquet(str(result.export_paths["wide_debug_parquet"]))
    hybrid_decisions = _read_parquet(
        str(result.hybrid_paths["hybrid_decisions_parquet"])
    )

    col_rows, col_quality, col_output = st.columns([1, 1, 2])
    col_rows.metric("Строк", f"{result.row_count:,}".replace(",", " "))
    col_quality.metric("Строк с проблемами качества", result.quality_issue_rows)
    col_output.write("События обслуживания")
    col_output.dataframe(
        build_maintenance_event_table(result.cfg, hybrid_decisions),
        use_container_width=True,
        hide_index=True,
    )
    plot_config = {
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
    }
    graph_label = st.radio(
        "График",
        list(GRAPH_CHOICES),
        horizontal=True,
    )
    plot_key = GRAPH_CHOICES[graph_label]
    figure = build_interactive_plot(
        plot_key,
        result.cfg,
        telemetry,
        hybrid_decisions,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config=plot_config,
    )

    if False:
        metrics = _read_json(str(result.ml_paths["ml_metrics_json"]))
        training = training_quality_summary(metrics)

        st.subheader("Качество модели на отложенных test-прогонах")
        test_columns = st.columns(4)
        test_columns[0].metric(
            "ML MAE", _metric_value(training["ml_mae_h"], " ч")
        )
        test_columns[1].metric(
            "Аналитика MAE",
            _metric_value(training["analytic_mae_h"], " ч"),
        )
        test_columns[2].metric(
            "Улучшение ML",
            _metric_value(training["improvement_percent"], "%"),
        )
        test_columns[3].metric("ML R²", _metric_value(training["ml_r2"]))
        st.plotly_chart(
            build_training_scenario_figure(metrics),
            use_container_width=True,
            config=plot_config,
        )

        st.subheader("Ошибка на текущем прогоне")
        current = current_run_quality(hybrid_decisions)
        current_columns = st.columns(3)
        current_columns[0].metric(
            "ML MAE", _metric_value(current["ml_mae_h"], " ч")
        )
        current_columns[1].metric(
            "Аналитика MAE",
            _metric_value(current["analytic_mae_h"], " ч"),
        )
        current_columns[2].metric(
            "Гибрид MAE",
            _metric_value(current["hybrid_mae_h"], " ч"),
        )
        st.plotly_chart(
            build_current_run_quality_figure(result.cfg, hybrid_decisions),
            use_container_width=True,
            config=plot_config,
        )
        st.caption(
            "Ошибка текущего прогона использует скрытый Oracle RUL симулятора. "
            "Она доступна только для демонстрации и не является test-метрикой модели."
        )


if __name__ == "__main__":
    main()
