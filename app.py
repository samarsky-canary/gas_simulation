from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from main import run_pipeline
from src.simulator.config import SCENARIO_OVERRIDES, load_config, load_config_with_overrides
from src.visualization import build_interactive_plot


BASE_CONFIG_PATH = Path("configs/base.yaml")
OUTPUT_ROOT = Path("outputs/ui_runs")

GRAPH_CHOICES = {
    "Давление до и после фильтра": "pressure",
    "Перепад deltaP": "delta_p",
    "Состояние фильтра": "state",
    "Сравнение остаточного ресурса": "rul_comparison",
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
    "maintenance_reset": "Обслуживание со сбросом засорения",
}
SCENARIO_CODES = {label: code for code, label in SCENARIO_LABELS.items()}


@st.cache_data(show_spinner=False)
def _read_parquet(path: str) -> pd.DataFrame:
    """Кэширует таблицы завершенного запуска между переключениями графиков."""
    return pd.read_parquet(path)


def main() -> None:
    st.set_page_config(
        page_title="Симуляция газового фильтра",
        layout="wide",
    )

    base_cfg = load_config(BASE_CONFIG_PATH)

    st.title("Симуляция газового фильтра")

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

        with st.form("simulation_form"):
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
            k_s_per_hour = st.number_input(
                "Базовая скорость роста засорения, 1/ч",
                min_value=0.0,
                value=float(base_cfg.k_s_per_hour),
                step=0.00001,
                format="%.8f",
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
        overrides = {
            "scenario_name": scenario_name,
            "duration_days": int(duration_days),
            "step_minutes": int(step_minutes),
            "k_s_per_hour": float(k_s_per_hour),
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
        st.info("Задайте параметры слева и запустите симуляцию.")
        return

    col_rows, col_quality, col_output = st.columns([1, 1, 2])
    col_rows.metric("Строк", f"{result.row_count:,}".replace(",", " "))
    col_quality.metric("Строк с проблемами качества", result.quality_issue_rows)
    col_output.write("Каталог результатов")
    col_output.code(str(result.output_dir), language="text")

    graph_label = st.radio(
        "График",
        list(GRAPH_CHOICES),
        horizontal=True,
    )
    plot_key = GRAPH_CHOICES[graph_label]
    telemetry = _read_parquet(str(result.export_paths["wide_debug_parquet"]))
    hybrid_decisions = _read_parquet(
        str(result.hybrid_paths["hybrid_decisions_parquet"])
    )
    figure = build_interactive_plot(
        plot_key,
        result.cfg,
        telemetry,
        hybrid_decisions,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
        },
    )

    with st.expander("Сводка решений"):
        st.text(result.decision_summary)


if __name__ == "__main__":
    main()
