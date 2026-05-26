from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from main import run_pipeline
from src.simulator.config import SCENARIO_OVERRIDES, load_config, load_config_with_overrides


BASE_CONFIG_PATH = Path("configs/base.yaml")
OUTPUT_ROOT = Path("outputs/ui_runs")

GRAPH_CHOICES = {
    "01 - расход": "q",
    "02 - давление до и после фильтра": "pressure",
    "03 - перепад deltaP": "delta_p",
    "06 - состояние фильтра": "state",
    "09 - сравнение RUL": "rul_comparison",
    "11 - периоды предпочтения RUL": "rul_source_periods",
}


def main() -> None:
    st.set_page_config(
        page_title="Симуляция газового фильтра",
        layout="wide",
    )

    base_cfg = load_config(BASE_CONFIG_PATH)

    st.title("Симуляция газового фильтра")

    with st.sidebar:
        st.header("Параметры запуска")
        with st.form("simulation_form"):
            scenario_names = list(SCENARIO_OVERRIDES)
            scenario_name = st.selectbox(
                "Сценарий",
                scenario_names,
                index=scenario_names.index(base_cfg.scenario_name),
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
            k_s_per_hour = st.number_input(
                "Базовая скорость роста засорения, 1/ч",
                min_value=0.0,
                value=float(base_cfg.k_s_per_hour),
                step=0.00001,
                format="%.8f",
            )
            submitted = st.form_submit_button("Запустить симуляцию", type="primary")

    if submitted:
        overrides = {
            "scenario_name": scenario_name,
            "duration_days": int(duration_days),
            "step_minutes": int(step_minutes),
            "k_s_per_hour": float(k_s_per_hour),
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
    plot_path = result.plot_paths[plot_key]
    st.image(str(plot_path), use_container_width=True)

    with st.expander("Сводка решений"):
        st.text(result.decision_summary)


if __name__ == "__main__":
    main()
