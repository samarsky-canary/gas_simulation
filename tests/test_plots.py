from __future__ import annotations

from src.simulator.config import ScenarioConfig
from src.simulator.runner import run_scenario
from src.visualization import build_plots


def test_build_plots_creates_png_files(tmp_path) -> None:
    cfg = ScenarioConfig(duration_days=1, step_minutes=30, p_missing=0, p_spike=0, p_stuck=0)
    df, _ = run_scenario(cfg)

    paths = build_plots(cfg, df, tmp_path)

    png_paths = [path for path in paths.values() if path.suffix == ".png"]
    assert set(paths) == {
        "q",
        "pressure",
        "delta_p",
        "state",
        "rul_comparison",
        "rul_source_periods",
        "description",
        "diagnostics",
    }
    assert len(png_paths) == 6
    assert all(path.exists() and path.stat().st_size > 0 for path in png_paths)
    assert paths["description"].exists()
    assert paths["diagnostics"].exists()
