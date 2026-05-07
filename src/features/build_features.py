from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    out = df.copy()
    one_hour = max(int(60 / step_minutes), 1)
    six_hours = max(int(360 / step_minutes), 1)
    day = max(int(1440 / step_minutes), 1)

    out["delta_p_ma_1h"] = out["delta_p_kpa"].rolling(one_hour, min_periods=1).mean()
    out["delta_p_std_1h"] = out["delta_p_kpa"].rolling(one_hour, min_periods=2).std()
    out["delta_p_ema_6h"] = out["delta_p_kpa"].ewm(span=six_hours, adjust=False).mean()
    out["q_mean_1h"] = out["q_m3h"].rolling(one_hour, min_periods=1).mean()
    out["q_cv_6h"] = (
        out["q_m3h"].rolling(six_hours, min_periods=2).std()
        / out["q_m3h"].rolling(six_hours, min_periods=1).mean().replace(0, np.nan)
    )
    out["time_above_warn_24h"] = (
        (out["state_obs"].isin(["warning", "critical"])).rolling(day, min_periods=1).sum()
        * step_minutes
        / 60.0
    )
    out["missing_rate_6h"] = (
        out["quality_code"].eq("missing").rolling(six_hours, min_periods=1).mean()
    )
    return out
