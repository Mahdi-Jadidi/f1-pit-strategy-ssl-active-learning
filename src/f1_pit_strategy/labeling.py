import numpy as np
import pandas as pd


def percentile_score(series: pd.Series, higher_is_more: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan); values = values.fillna(0 if pd.isna(values.median()) else values.median()); ranks = values.rank(method="average", pct=True); return ranks if higher_is_more else 1 - ranks


def make_strategy_labels(frame: pd.DataFrame, emergency_share: float = .10, undercut_share: float = .15, overcut_share: float = .06) -> pd.DataFrame:
    result = frame.copy(); stop_log = np.log1p(pd.to_numeric(result["stop_duration_ms"], errors="coerce").clip(lower=0))
    result["emergency_score"] = 3 * result["safety_car_likely"].fillna(0) + 1.35 * percentile_score(result["lap_time_delta"]) + .9 * percentile_score(stop_log) + .45 * percentile_score(result["recent_pace_delta"])
    result["undercut_score"] = 1.8 * percentile_score(result["approx_gap_ahead"], False) + .9 * percentile_score(result["laps_remaining"]) + .75 * percentile_score(result["laps_since_last_pit"], False) + .45 * percentile_score(result["pressure_index"])
    result["overcut_score"] = 1.55 * percentile_score(result["laps_since_last_pit"]) + .85 * percentile_score(result["recent_pace_delta"], False) + .65 * percentile_score(result["approx_gap_behind"]) + .35 * percentile_score(result["race_progress"])
    labels = pd.Series("Standard", index=result.index); available = pd.Series(True, index=result.index)
    for label, score, share in (("Emergency", "emergency_score", emergency_share), ("Undercut", "undercut_score", undercut_share), ("Overcut", "overcut_score", overcut_share)):
        selected = result.loc[available, score].nlargest(round(len(result) * share)).index; labels.loc[selected] = label; available.loc[selected] = False
    return result.assign(strategy=labels).drop(columns=["emergency_score", "undercut_score", "overcut_score"])
