import numpy as np
import pandas as pd


NUMERIC_FEATURES = ["year", "round", "stop_number", "pit_lap", "pit_lap_time_ms", "grid", "current_position", "driver_points_before", "constructor_points_before", "lap_time_delta", "recent_pace_delta", "laps_since_last_pit", "race_total_laps", "laps_remaining", "race_progress", "approx_gap_ahead", "approx_gap_behind", "pressure_index", "position_change_from_grid"]
CATEGORICAL_FEATURES = ["constructor_name", "circuit_name"]


def build_feature_table(tables: dict[str, pd.DataFrame], start_year: int = 2014, end_year: int = 2024) -> pd.DataFrame:
    races = tables["races"].query("@start_year <= year <= @end_year").copy()
    race_ids = races["raceId"]
    laps = tables["lap_times"].loc[lambda d: d.raceId.isin(race_ids)].copy()
    pits = tables["pit_stops"].loc[lambda d: d.raceId.isin(race_ids)].copy()
    for frame, columns in ((laps, ["lap", "position", "milliseconds"]), (pits, ["stop", "lap", "milliseconds"])):
        for column in columns: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    laps = laps.sort_values(["raceId", "driverId", "lap"])
    laps["rolling_pace"] = laps.groupby(["raceId", "driverId"])["milliseconds"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    race_median = laps.groupby(["raceId", "lap"])["milliseconds"].median().rename("field_lap_median")
    laps = laps.join(race_median, on=["raceId", "lap"]); laps["lap_time_delta"] = laps["milliseconds"] - laps["field_lap_median"]; laps["recent_pace_delta"] = laps["milliseconds"] - laps["rolling_pace"]
    lap_features = laps[["raceId", "driverId", "lap", "position", "milliseconds", "lap_time_delta", "recent_pace_delta"]].rename(columns={"lap": "pit_lap", "position": "current_position", "milliseconds": "pit_lap_time_ms"})
    pits = pits.rename(columns={"stop": "stop_number", "lap": "pit_lap", "milliseconds": "stop_duration_ms"}).merge(lap_features, on=["raceId", "driverId", "pit_lap"], how="left")
    pits = pits.sort_values(["raceId", "driverId", "pit_lap"]); pits["laps_since_last_pit"] = pits.groupby(["raceId", "driverId"])["pit_lap"].diff().fillna(pits["pit_lap"])
    race_meta = races[["raceId", "year", "round", "name", "circuitId"]].rename(columns={"name": "race_name"})
    race_laps = laps.groupby("raceId")["lap"].max().rename("race_total_laps")
    features = pits.merge(race_meta, on="raceId", how="left").join(race_laps, on="raceId")
    features["laps_remaining"] = features["race_total_laps"] - features["pit_lap"]; features["race_progress"] = features["pit_lap"] / features["race_total_laps"]
    features["pressure_index"] = features[["lap_time_delta", "recent_pace_delta"]].abs().mean(axis=1); features["safety_car_likely"] = ((features["lap_time_delta"] > features["lap_time_delta"].quantile(.9)) & (features["stop_duration_ms"] > features["stop_duration_ms"].median())).astype(int)
    results = tables["results"][["raceId", "driverId", "constructorId", "grid", "positionOrder"]].copy(); features = features.merge(results, on=["raceId", "driverId"], how="left"); features["position_change_from_grid"] = features["grid"] - features["positionOrder"]
    features = features.merge(tables["constructors"][["constructorId", "name"]].rename(columns={"name": "constructor_name"}), on="constructorId", how="left").merge(tables["circuits"][["circuitId", "name"]].rename(columns={"name": "circuit_name"}), on="circuitId", how="left")
    features["approx_gap_ahead"] = np.nan; features["approx_gap_behind"] = np.nan; features["driver_points_before"] = 0.0; features["constructor_points_before"] = 0.0
    return features.reset_index(drop=True).rename_axis("row_id").reset_index()
