from dataclasses import dataclass, fields

import pandas as pd

from ai_race_engineer.domains.common.data.frame import CarStatus, StateSample


@dataclass(frozen=True)
class LapRecord:
    """One completed lap of one car (row schema of the laps table).
    slot_id           int           game slot id
    lap               int           completed lap number
    time_s            float         s telemetry clock at the line
    lap_time_s        float | None  s
    sector1_s         float | None  s
    sector2_s         float | None  s
    sector3_s         float | None  s
    place             int           1-based overall position at the line
    pitted            bool          stopped in the pits (pit_state 3) during the lap
    stop_time_s       float | None  s stationary in the pits (None when not pitted or on the first recorded lap)
    fuel_used_l       float | None  litres (player only; None when pitted or on the first recorded lap)
    energy_used       float | None  fraction of virtual energy (player only; None like fuel_used_l)
    wear_delta_fl     float | None  fraction of tyre wear lost during the lap (player only; None like fuel_used_l)
    wear_delta_fr     float | None  fraction (player only)
    wear_delta_rl     float | None  fraction (player only)
    wear_delta_rr     float | None  fraction (player only)
    track_temp_c      float         degC at the line
    ambient_temp_c    float         degC at the line
    raining           float         0-1 at the line
    path_wetness_avg  float         0-1 at the line
    """

    slot_id: int
    lap: int
    time_s: float
    lap_time_s: float | None
    sector1_s: float | None
    sector2_s: float | None
    sector3_s: float | None
    place: int
    pitted: bool
    stop_time_s: float | None
    fuel_used_l: float | None
    energy_used: float | None
    wear_delta_fl: float | None
    wear_delta_fr: float | None
    wear_delta_rl: float | None
    wear_delta_rr: float | None
    track_temp_c: float
    ambient_temp_c: float
    raining: float
    path_wetness_avg: float


_CORNERS: tuple[str, ...] = ("fl", "fr", "rl", "rr")
_WEATHER: tuple[str, ...] = ("track_temp_c", "ambient_temp_c", "raining", "path_wetness_avg")


def lap_table(cars_state: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """One row per completed lap per car, detected where completed_laps increases; missing input columns give nulls.
    Fuel, energy and wear deltas are player only and null on pitted laps (refuel, tyre change) and on a car's first
    recorded lap (its start was not seen); stop_time_s is null on that first lap too.
    In:  cars_state  pd.DataFrame  cars_state_5hz rows (CarStatus columns, all cars)
         state       pd.DataFrame  player_state_5hz rows (StateSample columns)
    Out:             pd.DataFrame  LapRecord columns, in time order
    """
    cars = (
        cars_state.reindex(columns=[f.name for f in fields(CarStatus)])
        .astype({"time_s": float})
        .sort_values(["slot_id", "time_s"], kind="stable")
    )
    player_state = (
        state.reindex(columns=[f.name for f in fields(StateSample)]).astype({"time_s": float}).sort_values("time_s")
    )
    ends = cars[cars.groupby("slot_id")["completed_laps"].diff() > 0].rename(columns={"completed_laps": "lap"})
    laps = ends.merge(_lap_spans(cars), on=["slot_id", "lap"], how="left").sort_values("time_s", kind="stable")
    laps["lap_time_s"] = laps["last_lap_s"]
    laps["sector1_s"] = laps["last_sector1_s"]
    laps["sector2_s"] = laps["last_sector2_cum_s"] - laps["last_sector1_s"]
    laps["sector3_s"] = laps["last_lap_s"] - laps["last_sector2_cum_s"]
    start = _state_at(laps["start_s"], player_state)
    end = _state_at(laps["time_s"], player_state)
    clean = laps["is_player"].eq(True) & laps["start_seen"].eq(True) & ~laps["pitted"].eq(True)
    laps["fuel_used_l"] = (start["fuel_l"] - end["fuel_l"]).where(clean)
    laps["energy_used"] = (start["virtual_energy"] - end["virtual_energy"]).where(clean)
    for c in _CORNERS:
        laps[f"wear_delta_{c}"] = (start[f"tyre_wear_{c}"] - end[f"tyre_wear_{c}"]).where(clean)
    for column in _WEATHER:
        laps[column] = end[column]
    return laps.reindex(columns=[f.name for f in fields(LapRecord)]).reset_index(drop=True)


def _lap_spans(cars: pd.DataFrame) -> pd.DataFrame:
    """Per car and lap: start time, whether the start was seen, pit flag and time stationary in the pits.
    In:  cars  pd.DataFrame  CarStatus rows sorted by slot_id, time_s
    Out:       pd.DataFrame  slot_id, lap (completed_laps + 1), start_s (s), start_seen (False for the car's first
                             recorded lap), pitted, stop_time_s (s, None if not pitted or start not seen)
    """
    stopped = cars["pit_state"].eq(3)
    step_s = cars.groupby("slot_id")["time_s"].shift(-1) - cars["time_s"]
    spans = (
        cars.assign(stopped=stopped, stopped_s=step_s.where(stopped, 0.0))
        .groupby(["slot_id", "completed_laps"], as_index=False)
        .agg(start_s=("time_s", "min"), pitted=("stopped", "any"), stop_time_s=("stopped_s", "sum"))
    )
    spans["start_seen"] = spans["start_s"] > spans.groupby("slot_id")["start_s"].transform("min")
    spans["stop_time_s"] = spans["stop_time_s"].where(spans["pitted"] & spans["start_seen"])
    return spans.assign(lap=spans.pop("completed_laps") + 1)


def _state_at(time_s: pd.Series, state: pd.DataFrame) -> pd.DataFrame:
    """Latest player state at or before each time (as of), aligned to the times' index.
    In:  time_s  pd.Series     s telemetry clock (nulls give null rows)
         state   pd.DataFrame  StateSample rows sorted by time_s
    Out:         pd.DataFrame  StateSample columns, same index as time_s
    """
    times = time_s.dropna().sort_values().rename("time_s").rename_axis("row").reset_index()
    return pd.merge_asof(times, state, on="time_s").set_index("row").reindex(time_s.index)
