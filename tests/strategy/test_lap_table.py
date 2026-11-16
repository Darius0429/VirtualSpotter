import pandas as pd
import pytest

from ai_race_engineer.strategy.lap_table import lap_table


def _cars() -> pd.DataFrame:
    """Player (slot 0) completing a partial lap 1 at t=2, lap 2 at t=5 (with sectors) and lap 3 at t=8 (after a
    2 s stop); AI (slot 1) completing its partial first lap at t=3 after stopping in it.
    In:  -
    Out: pd.DataFrame  CarStatus columns used by lap_table
    """
    player = pd.DataFrame(
        {
            "time_s": range(10),
            "slot_id": 0,
            "is_player": True,
            "completed_laps": [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
            "place": [3, 3, 3, 3, 3, 2, 2, 2, 1, 1],
            "last_lap_s": [None, None, 90.0, 90.0, 90.0, 100.0, 100.0, 100.0, 110.0, 110.0],
            "last_sector1_s": [None, None, 29.0, 29.0, 29.0, 30.0, 30.0, 30.0, 31.0, 31.0],
            "last_sector2_cum_s": [None, None, 60.0, 60.0, 60.0, 65.0, 65.0, 65.0, 70.0, 70.0],
            "pit_state": [0, 0, 0, 0, 0, 0, 3, 3, 0, 0],
        }
    )
    ai = pd.DataFrame(
        {
            "time_s": range(10),
            "slot_id": 1,
            "is_player": False,
            "completed_laps": [5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            "place": 4,
            "last_lap_s": 99.0,
            "pit_state": [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    return pd.concat([player, ai], ignore_index=True)


def _state() -> pd.DataFrame:
    """Player state at t=0..9: lap 2 uses 3 l, lap 3 gets a partial refuel (net 2 l used) and new tyres.
    In:  -
    Out: pd.DataFrame  StateSample columns used by lap_table
    """
    return pd.DataFrame(
        {
            "time_s": [float(t) for t in range(10)],
            "fuel_l": [52.0, 51.0, 50.0, 49.0, 48.0, 47.0, 46.0, 46.5, 45.0, 44.0],
            "virtual_energy": [0.95, 0.92, 0.9, 0.87, 0.83, 0.8, 0.78, 0.79, 0.77, 0.74],
            "tyre_wear_fl": [1.0, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 1.0, 0.99, 0.98],
            "track_temp_c": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
            "raining": 0.0,
        }
    )


def test_one_row_per_completed_lap() -> None:
    """Each increase of completed_laps gives one row, in time order."""
    laps = lap_table(_cars(), _state())
    assert list(zip(laps["slot_id"], laps["lap"], strict=True)) == [(0, 1), (1, 6), (0, 2), (0, 3)]


def test_unfinished_first_lap_gives_no_row() -> None:
    """A car that never crosses the line has no laps."""
    cars = _cars().assign(completed_laps=3)
    assert lap_table(cars, _state()).empty


def test_lap_and_sector_times() -> None:
    """Sector 2 and 3 are split from the cumulative sector times."""
    lap = lap_table(_cars(), _state()).iloc[2]
    assert (lap["lap_time_s"], lap["sector1_s"], lap["sector2_s"], lap["sector3_s"]) == (100.0, 30.0, 35.0, 35.0)


def test_pit_stop() -> None:
    """Pitted and stop time come from the time spent with pit_state 3 during the lap."""
    laps = lap_table(_cars(), _state())
    assert list(laps["pitted"]) == [False, True, False, True]
    assert laps["stop_time_s"].iloc[3] == pytest.approx(2.0)


def test_player_deltas() -> None:
    """Fuel, energy and wear used over a full player lap."""
    lap = lap_table(_cars(), _state()).iloc[2]
    assert lap["fuel_used_l"] == pytest.approx(3.0)
    assert lap["energy_used"] == pytest.approx(0.1)
    assert lap["wear_delta_fl"] == pytest.approx(0.03)


def test_deltas_none_for_ai() -> None:
    """AI cars get no player deltas."""
    assert lap_table(_cars(), _state())[["fuel_used_l", "energy_used", "wear_delta_fl"]].iloc[1].isna().all()


def test_partial_refuel_and_tyre_change_give_none() -> None:
    """A pitted lap (partial refuel with a positive net use, new tyres) gets no fuel, energy or wear deltas."""
    assert lap_table(_cars(), _state())[["fuel_used_l", "energy_used", "wear_delta_fl"]].iloc[3].isna().all()


def test_first_recorded_lap_gives_none() -> None:
    """A car's first recorded lap (start not seen) gets no deltas and no stop time, even when pitted."""
    laps = lap_table(_cars(), _state())
    assert laps[["fuel_used_l", "energy_used", "wear_delta_fl", "stop_time_s"]].iloc[0].isna().all()
    assert pd.isna(laps["stop_time_s"].iloc[1])


def test_weather_and_place_at_the_line() -> None:
    """Place and weather are taken at the line crossing."""
    lap = lap_table(_cars(), _state()).iloc[2]
    assert (lap["place"], lap["track_temp_c"], lap["raining"]) == (2, 25.0, 0.0)


def test_missing_columns_give_null() -> None:
    """Inputs without optional columns give null values instead of failing."""
    cars = _cars()[["time_s", "slot_id", "completed_laps"]]
    laps = lap_table(cars, _state()[["time_s"]])
    assert len(laps) == 4
    assert laps[["lap_time_s", "sector2_s", "fuel_used_l", "track_temp_c", "path_wetness_avg"]].isna().all().all()
