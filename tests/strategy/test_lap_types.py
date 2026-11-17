import dataclasses

from ai_race_engineer.strategy.lap_types import (
    ABORTED_LAP,
    FIRST_LAP,
    IN_LAP,
    INVALID_LAP,
    LAP,
    OUT_LAP,
    RETIRED,
    LapTypeTracker,
)
from ai_race_engineer.telemetry.race_snapshot import RaceSnapshot
from ai_race_engineer.telemetry.snapshot import SessionType
from tests.oval import oval_lap


def _ended(*steps: tuple[int, float | None, dict[str, object]], race: bool = True) -> list[tuple[int, str, float]]:
    """What a tracker says ended while a car goes through some states one second apart.
    In:  steps  tuple[int, float | None, dict[str, object]]  completed laps, last lap time (s) and the fields of the
                                                             car that differ from a car on track
         race   bool                                         the session is a race, else qualifying
    Out:        list[tuple[int, str, float]]                 (lap, name, time s) of every lap that ended
    """
    tracker, ended = LapTypeTracker(), []
    base: RaceSnapshot = oval_lap(0)[0]
    kind = SessionType.RACE if race else SessionType.QUALIFY
    for second, (completed_laps, last_lap_s, fields) in enumerate(steps):
        vehicle = dataclasses.replace(
            base.player.vehicle, completed_laps=completed_laps, last_lap_s=last_lap_s, **fields
        )
        session = dataclasses.replace(base.session, elapsed_s=float(second), session_type=kind)
        green = dataclasses.replace(base.flags, game_phase=5)
        tracker.update(dataclasses.replace(base, vehicles=(vehicle,), session=session, flags=green))
        ended += [(end.lap, end.name, end.time_s) for end in tracker.ended]
    return ended


def test_lap_from_line_to_line_is_timed_or_invalid() -> None:
    """A lap already running when the car is first seen is left out; the next ones are timed laps with the game's
    time, or invalid laps when the game gives none."""
    assert _ended((0, None, {}), (1, 41.0, {}), (1, 41.0, {}), (2, 40.0, {}), (3, None, {})) == [
        (2, LAP, 40.0),
        (3, INVALID_LAP, 1.0),
    ]


def test_pit_stop_gives_an_in_lap_and_an_out_lap() -> None:
    """Driving into the pits ends an in-lap there; the lap from the pit exit to the line is an out-lap, whatever
    the lap counter did in the pit lane."""
    pits = {"in_pits": True}
    steps = ((0, None, {}), (1, 41.0, {}), (1, 41.0, {}), (1, 41.0, pits), (2, 60.0, pits), (2, 60.0, {}))
    assert _ended(*steps, (3, 50.0, {})) == [
        (2, IN_LAP, 2.0),
        (3, OUT_LAP, 1.0),
    ]


def test_jump_to_the_garage_aborts_the_lap_or_retires_the_car() -> None:
    """A car that is in the garage without having driven into the pit lane cut its lap short: an aborted lap in
    practice and qualifying, a retirement in a race."""
    garage = {"in_pits": True, "in_garage": True}
    steps = ((0, None, {}), (1, 41.0, {}), (1, 41.0, garage))
    assert _ended(*steps, race=False) == [(2, ABORTED_LAP, 1.0)]
    assert _ended(*steps) == [(2, RETIRED, 1.0)]


def test_garage_before_the_start_of_a_race_is_not_a_retirement() -> None:
    """Cars moved between the garage and the grid before the green flag have not retired."""
    tracker, base = LapTypeTracker(), oval_lap(0)[0]
    for second, fields in enumerate(({"in_pits": True}, {}, {"in_pits": True, "in_garage": True})):
        vehicle = dataclasses.replace(base.player.vehicle, **fields)
        session = dataclasses.replace(base.session, elapsed_s=float(second))
        tracker.update(dataclasses.replace(base, vehicles=(vehicle,), session=session))
    assert tracker.ended == ()


def test_first_lap_of_a_race_and_the_garage_after_the_flag() -> None:
    """The lap from the grid to the line is the first lap of the race, with the game's time; a car put in the
    garage after it has finished is not a retirement."""
    grid = {"in_pits": True}
    assert _ended((0, None, grid), (0, None, {}), (1, 112.0, {})) == [(1, FIRST_LAP, 112.0)]
    flag = dataclasses.replace(oval_lap(0)[0].flags, game_phase=8)
    tracker = LapTypeTracker()
    base = oval_lap(0)[0]
    for second, (laps, fields) in enumerate(((0, {}), (1, {}), (1, {"in_pits": True, "in_garage": True}))):
        vehicle = dataclasses.replace(base.player.vehicle, completed_laps=laps, last_lap_s=41.0, **fields)
        session = dataclasses.replace(base.session, elapsed_s=float(second))
        tracker.update(dataclasses.replace(base, vehicles=(vehicle,), session=session, flags=flag))
    assert tracker.ended == ()
