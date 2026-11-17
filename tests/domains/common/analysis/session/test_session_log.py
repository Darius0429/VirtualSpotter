import dataclasses

import pytest

from ai_race_engineer.domains.common.analysis.corners.lap_types import LapEnd
from ai_race_engineer.domains.common.analysis.field.field import CarEstimate
from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitStopSample
from ai_race_engineer.domains.common.analysis.session.session_log import SessionLog
from ai_race_engineer.domains.common.analysis.session.track_limits import TrackLimitStatus
from ai_race_engineer.domains.common.data.race_snapshot import CarResources, LapSectors, RaceSnapshot, TrackFlags
from ai_race_engineer.domains.common.data.snapshot import PlayerState, SessionState, SessionType, VehicleState

GREEN: int = 5
YELLOW: int = 6
OVER: int = 8
ESTIMATES: dict[int, CarEstimate] = {0: CarEstimate(0, 100.0, 3.0, 0.04, 0.75, (100.0,))}
LIMITS: TrackLimitStatus = TrackLimitStatus(0.25, 5.0, 0, ())


def _snapshot(
    elapsed_s: float,
    lap_start_s: float = 0.0,
    completed_laps: int = 0,
    virtual_energy: float = 1.0,
    fuel_l: float = 50.0,
    last_lap_s: float | None = None,
    in_pits: bool = False,
    game_phase: int = GREEN,
    sector_flags: tuple[int, int, int] = (11, 11, 11),
    session_type: SessionType = SessionType.RACE,
    track_name: str = "Track",
) -> RaceSnapshot:
    """Race snapshot of the player (slot 0, overall place 3) behind one car of its class and one of another class.
    In:  elapsed_s       float                 s session time
         lap_start_s     float                 s session time the player's lap started
         completed_laps  int                   the player's completed laps
         virtual_energy  float                 fraction on board
         fuel_l          float                 litres on board
         last_lap_s      float | None          s, the game's time of the last lap
         in_pits         bool
         game_phase      int                   mGamePhase
         sector_flags    tuple[int, int, int]  mSectorFlag values
         session_type    SessionType
         track_name      str
    Out:                 RaceSnapshot
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    me = VehicleState(
        **{
            **base,
            "is_player": True,
            "driver_name": "me",
            "vehicle_name": "my car",
            "class_name": "Hyper",
            "place": 3,
            "lap_start_s": lap_start_s,
            "completed_laps": completed_laps,
            "last_lap_s": last_lap_s,
            "in_pits": in_pits,
        }
    )
    others = (
        VehicleState(**{**base, "slot_id": 1, "class_name": "Hyper", "place": 1}),
        VehicleState(**{**base, "slot_id": 2, "class_name": "GT3", "place": 2}),
    )
    session = SessionState(session_type, elapsed_s, 3600.0, 1000, track_name, 5000.0, 20.0, 20.0)
    player = PlayerState(
        me, fuel_l, virtual_energy, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, "Medium", "Soft", (0.0,) * 4
    )
    return RaceSnapshot(
        session,
        player,
        (me, *others),
        (CarResources(0, fuel_l, 100.0, virtual_energy, 0),),
        TrackFlags(game_phase, 0, sector_flags, 0),
        (LapSectors(0, 30.0, 70.0),),
    )


def _log(
    samples: list[PitStopSample] | None = None,
    lap_ends: list[LapEnd] | None = None,
    count_yellow_flags: bool = True,
) -> SessionLog:
    """Session log reading a list of pit samples, a list of lap ends, fixed estimates and fixed track limits.
    In:  samples             list[PitStopSample] | None  the list the pit samples are read from (default: empty)
         lap_ends            list[LapEnd] | None         the list the lap ends are read from (default: empty)
         count_yellow_flags  bool                        the record gives the yellow-flag count
    Out:                     SessionLog
    """
    stops = samples if samples is not None else []
    ends = lap_ends if lap_ends is not None else []
    return SessionLog(
        lambda: tuple(stops),
        lambda: ESTIMATES,
        lambda: (),
        1,
        lambda: 1_000_000.0,
        lambda: tuple(ends),
        lambda: LIMITS,
        count_yellow_flags,
    )


def _two_laps(log: SessionLog) -> None:
    """Stand on the grid, get the green at 10 s (new lap start time, no crossing), drive the lap of the start in
    100 s and an invalidated lap of 104 s.
    In:  log  SessionLog
    Out:      None
    """
    log.update(_snapshot(0.0))
    log.update(_snapshot(10.0, lap_start_s=10.0))
    log.update(_snapshot(110.0, 110.0, 1, 0.955, 47.0, last_lap_s=100.0))
    log.update(_snapshot(214.0, 214.0, 2, 0.912, 44.1, last_lap_s=None))


def test_laps_are_timed_from_the_line_crossings() -> None:
    """Each lap count increase stores a lap with the time between the crossings, also for an invalidated lap; the
    crossing that starts the race stores none."""
    log = _log()
    _two_laps(log)
    first, second = log.record().laps
    assert (first.lap, first.start_s, first.lap_time_s, first.timed) == (1, 10.0, 100.0, True)
    assert (second.lap, second.start_s, second.lap_time_s, second.timed) == (2, 110.0, 104.0, False)


def test_energy_fuel_sectors_place_and_compounds_per_lap() -> None:
    """A lap carries what it used, its sector times, the place in class and the tyre compounds."""
    log = _log()
    _two_laps(log)
    first, second = log.record().laps
    assert (first.energy_used, first.fuel_used_l) == (pytest.approx(0.045), pytest.approx(3.0))
    assert (first.sector1_s, first.sector2_s, first.sector3_s) == (30.0, 40.0, 30.0)
    assert (second.sector1_s, second.sector2_s, second.sector3_s) == (30.0, 40.0, None)
    assert (first.class_place, first.compound_front, first.compound_rear) == (2, "Medium", "Soft")
    assert (first.pitted, first.green, second.green) == (False, False, True)


def test_pit_and_not_green_laps_are_flagged_and_not_clean() -> None:
    """A lap with the pits or with a phase other than green is flagged; only clean laps with energy are returned,
    and a refill gives no usage."""
    log = _log()
    _two_laps(log)
    log.update(_snapshot(260.0, 214.0, 2, 0.89, 43.0, game_phase=YELLOW))
    log.update(_snapshot(330.0, 330.0, 3, 0.87, 41.0, last_lap_s=116.0))
    log.update(_snapshot(400.0, 330.0, 3, 0.84, 40.0, in_pits=True))
    log.update(_snapshot(450.0, 450.0, 4, 1.0, 60.0, last_lap_s=120.0))
    third, fourth = log.record().laps[2:]
    assert (third.green, third.pitted) == (False, False)
    assert (fourth.green, fourth.pitted, fourth.energy_used, fourth.fuel_used_l) == (True, True, None, None)
    assert log.clean_laps() == ((104.0, pytest.approx(0.043)),)


def test_lap_in_progress_at_first_sight_has_no_usage() -> None:
    """Joined mid-lap, the first lap is timed from its start but its usage is unknown."""
    log = _log()
    log.update(_snapshot(500.0, lap_start_s=450.0, completed_laps=4, virtual_energy=0.8))
    log.update(_snapshot(555.0, 555.0, 5, 0.77, last_lap_s=105.0))
    lap = log.record().laps[0]
    assert (lap.lap, lap.lap_time_s, lap.energy_used, lap.green) == (5, 105.0, None, False)
    assert log.clean_laps() == ()


def test_yellow_flags_are_counted_when_they_turn_yellow() -> None:
    """Each sector flag turning yellow counts once, however long it stays."""
    log = _log()
    log.update(_snapshot(0.0))
    log.update(_snapshot(1.0, sector_flags=(1, 11, 11)))
    log.update(_snapshot(2.0, sector_flags=(1, 1, 11)))
    log.update(_snapshot(3.0, sector_flags=(11, 1, 11)))
    log.update(_snapshot(4.0, sector_flags=(1, 1, 11)))
    assert log.record().yellow_flags == 3


def test_yellow_flag_count_is_none_when_switched_off() -> None:
    """A domain that does not count yellow flags gives none in the record."""
    log = _log(count_yellow_flags=False)
    log.update(_snapshot(0.0))
    log.update(_snapshot(1.0, sector_flags=(1, 11, 11)))
    assert log.record().yellow_flags is None


def test_lap_type_comes_from_the_lap_ends() -> None:
    """A lap is named as the player's lap end at the crossing names it; another car's lap end does not count, and
    without one a timed lap is a lap and an untimed one has no type."""
    ends: list[LapEnd] = []
    log = _log(lap_ends=ends)
    log.update(_snapshot(0.0))
    crossing = _snapshot(100.0, 100.0, 1)
    ends[:] = [LapEnd(crossing.vehicles[1], 1, "lap", 99.0), LapEnd(crossing.player.vehicle, 1, "out lap", 100.0)]
    log.update(crossing)
    crossing = _snapshot(204.0, 204.0, 2)
    ends[:] = [LapEnd(crossing.player.vehicle, 2, "invalid lap", 104.0)]
    log.update(crossing)
    ends[:] = []
    log.update(_snapshot(304.0, 304.0, 3, last_lap_s=100.0))
    log.update(_snapshot(404.0, 404.0, 4))
    assert [lap.lap_type for lap in log.record().laps] == ["out lap", "invalid lap", "lap", None]


def test_crossing_in_the_pit_lane_gives_an_in_lap() -> None:
    """A lap that ends at the line in the pit lane has no lap end there and is an in lap."""
    log = _log()
    log.update(_snapshot(0.0))
    log.update(_snapshot(100.0, 100.0, 1, last_lap_s=100.0, in_pits=True))
    assert log.record().laps[0].lap_type == "in lap"


def test_record_carries_the_track_limits() -> None:
    """The record gives the track limits and penalties as the source has them."""
    log = _log()
    log.update(_snapshot(0.0))
    assert log.record().limits == LIMITS


def test_pit_stop_gives_an_automatic_note() -> None:
    """A new pit sample of the player adds a note; other cars' samples do not."""
    samples: list[PitStopSample] = []
    log = _log(samples)
    log.update(_snapshot(0.0))
    samples.append(PitStopSample(5, False, 30.0, 400.0, 20.0, 0.5, 30.0, None))
    log.update(_snapshot(100.0, completed_laps=0))
    samples.append(PitStopSample(0, True, 30.0, 400.0, 22.6, 0.53, 36.5, None))
    log.update(_snapshot(2600.0, lap_start_s=2500.0, completed_laps=22))
    log.update(_snapshot(2601.0, lap_start_s=2500.0, completed_laps=22))
    (note,) = log.record().notes
    assert (note.time_s, note.lap, note.auto) == (2600.0, 22, True)
    assert note.text == "Pit stop lap 22: 22.6 s stationary, plus 53 percent energy, plus 36.5 litres"


def test_add_note_is_stamped_with_time_and_lap() -> None:
    """A dictated note carries the session time and the lap in progress."""
    log = _log()
    _two_laps(log)
    note = log.add_note("understeer in turn 4")
    assert (note.time_s, note.lap, note.text, note.auto) == (214.0, 3, "understeer in turn 4", False)
    assert log.record().notes == (note,)


def test_record_of_the_running_session() -> None:
    """The record names the session, the player and the grid place and carries the sources' values."""
    log = _log()
    assert log.record() is None
    _two_laps(log)
    record = log.record()
    assert record.session_id.endswith("_race")
    assert record.started_unix == 1_000_000.0
    assert (record.track, record.kind, record.ended, record.time_s) == ("Track", "race", False, 214.0)
    assert (record.player_name, record.vehicle_name, record.class_name) == ("me", "my car", "Hyper")
    assert (record.grid_class_place, len(record.vehicles)) == (2, 3)
    assert (record.estimates, record.pit_samples, record.straights) == (ESTIMATES, (), ())


@pytest.mark.parametrize(
    "new",
    [
        _snapshot(5.0),
        _snapshot(300.0, session_type=SessionType.QUALIFY),
        _snapshot(300.0, track_name="Other"),
    ],
)
def test_new_session_puts_the_old_record_on_ended(new: RaceSnapshot) -> None:
    """The clock going back, another session type or another track ends the session: its record is queued and a
    new one starts empty."""
    log = _log()
    _two_laps(log)
    log.add_note("note")
    log.update(new)
    ended = log.ended.get_nowait()
    assert (ended.ended, len(ended.laps), len(ended.notes), ended.time_s) == (True, 2, 1, 214.0)
    assert log.ended.empty()
    assert (log.record().laps, log.record().notes, log.record().yellow_flags) == ((), (), 0)


def test_finished_is_false_until_the_finish_is_seen() -> None:
    """A session that is still running, or was left before its end, is not finished."""
    log = _log()
    _two_laps(log)
    assert log.record().finished is False


def test_finished_after_the_game_phase_is_over() -> None:
    """Once the game phase was over the session stays finished."""
    log = _log()
    _two_laps(log)
    log.update(_snapshot(300.0, 214.0, 2, game_phase=OVER))
    log.update(_snapshot(301.0, 214.0, 2))
    assert log.record().finished is True


def test_finished_after_the_player_took_the_flag() -> None:
    """The player's own finish status counts while the game phase is still green for the others."""
    log = _log()
    _two_laps(log)
    snapshot = _snapshot(300.0, 214.0, 2)
    log.update(dataclasses.replace(snapshot, resources=(CarResources(0, 40.0, 100.0, 0.9, 2),)))
    assert log.record().finished is False
    log.update(dataclasses.replace(snapshot, resources=(CarResources(0, 40.0, 100.0, 0.9, 1),)))
    assert log.record().finished is True


def test_finished_is_reset_by_a_new_session() -> None:
    """The ended record keeps the flag; the new session starts unfinished."""
    log = _log()
    _two_laps(log)
    log.update(_snapshot(300.0, 214.0, 2, game_phase=OVER))
    log.update(_snapshot(5.0))
    assert log.ended.get_nowait().finished is True
    assert log.record().finished is False
