import dataclasses
from collections.abc import Callable, Sequence

import pytest

from ai_race_engineer.events.timeline_builder import TimelineBuilder
from ai_race_engineer.strategy.pit_time import PitLane, PitModel, PitRates, PitStopSample
from ai_race_engineer.strategy.race_finish import CarInput, project_race
from ai_race_engineer.strategy.race_state import RaceState
from ai_race_engineer.strategy.rivals import Rival
from ai_race_engineer.strategy.timeline import TimelineEntry
from ai_race_engineer.telemetry.race_snapshot import CarResources, RaceSnapshot, TrackFlags
from ai_race_engineer.telemetry.snapshot import PlayerState, SessionState, SessionType, VehicleState

PIT: PitModel = PitModel(PitRates(3.0, 0.025, 1.0, base_s=1.0), PitLane(30.0, 500.0), 5000.0)
GRID: int = 4
GREEN: int = 5
FCY: int = 6
CLEAR: tuple[int, int, int] = (11, 11, 11)
PLACES: dict[int, int] = {1: 1, 0: 2, 2: 3, 3: 4}


class _Race:
    """A timeline builder fed snapshots of the player (slot 0) and three cars of its class, driver1 to driver3.
    builder  TimelineBuilder              3 s clear hold, 5 s position hold, 0.1 s best-lap gain, 1.5 % energy change,
                                          groups ahead/behind/strategy, 2 lap checks place hold, 30 s rival hold,
                                          1 litre fuel change
    _make    Callable[..., VehicleState]  vehicle factory
    """

    def __init__(self, make_vehicle: Callable[..., VehicleState]) -> None:
        """Build the builder.
        In:  make_vehicle  Callable[..., VehicleState]
        Out:               None
        """
        self.builder: TimelineBuilder = TimelineBuilder(
            1, [1, 2, 3], 3.0, 5.0, 0.1, 0.015, ["ahead", "behind", "strategy"], 2, 30.0, 1.0
        )
        self._make: Callable[..., VehicleState] = make_vehicle

    def step(
        self,
        time_s: float,
        places: dict[int, int] | None = None,
        phase: int = GREEN,
        sector_flags: tuple[int, int, int] = CLEAR,
        session_type: SessionType = SessionType.RACE,
        laps: int = 0,
        in_pits: Sequence[int] = (),
        in_garage: Sequence[int] = (),
        best_s: dict[int, float] | None = None,
        telemetry: Sequence[int] = (0, 1),
        state: RaceState | None = None,
        samples: Sequence[PitStopSample] = (),
        pre_race: bool = False,
        finished: bool = False,
        finish_status: int = 0,
        plan_recorded: bool | None = False,
        rivals_recorded: bool | None = False,
    ) -> list[TimelineEntry]:
        """Feed one snapshot.
        In:  time_s        float                    s session time
             places        dict[int, int] | None    slot id -> overall place (default: driver1, player, driver2,
                                                    driver3)
             phase         int                      mGamePhase
             sector_flags  tuple[int, int, int]     mSectorFlag values
             session_type  SessionType
             laps          int                      the player's completed laps
             in_pits       Sequence[int]            slot ids in the pits
             in_garage     Sequence[int]            slot ids in the garage
             best_s        dict[int, float] | None  slot id -> s best lap
             telemetry     Sequence[int]            slot ids with resources
             state         RaceState | None
             samples       Sequence[PitStopSample]
             pre_race      bool
             finished      bool
             finish_status  int                     the player's finish status (1 = finished)
             plan_recorded  bool | None             the timeline already has a plan entry; None = not known yet
             rivals_recorded  bool | None           the timeline already has a rival entry; None = not known yet
        Out:               list[TimelineEntry]
        """
        vehicles = tuple(
            self._make(
                slot_id=slot_id,
                is_player=slot_id == 0,
                class_name="Hyper",
                place=place,
                completed_laps=laps if slot_id == 0 else 0,
                sector_index=1,
                in_pits=slot_id in in_pits or slot_id in in_garage,
                in_garage=slot_id in in_garage,
                best_lap_s=(best_s or {}).get(slot_id),
            )
            for slot_id, place in (places or PLACES).items()
        )
        me = next(v for v in vehicles if v.is_player)
        session = SessionState(session_type, time_s, 3600.0, 1000, "Track", 5000.0, 20.0, 20.0)
        player = PlayerState(me, 50.0, 0.5, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, "", "", (0.0,) * 4)
        snapshot = RaceSnapshot(
            session,
            player,
            vehicles,
            tuple(CarResources(s, 50.0, 100.0, 0.5, finish_status if s == 0 else 0) for s in telemetry),
            TrackFlags(phase, 0, sector_flags, 0),
        )
        entries = self.builder.update(snapshot, state, samples, pre_race, finished, plan_recorded, rivals_recorded)
        return list(entries)


@pytest.fixture
def race(make_vehicle: Callable[..., VehicleState]) -> _Race:
    """A builder that has seen the first snapshot of a race under green.
    In:  make_vehicle  Callable[..., VehicleState]
    Out:               _Race
    """
    race = _Race(make_vehicle)
    race.step(0.0)
    race.step(1.0)
    return race


def _state(
    time_s: float,
    rivals: Sequence[Rival] = (),
    energy_per_lap: float | None = 0.045,
    remaining_s: float = 3400.0,
    before_green: bool = False,
    fuel_l_per_lap: float | None = None,
) -> RaceState:
    """Race state of the player (slot 0) between driver1 ahead and driver2, driver3 behind, each 0.1 lap apart.
    In:  time_s          float            s session time
         rivals          Sequence[Rival]
         energy_per_lap  float | None     fraction/lap of every car; None = usage unknown
         remaining_s     float            s to the end of the race
         before_green    bool
         fuel_l_per_lap  float | None     litres/lap of every car; given = cars without virtual energy, 60 of 100
                                          litres on board
    Out:                 RaceState
    """
    energy = None if fuel_l_per_lap is not None else 0.9
    progress = {1: 1.4, 0: 1.3, 2: 1.2, 3: 1.1}
    cars = tuple(
        CarInput(slot_id, "Hyper", laps, 100.0, False, False, 60.0, 100.0, energy, fuel_l_per_lap, energy_per_lap, 0.7)
        for slot_id, laps in progress.items()
    )
    return RaceState(
        time_s=time_s,
        player_slot_id=0,
        vehicles=(),
        cars=cars,
        estimates={},
        sectors={},
        pit=PIT,
        estimate=None,
        set_fuel_ratio=None,
        projection=project_race(cars, remaining_s, None, PIT),
        rivals=tuple(rivals),
        traffic_window_s=3.0,
        before_green=before_green,
    )


def _projected(state: RaceState, **changes: object) -> RaceState:
    """A race state with fields of the player's projection replaced.
    In:  state    RaceState
         changes  object     CarProjection field values
    Out:          RaceState
    """
    cars = {**state.projection.cars, 0: dataclasses.replace(state.projection.cars[0], **changes)}
    return dataclasses.replace(state, projection=dataclasses.replace(state.projection, cars=cars))


def _rival(slot_id: int, groups: tuple[str, ...], class_place: int, gap_s: float) -> Rival:
    """Rival of the player.
    In:  slot_id      int
         groups       tuple[str, ...]
         class_place  int
         gap_s        float  s, positive = ahead
    Out:              Rival
    """
    return Rival(
        slot_id,
        f"driver{slot_id}",
        groups,
        class_place,
        gap_s,
        0.0,
        False,
        0,
        100.0,
        None,
        0.0,
        class_place,
        gap_s,
        0,
        None,
        None,
    )


def _events(entries: Sequence[TimelineEntry]) -> list[tuple[str, object]]:
    """Entries as (kind, event) pairs.
    In:  entries  Sequence[TimelineEntry]
    Out:          list[tuple[str, object]]
    """
    return [(e.kind, e.fields["event"]) for e in entries]


def test_start_entry_on_the_first_snapshot_and_after_the_clock_goes_back(
    make_vehicle: Callable[..., VehicleState],
) -> None:
    """The first snapshot of a session gives the start entry; so does a session clock going backwards, and a
    change of session type."""
    race = _Race(make_vehicle)
    (start,) = race.step(100.0, laps=2)
    assert (start.time_s, start.lap, start.kind) == (100.0, 3, "session")
    assert start.text == "Engineer watching: race at Track, lap 3, P2 in class"
    assert start.fields == {"event": "start", "kind": "race", "track": "Track", "class_place": 2}
    assert race.step(101.0, laps=2) == []
    assert _events(race.step(5.0)) == [("session", "start")]
    (qualifying,) = race.step(6.0, session_type=SessionType.QUALIFY)
    assert qualifying.text == "Engineer watching: qualify at Track, lap 1, P2 in class"


def test_green_only_from_a_pre_green_phase(make_vehicle: Callable[..., VehicleState]) -> None:
    """The green flag of the race start is recorded once, with the place; the green after a full course yellow is
    no race start."""
    race = _Race(make_vehicle)
    race.step(0.0, phase=GRID)
    assert race.step(1.0, phase=GRID) == []
    (green,) = race.step(2.0)
    assert (green.text, green.fields) == (
        "Green flag, race started from P2 in class",
        {"event": "green", "class_place": 2},
    )
    race.step(3.0, phase=FCY)
    assert ("session", "green") not in _events(race.step(60.0))
    race.step(61.0, phase=GRID)
    assert race.step(62.0) == []


def test_finish_is_recorded_once(race: _Race) -> None:
    """The finish gives one entry with the place and the laps, however long the race stays finished."""
    (finish,) = race.step(3500.0, laps=34, finished=True, finish_status=1)
    assert finish.text == "Race finished P2 in class after 34 laps"
    assert finish.fields == {"event": "finish", "class_place": 2, "laps": 34}
    assert finish.lap == 34
    assert race.step(3501.0, laps=34, finished=True, finish_status=1) == []
    assert race.step(3502.0, laps=34, finished=False) == []
    assert race.step(3503.0, laps=34, finished=True, finish_status=1) == []


def test_finish_waits_for_the_players_own_flag(race: _Race) -> None:
    """The leader's flag ends the race for the tracker while the player still drives the lap: the finish is recorded
    when the player's own finish status says so, with the laps then completed."""
    assert race.step(3500.0, laps=32, finished=True) == []
    (finish,) = race.step(3530.0, laps=33, finished=True, finish_status=1)
    assert finish.text == "Race finished P2 in class after 33 laps"


def test_finish_of_a_player_without_resources_follows_the_tracker(race: _Race) -> None:
    """Without resources of the player the tracker's finish is used."""
    assert race.step(3500.0, laps=33, telemetry=(1,)) == []
    assert _events(race.step(3501.0, laps=33, telemetry=(1,), finished=True)) == [("session", "finish")]


def test_yellow_raised_once_and_cleared_after_the_hold(race: _Race) -> None:
    """A yellow is recorded once; a flicker inside the hold gives no second entry; the clear comes after the hold
    with the time the sector was yellow."""
    (yellow,) = race.step(10.0, sector_flags=(11, 1, 11))
    assert (yellow.text, yellow.fields) == (
        "Yellow flag in sector 2",
        {"event": "yellow", "sector": 2, "player_sector": 2},
    )
    assert race.step(11.0, sector_flags=(11, 1, 11)) == []
    assert race.step(20.0) == []
    assert race.step(21.0, sector_flags=(11, 1, 11)) == []
    assert race.step(24.0) == []
    assert race.step(26.0) == []
    (clear,) = race.step(27.0)
    assert (clear.text, clear.fields) == (
        "Sector 2 clear after 14 seconds",
        {"event": "clear", "sector": 2, "duration_s": 14.0},
    )
    assert race.step(40.0) == []


def test_sector_flags_are_silent_outside_green(race: _Race) -> None:
    """Under a full course yellow sector flags give no entries, and a value other than clear ends a yellow
    silently."""
    assert _events(race.step(10.0, phase=FCY, sector_flags=(1, 11, 11))) == [("flag", "fcy_start")]
    assert race.step(11.0, phase=FCY) == []
    assert race.step(12.0, phase=FCY, sector_flags=(1, 11, 11)) == []
    assert _events(race.step(13.0, sector_flags=(1, 11, 11))) == [("flag", "fcy_end")]
    assert race.step(14.0, sector_flags=(0, 11, 11)) == []
    assert race.step(20.0) == []


def test_full_course_yellow_start_and_end_with_duration(race: _Race) -> None:
    """The full course yellow is recorded when it starts and when the green returns, with how long it lasted."""
    (start,) = race.step(100.0, phase=FCY)
    assert (start.text, start.fields) == ("Full course yellow", {"event": "fcy_start"})
    assert race.step(150.0, phase=FCY) == []
    (end,) = race.step(195.0)
    assert end.text == "Green flag after 95 seconds of full course yellow"
    assert end.fields == {"event": "fcy_end", "duration_s": 95.0}


def test_pass_on_track_is_recorded_after_the_hold(race: _Race) -> None:
    """A car that changed side is recorded once the new order has held; until then nothing."""
    passed = {0: 1, 1: 2, 2: 3, 3: 4}
    assert race.step(10.0, places=passed) == []
    assert race.step(14.0, places=passed) == []
    (entry,) = race.step(15.0, places=passed)
    assert entry.text == "Passed driver1 for P1 in class, on track"
    assert entry.fields == {"event": "swap", "driver": "driver1", "gained": True, "class_place": 1, "via": "track"}
    assert race.step(30.0, places=passed) == []


def test_place_lost_during_the_players_stop_and_gained_at_theirs(race: _Race) -> None:
    """How a swap happened is taken at its first snapshot: the player in the pits, else the other car there."""
    lost = {1: 1, 2: 2, 0: 3, 3: 4}
    race.step(10.0, places=lost, in_pits=(0,))
    (entry,) = [e for e in race.step(15.0, places=lost) if e.kind == "position"]
    assert entry.text == "Lost P2 to driver2, now P3 in class, during your pit stop"
    assert (entry.fields["gained"], entry.fields["class_place"], entry.fields["via"]) == (False, 3, "your_stop")
    race.step(100.0, places=PLACES, in_pits=(2,))
    (entry,) = race.step(105.0, places=PLACES, in_pits=(2,))
    assert entry.text == "Passed driver2 for P2 in class, while driver2 was in the pits"
    assert entry.fields["via"] == "their_stop"


def test_swap_that_flips_back_gives_nothing(race: _Race) -> None:
    """A swap undone within the hold is not recorded, and the hold starts again with the next swap."""
    passed = {0: 1, 1: 2, 2: 3, 3: 4}
    assert race.step(10.0, places=passed) == []
    assert race.step(13.0) == []
    assert race.step(16.0, places=passed) == []
    assert race.step(20.0, places=passed) == []
    assert _events(race.step(21.0, places=passed)) == [("position", "swap")]


def test_place_changed_without_a_swap_has_no_cause(race: _Race) -> None:
    """A car ahead leaving the session moves the player up without a car changing side."""
    (entry,) = race.step(10.0, places={0: 2, 2: 3, 3: 4})
    assert entry.text == "Now P1 in class"
    assert entry.fields == {"event": "place", "driver": None, "gained": True, "class_place": 1, "via": None}


def test_no_position_entries_before_green_or_outside_a_race(make_vehicle: Callable[..., VehicleState]) -> None:
    """On the grid and in a qualifying session the order is followed without entries."""
    passed = {0: 1, 1: 2, 2: 3, 3: 4}
    race = _Race(make_vehicle)
    race.step(0.0, phase=GRID)
    assert race.step(1.0, places=passed, phase=GRID) == []
    assert race.step(10.0, places=passed, phase=GRID) == []
    qualifying = _Race(make_vehicle)
    qualifying.step(0.0, session_type=SessionType.QUALIFY)
    assert qualifying.step(1.0, places=passed, session_type=SessionType.QUALIFY) == []
    assert qualifying.step(10.0, places=passed, session_type=SessionType.QUALIFY) == []


def test_pit_in_with_prediction_for_a_rival_and_the_player(race: _Race) -> None:
    """A rival of a recorded group and the player get a pit-in entry with the predicted loss and rejoin place."""
    state = _state(1.0, [_rival(1, ("ahead",), 1, 10.0), _rival(3, ("pace",), 4, -20.0)])
    race.step(2.0, state=state)
    (rival,) = race.step(3.0, state=state, in_pits=(1,))
    assert rival.text == "driver1 pits from P1, expected time loss 25 seconds, rejoin P3 in class"
    assert rival.fields == {
        "event": "in",
        "driver": "driver1",
        "is_player": False,
        "class_place": 1,
        "predicted_stop_s": pytest.approx(25.0),
        "predicted_class_place": 3,
    }
    (player,) = race.step(4.0, state=state, in_pits=(1, 0))
    assert player.text == "You pit from P2, expected time loss 25 seconds, rejoin P4 in class"
    assert (player.fields["driver"], player.fields["is_player"]) == ("driver0", True)


def test_no_pit_in_for_a_non_rival_or_into_the_garage(race: _Race) -> None:
    """A car outside the recorded groups gives no entry; neither does a car that is in the garage. Without a race
    state the player's pit-in has no prediction."""
    state = _state(1.0, [_rival(3, ("pace",), 4, -20.0)])
    race.step(2.0, state=state)
    assert race.step(3.0, state=state, in_pits=(2, 3)) == []
    assert race.step(4.0, state=state, in_pits=(2, 3), in_garage=(0,)) == []
    race.step(5.0, state=state)
    (player,) = race.step(6.0, in_pits=(0,))
    assert player.text == "You pit from P2"
    assert (player.fields["predicted_stop_s"], player.fields["predicted_class_place"]) == (None, None)


def test_pit_out_pairs_the_sample_and_the_prediction(race: _Race) -> None:
    """The pit-out carries the measured pass of that stop, the place at the exit and the place predicted at the
    pit-in; an older pass of the same car is not used."""
    state = _state(1.0, [_rival(1, ("ahead",), 1, 10.0)])
    old = PitStopSample(1, False, 30.0, 500.0, 40.0, 0.2, 14.0, None)
    race.step(2.0, state=state, samples=(old,))
    race.step(3.0, state=state, in_pits=(1,), samples=(old,))
    new = PitStopSample(1, False, 30.0, 500.0, 31.2, 0.58, 40.6, None)
    after = {0: 1, 2: 2, 3: 3, 1: 4}
    out = [e for e in race.step(70.0, places=after, state=state, samples=(old, new)) if e.kind == "pit"]
    assert out[0].text == (
        "driver1 out after 31.2 seconds stationary, plus 58 percent energy, rejoined P4 in class, predicted P3"
    )
    assert out[0].fields == {
        "event": "out",
        "driver": "driver1",
        "is_player": False,
        "stationary_s": 31.2,
        "energy_added_percent": pytest.approx(58.0),
        "fuel_added_l": 40.6,
        "class_place": 4,
        "predicted_class_place": 3,
    }
    assert [e for e in race.step(71.0, places=after, state=state, samples=(old, new)) if e.kind == "pit"] == []


def test_pit_out_without_a_sample_or_without_resources(race: _Race) -> None:
    """A dropped pass gives no figures; a car without telemetry has no energy or fuel figure."""
    state = _state(1.0, [_rival(1, ("ahead",), 1, 10.0), _rival(2, ("behind",), 3, -10.0)])
    race.step(2.0, state=state)
    race.step(3.0, state=state, in_pits=(1, 2))
    sample = PitStopSample(2, False, 30.0, 500.0, 25.0, 0.0, 0.0, None)
    dropped, blind = race.step(60.0, state=state, samples=(sample,))
    assert dropped.text == "driver1 out of the pits, rejoined P1 in class, predicted P3"
    assert (dropped.fields["stationary_s"], dropped.fields["energy_added_percent"]) == (None, None)
    assert blind.text == "driver2 out after 25.0 seconds stationary, rejoined P3 in class, predicted P4"
    assert (blind.fields["stationary_s"], blind.fields["energy_added_percent"], blind.fields["fuel_added_l"]) == (
        25.0,
        None,
        None,
    )


def test_class_best_and_personal_best_with_the_gain_threshold(race: _Race) -> None:
    """A fastest lap of the class and a personal best are recorded when they improve by the threshold; the player
    setting the fastest lap of the class gives one entry."""
    (first,) = race.step(110.0, best_s={1: 107.9})
    assert first.text == "Fastest lap in class: driver1, 1:47.9"
    assert first.fields == {"event": "class_best", "driver": "driver1", "lap_s": 107.9, "is_player": False}
    (personal,) = race.step(112.0, best_s={1: 107.9, 0: 108.2})
    assert personal.text == "Personal best, 1:48.2"
    assert personal.fields == {"event": "personal_best", "driver": "driver0", "lap_s": 108.2, "is_player": True}
    assert race.step(220.0, best_s={1: 107.85, 0: 108.15}) == []
    (mine,) = race.step(330.0, best_s={1: 107.85, 0: 107.5})
    assert (mine.text, mine.fields["event"]) == ("Fastest lap in class: driver0, 1:47.5", "class_best")
    assert mine.fields["is_player"] is True
    assert race.step(331.0, best_s={1: 107.85, 0: 107.5}) == []


def test_best_laps_at_the_first_snapshot_are_not_recorded(make_vehicle: Callable[..., VehicleState]) -> None:
    """The best laps a session is joined with are taken as they are."""
    race = _Race(make_vehicle)
    race.step(500.0, best_s={1: 107.9, 0: 108.2})
    assert race.step(501.0, best_s={1: 107.9, 0: 108.2}) == []


def test_rival_set_at_the_first_settled_state_then_new_and_gone(race: _Race) -> None:
    """Before the green and on the pre-race figures no rivals are recorded; the first settled state gives the set,
    later states the cars entering and leaving it. A car that only changes group gives nothing."""
    ahead, behind = _rival(1, ("ahead", "pace"), 1, 10.0), _rival(2, ("behind",), 3, -8.0)
    grid = race.step(2.0, state=_state(2.0, [ahead, behind], before_green=True), pre_race=True)
    assert _events(grid) == [("plan", "initial")]
    assert race.step(3.0, state=_state(3.0, [ahead, behind]), pre_race=True) == []
    rivals, _ = race.step(4.0, state=_state(4.0, [ahead, behind]))
    assert rivals.text == "Rivals: driver1 ahead, driver2 behind"
    assert rivals.fields == {"event": "set", "drivers": "driver1, driver2"}
    regrouped = _rival(2, ("behind", "strategy"), 3, -8.0)
    assert race.step(5.0, state=_state(5.0, [ahead, regrouped])) == []
    strategy = _rival(3, ("strategy",), 4, -12.3)
    changed = [ahead, strategy, _rival(2, ("pace",), 3, -8.0)]
    assert race.step(6.0, state=_state(6.0, changed)) == []
    assert race.step(35.0, state=_state(35.0, changed)) == []
    new, gone = sorted(race.step(36.0, state=_state(36.0, changed)), key=lambda e: e.text, reverse=True)
    assert new.text == "driver3 became a rival (strategy), P4 in class, 12.3 seconds behind"
    assert new.fields == {"event": "new", "driver": "driver3", "groups": "strategy", "class_place": 4, "gap_s": -12.3}
    assert (gone.text, gone.fields) == ("driver2 is no longer a rival", {"event": "gone", "driver": "driver2"})
    assert race.step(37.0, state=_state(37.0, changed)) == []


def test_rival_that_leaves_and_returns_within_the_hold_gives_nothing(race: _Race) -> None:
    """A car out of the set, or in it, for less than the hold is not recorded, and the hold starts again."""
    ahead, behind, strategy = (
        _rival(1, ("ahead",), 1, 10.0),
        _rival(2, ("behind",), 3, -8.0),
        _rival(3, ("strategy",), 4, -12.3),
    )
    race.step(2.0, state=_state(2.0, [ahead, behind]))
    assert race.step(10.0, state=_state(10.0, [ahead, strategy])) == []
    assert race.step(30.0, state=_state(30.0, [ahead, behind])) == []
    assert race.step(45.0, state=_state(45.0, [ahead, behind])) == []
    assert race.step(50.0, state=_state(50.0, [ahead])) == []
    assert race.step(79.0, state=_state(79.0, [ahead])) == []
    assert _events(race.step(80.0, state=_state(80.0, [ahead]))) == [("rival", "gone")]


def test_rivals_and_plan_only_on_a_new_state(race: _Race) -> None:
    """The same state seen again gives no entry."""
    state = _state(2.0, [_rival(1, ("ahead",), 1, 10.0)])
    assert _events(race.step(2.0, state=state)) == [("rival", "set"), ("plan", "initial")]
    assert race.step(3.0, state=state, laps=1) == []


def test_plan_initial_and_no_entry_while_nothing_changes(race: _Race) -> None:
    """The first race state gives the plan with its basis; the same plan after a completed lap gives nothing, and a
    different plan is not compared before the player completes a lap."""
    (plan,) = [e for e in race.step(2.0, state=_state(2.0), pre_race=True) if e.kind == "plan"]
    assert (
        plan.text == "Plan (pre-race figures): 35 laps, 1 stop, pit end of lap 21, add 62 percent, finish P2 in class"
    )
    assert plan.fields == {
        "event": "initial",
        "basis": "pre_race",
        "race_laps": 35,
        "stops": 1,
        "pit_lap": 21,
        "first_stop_add_percent": pytest.approx(61.65),
        "projected_class_place": 2,
    }
    assert race.step(3.0, state=_state(3.0, energy_per_lap=0.05), pre_race=True) == []
    assert race.step(110.0, state=_state(110.0), pre_race=True, laps=1) == []


def test_plan_change_names_what_changed(race: _Race) -> None:
    """After a completed lap a different pit lap and first-stop amount give a change entry with the old values."""
    race.step(2.0, state=_state(2.0), pre_race=True)
    (change,) = race.step(110.0, state=_state(110.0, energy_per_lap=0.05), pre_race=True, laps=1)
    assert change.text == "Plan change: pit end of lap 19 (was 21), add 79 percent (was 62 percent)"
    assert (change.fields["event"], change.fields["changed"]) == ("change", "pit_lap, first_stop_add")
    assert (change.fields["pit_lap"], change.fields["first_stop_add_percent"]) == (19, pytest.approx(78.5))
    assert race.step(220.0, state=_state(220.0, energy_per_lap=0.05), pre_race=True, laps=2) == []


def test_plan_energy_threshold(race: _Race) -> None:
    """A first-stop amount that moved by less than the threshold is no change; the drift is measured against the
    last recorded plan."""
    race.step(2.0, state=_state(2.0), pre_race=True)
    assert race.step(110.0, state=_state(110.0, energy_per_lap=0.0453), pre_race=True, laps=1) == []
    (change,) = race.step(220.0, state=_state(220.0, energy_per_lap=0.0455), pre_race=True, laps=2)
    assert change.fields["changed"] == "first_stop_add"
    assert change.text == "Plan change: add 63 percent (was 62 percent)"


def test_plan_basis_switch_gives_the_full_plan(race: _Race) -> None:
    """When the measured laps replace the pre-race figures the whole plan is recorded again."""
    race.step(2.0, state=_state(2.0), pre_race=True)
    (basis,) = [e for e in race.step(3.0, state=_state(3.0)) if e.kind == "plan"]
    assert (
        basis.text
        == "Plan now from measured laps: 35 laps, 1 stop, pit end of lap 21, add 62 percent, finish P2 in class"
    )
    assert (basis.fields["event"], basis.fields["basis"]) == ("basis", "measured")
    assert [e for e in race.step(4.0, state=_state(4.0)) if e.kind == "plan"] == []


def test_plan_change_to_no_stop_names_the_stop_only(race: _Race) -> None:
    """When the stop is gone the text says so once; the fields name everything that changed."""
    race.step(2.0, state=_state(2.0), pre_race=True)
    (change,) = race.step(110.0, state=_state(110.0, remaining_s=1500.0), pre_race=True, laps=1)
    assert change.text == "Plan change: 17 laps (was 35 laps), no stop (was 1 stop)"
    assert change.fields["changed"] == "race_laps, stops, pit_lap, first_stop_add"
    assert (change.fields["pit_lap"], change.fields["first_stop_add_percent"]) == (None, 0.0)


def test_initial_plan_waits_for_a_usage_figure_or_the_green(make_vehicle: Callable[..., VehicleState]) -> None:
    """On the grid the plan is not recorded while the player's usage is unknown; it comes with the first usage
    figure, or at the green without one."""
    race = _Race(make_vehicle)
    race.step(0.0, phase=GRID)
    unknown = _state(1.0, energy_per_lap=None, before_green=True)
    assert race.step(1.0, phase=GRID, state=unknown, pre_race=True) == []
    known = _state(2.0, before_green=True)
    assert _events(race.step(2.0, phase=GRID, state=known, pre_race=True)) == [("plan", "initial")]
    late = _Race(make_vehicle)
    late.step(0.0, phase=GRID)
    assert late.step(1.0, phase=GRID, state=unknown, pre_race=True) == []
    (plan,) = [e for e in late.step(2.0, state=_state(2.0, energy_per_lap=None), pre_race=True) if e.kind == "plan"]
    assert plan.text == "Plan (pre-race figures): 36 laps, no stop, finish P2 in class"


def test_projected_place_change_must_hold_for_two_lap_checks(race: _Race) -> None:
    """A projected place that differs at one lap check only is no change; the same new place at two checks in a row
    is recorded once."""
    race.step(2.0, state=_state(2.0), pre_race=True)
    assert race.step(110.0, state=_projected(_state(110.0), class_place=4), pre_race=True, laps=1) == []
    assert race.step(220.0, state=_state(220.0), pre_race=True, laps=2) == []
    assert race.step(330.0, state=_projected(_state(330.0), class_place=3), pre_race=True, laps=3) == []
    assert race.step(440.0, state=_projected(_state(440.0), class_place=4), pre_race=True, laps=4) == []
    (change,) = race.step(550.0, state=_projected(_state(550.0), class_place=4), pre_race=True, laps=5)
    assert change.text == "Plan change: finish P4 in class (was P2)"
    assert (change.fields["changed"], change.fields["projected_class_place"]) == ("projected_class_place", 4)
    assert race.step(660.0, state=_projected(_state(660.0), class_place=4), pre_race=True, laps=6) == []


def test_other_plan_changes_stay_immediate_while_a_place_is_held_back(race: _Race) -> None:
    """A pit lap change is recorded at once and keeps the recorded place while the new place has not held."""
    race.step(2.0, state=_state(2.0), pre_race=True)
    moved = _projected(_state(110.0, energy_per_lap=0.05), class_place=4)
    (change,) = race.step(110.0, state=moved, pre_race=True, laps=1)
    assert (change.fields["changed"], change.fields["projected_class_place"]) == ("pit_lap, first_stop_add", 2)


def test_own_completed_stop_gives_no_plan_change(race: _Race) -> None:
    """The stop fewer, the pit lap and the amount that follow from the player's own stop are taken over without an
    entry, once the player is out of the pits; another change at that check is still recorded."""
    stop = PitStopSample(0, True, 30.0, 500.0, 22.0, 0.5, 35.0, None)
    race.step(2.0, state=_state(2.0), pre_race=True)
    stopped = _projected(_state(110.0), stops_remaining=0, pit_lap=None)
    in_lane = race.step(110.0, state=stopped, pre_race=True, laps=1, in_pits=(0,))
    assert _events(in_lane) == [("pit", "in")]
    stopped = _projected(_state(130.0), stops_remaining=0, pit_lap=None)
    assert _events(race.step(130.0, state=stopped, pre_race=True, laps=1, samples=(stop,))) == [("pit", "out")]
    stopped = _projected(_state(240.0), stops_remaining=0, pit_lap=None)
    assert race.step(240.0, state=stopped, pre_race=True, laps=2, samples=(stop,)) == []


def test_another_change_at_the_check_after_a_stop_is_recorded(race: _Race) -> None:
    """A shorter race seen at the lap check after the player's stop is a change; the stop itself is not named."""
    stop = PitStopSample(0, True, 30.0, 500.0, 22.0, 0.5, 35.0, None)
    race.step(2.0, state=_state(2.0), pre_race=True)
    shorter = _projected(_state(130.0, remaining_s=3000.0), stops_remaining=0, pit_lap=None)
    (change,) = race.step(130.0, state=shorter, pre_race=True, laps=1, samples=(stop,))
    assert change.fields["changed"] == "race_laps"
    assert change.text == "Plan change: 32 laps (was 35 laps)"


def test_no_finish_entry_outside_a_race(make_vehicle: Callable[..., VehicleState]) -> None:
    """A qualifying session that ends gives no "Race finished" entry."""
    qualifying = _Race(make_vehicle)
    qualifying.step(0.0, session_type=SessionType.QUALIFY)
    assert qualifying.step(900.0, session_type=SessionType.QUALIFY, laps=8, finished=True, finish_status=1) == []


def test_yellow_first_seen_outside_green_clears_without_an_entry(race: _Race) -> None:
    """A yellow that got no entry because it was raised under a full course yellow gets no clear entry either."""
    race.step(10.0, phase=FCY)
    assert race.step(11.0, phase=FCY, sector_flags=(11, 1, 11)) == []
    assert _events(race.step(12.0, sector_flags=(11, 1, 11))) == [("flag", "fcy_end")]
    assert race.step(13.0) == []
    assert race.step(17.0) == []
    (yellow,) = race.step(30.0, sector_flags=(11, 1, 11))
    assert yellow.fields["event"] == "yellow"
    race.step(40.0)
    assert _events(race.step(44.0)) == [("flag", "clear")]


def test_yellow_already_up_at_the_session_start_clears_without_an_entry(
    make_vehicle: Callable[..., VehicleState],
) -> None:
    """A yellow the session is joined with has no entry, so its clear has none."""
    race = _Race(make_vehicle)
    race.step(100.0, sector_flags=(1, 11, 11))
    assert race.step(101.0, sector_flags=(1, 11, 11)) == []
    assert race.step(102.0) == []
    assert race.step(106.0) == []


def test_fuel_car_plan_change_uses_the_litre_threshold(race: _Race) -> None:
    """For a car without virtual energy the first-stop amount is in litres and must move by the litre threshold,
    not by the energy fraction."""
    race.step(2.0, state=_state(2.0, fuel_l_per_lap=3.0), pre_race=True)
    assert race.step(110.0, state=_state(110.0, fuel_l_per_lap=3.02), pre_race=True, laps=1) == []
    (change,) = race.step(220.0, state=_state(220.0, fuel_l_per_lap=3.04), pre_race=True, laps=2)
    assert change.fields["changed"] == "first_stop_add"
    assert change.text == "Plan change: add 45 litres (was 44 litres)"


def test_plan_and_rival_set_wait_until_the_earlier_entries_are_known(race: _Race) -> None:
    """While it is not known what the session's timeline already has, neither the initial plan nor the rival set
    is made; they come with the first state after that."""
    rivals = [_rival(1, ("ahead",), 1, 10.0)]
    assert race.step(2.0, state=_state(2.0, rivals), plan_recorded=None, rivals_recorded=None) == []
    assert _events(race.step(3.0, state=_state(3.0, rivals))) == [("rival", "set"), ("plan", "initial")]


def test_restart_does_not_repeat_the_initial_plan_or_the_rival_set(race: _Race) -> None:
    """With a plan and a rival set already in the timeline both are taken as they are; later changes are
    recorded against them."""
    rivals = [_rival(1, ("ahead",), 1, 10.0)]
    assert race.step(2.0, state=_state(2.0, rivals), plan_recorded=True, rivals_recorded=True) == []
    assert race.step(3.0, state=_state(3.0, rivals), plan_recorded=True, rivals_recorded=True) == []
    changed = _state(110.0, [*rivals, _rival(2, ("behind",), 3, -8.0)], energy_per_lap=0.05)
    assert _events(race.step(110.0, state=changed, laps=1, plan_recorded=True, rivals_recorded=True)) == [
        ("plan", "change")
    ]
    later = _state(141.0, [*rivals, _rival(2, ("behind",), 3, -8.0)], energy_per_lap=0.05)
    assert _events(race.step(141.0, state=later, laps=1, plan_recorded=True, rivals_recorded=True)) == [
        ("rival", "new")
    ]


def test_pit_in_of_a_car_that_left_the_session_is_forgotten(race: _Race) -> None:
    """A car that disappears while in the pits gives no pit-out when its slot is seen on track again."""
    state = _state(1.0, [_rival(2, ("behind",), 3, -10.0)])
    race.step(2.0, state=state)
    assert _events(race.step(3.0, state=state, in_pits=(2,))) == [("pit", "in")]
    without = {1: 1, 0: 2, 3: 3}
    assert [e for e in race.step(4.0, places=without, state=state) if e.kind == "pit"] == []
    assert [e for e in race.step(5.0, state=state) if e.kind == "pit"] == []


def test_no_pit_entries_after_the_player_has_finished(race: _Race) -> None:
    """Driving into the pits after the chequered flag is no pit stop."""
    race.step(3500.0, laps=34, finished=True, finish_status=1)
    assert race.step(3560.0, laps=34, finished=True, finish_status=1, in_pits=(0,)) == []
    assert race.step(3600.0, laps=34, finished=True, finish_status=1) == []
