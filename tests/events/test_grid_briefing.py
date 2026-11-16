import dataclasses
from collections.abc import Callable

from ai_race_engineer.events.base import Priority
from ai_race_engineer.events.grid_briefing import GridBriefing
from ai_race_engineer.strategy.pit_time import PitLane, PitModel, PitRates
from ai_race_engineer.strategy.priors import EventPriors, PriorSource
from ai_race_engineer.strategy.race_finish import CarInput, project_race
from ai_race_engineer.strategy.race_state import RaceState
from ai_race_engineer.strategy.rivals import Rival
from ai_race_engineer.telemetry.race_snapshot import CarTyres, RaceSnapshot, TrackFlags
from ai_race_engineer.telemetry.snapshot import Snapshot, VehicleState

PIT: PitModel = PitModel(PitRates(2.0, 0.02, 1.0), PitLane(30.0, 500.0), 5000.0)
PRIORS: EventPriors = EventPriors("Track", "Car A", {"driver0": 106.0, "driver1": 105.5}, {})
SOURCES: dict[int, PriorSource] = {0: PriorSource("qualifying", "game")}


class _Tracker:
    """Stand-in for the race tracker: holds the state the test sets.
    state     RaceState | None
    finished  bool
    """

    def __init__(self) -> None:
        """Start without a state.
        In:  -
        Out: None
        """
        self.state: RaceState | None = None
        self.finished: bool = False


def _state(make_vehicle: Callable[..., VehicleState], before_green: bool = True) -> RaceState:
    """Race state of the player (slot 0) with the car ahead in class (slot 1).
    In:  make_vehicle  Callable[..., VehicleState]
         before_green  bool
    Out:               RaceState
    """
    cars = tuple(CarInput(s, "GT3", 0.0, 106.0, False, False, 60.0, 100.0, 1.0, 3.0, 0.045, 0.66) for s in (0, 1))
    rival = Rival(1, "driver1", ("ahead",), 1, 0.0, 0.0, False, 0, 106.0, None, 0.0, 1, 0.0, 1, 22.0, 22)
    return RaceState(
        time_s=10.0,
        player_slot_id=0,
        vehicles=tuple(make_vehicle(slot_id=s, is_player=s == 0, place=2 - s) for s in (0, 1)),
        cars=cars,
        estimates={},
        sectors={},
        pit=PIT,
        estimate=None,
        set_fuel_ratio=None,
        projection=project_race(cars, 3600.0, None, PIT),
        rivals=(rival,),
        traffic_window_s=3.0,
        before_green=before_green,
    )


def _moving(
    briefing: GridBriefing,
    make_snapshot: Callable[..., Snapshot],
    make_vehicle: Callable[..., VehicleState],
    speed_mps: float,
) -> None:
    """Feed the briefing a snapshot with the player at a speed.
    In:  briefing       GridBriefing
         make_snapshot  Callable[..., Snapshot]
         make_vehicle   Callable[..., VehicleState]
         speed_mps      float
    Out:                None
    """
    base = make_snapshot(player=make_vehicle(slot_id=0, is_player=True, speed_mps=speed_mps))
    tyres = (CarTyres(0, "Medium", "Medium", (1.0,) * 4), CarTyres(1, "Hard", "Hard", (1.0,) * 4))
    briefing.update(
        RaceSnapshot(base.session, base.player, base.vehicles, (), TrackFlags(0, 0, (11, 11, 11), 0), (), None, tyres)
    )


def test_fires_once_before_the_green(
    make_snapshot: Callable[..., Snapshot], make_vehicle: Callable[..., VehicleState]
) -> None:
    """Standing on the grid the briefing is given once, at once, from the qualifying times and the tyres."""
    tracker = _Tracker()
    briefing = GridBriefing(tracker, lambda: PRIORS, lambda: SOURCES, 3, 5.0, 0.005, 20.0)
    assert briefing.check(make_snapshot(), 0.0) is None
    tracker.state = _state(make_vehicle)
    _moving(briefing, make_snapshot, make_vehicle, 0.0)
    callout = briefing.check(make_snapshot(), 0.0)
    assert (callout.kind, callout.priority) == ("grid_briefing", Priority.SPOTTER)
    assert callout.text == "Ahead driver1, P1 in class, 0.5 seconds faster in qualifying, on Hard tyres."
    assert briefing.check(make_snapshot(), 1.0) is None


def test_rolling_start_waits_for_a_straight(
    make_snapshot: Callable[..., Snapshot], make_vehicle: Callable[..., VehicleState]
) -> None:
    """With the car already rolling the briefing is a normal callout."""
    tracker = _Tracker()
    tracker.state = _state(make_vehicle)
    briefing = GridBriefing(tracker, lambda: PRIORS, lambda: SOURCES, 3, 5.0, 0.005, 20.0)
    _moving(briefing, make_snapshot, make_vehicle, 30.0)
    assert briefing.check(make_snapshot(), 0.0).priority == Priority.STRATEGY


def test_not_after_the_green_and_not_without_priors(
    make_snapshot: Callable[..., Snapshot], make_vehicle: Callable[..., VehicleState]
) -> None:
    """Once the race is green, or while the priors are not known yet, nothing is said."""
    tracker = _Tracker()
    tracker.state = _state(make_vehicle, before_green=False)
    assert (
        GridBriefing(tracker, lambda: PRIORS, lambda: SOURCES, 3, 5.0, 0.005, 20.0).check(make_snapshot(), 0.0) is None
    )
    tracker.state = _state(make_vehicle)
    assert GridBriefing(tracker, lambda: None, lambda: SOURCES, 3, 5.0, 0.005, 20.0).check(make_snapshot(), 0.0) is None


def test_reset_re_arms(make_snapshot: Callable[..., Snapshot], make_vehicle: Callable[..., VehicleState]) -> None:
    """After a reset the next race start gets its briefing."""
    tracker = _Tracker()
    tracker.state = _state(make_vehicle)
    briefing = GridBriefing(tracker, lambda: PRIORS, lambda: SOURCES, 3, 5.0, 0.005, 20.0)
    assert briefing.check(make_snapshot(), 0.0) is not None
    assert briefing.check(make_snapshot(), 1.0) is None
    briefing.reset()
    assert briefing.check(make_snapshot(), 2.0) is not None
    assert dataclasses.asdict(briefing.cars()[0])["relation"] == "ahead"


def test_waits_for_the_qualifying_times_and_the_energy_figure(
    make_snapshot: Callable[..., Snapshot], make_vehicle: Callable[..., VehicleState]
) -> None:
    """While the qualifying times or the player's energy figure are missing the briefing is held; it is given as
    soon as both are known."""
    tracker = _Tracker()
    tracker.state = _state(make_vehicle)
    priors = [EventPriors("Track", "Car A", {}, {})]
    sources = [{0: PriorSource("history_class", "none")}]
    briefing = GridBriefing(tracker, lambda: priors[0], lambda: sources[0], 3, 5.0, 0.005, 20.0)
    assert briefing.check(make_snapshot(), 0.0) is None
    priors[0] = PRIORS
    tracker.state = dataclasses.replace(tracker.state, time_s=15.0)
    assert briefing.check(make_snapshot(), 0.0) is None
    sources[0] = SOURCES
    tracker.state = dataclasses.replace(tracker.state, time_s=16.0)
    assert "in qualifying" in briefing.check(make_snapshot(), 0.0).text
    assert briefing.check(make_snapshot(), 0.0) is None


def test_given_without_them_after_the_wait_and_only_before_the_green(
    make_snapshot: Callable[..., Snapshot], make_vehicle: Callable[..., VehicleState]
) -> None:
    """After max_wait_s of session time the briefing is given with what is known; a reset starts the wait again,
    and once the race is green a briefing that was still held is not given."""
    tracker = _Tracker()
    tracker.state = _state(make_vehicle)
    briefing = GridBriefing(tracker, lambda: EventPriors("Track", "Car A", {}, {}), dict, 3, 5.0, 0.005, 20.0)
    assert briefing.check(make_snapshot(), 0.0) is None
    tracker.state = dataclasses.replace(tracker.state, time_s=29.9)
    assert briefing.check(make_snapshot(), 0.0) is None
    tracker.state = dataclasses.replace(tracker.state, time_s=30.0)
    assert briefing.check(make_snapshot(), 0.0).text == "Ahead driver1, P1 in class."
    briefing.reset()
    tracker.state = dataclasses.replace(tracker.state, time_s=40.0)
    assert briefing.check(make_snapshot(), 0.0) is None
    tracker.state = dataclasses.replace(tracker.state, time_s=70.0, before_green=False)
    assert briefing.check(make_snapshot(), 0.0) is None
