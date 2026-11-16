import dataclasses
from collections.abc import Callable

import pytest

from ai_race_engineer.events.base import Callout, Priority
from ai_race_engineer.events.rival_watch import RivalWatch
from ai_race_engineer.strategy.pit_time import PitLane, PitModel, PitRates
from ai_race_engineer.strategy.race_finish import CarInput, project_race
from ai_race_engineer.strategy.race_state import RaceState
from ai_race_engineer.strategy.rivals import Rival
from ai_race_engineer.telemetry.snapshot import Snapshot, VehicleState

PIT: PitModel = PitModel(
    PitRates(fuel_rate_l_per_s=2.0, energy_rate_per_s=0.02, scale=1.0), PitLane(30.0, 500.0), 5000.0
)


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


class _Race:
    """A rival watch on a stand-in tracker, fed one race state per step.
    tracker  _Tracker
    watch    RivalWatch                   0.3 s pace change, 2 laps low, groups ahead/behind/strategy, 60 s minimum
    _make    Callable[..., VehicleState]  vehicle factory
    _time_s  float                        s session clock of the last state
    """

    def __init__(self, make_vehicle: Callable[..., VehicleState]) -> None:
        """Build the watch.
        In:  make_vehicle  Callable[..., VehicleState]
        Out:               None
        """
        self.tracker: _Tracker = _Tracker()
        self.watch: RivalWatch = RivalWatch(self.tracker, 0.3, 2.0, ["ahead", "behind", "strategy"], 60.0, 5)
        self._make: Callable[..., VehicleState] = make_vehicle
        self._time_s: float = 0.0

    def step(
        self,
        *rivals: Rival,
        places: dict[int, int] | None = None,
        dt_s: float = 1.0,
        laps: dict[int, float] | None = None,
        before_green: bool = False,
    ) -> Callout | None:
        """Publish a new race state with these rivals and check once.
        In:  rivals  Rival             rivals of the new state
             places  dict[int, int] | None  slot id -> overall place (default: the player 5th, rivals ahead of it)
             dt_s    float             s of session time since the previous state
             laps    dict[int, float] | None  slot id -> race distance in laps (default 10)
             before_green  bool        the race has not started
        Out:         Callout | None
        """
        self._time_s += dt_s
        slots = (0, *(r.slot_id for r in rivals))
        place_of = places or {slot_id: 5 if slot_id == 0 else slot_id for slot_id in slots}
        cars = tuple(
            CarInput(slot_id, "GT3", 10.0 + slot_id / 10, 100.0, False, False, 50.0, 100.0, 0.5, 2.0, 0.125, None)
            for slot_id in slots
        )
        self.tracker.state = RaceState(
            time_s=self._time_s,
            player_slot_id=0,
            vehicles=tuple(
                self._make(
                    slot_id=s,
                    is_player=s == 0,
                    place=place_of[s],
                    completed_laps=int((laps or {}).get(s, 10.0)),
                    lap_distance_m=(laps or {}).get(s, 10.0) % 1 * PIT.track_length_m,
                    in_pits=any(r.slot_id == s and r.in_pits for r in rivals),
                )
                for s in slots
            ),
            cars=cars,
            estimates={},
            sectors={},
            pit=PIT,
            estimate=None,
            set_fuel_ratio=None,
            projection=project_race(cars, 1000.0, None, PIT),
            rivals=rivals,
            traffic_window_s=3.0,
            before_green=before_green,
        )
        return self.check()

    def check(self) -> Callout | None:
        """Check without a new state.
        In:  -
        Out: Callout | None
        """
        return self.watch.check(None, 0.0)


@pytest.fixture
def race(make_vehicle: Callable[..., VehicleState]) -> _Race:
    """A rival watch ready to be fed states.
    In:  make_vehicle  Callable[..., VehicleState]
    Out:               _Race
    """
    return _Race(make_vehicle)


def _rival(slot_id: int = 1, **overrides: object) -> Rival:
    """Rival ahead on the player's pace with no stop left, and the given overrides.
    In:  slot_id    int
         overrides  object  Rival field values
    Out:            Rival
    """
    rival = Rival(
        slot_id=slot_id,
        driver_name=f"driver{slot_id}",
        groups=("ahead",),
        class_place=4,
        gap_s=5.0,
        lap_fraction=0.5,
        in_pits=False,
        pitstops=0,
        lap_time_s=100.0,
        best_lap_s=99.0,
        pace_delta_s=0.0,
        projected_class_place=4,
        projected_gap_s=5.0,
        stops_remaining=0,
        laps_on_board=8.0,
        pit_lap=None,
    )
    return dataclasses.replace(rival, **overrides)


def test_nothing_without_a_state(race: _Race, make_snapshot: Callable[..., Snapshot]) -> None:
    """While the tracker has no race state the watch is silent."""
    assert race.watch.check(make_snapshot(), 0.0) is None


def test_first_state_seeds_silently(race: _Race) -> None:
    """The rivals present at the start are not announced."""
    assert race.step(_rival(1), _rival(2)) is None
    assert race.step(_rival(1), _rival(2)) is None


def test_new_rival_once(race: _Race) -> None:
    """A car entering a watched group is announced once, at strategy priority, with its slot id."""
    race.step(_rival(1))
    callout = race.step(_rival(1), _rival(2, groups=("strategy",)))
    assert (callout.kind, callout.priority, callout.facts["slot_id"]) == ("rival_new", Priority.STRATEGY, 2)
    assert callout.text == "New rival driver2, P4 in class, 5.0 seconds ahead"
    assert race.step(_rival(1), _rival(2, groups=("strategy",))) is None


def test_rival_gone_once(race: _Race) -> None:
    """A strategy rival leaving the watched groups is announced once."""
    race.step(_rival(1), _rival(2, groups=("strategy",)))
    callout = race.step(_rival(1))
    assert (callout.kind, callout.text, callout.facts["slot_id"]) == ("rival_gone", "driver2 is no longer a rival", 2)
    assert race.step(_rival(1)) is None


def test_unwatched_groups_are_ignored(race: _Race) -> None:
    """A car that is only a pace rival is not announced, and moving to that group counts as gone."""
    race.step(_rival(1, groups=("strategy",)))
    assert race.step(_rival(1, groups=("strategy",)), _rival(2, groups=("pace",))) is None
    assert race.step(_rival(1, groups=("pace",))).kind == "rival_gone"


def test_swap_of_the_car_ahead_gives_one_callout(race: _Race) -> None:
    """Another car taking over as the class neighbour is one position callout, without a gone and a new rival."""
    race.step(_rival(1), _rival(3, groups=("behind",)), places={0: 5, 1: 4, 2: 3, 3: 6})
    callout = race.step(_rival(2, gap_s=4.0), _rival(3, groups=("behind",)), places={0: 5, 1: 3, 2: 4, 3: 6})
    assert (callout.kind, callout.text) == ("rival_position", "driver2 is now the car ahead, 4.0 seconds")
    assert callout.facts["slot_id"] == 2
    assert race.check() is None


def test_player_passing_the_car_ahead_gives_one_callout(race: _Race) -> None:
    """The player taking a place changes both neighbours: only the pass itself is announced."""
    race.step(_rival(1), _rival(3, groups=("behind",)), places={0: 5, 1: 4, 2: 3, 3: 6})
    callout = race.step(_rival(2), _rival(1, groups=("behind",)), places={0: 4, 1: 5, 2: 3, 3: 6})
    assert (callout.kind, callout.text) == ("rival_position", "You are now ahead of driver1")
    assert race.check() is None


def test_nothing_once_the_race_is_over(race: _Race) -> None:
    """After the chequered flag pending callouts are dropped and nothing new is said."""
    race.step(_rival(1), _rival(2))
    race.tracker.finished = True
    assert race.step(_rival(1, in_pits=True), _rival(2, best_lap_s=98.0)) is None
    assert race.check() is None


def test_rival_pit_with_predicted_rejoin(race: _Race) -> None:
    """A rival entering the pits is announced with its expected stop time (20 s lane loss + 25 s for half a charge),
    rejoin place and gap to the player."""
    race.step(_rival(1))
    callout = race.step(_rival(1, in_pits=True))
    assert (callout.kind, callout.priority, callout.facts["slot_id"]) == ("rival_pit", Priority.STRATEGY, 1)
    assert callout.facts["stop_time_s"] == pytest.approx(45.0)
    assert callout.facts["rejoin_class_place"] == 2
    assert callout.facts["rejoin_gap_s"] == pytest.approx(-40.0)
    assert callout.text == "driver1 pits, about 45 seconds, rejoins P2 in class, 40.0 seconds behind"
    assert race.step(_rival(1, in_pits=True)) is None


def test_rival_out_with_gap(race: _Race) -> None:
    """A rival leaving the pits is announced with its gap."""
    race.step(_rival(1, in_pits=True))
    callout = race.step(_rival(1, gap_s=-12.0))
    assert (callout.kind, callout.text) == ("rival_out", "driver1 is out of the pits, 12.0 seconds behind")
    assert race.step(_rival(1, gap_s=-12.0)) is None


def test_rival_position_flip(race: _Race) -> None:
    """The order between rival and player flipping is announced in both directions."""
    race.step(_rival(1), places={0: 5, 1: 4})
    callout = race.step(_rival(1), places={0: 4, 1: 5})
    assert (callout.kind, callout.text) == ("rival_position", "You are now ahead of driver1")
    assert race.step(_rival(1), places={0: 4, 1: 5}) is None
    assert race.step(_rival(1), places={0: 5, 1: 4}).text == "driver1 is now ahead of you"


def test_rival_personal_best(race: _Race) -> None:
    """A rival improving its best lap is announced."""
    race.step(_rival(1))
    callout = race.step(_rival(1, best_lap_s=98.5))
    assert (callout.kind, callout.text) == ("rival_pace", "driver1 personal best, 1:38.5")
    assert race.step(_rival(1, best_lap_s=98.5)) is None


def test_rival_pace_change_since_last_announced(race: _Race) -> None:
    """Pace is announced when it has moved by pace_change_s since the last announcement, not for less."""
    race.step(_rival(1))
    assert race.step(_rival(1, lap_time_s=100.2, pace_delta_s=0.2)) is None
    callout = race.step(_rival(1, lap_time_s=100.4, pace_delta_s=0.4))
    assert (callout.kind, callout.text) == ("rival_pace", "driver1 pace 1:40.4, 0.4 slower than you")
    assert race.step(_rival(1, lap_time_s=100.6, pace_delta_s=0.6)) is None
    assert race.step(_rival(1, lap_time_s=100.0, pace_delta_s=0.0)).kind == "rival_pace"


def test_rival_low_once_per_stint(race: _Race) -> None:
    """A rival that must stop and has few laps on board is announced once, and again after it refuelled."""
    race.step(_rival(1, stops_remaining=1, laps_on_board=2.5))
    callout = race.step(_rival(1, stops_remaining=1, laps_on_board=1.9))
    assert (callout.kind, callout.text) == ("rival_low", "driver1 has 1.9 laps on board, must pit")
    assert race.step(_rival(1, stops_remaining=1, laps_on_board=1.2)) is None
    assert race.step(_rival(1, stops_remaining=1, laps_on_board=8.0)) is None
    assert race.step(_rival(1, stops_remaining=1, laps_on_board=1.9)).kind == "rival_low"


def test_rival_low_is_not_repeated_on_small_rises(race: _Race) -> None:
    """Laps on board wobbling below the threshold give exactly one callout."""
    callouts = [
        race.step(_rival(1, stops_remaining=1, laps_on_board=laps)) for laps in (2.5, 1.95, 1.94, 1.97, 1.96, 1.99)
    ]
    assert [c.kind for c in callouts if c is not None] == ["rival_low"]


def test_rival_low_survives_leaving_and_returning(race: _Race) -> None:
    """A rival that drops out of the list and comes back in the same stint is not announced as low again."""
    low = _rival(2, groups=("strategy",), stops_remaining=1, laps_on_board=1.5)
    race.step(_rival(1), _rival(2, groups=("strategy",), stops_remaining=1, laps_on_board=2.5))
    assert race.step(_rival(1), low).kind == "rival_low"
    assert race.step(_rival(1)).kind == "rival_gone"
    assert race.step(_rival(1), low, dt_s=90.0).kind == "rival_new"
    assert race.step(_rival(1), low) is None


def test_returning_rival_is_not_announced_again_within_the_minimum_time(race: _Race) -> None:
    """A car announced as a new rival less than 60 s ago is not announced as new again; later it is."""
    back = _rival(2, groups=("strategy",))
    race.step(_rival(1))
    assert race.step(_rival(1), back).kind == "rival_new"
    assert race.step(_rival(1), dt_s=20.0).kind == "rival_gone"
    assert race.step(_rival(1), back, dt_s=20.0) is None
    assert race.step(_rival(1), dt_s=10.0).kind == "rival_gone"
    assert race.step(_rival(1), back, dt_s=10.0).kind == "rival_new"


@pytest.mark.parametrize("pitted", [{"pitstops": 1}, {"in_pits": True}])
def test_rival_low_rearmed_by_a_pit_visit(race: _Race, pitted: dict[str, object]) -> None:
    """A pit visit starts a new stint, also when the rival took too little to get above the threshold."""
    race.step(_rival(1, stops_remaining=1, laps_on_board=2.5))
    assert race.step(_rival(1, stops_remaining=1, laps_on_board=1.5)).kind == "rival_low"
    callouts = [
        race.step(_rival(1, stops_remaining=1, laps_on_board=1.8, **pitted)),
        race.check(),
        race.step(_rival(1, stops_remaining=1, laps_on_board=1.7, pitstops=1)),
        race.check(),
        race.step(_rival(1, stops_remaining=1, laps_on_board=1.6, pitstops=1)),
    ]
    assert [c.kind for c in callouts if c is not None].count("rival_low") == 1


def test_rival_low_needs_a_stop(race: _Race) -> None:
    """Few laps on board is no news for a rival that reaches the flag."""
    race.step(_rival(1, laps_on_board=2.5))
    assert race.step(_rival(1, laps_on_board=1.5)) is None


def test_several_changes_come_one_per_check(race: _Race) -> None:
    """Changes found in one state are queued and returned one per check."""
    race.step(_rival(1), _rival(2))
    first = race.step(_rival(1, in_pits=True), _rival(2, best_lap_s=98.0))
    second = race.check()
    assert (first.kind, second.kind) == ("rival_pit", "rival_pace")
    assert race.check() is None


def test_disabled_gives_nothing_and_no_flood_on_resume(race: _Race) -> None:
    """While switched off the watch keeps following the rivals but says nothing, also not afterwards."""
    race.step(_rival(1))
    race.watch.set_enabled(False)
    assert race.step(_rival(1, in_pits=True), _rival(2)) is None
    assert race.step(_rival(1), _rival(2)) is None
    race.watch.set_enabled(True)
    assert race.check() is None
    assert race.step(_rival(1), _rival(2)) is None
    assert race.step(_rival(1), _rival(2, in_pits=True)).kind == "rival_pit"


def test_reset_keeps_enabled_and_seeds_again(race: _Race) -> None:
    """A reset forgets the rivals and pending callouts but not the switch."""
    race.step(_rival(1))
    race.watch.set_enabled(False)
    race.watch.reset()
    assert race.watch.enabled is False
    race.watch.set_enabled(True)
    race.step(_rival(1), _rival(2))
    race.step(_rival(1, in_pits=True), _rival(2, in_pits=True))
    race.watch.reset()
    assert race.watch.enabled is True
    assert race.step(_rival(1), _rival(2), _rival(3)) is None
    assert race.step(_rival(1), _rival(2), _rival(3)) is None


def test_car_in_the_pits_or_on_its_out_lap_is_not_a_new_rival(race: _Race) -> None:
    """A car that comes into the strategy window while in the pits is announced only once its out-lap is over."""
    new = _rival(2, groups=("strategy",))
    race.step(_rival(1))
    assert race.step(_rival(1), dataclasses.replace(new, in_pits=True), laps={2: 10.0}) is None
    assert race.step(_rival(1), new, laps={2: 10.5}) is None
    assert race.step(_rival(1), new, laps={2: 10.99}) is None
    assert race.step(_rival(1), new, laps={2: 11.0}).kind == "rival_new"


def test_rival_that_pits_stays_a_rival(race: _Race) -> None:
    """A car that already is a rival stays one through its stop: its pit and out callouts come, no gone or new."""
    rival = _rival(2, groups=("strategy",))
    race.step(_rival(1), rival)
    assert race.step(_rival(1), dataclasses.replace(rival, in_pits=True), laps={2: 10.2}).kind == "rival_pit"
    assert race.step(_rival(1), rival, laps={2: 10.3}).kind == "rival_out"
    assert race.step(_rival(1), rival, laps={2: 10.4}) is None


def test_silent_before_the_green_and_no_flood_after_it(race: _Race) -> None:
    """Before the green the rivals are followed without a callout (cars leaving the garage, rivals coming and
    going); the first callout is about a change after the start."""
    assert race.step(_rival(1), before_green=True) is None
    assert race.step(_rival(1, in_pits=True), _rival(2, groups=("strategy",)), before_green=True) is None
    assert race.step(_rival(1), before_green=True) is None
    assert race.check() is None
    assert race.step(_rival(1)) is None
    assert race.step(_rival(1), _rival(3, groups=("strategy",))).kind == "rival_new"


def test_silent_through_the_first_laps_and_no_flood_at_the_release(race: _Race) -> None:
    """While the player has completed fewer than 5 laps the rivals are followed without a callout. The state in
    which lap 5 is completed is silent too (every pace switches to the measured one there and a rival drops out);
    after it only real changes are announced, against the silently kept state."""
    assert race.step(_rival(1), laps={0: 3.5}) is None
    assert race.step(_rival(1), _rival(2, groups=("strategy",)), laps={0: 4.2}) is None
    assert race.step(_rival(1, lap_time_s=101.0), _rival(2, groups=("strategy",)), laps={0: 4.9}) is None
    assert race.check() is None
    assert race.step(_rival(1, lap_time_s=103.0), laps={0: 5.0}) is None
    assert race.check() is None
    assert race.step(_rival(1, lap_time_s=103.0), laps={0: 5.1}) is None
    callout = race.step(_rival(1, lap_time_s=104.0), laps={0: 5.2})
    assert callout.kind == "rival_pace"
    assert race.check() is None


def test_rival_low_at_the_release_is_still_called(race: _Race) -> None:
    """A rival that became low during the silent laps is called once the callouts start, and a strategy rival seen
    meanwhile is not muted as already announced."""
    low = _rival(1, stops_remaining=1, laps_on_board=1.5)
    assert race.step(_rival(1, stops_remaining=1), laps={0: 4.2}) is None
    assert race.step(low, _rival(2, groups=("strategy",)), laps={0: 4.5}) is None
    assert race.step(low, laps={0: 5.0}) is None
    assert race.check() is None
    assert race.step(low, laps={0: 5.1}).kind == "rival_low"
    assert race.step(low, _rival(2, groups=("strategy",)), laps={0: 5.2}).kind == "rival_new"
    assert race.step(low, _rival(2, groups=("strategy",)), laps={0: 5.3}) is None
