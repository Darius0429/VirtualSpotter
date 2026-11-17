import dataclasses

import pytest

from ai_race_engineer.domains.common.tools.plan_tools import PlanTools
from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitLane, PitModel, PitRates
from ai_race_engineer.domains.common.analysis.field.priors import PriorSource
from ai_race_engineer.domains.common.analysis.strategy.race_finish import CarInput, project_race
from ai_race_engineer.domains.common.analysis.field.race_state import RaceState
from ai_race_engineer.domains.common.analysis.session.session_log import Note
from ai_race_engineer.domains.common.data.snapshot import VehicleState

PIT: PitModel = PitModel(PitRates(3.0, 0.025, 1.0, base_s=1.0), PitLane(28.0, 417.4), 5386.8)


class _Tracker:
    """Stand-in for the race tracker.
    state     RaceState | None
    finished  bool
    """

    def __init__(self, state: RaceState | None) -> None:
        """Hold a state.
        In:  state  RaceState | None
        Out:        None
        """
        self.state: RaceState | None = state
        self.finished: bool = False


class _Field:
    """Stand-in for the prior field tracker.
    sources   dict[int, PriorSource]
    pre_race  bool
    """

    def __init__(self, pre_race: bool) -> None:
        """Set the sources of the player and whether the pre-race figures are in use.
        In:  pre_race  bool
        Out:           None
        """
        self.sources: dict[int, PriorSource] = {0: PriorSource("qualifying", "game")}
        self.pre_race: bool = pre_race


class _SessionLog:
    """Stand-in for the session log.
    notes  list[Note]
    """

    def __init__(self) -> None:
        """Start without notes.
        In:  -
        Out: None
        """
        self.notes: list[Note] = []

    def clean_laps(self) -> tuple[tuple[float, float], ...]:
        """No clean laps yet.
        In:  -
        Out: tuple[tuple[float, float], ...]
        """
        return ()

    def add_note(self, text: str) -> Note:
        """Keep a note on lap 3.
        In:  text  str
        Out:       Note
        """
        self.notes.append(Note(250.0, 3, text, False))
        return self.notes[-1]

    def record(self) -> None:
        """No session record.
        In:  -
        Out: None
        """
        return None


class _Keeper:
    """Stand-in for the session keeper without sessions."""

    def summary(self) -> None:
        """No running session.
        In:  -
        Out: None
        """
        return None

    def past_summary(self, kind: str, skip_running: bool) -> None:
        """No earlier session.
        In:  kind          str
             skip_running  bool
        Out:               None
        """
        return None

    def priors(self) -> None:
        """No priors.
        In:  -
        Out: None
        """
        return None


class _Briefing:
    """Stand-in for the grid briefing without cars."""

    def cars(self) -> tuple[()]:
        """No cars.
        In:  -
        Out: tuple[()]
        """
        return ()


def _state(energy_per_lap: float | None = 0.045, max_laps: int | None = None) -> RaceState:
    """Race state of the player alone on the grid of a 60-minute race at a 107.1 s lap.
    In:  energy_per_lap  float | None  fraction/lap; None = usage unknown
         max_laps        int | None    lap limit instead of the clock
    Out:                 RaceState
    """
    car = CarInput(0, "Hyper", 0.0, 107.1, False, False, 70.0, 100.0, 1.0, None, energy_per_lap, 0.7)
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    vehicle = VehicleState(**{**base, "is_player": True, "driver_name": "me", "class_name": "Hyper", "place": 1})
    return RaceState(
        time_s=10.0,
        player_slot_id=0,
        vehicles=(vehicle,),
        cars=(car,),
        estimates={},
        sectors={},
        pit=PIT,
        estimate=None,
        set_fuel_ratio=None,
        projection=project_race((car,), None if max_laps else 3600.0, max_laps, PIT),
        rivals=(),
        traffic_window_s=3.0,
        before_green=True,
    )


def _tools(state: RaceState | None, pre_race: bool = True) -> PlanTools:
    """Plan tools on stand-ins.
    In:  state     RaceState | None
         pre_race  bool
    Out:           PlanTools
    """
    return PlanTools(_Tracker(state), None, 3.0, _Field(pre_race), _SessionLog(), _Keeper(), _Briefing(), 0.05, 5, 0.01)


def test_without_a_state_every_tool_says_so() -> None:
    """With no race state the plan names the reason and the other tools are unavailable or empty."""
    tools = _tools(None)
    assert tools.race_plan() == {"available": False, "reason": "no reference pace"}
    assert tools.saving() == {"available": False}
    assert tools.my_race() == {"available": False}
    assert tools.pit_stop() == {"available": False}
    assert tools.session_summary() == {"available": False}
    assert tools.session_summary("last_race") == {"available": False}
    assert tools.driver_laps() == {"available": False}
    assert tools.track_history() == {"available": False}
    assert (tools.grid_briefing(), tools.notes()) == ([], [])


def test_plan_with_first_stop_time_and_basis() -> None:
    """The owner's example as tool output: 34 laps, pit at the end of lap 22, 53 % at a stop of 19.7 s lane loss
    plus 22.2 s standing; the basis and the sources are named."""
    plan = _tools(_state()).race_plan()
    assert (plan["plan_basis"], plan["pace_source"], plan["usage_source"]) == ("pre_race", "qualifying", "game")
    assert (plan["race_laps"], plan["stops"], plan["pit_at_end_of_lap"], plan["net_leader"]) == (34, 1, 22, "me")
    assert (plan["pace_used"], plan["average_lap_with_stops"]) == ("1:47.1", "1:48.3")
    assert plan["energy_to_add_total_percent"] == pytest.approx(53.0)
    assert plan["first_stop_add_percent"] == pytest.approx(53.0)
    assert plan["fuel_to_add_total_l"] == pytest.approx(37.1)
    assert plan["first_stop_time_s"] == pytest.approx(19.7 + 22.2, abs=0.01)
    assert plan["min_start_percent_for_min_stops"] == pytest.approx(54.0)
    assert _tools(_state(), pre_race=False).race_plan()["plan_basis"] == "measured"


def test_unknown_usage_of_the_net_leader_marks_the_laps_as_approximate() -> None:
    """While the net leader has no usage figure it is projected without stops: the plan and the finish say so."""
    known, unknown = _tools(_state()), _tools(_state(energy_per_lap=None))
    assert (known.race_plan()["laps_approximate"], known.race_finish()["laps_approximate"]) == (False, False)
    assert (unknown.race_plan()["laps_approximate"], unknown.race_finish()["laps_approximate"]) == (True, True)


def test_plan_without_a_stop_has_no_stop_time() -> None:
    """A race the energy on board covers has no pit lap and no stop time."""
    plan = _tools(_state(max_laps=10)).race_plan()
    assert (plan["stops"], plan["pit_at_end_of_lap"], plan["first_stop_time_s"]) == (0, None, None)
    assert plan["first_stop_add_percent"] == 0.0


def test_plan_with_unknown_usage_keeps_laps_and_sources() -> None:
    """Without a usage the plan has the laps, the pace and the sources with no stops, and no amounts."""
    plan = _tools(_state(energy_per_lap=None)).race_plan()
    assert (plan["race_laps"], plan["stops"], plan["pace_used"]) == (34, 0, "1:47.1")
    assert "pit_at_end_of_lap" not in plan
    assert _tools(_state(energy_per_lap=None)).saving() == {"available": False, "reason": "no energy usage"}


def test_saving_entries_and_basis() -> None:
    """With a stop ahead the two standing checks are given; the asked ones are added; the basis is named."""
    saving = _tools(_state()).saving(laps_needed=23.0, shorten_stop_s=2.0)
    assert {"one_more_lap_before_stop", "skip_last_stop", "reach_laps", "shorten_stop"} <= saving.keys()
    assert (saving["plan_basis"], saving["laps_measured"], saving["max_saving_percent"]) == ("pre_race", 0, 5.0)
    assert saving["reach_laps"]["hold_for_laps"] == 23.0
    assert saving["reach_laps"]["within_lowest_shown"] is None
    assert set(_tools(_state(max_laps=10)).saving()) == {
        "plan_basis",
        "current_usage_percent_per_lap",
        "lowest_usage_shown_percent_per_lap",
        "max_saving_percent",
        "laps_measured",
    }


def test_my_race_and_pit_stop_name_the_basis() -> None:
    """The existing answers carry the basis of the plan too."""
    tools = _tools(_state())
    assert (tools.my_race()["plan_basis"], tools.my_race()["pit_at_end_of_lap"]) == ("pre_race", 22)
    assert tools.pit_stop()["plan_basis"] == "pre_race"


def test_add_note_confirms_the_lap() -> None:
    """A note is handed to the session log and confirmed with its lap."""
    assert _tools(_state()).add_note("understeer in turn 4") == "Noted on lap 3."


def test_all_fifteen_tools_are_registered() -> None:
    """The seven race tools and the eight plan tools are offered to the agent."""
    names = [tool.__name__ for tool in _tools(None).tools()]
    assert len(names) == len(set(names)) == 15
    assert names[-8:] == [
        "race_plan",
        "saving",
        "grid_briefing",
        "session_summary",
        "driver_laps",
        "add_note",
        "notes",
        "track_history",
    ]


class _StoredKeeper(_Keeper):
    """Stand-in for the session keeper with one stored race.
    yellow_flags  int | None  the yellow-flag count of the stored summary
    """

    def __init__(self, yellow_flags: int | None) -> None:
        """Store the count.
        In:  yellow_flags  int | None
        Out:               None
        """
        self.yellow_flags: int | None = yellow_flags

    def past_summary(self, kind: str, skip_running: bool) -> dict[str, object]:
        """The stored race.
        In:  kind          str
             skip_running  bool
        Out:               dict[str, object]
        """
        return {"kind": kind, "yellow_flags": self.yellow_flags, "player": {"laps": 3, "lap_list": []}}


def test_session_summary_leaves_out_yellow_flags_that_are_not_counted() -> None:
    """A summary without a yellow-flag count has no such key; a counted one keeps it."""
    uncounted = PlanTools(None, None, 3.0, None, None, _StoredKeeper(None), None, 0.05, 5, 0.01)
    assert uncounted.session_summary("race") == {"kind": "race", "player": {"laps": 3}}
    counted = PlanTools(None, None, 3.0, None, None, _StoredKeeper(4), None, 0.05, 5, 0.01)
    assert counted.session_summary("race")["yellow_flags"] == 4
