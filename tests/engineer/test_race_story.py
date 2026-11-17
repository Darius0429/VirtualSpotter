import queue
from types import SimpleNamespace

from ai_race_engineer.domains.lmu.energy_tools import EnergyTools
from ai_race_engineer.domains.common.tools.history_tools import HistoryTools
from ai_race_engineer.engineer.race_story import RaceStory
from ai_race_engineer.domains.common.callouts.base import Callout
from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitRates
from ai_race_engineer.domains.common.analysis.session.timeline import TimelineEntry
from tests.domains.common.analysis.session.test_lap_digest import _digests


class _Timeline:
    """Stand-in for the timeline recorder.
    stored  list[TimelineEntry]
    """

    def __init__(self) -> None:
        """Start empty.
        In:  -
        Out: None
        """
        self.stored: list[TimelineEntry] = []

    def entries(self) -> tuple[TimelineEntry, ...]:
        """The entries held.
        In:  -
        Out: tuple[TimelineEntry, ...]
        """
        return tuple(self.stored)

    def add(self, kind: str, text: str, fields: dict[str, object]) -> TimelineEntry:
        """Keep an entry.
        In:  kind    str
             text    str
             fields  dict[str, object]
        Out:         TimelineEntry
        """
        self.stored.append(TimelineEntry(0.0, 0, kind, text, fields))
        return self.stored[-1]


class _Writer:
    """Stand-in for the note writer: one note per lap and a fixed summary.
    digests  list[dict[str, object]]  the lap facts it was given
    stories  list[list[str]]          the notes so far it was given with each
    """

    def __init__(self) -> None:
        """Start without calls.
        In:  -
        Out: None
        """
        self.digests: list[dict[str, object]] = []
        self.stories: list[list[str]] = []

    def notes(self, digest: dict[str, object], story: list[str]) -> list[str]:
        """One note for the lap.
        In:  digest  dict[str, object]
             story   list[str]
        Out:         list[str]
        """
        self.digests.append(digest)
        self.stories.append(list(story))
        return [f"you are P{digest['my_place']}"]

    def summary(self, story: list[str], timeline: list[str], laps: tuple[dict[str, object], ...]) -> str:
        """Summary naming how many notes and laps there were.
        In:  story     list[str]
             timeline  list[str]
             laps      tuple[dict[str, object], ...]
        Out:           str
        """
        return f"Good race, {len(story)} notes, {len(laps)} laps."


def test_each_finished_race_lap_gets_notes_and_the_flag_a_spoken_summary() -> None:
    """Each lap added to the history is given to the writer with the notes so far and its notes join the timeline;
    at the flag the summary is stored and spoken once."""
    lap_9, lap_10 = _digests()
    history = SimpleNamespace(laps=())
    tracker = SimpleNamespace(finished=False)
    timeline, writer, callouts = _Timeline(), _Writer(), queue.Queue()
    story = RaceStory(history, tracker, timeline, writer, callouts)
    story.update(None)
    history.laps = (lap_9,)
    story.update(None)
    story.update(None)
    history.laps = (lap_9, lap_10)
    story.update(None)
    story.step()
    story.step()
    assert [d["lap"] for d in writer.digests] == [9, 10]
    assert writer.stories == [[], ["Lap 9: you are P3"]]
    assert [e.text for e in timeline.stored] == ["Lap 9: you are P3", "Lap 10: you are P2"]
    tracker.finished = True
    story.update(None)
    story.update(None)
    story.step()
    callout: Callout = callouts.get_nowait()
    assert (callout.kind, callout.text) == ("race_summary", "Good race, 2 notes, 2 laps.")
    assert timeline.stored[-1].fields == {"event": "summary"}
    assert callouts.empty()


def test_gap_history_gives_the_trend_against_a_car() -> None:
    """Against Bob over the two laps: he was 1 s ahead and is 5 s behind, so the driver gained 6 s in one lap; with
    no name the car directly ahead in class is taken."""
    tools = HistoryTools(SimpleNamespace(laps=tuple(_digests())))
    bob = tools.gap_history("bob", 0)
    assert [(row["lap"], row["gap_s"], row["their_place"]) for row in bob["laps"]] == [(9, 1.0, 2), (10, -5.0, 3)]
    assert (bob["gap_change_s"], bob["gained_s"], bob["over_laps"]) == (-6.0, 6.0, 1)
    assert tools.gap_history("", 1)["driver"] == "Ann"
    assert tools.gap_history("nobody", 0) == {"available": False}
    assert HistoryTools(SimpleNamespace(laps=())).gap_history("", 0) == {"available": False}


def test_pit_energy_gives_the_level_to_leave_with_and_what_saving_can_make_up() -> None:
    """With a stop at the end of lap 22 and 12 laps after it at 4.5 % a lap the driver leaves with 54 %; lifting
    and coasting makes up 3.0 to 3.6 %, so the shortest fill is 50.4 %. Without a stop left, 5 laps short by 2 %
    are out of reach of 1.25 to 1.5 %."""

    def tracker(pit_lap: int | None, progress: float, energy: float, remaining: float) -> SimpleNamespace:
        car = SimpleNamespace(slot_id=0, virtual_energy=energy, energy_per_lap=0.045, progress_laps=progress)
        projection = SimpleNamespace(cars={0: SimpleNamespace(pit_lap=pit_lap, laps_remaining=remaining)})
        pit = SimpleNamespace(rates=PitRates(fuel_rate_l_per_s=2.0, energy_rate_per_s=0.025, scale=1.0, base_s=0.0))
        return SimpleNamespace(state=SimpleNamespace(player_slot_id=0, cars=[car], projection=projection, pit=pit))

    stop = EnergyTools(tracker(22, 20.0, 0.14, 14.0), (0.25, 0.30), 0.3).pit_energy()
    assert (stop["energy_at_stop_percent"], stop["laps_after_stop"]) == (5.0, 12.0)
    assert (stop["leave_pits_with_percent"], stop["add_percent"]) == (54.0, 49.0)
    assert stop["can_save_after_stop_percent"] == [3.0, 3.6]
    assert stop["shortest_fill_leave_with_percent"] == 50.4
    later = stop["one_lap_later"]
    assert (later["pit_at_end_of_lap"], later["for_laps"], later["within_reach"]) == (23, 3.0, True)
    assert (later["save_percent_per_lap"], later["energy_not_added_percent"]) == (0.0, 0.0)
    close = EnergyTools(tracker(22, 20.0, 0.132, 14.0), (0.25, 0.30), 0.3).pit_energy()["one_lap_later"]
    assert (close["save_percent_per_lap"], close["within_reach"], close["energy_not_added_percent"]) == (0.1, True, 0.3)
    assert (close["stop_shorter_by_s"], close["saving_costs_s"], close["net_gain_s"]) == (0.1, 0.9, -0.8)
    flag = EnergyTools(tracker(None, 29.0, 0.205, 5.0), (0.25, 0.30), 0.3).pit_energy()
    assert (flag["short_by_percent"], flag["can_save_to_flag_percent"], flag["reachable_by_saving"]) == (
        2.0,
        [1.25, 1.5],
        False,
    )
