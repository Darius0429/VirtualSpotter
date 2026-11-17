from collections.abc import Callable

from ai_race_engineer.domains.common.tools.plan_tools import PlanTools
from ai_race_engineer.domains.common.callouts.grid_briefing import GridBriefing
from ai_race_engineer.domains.common.callouts.rival_watch import RivalWatch
from ai_race_engineer.domains.common.store.session_keeper import SessionKeeper
from ai_race_engineer.domains.common.store.timeline_recorder import TimelineRecorder
from ai_race_engineer.domains.common.analysis.field.prior_field import PriorFieldTracker
from ai_race_engineer.domains.common.analysis.field.race_state import RaceTracker
from ai_race_engineer.domains.common.analysis.session.session_log import SessionLog
from ai_race_engineer.domains.common.analysis.session.timeline import select_entries


class TimelineTools(PlanTools):
    """The plan tools with the session timeline in place of the notes list: what happened earlier in the session.
    _timeline     TimelineRecorder  the entries of the running session
    _max_entries  int               entries the timeline tool returns at most
    """

    def __init__(
        self,
        tracker: RaceTracker,
        rival_watch: RivalWatch,
        traffic_horizon_laps: float,
        field: PriorFieldTracker,
        session_log: SessionLog,
        keeper: SessionKeeper,
        briefing: GridBriefing,
        timeline: TimelineRecorder,
        max_entries: int,
        max_saving_fraction: float,
        min_laps: int,
        min_spread: float,
    ) -> None:
        """Store the sources, the timeline and the saving parameters.
        In:  tracker               RaceTracker
             rival_watch           RivalWatch
             traffic_horizon_laps  float              laps within which other-class cars are reported as traffic
             field                 PriorFieldTracker
             session_log           SessionLog
             keeper                SessionKeeper
             briefing              GridBriefing
             timeline              TimelineRecorder
             max_entries           int                entries the timeline tool returns at most (newest kept)
             max_saving_fraction   float              share of the usage per lap that is realistic to save
             min_laps              int                clean laps needed before a cost is estimated
             min_spread            float              the lowest lap must use this share less than the mean before a
                                                      cost is estimated
        Out:                       None
        """
        super().__init__(
            tracker,
            rival_watch,
            traffic_horizon_laps,
            field,
            session_log,
            keeper,
            briefing,
            max_saving_fraction,
            min_laps,
            min_spread,
        )
        self._timeline: TimelineRecorder = timeline
        self._max_entries: int = max_entries

    def timeline(self, from_lap: int = 0, to_lap: int = 0, kind: str = "") -> dict[str, object]:
        """What happened earlier in this session and when, oldest first: session start, green flag and finish
        ("session"), yellow flags and full course yellows ("flag"), places gained and lost in class and how
        ("position"), pit stops of the driver and the rivals with the predicted and the actual rejoin ("pit"),
        fastest laps in class and personal bests ("best_lap"), who became or stopped being a rival ("rival"), the
        race plan and its changes ("plan"), the driver's notes ("note"), earlier questions with their answers
        ("question") and the engineer's own notes of what mattered in the driver's race, lap by lap, with the
        summary at the flag ("story"), contact between cars or a car going off on its own ("incident"), the
        driver's track-limit offences with the place and the points against the limit ("track_limits") and changes
        of the number of outstanding penalties ("penalty").
        For how the race went, read the "story" entries first.
        Laps are counted as the driver counts them. Give a lap range and a kind to narrow it down;
        when more entries match than fit, the newest are returned and the number of older ones left out is given.
        In:  from_lap  int  first lap of the range; 0 = from the start
             to_lap    int  last lap of the range; 0 = to the end
             kind      str  "session" | "flag" | "position" | "pit" | "best_lap" | "rival" | "plan" | "note" |
                            "question" | "story" | "incident" | "track_limits" | "penalty"; empty = all
        Out:           dict[str, object]  {"entries": [{"lap", "time_s", "kind", "text"}],
                                          "older_entries_left_out": int}
        """
        entries, left_out = select_entries(self._timeline.entries(), from_lap, to_lap, kind, self._max_entries)
        return {
            "entries": [{"lap": e.lap, "time_s": e.time_s, "kind": e.kind, "text": e.text} for e in entries],
            "older_entries_left_out": left_out,
        }

    def add_note(self, text: str) -> str:
        """Store a note of the driver with the session time and the lap; it is kept in the session file and in the
        timeline.
        In:  text  str  the note
        Out:       str  confirmation sentence
        """
        confirmation = super().add_note(text)
        self._timeline.add("note", text, {})
        return confirmation

    def tools(self) -> list[Callable[..., object]]:
        """The functions an agent may call.
        In:  -
        Out: list[Callable[..., object]]  the plan tools without notes, plus timeline
        """
        return [*(tool for tool in super().tools() if tool.__name__ != "notes"), self.timeline]
