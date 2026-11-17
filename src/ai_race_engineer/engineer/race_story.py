import queue

from ai_race_engineer.engineer.race_notes import RaceNoteWriter
from ai_race_engineer.domains.common.callouts.base import Callout, Priority
from ai_race_engineer.domains.common.store.timeline_recorder import TimelineRecorder
from ai_race_engineer.domains.common.analysis.session.lap_history import LapHistory
from ai_race_engineer.domains.common.analysis.field.race_state import RaceTracker
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot

FINISH: None = None
STORY: str = "story"
SUMMARY_KINDS: tuple[str, ...] = ("session", "flag", "position", "pit", "plan", "note", "incident")


class RaceStory:
    """Keeps the story of the driver's race: after every lap of a race the facts of the lap go to the note writer,
    whose notes join the session timeline as "story" entries, and at the flag the race is summed up and spoken.
    The telemetry thread only queues the facts; the model is called in this block's own thread.
    _history   LapHistory                            the facts of every finished race lap
    _tracker   RaceTracker                           tells when the race is over for the player
    _timeline  TimelineRecorder                      the session timeline the notes are added to
    _writer    RaceNoteWriter
    _callouts  queue.Queue[Callout]                  output queue of the spoken summary
    _work      queue.Queue[dict[str, object] | None]  lap facts waiting for the writer; None = the race is over
    _queued    int                                   laps of the history already queued
    _summed    bool                                  the summary of this race was asked for
    """

    def __init__(
        self,
        history: LapHistory,
        tracker: RaceTracker,
        timeline: TimelineRecorder,
        writer: RaceNoteWriter,
        callouts: queue.Queue[Callout],
    ) -> None:
        """Store the blocks to wire together.
        In:  history   LapHistory
             tracker   RaceTracker
             timeline  TimelineRecorder
             writer    RaceNoteWriter
             callouts  queue.Queue[Callout]  output queue
        Out:           None
        """
        self._history: LapHistory = history
        self._tracker: RaceTracker = tracker
        self._timeline: TimelineRecorder = timeline
        self._writer: RaceNoteWriter = writer
        self._callouts: queue.Queue[Callout] = callouts
        self._work: queue.Queue[dict[str, object] | None] = queue.Queue()
        self._queued: int = 0
        self._summed: bool = False

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer, runs after the lap history): queue the facts of a lap that was just
        added to the history, and the summary once the race is over for the player.
        In:  snapshot  RaceSnapshot  (unused: the facts come from the history and the tracker)
        Out:           None
        """
        laps = self._history.laps
        if len(laps) < self._queued:
            self._queued, self._summed = 0, False
        for digest in laps[self._queued :]:
            self._work.put(digest)
        self._queued = len(laps)
        if laps and self._tracker.finished and not self._summed:
            self._summed = True
            self._work.put(FINISH)

    def run(self) -> None:
        """Write notes and the summary as the laps come in (run in a daemon thread).
        In:  -
        Out: None
        """
        while True:
            self.step()

    def step(self) -> None:
        """Handle the next queued lap or the end of the race; blocks until there is one.
        In:  -
        Out: None
        """
        work = self._work.get()
        entries = self._timeline.entries()
        story = [entry.text for entry in entries if entry.kind == STORY]
        if work is FINISH:
            timeline = [f"lap {e.lap}: {e.text}" for e in entries if e.kind in SUMMARY_KINDS]
            text = self._writer.summary(story, timeline, self._history.laps)
            if text:
                self._timeline.add(STORY, text, {"event": "summary"})
                self._callouts.put(Callout("race_summary", text, Priority.STRATEGY, {}))
            return
        for note in self._writer.notes(work, story):
            prefix = f"Lap {work['lap']}: "
            self._timeline.add(STORY, prefix + note.removeprefix(prefix), {"event": "note"})
