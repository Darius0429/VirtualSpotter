from ai_race_engineer.domains.base import FINISH, Domain
from ai_race_engineer.engineer.race_notes import RaceNoteWriter

STORY: str = "story"
SUMMARY_KINDS: tuple[str, ...] = ("session", "flag", "position", "pit", "plan", "note", "incident")


class RaceStory:
    """Keeps the story of the driver's race: after every lap of a race the facts of the lap go to the note writer,
    whose notes join the session timeline as "story" entries, and at the flag the race is summed up and spoken.
    The domain only queues the facts; the model is called in this block's own thread.
    _domain  Domain          gives the laps and the finish, holds the session timeline and speaks the summary
    _writer  RaceNoteWriter
    """

    def __init__(self, domain: Domain, writer: RaceNoteWriter) -> None:
        """Store the blocks to wire together.
        In:  domain  Domain
             writer  RaceNoteWriter
        Out:         None
        """
        self._domain: Domain = domain
        self._writer: RaceNoteWriter = writer

    def run(self) -> None:
        """Write notes and the summary as the laps come in (run in a daemon thread).
        In:  -
        Out: None
        """
        while True:
            self.step()

    def step(self) -> None:
        """Handle the next finished lap or the end of the race; blocks until there is one.
        In:  -
        Out: None
        """
        work = self._domain.wait_happening()
        entries = self._domain.happenings()
        story = [entry.text for entry in entries if entry.kind == STORY]
        if work.kind == FINISH:
            timeline = [f"lap {e.lap}: {e.text}" for e in entries if e.kind in SUMMARY_KINDS]
            text = self._writer.summary(story, timeline, work.facts["laps"])
            if text:
                self._domain.record(STORY, text, {"event": "summary"})
                self._domain.say("race_summary", text, False)
            return
        for note in self._writer.notes(dict(work.facts), story):
            prefix = f"Lap {work.facts['lap']}: "
            self._domain.record(STORY, prefix + note.removeprefix(prefix), {"event": "note"})
