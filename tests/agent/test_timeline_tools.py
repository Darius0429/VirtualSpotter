from ai_race_engineer.agent.timeline_tools import TimelineTools
from ai_race_engineer.strategy.session_log import Note
from ai_race_engineer.strategy.timeline import TimelineEntry


class _Tracker:
    """Stand-in for the race tracker without a state.
    state     None
    finished  bool
    """

    def __init__(self) -> None:
        """Start without a state.
        In:  -
        Out: None
        """
        self.state: None = None
        self.finished: bool = False


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

    def add_note(self, text: str) -> Note:
        """Keep a note on lap 3.
        In:  text  str
        Out:       Note
        """
        self.notes.append(Note(250.0, 3, text, False))
        return self.notes[-1]


class _Timeline:
    """Stand-in for the timeline recorder.
    stored  list[TimelineEntry]
    """

    def __init__(self, stored: list[TimelineEntry]) -> None:
        """Hold the entries.
        In:  stored  list[TimelineEntry]
        Out:         None
        """
        self.stored: list[TimelineEntry] = stored

    def entries(self) -> tuple[TimelineEntry, ...]:
        """The entries held.
        In:  -
        Out: tuple[TimelineEntry, ...]
        """
        return tuple(self.stored)

    def add(self, kind: str, text: str, fields: dict[str, object]) -> TimelineEntry:
        """Keep an entry on lap 3.
        In:  kind    str
             text    str
             fields  dict[str, object]
        Out:         TimelineEntry
        """
        self.stored.append(TimelineEntry(250.0, 3, kind, text, dict(fields)))
        return self.stored[-1]


def _tools(stored: list[TimelineEntry], session_log: _SessionLog | None = None) -> TimelineTools:
    """Timeline tools on stand-ins, returning at most 2 entries.
    In:  stored       list[TimelineEntry]  the timeline's entries
         session_log  _SessionLog | None
    Out:              TimelineTools
    """
    return TimelineTools(
        _Tracker(), None, 3.0, None, session_log or _SessionLog(), None, None, _Timeline(stored), 2, 0.05, 5, 0.01
    )


def _stored() -> list[TimelineEntry]:
    """Three entries on laps 1, 4 and 6.
    In:  -
    Out: list[TimelineEntry]
    """
    return [
        TimelineEntry(5.0, 1, "session", "Green flag, race started from P5 in class", {"event": "green"}),
        TimelineEntry(400.0, 4, "pit", "driver3 pits from P4", {"event": "in", "driver": "driver3"}),
        TimelineEntry(650.0, 6, "position", "Passed driver3 for P4 in class, on track", {"event": "swap"}),
    ]


def test_timeline_returns_text_entries_without_fields() -> None:
    """Each entry has the lap, the time, the kind and the text; the structured fields stay out."""
    assert _tools(_stored()).timeline(from_lap=4, to_lap=4) == {
        "entries": [{"lap": 4, "time_s": 400.0, "kind": "pit", "text": "driver3 pits from P4"}],
        "older_entries_left_out": 0,
    }
    assert _tools(_stored()).timeline(kind="position")["entries"][0]["lap"] == 6
    assert _tools([]).timeline() == {"entries": [], "older_entries_left_out": 0}


def test_timeline_is_capped_at_the_newest_entries() -> None:
    """More matches than the cap: the newest are returned and the rest counted."""
    result = _tools(_stored()).timeline()
    assert [e["lap"] for e in result["entries"]] == [4, 6]
    assert result["older_entries_left_out"] == 1


def test_add_note_reaches_the_session_log_and_the_timeline() -> None:
    """A note is stored in the session log, confirmed with its lap, and added to the timeline as a note entry."""
    stored: list[TimelineEntry] = []
    session_log = _SessionLog()
    assert _tools(stored, session_log).add_note("understeer in turn 4") == "Noted on lap 3."
    assert [note.text for note in session_log.notes] == ["understeer in turn 4"]
    assert [(e.kind, e.text, e.fields) for e in stored] == [("note", "understeer in turn 4", {})]


def test_timeline_replaces_notes_in_the_fifteen_tools() -> None:
    """The agent is offered the timeline instead of the notes list; the number of tools stays fifteen."""
    names = [tool.__name__ for tool in _tools([]).tools()]
    assert len(names) == len(set(names)) == 15
    assert "notes" not in names
    assert names[-1] == "timeline"
    assert "add_note" in names
