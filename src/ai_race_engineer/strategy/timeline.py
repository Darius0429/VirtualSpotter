import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

type Fields = dict[str, str | float | int | bool | None]

STAMP_FORMAT: str = "%Y%m%d_%H%M%S"
TIMELINE_SUFFIX: str = ".timeline.jsonl"


@dataclass(frozen=True)
class TimelineEntry:
    """One event of the session.
    time_s  float   s session time
    lap     int     the player's lap in progress, 1-based (completed laps + 1)
    kind    str     "session" | "flag" | "position" | "pit" | "best_lap" | "rival" | "plan" | "note" | "question" |
                    "story"
    text    str     short sentence the agent can quote
    fields  Fields  structured values, units in the key suffix; "event" names the sub-kind
    """

    time_s: float
    lap: int
    kind: str
    text: str
    fields: Fields


def select_entries(
    entries: Sequence[TimelineEntry], from_lap: int, to_lap: int, kind: str, max_entries: int
) -> tuple[tuple[TimelineEntry, ...], int]:
    """Entries of a lap range and kind, the newest max_entries kept in time order.
    In:  entries      Sequence[TimelineEntry]  oldest first
         from_lap     int                      first lap, inclusive; 0 = from the start
         to_lap       int                      last lap, inclusive; 0 = to the end
         kind         str                      entry kind; "" = all
         max_entries  int                      entries returned at most
    Out:              tuple[tuple[TimelineEntry, ...], int]  (entries, number of older matching entries left out)
    """
    matching = [
        e for e in entries if e.lap >= from_lap and (to_lap == 0 or e.lap <= to_lap) and (kind == "" or e.kind == kind)
    ]
    left_out = max(0, len(matching) - max_entries)
    return tuple(matching[left_out:]), left_out


def file_name(kind: str, started_unix: float) -> str:
    """Name of a session's timeline file.
    In:  kind          str    "practice" | "qualify" | "race" | ...
         started_unix  float  s unix time the session started
    Out:               str    "<YYYYmmdd_HHMMSS>_<kind>.timeline.jsonl", the stamp in local time
    """
    return f"{datetime.fromtimestamp(started_unix).strftime(STAMP_FORMAT)}_{kind}{TIMELINE_SUFFIX}"


def matching_file(names: Sequence[str], kind: str, started_unix: float, margin_s: float) -> str | None:
    """The timeline file of the same session among the files of a folder.
    In:  names         Sequence[str]  file names
         kind          str            "practice" | "qualify" | "race" | ...
         started_unix  float          s unix time the session started
         margin_s      float          s the start stamps may differ
    Out:               str | None     the name of the same kind whose stamp is nearest within the margin; None
                                      without one
    """
    stamps = _stamps(names, kind)
    near = {name: abs(stamp - started_unix) for name, stamp in stamps.items() if abs(stamp - started_unix) <= margin_s}
    return min(near, key=near.get, default=None)


def files_in_window(names: Sequence[str], kind: str, from_unix: float, to_unix: float) -> list[str]:
    """The timeline files of a kind stamped within a time window.
    In:  names      Sequence[str]  file names
         kind       str            "practice" | "qualify" | "race" | ...
         from_unix  float          s unix time, earliest stamp
         to_unix    float          s unix time, latest stamp
    Out:            list[str]      names, newest stamp first
    """
    stamps = _stamps(names, kind)
    return sorted(
        (name for name, stamp in stamps.items() if from_unix <= stamp <= to_unix), key=stamps.get, reverse=True
    )


def _stamps(names: Sequence[str], kind: str) -> dict[str, float]:
    """Start stamps of the timeline files of a kind.
    In:  names  Sequence[str]  file names
         kind   str            "practice" | "qualify" | "race" | ...
    Out:        dict[str, float]  name -> s unix time of its stamp (local time), for the names that are timeline
                                  files of the kind
    """
    pattern = re.compile(rf"(\d{{8}}_\d{{6}})_{re.escape(kind)}{re.escape(TIMELINE_SUFFIX)}")
    return {
        m.group(0): datetime.strptime(m.group(1), STAMP_FORMAT).timestamp() for m in map(pattern.fullmatch, names) if m
    }
