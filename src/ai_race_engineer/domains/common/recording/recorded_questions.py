import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path


class RecordedQuestions:
    """The questions the driver asked in a recorded run, asked again in a replay at the moments they were asked.
    Read from the run's log stream "state" (<run>/engineer/state.jsonl, records with reason "question").
    _questions  list[tuple[float, str]]  (s unix time at which it was asked, question), oldest first
    _now_unix   Callable[[], float]      s unix time of the recording at the replay time
    _poll_s     float                    s between two looks at the clock
    _next       int                      index of the question asked next
    """

    def __init__(
        self, state_file: Path, first_wall_s: float, last_wall_s: float, now_unix: Callable[[], float], poll_s: float
    ) -> None:
        """Read the questions asked while the recording ran; a file that cannot be read raises OSError, a line that
        is not JSON (cut off by a hard kill) is left out.
        In:  state_file    Path                 the "state" stream of the recorded run
             first_wall_s  float                s unix time of the recording's first row
             last_wall_s   float                s unix time of the recording's last row
             now_unix      Callable[[], float]  s unix time of the recording at the replay time
             poll_s        float                s between two looks at the clock
        Out:               None
        """
        lines = state_file.read_text(encoding="utf-8").splitlines()
        self._questions: list[tuple[float, str]] = questions_within(_records(lines), first_wall_s, last_wall_s)
        self._now_unix: Callable[[], float] = now_unix
        self._poll_s: float = poll_s
        self._next: int = 0

    def __call__(self) -> str:
        """The next recorded question; blocks until the replay reaches the moment it was asked, and for good once
        every question has been asked.
        In:  -
        Out: str  the question
        """
        if self._next == len(self._questions):
            threading.Event().wait()
        wall_s, question = self._questions[self._next]
        self._next += 1
        while self._now_unix() < wall_s:
            time.sleep(self._poll_s)
        return question


def questions_within(
    records: Iterable[Mapping[str, object]], first_wall_s: float, last_wall_s: float
) -> list[tuple[float, str]]:
    """The questions among the records of a "state" stream that were asked within a time range.
    In:  records       Iterable[Mapping[str, object]]  a question has reason "question", wall_s and question
         first_wall_s  float                           s unix time, start of the range (included)
         last_wall_s   float                           s unix time, end of the range (included)
    Out:               list[tuple[float, str]]         (s unix time, question), oldest first, in the order logged
                                                       when asked at the same time
    """
    asked = [r for r in records if r.get("reason") == "question" and first_wall_s <= r["wall_s"] <= last_wall_s]
    return [(r["wall_s"], r["question"]) for r in sorted(asked, key=lambda r: r["wall_s"])]


def _records(lines: Iterable[str]) -> list[dict[str, object]]:
    """The records of JSON lines.
    In:  lines  Iterable[str]            one JSON object per line
    Out:        list[dict[str, object]]  the lines that are not JSON are left out
    """
    records: list[dict[str, object]] = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records
