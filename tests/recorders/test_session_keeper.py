import dataclasses
import os
import queue
from pathlib import Path

import pytest

from ai_race_engineer.recorders import session_keeper
from ai_race_engineer.recorders.history_store import HistoryStore
from ai_race_engineer.recorders.session_keeper import SessionKeeper
from ai_race_engineer.recorders.session_store import SessionStore
from ai_race_engineer.strategy.pit_time import PitRates
from ai_race_engineer.strategy.session_log import SessionRecord
from ai_race_engineer.telemetry.results import SessionResults
from ai_race_engineer.telemetry.snapshot import VehicleState
from tests.telemetry.test_results import QUALIFY, RACE

PRACTICE: str = QUALIFY.replace("Qualify>", "Practice1>").replace("<DateTime>1200", "<DateTime>1500")
NOW_UNIX: float = 2100.0


class _Log:
    """Stand-in for the session log: gives the record the test sets.
    ended    queue.Queue[SessionRecord]
    running  SessionRecord | None
    """

    def __init__(self, running: SessionRecord | None) -> None:
        """Hold the running session's record.
        In:  running  SessionRecord | None
        Out:          None
        """
        self.ended: queue.Queue[SessionRecord] = queue.Queue()
        self.running: SessionRecord | None = running

    def record(self) -> SessionRecord | None:
        """The running session.
        In:  -
        Out: SessionRecord | None
        """
        return self.running


class _RunLog:
    """Stand-in for the run log: keeps what is written.
    records  list[tuple[str, dict[str, object]]]
    """

    def __init__(self) -> None:
        """Start empty.
        In:  -
        Out: None
        """
        self.records: list[tuple[str, dict[str, object]]] = []

    def write(self, stream: str, record: dict[str, object]) -> None:
        """Keep a record.
        In:  stream  str
             record  dict[str, object]
        Out:         None
        """
        self.records.append((stream, record))


class _Game:
    """What the test changes while the keeper runs: the results file pointed to and the green.
    path   Path | None
    green  bool
    """

    def __init__(self, path: Path | None) -> None:
        """Point to a file, before the green.
        In:  path  Path | None
        Out:       None
        """
        self.path: Path | None = path
        self.green: bool = False


def _record(session_id: str = "s2_race", kind: str = "race", started_unix: float = 2010.0) -> SessionRecord:
    """Record of a session of "Driver One" on "Test Course" without laps.
    In:  session_id    str
         kind          str
         started_unix  float  s unix time
    Out:               SessionRecord
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    player = VehicleState(**{**base, "is_player": True, "driver_name": "Driver One", "class_name": "Hyper", "place": 1})
    return SessionRecord(
        session_id,
        started_unix,
        "Test Course",
        kind,
        "Driver One",
        "Team A 2026 #1:WEC",
        "Hyper",
        50.0,
        False,
        (player,),
        1,
        (),
        (),
        0,
        (),
        {},
        (),
    )


def _keeper(tmp_path: Path, log: _Log, game: _Game) -> SessionKeeper:
    """Keeper on stand-ins that writes below tmp_path and never rewrites the running summary by itself.
    In:  tmp_path  Path
         log       _Log
         game      _Game
    Out:           SessionKeeper
    """
    return SessionKeeper(
        log,
        SessionStore(tmp_path / "sessions"),
        HistoryStore(tmp_path / "history.jsonl"),
        lambda: game.path,
        PitRates(3.0, 0.025, 1.0),
        _RunLog(),
        1e9,
        5,
        "TrackCourse",
        5,
        6.0,
        lambda: NOW_UNIX,
        1.0,
        300.0,
        lambda: game.green,
    )


def _file(folder: Path, name: str, text: str) -> Path:
    """Results file in a folder.
    In:  folder  Path
         name    str   file name
         text    str   XML document
    Out:         Path
    """
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text(text)
    return path


def test_results_file_is_read_again_only_when_it_changed_and_not_after_the_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without qualifying times the pointed file is read once; a new modification time or path reads it again,
    until the green."""
    read: list[Path] = []

    def counted(path: Path) -> SessionResults | None:
        """Read a results file and note it.
        In:  path  Path
        Out:       SessionResults | None
        """
        read.append(path)
        return session_keeper.latest_results(path.parent, "practice")

    monkeypatch.setattr(session_keeper, "read_results", counted)
    practice = _file(tmp_path / "results", "2026_01_01_10_00_00-11P1.xml", PRACTICE)
    game = _Game(practice)
    keeper = _keeper(tmp_path, _Log(_record()), game)
    for _ in range(3):
        keeper.step()
    assert (len(read), keeper.priors().qualifying_best_s) == (1, {})
    os.utime(practice, (5000.0, 5000.0))
    keeper.step()
    keeper.step()
    assert len(read) == 2
    game.path = _file(tmp_path / "results", "2026_01_02_10_00_00-22P1.xml", PRACTICE)
    keeper.step()
    assert len(read) == 3
    game.green = True
    os.utime(game.path, (6000.0, 6000.0))
    keeper.step()
    assert len(read) == 3


def test_qualifying_of_an_earlier_event_is_not_taken(tmp_path: Path) -> None:
    """The newest qualifying file of the folder is used only when it is not older than the pointed file; once the
    game points to the qualifying file itself, that one gives the times."""
    folder = tmp_path / "results"
    _file(folder, "2026_01_01_09_00_00-11Q1.xml", QUALIFY)
    game = _Game(_file(folder, "2026_01_01_10_00_00-22P1.xml", PRACTICE))
    keeper = _keeper(tmp_path, _Log(_record()), game)
    keeper.step()
    assert keeper.priors().qualifying_best_s == {}
    _file(folder, "2026_01_01_11_00_00-33Q1.xml", QUALIFY.replace("<DateTime>1200", "<DateTime>1600"))
    os.utime(game.path, (5000.0, 5000.0))
    keeper.step()
    assert keeper.priors().qualifying_best_s == {"Driver One": 100.5}


def test_pointed_qualifying_file_gives_the_times(tmp_path: Path) -> None:
    """A qualifying file the game points to is this event's."""
    game = _Game(_file(tmp_path / "results", "2026_01_01_09_00_00-11Q1.xml", QUALIFY))
    keeper = _keeper(tmp_path, _Log(_record()), game)
    keeper.step()
    assert keeper.priors().qualifying_best_s == {"Driver One": 100.5}
    assert [e.session_id for e in keeper.entries()] == ["2026_01_01_09_00_00-11Q1"]


def test_own_results_file_goes_into_the_history_only_when_the_session_has_ended(tmp_path: Path) -> None:
    """The file of the running session is not stored while it runs, also not by close; it is stored once when the
    session has ended, and the history is not used for the session's own priors."""
    game = _Game(_file(tmp_path / "results", "2026_01_01_12_00_00-44R1.xml", RACE))
    log = _Log(_record())
    keeper = _keeper(tmp_path, log, game)
    keeper.step()
    keeper.close()
    assert keeper.entries() == ()
    again = _keeper(tmp_path, log, game)
    log.ended.put(dataclasses.replace(_record(), ended=True))
    log.running = _record("s3_practice", "practice", 2050.0)
    again.step()
    again.step()
    assert [e.session_id for e in again.entries()] == ["2026_01_01_12_00_00-44R1"]
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 1
    assert again.priors().car_history is None


def test_step_after_close_does_nothing(tmp_path: Path) -> None:
    """Once closed the keeper writes nothing more, so the final write is the last one."""
    log = _Log(_record())
    keeper = _keeper(tmp_path, log, _Game(None))
    keeper.close()
    written = sorted(p.name for p in (tmp_path / "sessions").iterdir())
    log.running = _record("s9_race")
    keeper.step()
    assert sorted(p.name for p in (tmp_path / "sessions").iterdir()) == written == ["s2_race_Test_Course.yaml"]
    assert keeper.priors() is None


def test_close_stores_the_sessions_that_ended_since_the_last_step(tmp_path: Path) -> None:
    """A session that ended just before the shutdown is stored by close, like the running one."""
    log = _Log(_record("s3_practice", "practice", 2050.0))
    keeper = _keeper(tmp_path, log, _Game(None))
    log.ended.put(dataclasses.replace(_record(), ended=True))
    keeper.close()
    assert log.ended.empty()
    assert sorted(p.name for p in (tmp_path / "sessions").iterdir()) == [
        "s2_race_Test_Course.yaml",
        "s3_practice_Test_Course.yaml",
    ]
