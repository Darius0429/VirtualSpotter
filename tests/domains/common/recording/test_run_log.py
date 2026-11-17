import json
from pathlib import Path

from ai_race_engineer.domains.common.callouts.base import Callout, Priority
from ai_race_engineer.domains.common.recording.run_log import RunLog
from ai_race_engineer.domains.common.data.snapshot import SessionType


def _lines(folder: Path, stream: str) -> list[dict[str, object]]:
    """Records of one stream of a log folder.
    In:  folder  Path  folder the log was given
         stream  str   file name without extension
    Out:         list[dict[str, object]]
    """
    return [json.loads(line) for line in (folder / f"{stream}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_records_are_stamped_and_written_to_their_stream(tmp_path: Path) -> None:
    """Each record is one JSON line in its stream's file in the given folder (created by the log), with a sequence
    number, wall time and session time."""
    folder = tmp_path / "run" / "engineer"
    log = RunLog(folder, True, lambda: 12.5)
    log.write("callouts", {"event": "emitted"})
    log.write("health", {"gap_s": 0.05})
    log.write("callouts", {"event": "spoken"})
    log.close()
    callouts = _lines(folder, "callouts")
    assert [(r["seq"], r["event"], r["session_s"]) for r in callouts] == [(0, "emitted", 12.5), (2, "spoken", 12.5)]
    assert all(r["wall_s"] > 0 for r in callouts)
    assert _lines(folder, "health")[0]["gap_s"] == 0.05
    assert log.queue_size() == 0


def test_dataclasses_enums_paths_and_messages_serialise(tmp_path: Path) -> None:
    """Dataclasses become objects, enums their names, paths and unknown values text, messages their dict form, a
    number that is not finite null, and keys text."""
    with RunLog(tmp_path, True, lambda: 0.0) as log:
        log.write(
            "state",
            {
                "callout": Callout("spotter", "car left", Priority.SPOTTER, {"slot_id": 7}),
                "session_type": SessionType.RACE,
                "path": Path("data") / "logs",
                "by_slot": {3: (1.5, float("inf"))},
            },
        )
    (record,) = _lines(tmp_path, "state")
    assert record["callout"] == {"kind": "spotter", "text": "car left", "priority": "SPOTTER", "facts": {"slot_id": 7}}
    assert record["session_type"] == "RACE"
    assert record["path"] == str(Path("data") / "logs")
    assert record["by_slot"] == {"3": [1.5, None]}


def test_disabled_log_writes_nothing(tmp_path: Path) -> None:
    """With logging off no record is queued and no folder is made."""
    log = RunLog(tmp_path, False, lambda: 0.0)
    log.write("callouts", {"event": "emitted"})
    log.close()
    assert log.queue_size() == 0
    assert list(tmp_path.iterdir()) == []


def test_record_that_cannot_be_written_does_not_stop_the_log(tmp_path: Path) -> None:
    """A record with text that is not valid Unicode or with a value that contains itself is replaced by a short note
    of the failure; the records after it are still written."""
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    log = RunLog(tmp_path, True, lambda: 1.0)
    log.write("log", {"message": "bad \ud800 text"})
    log.write("log", {"value": cyclic})
    log.write("log", {"message": "fine"})
    log.close()
    first, second, third = _lines(tmp_path, "log")
    assert (first["seq"], "UnicodeEncodeError" in first["unwritable_record"]) == (0, True)
    assert (second["seq"], "RecursionError" in second["unwritable_record"]) == (1, True)
    assert third["message"] == "fine"


def test_writing_is_off_once_the_writer_has_ended(tmp_path: Path) -> None:
    """After the writer thread ends nothing is queued any more."""
    log = RunLog(tmp_path, True, lambda: 1.0)
    log.write("log", {"message": "one"})
    log.close()
    log.write("log", {"message": "two"})
    assert log.queue_size() == 0
    assert len(_lines(tmp_path, "log")) == 1
