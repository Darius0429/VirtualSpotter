import os
from pathlib import Path

from ai_race_engineer.domains.common.recording.run_folder import RunFolder, run_to_join

LIVE: frozenset[int] = frozenset({10, 11})


def _alive(pid: int) -> bool:
    """Fake liveness: only the processes of LIVE run.
    In:  pid  int  process id
    Out:      bool
    """
    return pid in LIVE


def test_engineer_joins_the_newest_run_of_a_live_recorder() -> None:
    """A program joins the newest run folder when the other kind is alive in it."""
    runs = {"20261117_190000": {"recorder": 10}, "20261117_203000": {"recorder": 11}}
    assert run_to_join(runs, "engineer", "recorder", _alive) == "20261117_203000"
    assert run_to_join({"20261117_203000": {"engineer": 10}}, "recorder", "engineer", _alive) == "20261117_203000"


def test_a_dead_lock_of_the_other_kind_is_not_joined() -> None:
    """A lock left by a killed program does not count."""
    assert run_to_join({"20261117_203000": {"recorder": 99}}, "engineer", "recorder", _alive) is None


def test_a_run_with_a_live_program_of_the_own_kind_is_not_joined() -> None:
    """A second engineer gets its own run folder; after the first one ended (or was killed) the next one joins."""
    assert run_to_join({"20261117_203000": {"recorder": 10, "engineer": 11}}, "engineer", "recorder", _alive) is None
    assert run_to_join({"20261117_203000": {"recorder": 10, "engineer": 99}}, "engineer", "recorder", _alive) == (
        "20261117_203000"
    )


def test_newest_run_with_a_live_program_of_the_other_kind_is_joined() -> None:
    """A newer run folder without a recorder (a replay) does not hide the live recorder of an older one; of two
    live recorders the newer is joined."""
    runs = {"20261117_190000": {"recorder": 10}, "20261117_203000": {}}
    assert run_to_join(runs, "engineer", "recorder", _alive) == "20261117_190000"
    runs = {"20261117_190000": {"recorder": 10}, "20261117_203000": {"recorder": 11}}
    assert run_to_join(runs, "engineer", "recorder", _alive) == "20261117_203000"


def test_a_replay_and_a_first_start_never_join() -> None:
    """With no kind to join (replay) or no run folder yet, a new run folder is created."""
    assert run_to_join({"20261117_203000": {"recorder": 10}}, "engineer", None, _alive) is None
    assert run_to_join({}, "engineer", "recorder", _alive) is None


def test_run_folder_writes_its_lock_is_joined_and_removes_the_lock(tmp_path: Path) -> None:
    """The first program creates the run folder with its lock (this process id), the other kind joins it while the
    first is alive, and closing removes only the own lock."""
    with RunFolder(tmp_path, "recorder", "engineer") as recorder:
        assert (recorder.path / "recorder.lock").read_text().split()[0] == str(os.getpid())
        with RunFolder(tmp_path, "engineer", "recorder") as engineer:
            assert engineer.path == recorder.path
            assert (engineer.path / "engineer.lock").exists()
        assert sorted(file.name for file in recorder.path.iterdir()) == ["recorder.lock"]
    assert list(recorder.path.iterdir()) == []
    assert [folder.name for folder in tmp_path.iterdir()] == [recorder.path.name]
