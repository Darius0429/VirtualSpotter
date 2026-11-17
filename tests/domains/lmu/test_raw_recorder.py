from ai_race_engineer.domains.lmu.raw_recorder import Decision, SessionGate, session_name

QUALIFY: tuple[int, str] = (5, "Bahrain")
RACE: tuple[int, str] = (10, "Bahrain")


def _opened(key: tuple[int, str], start_et: float) -> SessionGate:
    """A gate with an open folder: the session was seen at start_et and its clock ran 0.2 s.
    In:  key       tuple[int, str]  (session number, track name)
         start_et  float            s session time
    Out:           SessionGate      one scoring tick recorded
    """
    gate = SessionGate(finish_grace_s=5.0, restart_back_s=1.0)
    gate.update(key, start_et, 0.0)
    gate.update(key, start_et + 0.2, 0.2)
    gate.tick(start_et + 0.2)
    return gate


def test_folder_is_only_opened_once_the_session_clock_runs() -> None:
    """A session whose clock stands still (results screen, loading) is not recorded; the first poll with a moved
    clock opens the folder and the polls after it are recorded."""
    gate = SessionGate(finish_grace_s=5.0, restart_back_s=1.0)
    assert gate.update(QUALIFY, 30.0, 0.0) is Decision.SKIP
    assert gate.update(QUALIFY, 30.0, 0.1) is Decision.SKIP
    assert gate.update(QUALIFY, 30.2, 0.2) is Decision.OPEN
    assert gate.update(QUALIFY, 30.2, 0.3) is Decision.RECORD


def test_new_session_closes_and_waits_for_its_clock() -> None:
    """A new session number closes the folder; the next one is opened when the new session's clock runs."""
    gate = _opened(QUALIFY, 30.0)
    assert gate.update(RACE, 0.0, 1.0) is Decision.CLOSE
    assert gate.update(RACE, 0.0, 1.1) is Decision.SKIP
    assert gate.update(RACE, 0.2, 1.2) is Decision.OPEN


def test_new_track_closes() -> None:
    """The same session number on another track is another session."""
    gate = _opened(QUALIFY, 30.0)
    assert gate.update((5, "Monza"), 30.4, 1.0) is Decision.CLOSE


def test_clock_jumping_back_is_a_restart() -> None:
    """The session clock jumping back by more than a second closes the folder and a new one is opened when the
    clock runs again; a clock that only stands still or steps back a little goes on in the same folder."""
    gate = _opened(RACE, 100.0)
    assert gate.update(RACE, 100.2, 1.0) is Decision.RECORD
    assert gate.update(RACE, 99.8, 1.1) is Decision.RECORD
    assert gate.update(RACE, 0.0, 1.2) is Decision.CLOSE
    assert gate.update(RACE, 0.0, 1.3) is Decision.SKIP
    assert gate.update(RACE, 0.2, 1.4) is Decision.OPEN


def test_finish_closes_after_the_grace_time_and_nothing_more_is_recorded() -> None:
    """The folder stays open finish_grace_s after the player was first seen finished, then it is closed and stays
    closed while the same session goes on."""
    gate = _opened(RACE, 100.0)
    gate.finish(10.0)
    gate.finish(12.0)
    assert gate.update(RACE, 110.0, 14.9) is Decision.RECORD
    assert gate.update(RACE, 110.2, 15.0) is Decision.FINISH
    assert gate.update(RACE, 110.4, 15.2) is Decision.SKIP
    assert gate.update(RACE, 500.0, 400.0) is Decision.SKIP


def test_restart_or_next_session_after_the_finish_is_recorded_again() -> None:
    """After the finish a restart of the session, or the next session, opens a new folder once its clock runs."""
    gate = _opened(RACE, 100.0)
    gate.finish(10.0)
    gate.update(RACE, 110.0, 15.0)
    assert gate.update(RACE, 0.0, 20.0) is Decision.SKIP
    assert gate.update(RACE, 0.2, 20.2) is Decision.OPEN
    gate.finish(30.0)
    assert gate.update(RACE, 20.0, 35.0) is Decision.FINISH
    assert gate.update(QUALIFY, 0.0, 40.0) is Decision.SKIP
    assert gate.update(QUALIFY, 0.2, 40.2) is Decision.OPEN


def test_scoring_tick_is_new_when_the_session_clock_moved() -> None:
    """The first poll of a folder is a scoring tick; after it only polls with another session time are."""
    gate = SessionGate(finish_grace_s=5.0, restart_back_s=1.0)
    gate.update(QUALIFY, 30.0, 0.0)
    gate.update(QUALIFY, 30.2, 0.2)
    assert gate.tick(30.2) is True
    assert gate.tick(30.2) is False
    assert gate.tick(30.4) is True


def test_session_names() -> None:
    """The game's session numbers have names; an unknown number is named after itself."""
    assert [session_name(number) for number in (0, 1, 5, 9, 10, 13)] == [
        "testday",
        "practice1",
        "qualify1",
        "warmup",
        "race1",
        "race4",
    ]
    assert session_name(14) == "session14"
