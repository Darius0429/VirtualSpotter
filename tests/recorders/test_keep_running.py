import logging
import threading

import pytest

from ai_race_engineer.recorders.keep_running import keep_running


def test_failed_target_is_started_again_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A target that raises twice is called a third time; both exceptions are logged with their traceback."""
    calls: list[int] = []

    def target() -> None:
        """Fail on the first two calls.
        In:  -
        Out: None
        """
        calls.append(len(calls))
        if len(calls) <= 2:
            raise ValueError(f"failure {len(calls)}")

    with caplog.at_level(logging.ERROR):
        keep_running(target, 0.0, threading.Event())()
    assert calls == [0, 1, 2]
    assert [r.exc_info[0] for r in caplog.records] == [ValueError, ValueError]
    assert "failure 1" in caplog.text and "failure 2" in caplog.text


def test_failure_during_shutdown_is_quiet_and_final(caplog: pytest.LogCaptureFixture) -> None:
    """Once shutdown has begun a failing target is neither logged nor started again."""
    calls: list[int] = []
    stopping = threading.Event()

    def target() -> None:
        """Fail on every call.
        In:  -
        Out: None
        """
        calls.append(len(calls))
        raise OSError("source closed")

    stopping.set()
    with caplog.at_level(logging.ERROR):
        keep_running(target, 0.0, stopping)()
    assert (calls, caplog.records) == ([0], [])
