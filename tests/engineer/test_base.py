import pytest

from ai_race_engineer.engineer import base
from ai_race_engineer.engineer.base import answer_with_retries, configure_retries


class _Model:
    """A model round trip that fails a number of times before it answers.
    calls      int  round trips made
    _failures  int  round trips that fail first
    """

    def __init__(self, failures: int) -> None:
        """Set how often it fails.
        In:  failures  int
        Out:           None
        """
        self.calls: int = 0
        self._failures: int = failures

    def ask(self) -> str:
        """One round trip.
        In:  -
        Out: str  "answer"; raises while failures are left
        """
        self.calls += 1
        if self.calls <= self._failures:
            raise ConnectionError("no network")
        return "answer"


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Three tries with 2 s between them and "say again" as the error reply; the waits are recorded, not slept.
    In:  monkeypatch  pytest.MonkeyPatch
    Out:              list[float]  s of every wait
    """
    slept: list[float] = []
    monkeypatch.setattr(base.time, "sleep", slept.append)
    monkeypatch.setattr(base, "_retries", base._retries)
    configure_retries(3, 2.0, "say again")
    return slept


def test_answer_after_failed_tries(waits: list[float]) -> None:
    """Two failures are followed by a wait each and the third try answers."""
    model = _Model(failures=2)
    assert answer_with_retries(model.ask) == "answer"
    assert (model.calls, waits) == (3, [2.0, 2.0])


def test_error_reply_without_a_wait_after_the_last_try(waits: list[float]) -> None:
    """When every try fails the configured reply is given, and there is no wait after the last failure."""
    model = _Model(failures=5)
    assert answer_with_retries(model.ask) == "say again"
    assert (model.calls, waits) == (3, [2.0, 2.0])


def test_no_wait_when_the_first_try_answers(waits: list[float]) -> None:
    """A round trip that works is made once."""
    model = _Model(failures=0)
    assert answer_with_retries(model.ask) == "answer"
    assert (model.calls, waits) == (1, [])
