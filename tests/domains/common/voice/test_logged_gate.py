from collections.abc import Mapping

import pytest

from ai_race_engineer.domains.common.callouts.base import Callout, Priority
from ai_race_engineer.domains.common.voice.logged_gate import LoggedSpeechGate


class _Log:
    """Stands in for the run log: keeps what is written.
    records  list[tuple[str, Mapping[str, object]]]  (stream, record) in order
    """

    def __init__(self) -> None:
        """Start empty.
        In:  -
        Out: None
        """
        self.records: list[tuple[str, Mapping[str, object]]] = []

    def write(self, stream: str, record: Mapping[str, object]) -> None:
        """Keep one record.
        In:  stream  str
             record  Mapping[str, object]
        Out:         None
        """
        self.records.append((stream, record))


def _callout(kind: str, text: str, priority: Priority = Priority.STRATEGY) -> Callout:
    """Callout about car 7.
    In:  kind      str
         text      str
         priority  Priority
    Out:           Callout
    """
    return Callout(kind, text, priority, {"slot_id": 7})


def test_replaced_callout_is_logged() -> None:
    """A waiting callout replaced by a newer one of the same kind and car is logged with the new text."""
    log = _Log()
    gate = LoggedSpeechGate(log, Priority.SPOTTER, 15.0)
    old = _callout("rival_pace", "old")
    gate.push(old, 0.0)
    gate.push(_callout("rival_pace", "new"), 1.0)
    assert log.records == [("callouts", {"event": "replaced", "callout": old, "by": "new"})]


def test_expired_callout_is_logged() -> None:
    """A callout that waited longer than the limit is logged as expired and not released."""
    log = _Log()
    gate = LoggedSpeechGate(log, Priority.SPOTTER, 15.0)
    old = _callout("rival_pace", "old")
    gate.push(old, 0.0)
    assert gate.pop(True, 16.0) is None
    assert log.records == [("callouts", {"event": "expired", "callout": old, "waited_s": 16.0})]


def test_released_callout_is_logged_with_its_wait() -> None:
    """The callout the gate lets through is logged with the time it waited; the gate itself works as before."""
    log = _Log()
    gate = LoggedSpeechGate(log, Priority.SPOTTER, 15.0)
    waiting, urgent = _callout("rival_pace", "pace"), _callout("spotter", "car left", Priority.SPOTTER)
    gate.push(waiting, 0.0)
    gate.push(urgent, 1.0)
    assert gate.pop(False, 1.5) is urgent
    assert gate.pop(False, 2.0) is None
    assert gate.pop(True, 4.0) is waiting
    assert log.records == [
        ("callouts", {"event": "released", "callout": urgent, "waited_s": pytest.approx(0.5)}),
        ("callouts", {"event": "released", "callout": waiting, "waited_s": pytest.approx(4.0)}),
    ]
