from ai_race_engineer.domains.common.callouts.base import Callout, Priority
from ai_race_engineer.domains.common.voice.speech_gate import SpeechGate


def _gate() -> SpeechGate:
    """Gate that speaks spotter callouts at once and drops waiting ones after 15 s.
    In:  -
    Out: SpeechGate
    """
    return SpeechGate(Priority.SPOTTER, 15.0)


def _callout(kind: str, priority: Priority = Priority.STRATEGY, slot_id: int | None = None, text: str = "") -> Callout:
    """Callout of a kind, optionally about a car.
    In:  kind      str
         priority  Priority
         slot_id   int | None  car the callout is about
         text      str         default: the kind
    Out:           Callout
    """
    return Callout(kind, text or kind, priority, {} if slot_id is None else {"slot_id": slot_id})


def test_empty_gate_gives_nothing() -> None:
    """Nothing pushed, nothing to speak."""
    assert _gate().pop(True, 0.0) is None


def test_urgent_bypasses_the_straight() -> None:
    """A callout at the urgent priority is spoken in a corner too."""
    gate = _gate()
    gate.push(_callout("spotter", Priority.SPOTTER), 0.0)
    assert gate.pop(False, 0.1).kind == "spotter"
    assert gate.pop(False, 0.2) is None


def test_non_urgent_waits_for_a_straight() -> None:
    """A strategy callout is held in a corner and spoken on the next straight, once."""
    gate = _gate()
    gate.push(_callout("low_fuel"), 0.0)
    assert gate.pop(False, 1.0) is None
    assert gate.pop(True, 2.0).kind == "low_fuel"
    assert gate.pop(True, 3.0) is None


def test_urgent_goes_before_waiting_callouts_oldest_first() -> None:
    """On a straight the urgent callouts still come first, in the order they arrived."""
    gate = _gate()
    gate.push(_callout("low_fuel"), 0.0)
    gate.push(_callout("spotter", Priority.SPOTTER), 1.0)
    gate.push(_callout("answer", Priority.SPOTTER), 2.0)
    assert [gate.pop(True, 3.0).kind for _ in range(3)] == ["spotter", "answer", "low_fuel"]


def test_waiting_callout_expires() -> None:
    """A non-urgent callout older than max_age_s is dropped; at exactly that age it is still spoken."""
    gate = _gate()
    gate.push(_callout("low_fuel"), 0.0)
    assert gate.pop(False, 15.1) is None
    assert gate.pop(True, 15.2) is None
    gate.push(_callout("low_fuel"), 20.0)
    assert gate.pop(True, 35.0).kind == "low_fuel"


def test_urgent_callout_does_not_expire() -> None:
    """Urgent callouts are spoken however long the previous sentence took."""
    gate = _gate()
    gate.push(_callout("spotter", Priority.SPOTTER), 0.0)
    assert gate.pop(False, 60.0).kind == "spotter"


def test_same_kind_and_slot_is_replaced() -> None:
    """A newer callout of the same kind about the same car replaces the waiting one and takes the new time."""
    gate = _gate()
    gate.push(_callout("rival_pace", slot_id=7, text="old"), 0.0)
    gate.push(_callout("rival_pace", slot_id=7, text="new"), 10.0)
    assert gate.pop(True, 20.0).text == "new"
    assert gate.pop(True, 20.0) is None


def test_replaced_urgent_callout() -> None:
    """A sector clearing replaces the yellow of that sector that was not yet spoken."""
    gate = _gate()
    gate.push(_callout("yellow_sector_2", Priority.SPOTTER, text="Yellow flag, sector 2"), 0.0)
    gate.push(_callout("spotter", Priority.SPOTTER), 0.5)
    gate.push(_callout("yellow_sector_2", Priority.SPOTTER, text="Sector 2 clear"), 1.0)
    assert [gate.pop(False, 2.0).text for _ in range(2)] == ["spotter", "Sector 2 clear"]
    assert gate.pop(False, 2.0) is None


def test_different_slots_are_kept() -> None:
    """Callouts of the same kind about different cars both stay, oldest first."""
    gate = _gate()
    gate.push(_callout("rival_pit", slot_id=7), 0.0)
    gate.push(_callout("rival_pit", slot_id=8), 1.0)
    assert [gate.pop(True, 2.0).facts["slot_id"] for _ in range(2)] == [7, 8]


def test_priority_order_then_age() -> None:
    """On a straight the highest priority goes first, the oldest first within a priority."""
    gate = _gate()
    gate.push(_callout("fastest_lap", Priority.INFO), 0.0)
    gate.push(_callout("low_fuel", Priority.STRATEGY), 1.0)
    gate.push(_callout("low_energy", Priority.STRATEGY), 2.0)
    assert [gate.pop(True, 3.0).kind for _ in range(3)] == ["low_fuel", "low_energy", "fastest_lap"]


def test_waited_time_of_the_last_callout() -> None:
    """The gate tells how long the callout it just released had waited."""
    gate = _gate()
    gate.push(_callout("low_fuel"), 10.0)
    gate.push(_callout("spotter", Priority.SPOTTER), 12.0)
    gate.pop(False, 12.5)
    assert gate.waited_s() == 0.5
    gate.pop(True, 16.0)
    assert gate.waited_s() == 6.0
