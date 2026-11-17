from ai_race_engineer.domains.common.callouts.base import Callout, Priority

type Key = tuple[str, str | float | int | None]


class SpeechGate:
    """Decides which callout is spoken next: urgent ones at once, the rest on straights while they are fresh.
    _urgent_priority  Priority                          at or above this a callout is urgent
    _max_age_s        float                             s after which a waiting non-urgent callout is dropped
    _urgent           dict[Key, tuple[Callout, float]]  (kind, slot id) -> (callout, s monotonic clock when pushed)
    _waiting          dict[Key, tuple[Callout, float]]  the same for non-urgent callouts
    _waited_s         float                             s the callout returned last had waited
    """

    def __init__(self, urgent_priority: Priority, max_age_s: float) -> None:
        """Set the urgency threshold and the expiry.
        In:  urgent_priority  Priority  at or above this a callout is spoken at once
             max_age_s        float     s after which a waiting non-urgent callout is dropped
        Out:                  None
        """
        self._urgent_priority: Priority = urgent_priority
        self._max_age_s: float = max_age_s
        self._urgent: dict[Key, tuple[Callout, float]] = {}
        self._waiting: dict[Key, tuple[Callout, float]] = {}
        self._waited_s: float = 0.0

    def push(self, callout: Callout, now_s: float) -> None:
        """Queue a callout; a waiting callout of the same kind and car is replaced and takes the new time.
        In:  callout  Callout
             now_s    float    s monotonic clock
        Out:          None
        """
        key: Key = (callout.kind, callout.facts.get("slot_id"))
        self._urgent.pop(key, None)
        self._waiting.pop(key, None)
        queue = self._urgent if callout.priority >= self._urgent_priority else self._waiting
        queue[key] = (callout, now_s)

    def pop(self, straight: bool, now_s: float) -> Callout | None:
        """Next callout to speak now.
        In:  straight  bool   the driver is on a straight
             now_s     float  s monotonic clock
        Out:           Callout | None  the oldest urgent callout; else, on a straight, the oldest of the
                                       highest-priority fresh callouts; else None
        """
        self._waiting = {k: v for k, v in self._waiting.items() if now_s - v[1] <= self._max_age_s}
        if self._urgent:
            callout, pushed_s = self._urgent.pop(next(iter(self._urgent)))
        elif straight and self._waiting:
            callout, pushed_s = self._waiting.pop(max(self._waiting, key=lambda k: self._waiting[k][0].priority))
        else:
            return None
        self._waited_s = now_s - pushed_s
        return callout

    def waited_s(self) -> float:
        """How long the callout returned by the last pop had waited in the gate.
        In:  -
        Out: float  s on the clock of push and pop
        """
        return self._waited_s
