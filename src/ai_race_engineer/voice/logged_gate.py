from ai_race_engineer.events.base import Callout, Priority
from ai_race_engineer.recorders.run_log import RunLog
from ai_race_engineer.voice.speech_gate import Key, SpeechGate


class LoggedSpeechGate(SpeechGate):
    """Speech gate that writes what happens to each callout to the run log (stream "callouts"): "replaced" by a newer
    one of the same kind and car, "expired" while waiting, "released" to be spoken.
    _log  RunLog
    """

    def __init__(self, log: RunLog, urgent_priority: Priority, max_age_s: float) -> None:
        """Set the log, the urgency threshold and the expiry.
        In:  log              RunLog
             urgent_priority  Priority  at or above this a callout is spoken at once
             max_age_s        float     s after which a waiting non-urgent callout is dropped
        Out:                  None
        """
        super().__init__(urgent_priority, max_age_s)
        self._log: RunLog = log

    def push(self, callout: Callout, now_s: float) -> None:
        """Queue a callout, logging the waiting one it replaces.
        In:  callout  Callout
             now_s    float    s monotonic clock
        Out:          None
        """
        key: Key = (callout.kind, callout.facts.get("slot_id"))
        replaced = self._urgent.get(key) or self._waiting.get(key)
        if replaced is not None:
            self._log.write("callouts", {"event": "replaced", "callout": replaced[0], "by": callout.text})
        super().push(callout, now_s)

    def pop(self, straight: bool, now_s: float) -> Callout | None:
        """Next callout to speak now, logging the ones that expire and the one released.
        In:  straight  bool   the driver is on a straight
             now_s     float  s monotonic clock
        Out:           Callout | None
        """
        for callout, pushed_s in self._waiting.values():
            if now_s - pushed_s > self._max_age_s:
                self._log.write("callouts", {"event": "expired", "callout": callout, "waited_s": now_s - pushed_s})
        callout = super().pop(straight, now_s)
        if callout is not None:
            self._log.write("callouts", {"event": "released", "callout": callout, "waited_s": self.waited_s()})
        return callout
