import queue

from ai_race_engineer.domains.common.callouts.base import Callout
from ai_race_engineer.domains.common.recording.run_log import RunLog


class LoggedCalloutQueue(queue.Queue[Callout]):
    """Callout queue that writes every callout put on it to the run log (stream "callouts", event "emitted").
    _log  RunLog
    """

    def __init__(self, log: RunLog) -> None:
        """Make an unbounded queue.
        In:  log  RunLog
        Out:      None
        """
        super().__init__()
        self._log: RunLog = log

    def put(self, item: Callout, block: bool = True, timeout: float | None = None) -> None:
        """Log the callout, then queue it.
        In:  item     Callout
             block    bool          as queue.Queue.put
             timeout  float | None  s, as queue.Queue.put
        Out:          None
        """
        self._log.write("callouts", {"event": "emitted", "callout": item})
        super().put(item, block, timeout)
