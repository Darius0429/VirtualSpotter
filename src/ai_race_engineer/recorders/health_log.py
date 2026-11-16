import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ai_race_engineer.recorders.run_log import RunLog


class HealthLog:
    """Timing health of the run: how long the timed calls take, the longest gap between telemetry reads and the queue
    sizes, written to the run log once per period.
    _log         RunLog
    _period_s    float                            s between records
    _sizes       Mapping[str, Callable[[], int]]  name -> current size of a queue
    _lock        threading.Lock                   guards the two statistics tables
    _calls       dict[str, _Timing]               name -> timed calls since the last record
    _gaps        dict[str, _Timing]               name -> gaps between calls since the last record
    _last_s      dict[str, float]                 name -> s monotonic clock of the last gap call
    """

    def __init__(self, log: RunLog, period_s: float, sizes: Mapping[str, Callable[[], int]]) -> None:
        """Store the log, the period and the queues to watch.
        In:  log       RunLog
             period_s  float                            s between records
             sizes     Mapping[str, Callable[[], int]]  name -> current size of a queue (non-blocking)
        Out:           None
        """
        self._log: RunLog = log
        self._period_s: float = period_s
        self._sizes: Mapping[str, Callable[[], int]] = sizes
        self._lock: threading.Lock = threading.Lock()
        self._calls: dict[str, _Timing] = {}
        self._gaps: dict[str, _Timing] = {}
        self._last_s: dict[str, float] = {}

    def timed[**P, R](self, name: str, call: Callable[P, R]) -> Callable[P, R]:
        """The same call, timed and counted under a name.
        In:  name  str             name in the health record
             call  Callable[P, R]
        Out:       Callable[P, R]  calls through; an exception passes on and is still timed
        """

        def timed_call(*args: P.args, **kwargs: P.kwargs) -> R:
            """Run the call and record its duration.
            In:  args, kwargs  the call's arguments
            Out:               R  the call's result
            """
            start_s = time.perf_counter()
            try:
                return call(*args, **kwargs)
            finally:
                self._add(False, name, time.perf_counter() - start_s)

        return timed_call

    def gap(self, name: str) -> Callable[[object], None]:
        """Observer that records the time between its calls; as the first observer of the telemetry source it shows
        any stall of the telemetry loop.
        In:  name  str  name in the health record
        Out:       Callable[[object], None]  takes the snapshot (unused)
        """

        def observe(snapshot: object) -> None:
            """Record the time since the previous call.
            In:  snapshot  object  (unused)
            Out:           None
            """
            now_s = time.perf_counter()
            if name in self._last_s:
                self._add(True, name, now_s - self._last_s[name])
            self._last_s[name] = now_s

        return observe

    def run(self) -> None:
        """Write one health record every period_s, forever (run in a daemon thread).
        In:  -
        Out: None
        """
        while True:
            time.sleep(self._period_s)
            with self._lock:
                calls, gaps, self._calls, self._gaps = self._calls, self._gaps, {}, {}
            self._log.write(
                "health",
                {
                    "period_s": self._period_s,
                    "calls": {name: timing.summary() for name, timing in calls.items()},
                    "gaps": {name: timing.summary() for name, timing in gaps.items()},
                    "sizes": {name: size() for name, size in self._sizes.items()},
                },
            )

    def _add(self, is_gap: bool, name: str, duration_s: float) -> None:
        """Add one measurement to the statistics of the running period.
        In:  is_gap      bool   a gap between calls (False = a timed call)
             name        str
             duration_s  float  s
        Out:             None
        """
        with self._lock:
            timing = (self._gaps if is_gap else self._calls).setdefault(name, _Timing())
            timing.count += 1
            timing.total_s += duration_s
            timing.max_s = max(timing.max_s, duration_s)


@dataclass
class _Timing:
    """Durations collected under one name in one period.
    count    int    measurements
    total_s  float  s, sum
    max_s    float  s, longest
    """

    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def summary(self) -> dict[str, float | int]:
        """The figures written to the health record.
        In:  -
        Out: dict[str, float | int]  count, mean_s, max_s
        """
        return {"count": self.count, "mean_s": self.total_s / self.count, "max_s": self.max_s}
