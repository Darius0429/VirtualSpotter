import time

from ai_race_engineer.telemetry.pit_estimate import PitEstimate, PitEstimateSource


class PitEstimatePoller:
    """Reads the pit estimate periodically in its own thread, so the telemetry loop never waits for the network.
    _source    PitEstimateSource
    _period_s  float               s between reads
    _latest    PitEstimate | None  replaced, never mutated (read from other threads)
    """

    def __init__(self, source: PitEstimateSource, period_s: float) -> None:
        """Set the source and the polling period.
        In:  source    PitEstimateSource
             period_s  float  s
        Out:           None
        """
        self._source: PitEstimateSource = source
        self._period_s: float = period_s
        self._latest: PitEstimate | None = None

    def run(self) -> None:
        """Read forever every period_s (run in a daemon thread).
        In:  -
        Out: None
        """
        while True:
            self._latest = self._source.read()
            time.sleep(self._period_s)

    def latest(self) -> PitEstimate | None:
        """Estimate of the last read.
        In:  -
        Out: PitEstimate | None  None before the first read or when the game gave none
        """
        return self._latest
