from collections import deque


class LapConsumption:
    """Rolling average of an amount (fuel litres or virtual energy fraction) used per completed lap, optionally seeded.
    _window               deque[float]  last measured per-lap amounts
    _seed_per_lap         float | None  amount/lap used until the first measured lap
    _lap_start_amount     float | None  amount at the start of the current lap
    _last_completed_laps  int | None    completed laps at the previous update
    """

    def __init__(self, window_laps: int, seed_per_lap: float | None = None) -> None:
        """Set the averaging window and optional seed.
        In:  window_laps   int           laps to average over
             seed_per_lap  float | None  litres/lap or fraction/lap
        Out:               None
        """
        self._window: deque[float] = deque(maxlen=window_laps)
        self._seed_per_lap: float | None = seed_per_lap
        self._lap_start_amount: float | None = None
        self._last_completed_laps: int | None = None

    def update(self, completed_laps: int, amount: float) -> float | None:
        """Feed the current lap count and amount; laps where the amount rose (refuel) are skipped.
        In:  completed_laps  int    laps
             amount          float  litres or fraction
        Out:                 float | None  average litres/lap or fraction/lap; the seed before the first measured lap;
                                           None if neither
        """
        if self._last_completed_laps is None or completed_laps < self._last_completed_laps:
            self._window.clear()
            self._lap_start_amount = amount
        elif completed_laps > self._last_completed_laps:
            if amount <= self._lap_start_amount:
                self._window.append(self._lap_start_amount - amount)
            self._lap_start_amount = amount
        self._last_completed_laps = completed_laps
        if self._window:
            return sum(self._window) / len(self._window)
        return self._seed_per_lap


def amount_to_finish(per_lap: float, laps_remaining: float) -> float:
    """Amount needed to finish the race.
    In:  per_lap         float  litres/lap or fraction/lap
         laps_remaining  float  laps
    Out:                 float  litres or fraction
    """
    return per_lap * laps_remaining


def amount_to_add(to_finish: float, current: float) -> float:
    """Amount to add at the next stop.
    In:  to_finish  float  litres or fraction needed to finish
         current    float  litres or fraction on board
    Out:            float  litres or fraction (negative = surplus)
    """
    return to_finish - current
