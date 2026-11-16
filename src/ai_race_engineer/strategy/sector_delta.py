from dataclasses import dataclass

type Splits = tuple[float | None, float | None, float | None]


@dataclass(frozen=True)
class SectorDelta:
    """The player's sector times against another car's.
    last_delta_s        Splits        s per sector on the last lap, player - other (positive = player slower);
                                      None for a sector either car has no time for
    best_delta_s        Splits        s per sector on the best sectors, player - other
    last_total_delta_s  float | None  s, sum of the last-lap deltas; None unless all three sectors are known
    best_total_delta_s  float | None  s, sum of the best-sector deltas; None unless all three sectors are known
    worst_sector        int | None    sector number (1-3) where the player loses the most: on the last lap when it
                                      has a delta, else on the best sectors; None when the player loses nowhere
    """

    last_delta_s: Splits
    best_delta_s: Splits
    last_total_delta_s: float | None
    best_total_delta_s: float | None
    worst_sector: int | None


def sector_delta(player_last: Splits, player_best: Splits, other_last: Splits, other_best: Splits) -> SectorDelta:
    """Compare the player's sector times with another car's.
    In:  player_last  Splits  s (S1, S2, S3) of the player's last lap
         player_best  Splits  s (S1, S2, S3) of the player's best sectors
         other_last   Splits  s (S1, S2, S3) of the other car's last lap
         other_best   Splits  s (S1, S2, S3) of the other car's best sectors
    Out:              SectorDelta
    """
    last = _deltas(player_last, other_last)
    best = _deltas(player_best, other_best)
    return SectorDelta(last, best, _total(last), _total(best), _worst_sector(last) or _worst_sector(best))


def splits(sector1_s: float | None, sector2_cum_s: float | None, lap_s: float | None) -> Splits:
    """Individual sector times of a lap of which some times may be missing.
    In:  sector1_s      float | None  s
         sector2_cum_s  float | None  s (sector 1 + sector 2)
         lap_s          float | None  s
    Out:                Splits        s (S1, S2, S3); None for a sector that cannot be derived
    """
    return (
        sector1_s,
        None if sector1_s is None or sector2_cum_s is None else sector2_cum_s - sector1_s,
        None if sector2_cum_s is None or lap_s is None else lap_s - sector2_cum_s,
    )


def _deltas(player: Splits, other: Splits) -> Splits:
    """Per-sector difference of two cars.
    In:  player  Splits  s
         other   Splits  s
    Out:         Splits  s player - other; None where either time is missing
    """
    first, second, third = (None if p is None or o is None else p - o for p, o in zip(player, other, strict=True))
    return first, second, third


def _total(deltas: Splits) -> float | None:
    """Sum of the three sector deltas.
    In:  deltas  Splits  s
    Out:         float | None  s; None when a sector is missing
    """
    return None if None in deltas else sum(deltas)


def _worst_sector(deltas: Splits) -> int | None:
    """Sector where the player loses the most.
    In:  deltas  Splits  s player - other
    Out:         int | None  sector number 1-3; None when no known delta is positive
    """
    losses = {index + 1: delta for index, delta in enumerate(deltas) if delta is not None and delta > 0}
    return max(losses, key=losses.get) if losses else None
