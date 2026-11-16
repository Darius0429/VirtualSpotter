def sector_splits_s(sector1_s: float, sector2_cum_s: float, lap_s: float) -> tuple[float, float, float]:
    """Individual sector times of a lap.
    In:  sector1_s      float  s
         sector2_cum_s  float  s (sector 1 + sector 2)
         lap_s          float  s
    Out:                tuple[float, float, float]  s (S1, S2, S3)
    """
    return sector1_s, sector2_cum_s - sector1_s, lap_s - sector2_cum_s


def sector_deltas_s(
    other_splits: tuple[float, float, float], player_splits: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Per-sector difference between another car and the player.
    In:  other_splits   tuple[float, float, float]  s (S1, S2, S3)
         player_splits  tuple[float, float, float]  s (S1, S2, S3)
    Out:                tuple[float, float, float]  s other - player (negative = other faster)
    """
    return (
        other_splits[0] - player_splits[0],
        other_splits[1] - player_splits[1],
        other_splits[2] - player_splits[2],
    )
