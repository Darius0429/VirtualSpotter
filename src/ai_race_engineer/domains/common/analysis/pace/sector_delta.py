type Splits = tuple[float | None, float | None, float | None]


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
