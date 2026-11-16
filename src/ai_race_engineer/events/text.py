def lap_time_text(lap_s: float) -> str:
    """Lap time as spoken/printed text, rounded to tenths.
    In:  lap_s  float  s
    Out:        str    "m:ss.s", e.g. 112.34 -> "1:52.3"
    """
    minutes, tenths = divmod(round(lap_s * 10), 600)
    return f"{minutes}:{tenths // 10:02d}.{tenths % 10}"


def delta_text(delta_s: float) -> str:
    """Signed time delta as text, rounded to hundredths.
    In:  delta_s  float  s
    Out:          str    e.g. "+0.12", "-0.30"
    """
    return f"{delta_s:+.2f}"
