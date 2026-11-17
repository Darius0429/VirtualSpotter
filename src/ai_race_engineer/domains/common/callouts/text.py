def lap_time_text(lap_s: float) -> str:
    """Lap time as spoken/printed text, cut to the tenth below as on a timing screen.
    In:  lap_s  float  s
    Out:        str    "m:ss.s", e.g. 112.39 -> "1:52.3", 105.999 -> "1:45.9"
    """
    minutes, tenths = divmod(int(lap_s * 10 + 1e-6), 600)
    return f"{minutes}:{tenths // 10:02d}.{tenths % 10}"


def delta_text(delta_s: float) -> str:
    """Signed time delta as text, rounded to hundredths.
    In:  delta_s  float  s
    Out:          str    e.g. "+0.12", "-0.30"
    """
    return f"{delta_s:+.2f}"
