from ai_race_engineer.events.text import delta_text, lap_time_text


def test_lap_time_text() -> None:
    """Seconds format as m:ss.s."""
    assert lap_time_text(112.34) == "1:52.3"


def test_lap_time_rounds_across_minute() -> None:
    """Rounding up to a full minute carries into the minutes."""
    assert lap_time_text(59.96) == "1:00.0"


def test_lap_time_pads_seconds() -> None:
    """Seconds below ten are zero-padded."""
    assert lap_time_text(65.04) == "1:05.0"


def test_delta_text_signed() -> None:
    """Deltas carry an explicit sign and two decimals."""
    assert delta_text(0.123) == "+0.12"
    assert delta_text(-0.3) == "-0.30"
