import pytest

from ai_race_engineer.voice.timed_print_tts import TimedPrintTextToSpeech, stamp_text


@pytest.mark.parametrize(
    ("time_s", "delay_s", "expected"),
    [
        (0.0, 0.0, "00:00"),
        (2467.9, 0.0, "41:07"),
        (2467.0, 1.0, "41:07"),
        (2467.0, 6.4, "41:07 +6s"),
        (3599.9, 0.0, "59:59"),
        (3600.0, 0.0, "1:00:00"),
        (3725.0, 12.0, "1:02:05 +12s"),
    ],
)
def test_stamp_text(time_s: float, delay_s: float, expected: str) -> None:
    """Minutes and seconds, hours from one hour on, and the queue delay when it is over a second."""
    assert stamp_text(time_s, delay_s) == expected


def test_speak_prints_the_session_time_and_the_scaled_delay(capsys: pytest.CaptureFixture[str]) -> None:
    """The printed time is the clock's; the waiting time is wall time, scaled to session time."""
    tts = TimedPrintTextToSpeech(lambda: 2467.0, lambda: 0.3, 20.0)
    tts.speak("Box this lap")
    TimedPrintTextToSpeech(lambda: 65.0).speak("Green flag")
    assert capsys.readouterr().out == "[41:07 +6s] Box this lap\n[01:05] Green flag\n"
