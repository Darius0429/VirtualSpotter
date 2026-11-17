from collections.abc import Callable

from ai_race_engineer.domains.common.voice.base import TextToSpeech


class TimedPrintTextToSpeech(TextToSpeech):
    """Prints the text with the session time it is spoken at, and how long it waited when that was over a second:
    "[41:07] text", "[41:07 +6s] text"; for debugging.
    _clock       Callable[[], float]         s session time now
    _waited      Callable[[], float] | None  wall s the text waited before it was spoken; None = not shown
    _time_scale  float                       session s per wall s (the replay speed)
    """

    def __init__(
        self, clock: Callable[[], float], waited: Callable[[], float] | None = None, time_scale: float = 1.0
    ) -> None:
        """Set the clock and the optional source of the waiting time.
        In:  clock       Callable[[], float]         s session time now
             waited      Callable[[], float] | None  wall s the text being spoken waited in the queue
             time_scale  float                       session s per wall s (the replay speed)
        Out:             None
        """
        self._clock: Callable[[], float] = clock
        self._waited: Callable[[], float] | None = waited
        self._time_scale: float = time_scale

    def speak(self, text: str) -> None:
        """Print a text with its time stamp.
        In:  text  str
        Out:       None
        """
        delay_s = 0.0 if self._waited is None else self._waited() * self._time_scale
        print(f"[{stamp_text(self._clock(), delay_s)}] {text}", flush=True)


def stamp_text(time_s: float, delay_s: float) -> str:
    """Time stamp of a spoken text.
    In:  time_s   float  s session time it is spoken at
         delay_s  float  s it waited before
    Out:          str    "mm:ss", "h:mm:ss" from one hour on; " +Ns" appended when the delay is over 1 s
    """
    minutes, seconds = divmod(int(time_s), 60)
    hours, minutes = divmod(minutes, 60)
    clock = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
    return f"{clock} +{delay_s:.0f}s" if delay_s > 1 else clock
