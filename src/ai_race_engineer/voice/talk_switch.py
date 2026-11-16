from enum import Enum


class TalkMode(Enum):
    """How the talk button controls recording: HOLD records while held, TOGGLE starts and stops on a press."""

    HOLD = "HOLD"
    TOGGLE = "TOGGLE"


class TalkAction(Enum):
    """What to do with the recording after a button sample."""

    NONE = "NONE"
    START = "START"
    STOP = "STOP"


class TalkSwitch:
    """Turns the sampled talk-button state into recording start and stop actions.
    _mode      TalkMode
    _max_s     float         s, a recording is cut at this length
    _previous  bool          button state at the previous sample
    _start_s   float | None  s, monotonic time the recording started; None while idle
    """

    def __init__(self, mode: TalkMode, max_s: float) -> None:
        """Store the mode and the longest recording; starts idle with the button up.
        In:  mode   TalkMode
             max_s  float     s
        Out:        None
        """
        self._mode: TalkMode = mode
        self._max_s: float = max_s
        self._previous: bool = False
        self._start_s: float | None = None

    def update(self, pressed: bool, time_s: float) -> TalkAction:
        """Take one button sample: START on a press while idle; STOP on the release (HOLD), on the next press
        (TOGGLE), or once the recording is max_s long.
        In:  pressed  bool        talk button is down
             time_s   float       s, monotonic
        Out:          TalkAction
        """
        rising = pressed and not self._previous
        falling = self._previous and not pressed
        self._previous = pressed
        if self._start_s is None:
            if rising:
                self._start_s = time_s
                return TalkAction.START
            return TalkAction.NONE
        ended = falling if self._mode is TalkMode.HOLD else rising
        if ended or time_s - self._start_s >= self._max_s:
            self._start_s = None
            return TalkAction.STOP
        return TalkAction.NONE

    def reset(self, pressed: bool) -> None:
        """Go idle and take the current button state as the previous one.
        In:  pressed  bool  talk button is down
        Out:          None
        """
        self._start_s = None
        self._previous = pressed
