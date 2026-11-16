import logging
import queue
import time
from collections.abc import Callable

from ai_race_engineer.events.base import Callout
from ai_race_engineer.voice.base import TextToSpeech
from ai_race_engineer.voice.speech_gate import SpeechGate

logger = logging.getLogger(__name__)


class SpeechLoop:
    """Speaks the callouts the gate lets through, one at a time.
    _callouts  queue.Queue[Callout]  input queue (events and answers)
    _gate      SpeechGate
    _tts       TextToSpeech
    _straight  Callable[[], bool]    whether the driver is on a straight
    _poll_hz   float                 Hz
    """

    def __init__(
        self,
        callouts: queue.Queue[Callout],
        gate: SpeechGate,
        tts: TextToSpeech,
        straight: Callable[[], bool],
        poll_hz: float,
    ) -> None:
        """Store the blocks to wire together.
        In:  callouts  queue.Queue[Callout]  input queue
             gate      SpeechGate
             tts       TextToSpeech
             straight  Callable[[], bool]    whether the driver is on a straight
             poll_hz   float                 Hz
        Out:           None
        """
        self._callouts: queue.Queue[Callout] = callouts
        self._gate: SpeechGate = gate
        self._tts: TextToSpeech = tts
        self._straight: Callable[[], bool] = straight
        self._poll_hz: float = poll_hz

    def run(self) -> None:
        """Speak forever: move new callouts into the gate, speak what it releases, else wait one poll.
        In:  -
        Out: None
        """
        while True:
            now_s = time.monotonic()
            while not self._callouts.empty():
                self._gate.push(self._callouts.get(), now_s)
            callout = self._gate.pop(self._straight(), now_s)
            if callout is None:
                time.sleep(1 / self._poll_hz)
                continue
            try:
                self._tts.speak(callout.text)
            except Exception:
                logger.exception("speech failed")
