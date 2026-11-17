import time

from ai_race_engineer.domains.common.recording.run_log import RunLog
from ai_race_engineer.domains.common.voice.base import TextToSpeech


class LoggedTextToSpeech(TextToSpeech):
    """Wraps a text-to-speech engine and writes what was spoken to the run log (stream "callouts").
    _tts  TextToSpeech  the wrapped engine
    _log  RunLog
    """

    def __init__(self, tts: TextToSpeech, log: RunLog) -> None:
        """Store the engine and the log.
        In:  tts  TextToSpeech
             log  RunLog
        Out:      None
        """
        self._tts: TextToSpeech = tts
        self._log: RunLog = log

    def speak(self, text: str) -> None:
        """Say a text with the wrapped engine; logs "spoken" with the duration, or "speech_failed" and raises again.
        In:  text  str
        Out:       None
        """
        start_s = time.monotonic()
        try:
            self._tts.speak(text)
        except Exception as error:
            self._log.write("callouts", {"event": "speech_failed", "text": text, "error": repr(error)})
            raise
        self._log.write("callouts", {"event": "spoken", "text": text, "duration_s": time.monotonic() - start_s})
