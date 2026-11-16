import time

import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.recorders.run_log import RunLog
from ai_race_engineer.voice.base import SpeechToText


class LoggedSpeechToText(SpeechToText):
    """Wraps a speech-to-text engine and writes each transcription to the run log (stream "conversations").
    _stt             SpeechToText  the wrapped engine
    _log             RunLog
    _sample_rate_hz  int           Hz, of the audio given to transcribe
    """

    def __init__(self, stt: SpeechToText, log: RunLog, sample_rate_hz: int) -> None:
        """Store the engine, the log and the sample rate.
        In:  stt             SpeechToText
             log             RunLog
             sample_rate_hz  int           Hz
        Out:                 None
        """
        self._stt: SpeechToText = stt
        self._log: RunLog = log
        self._sample_rate_hz: int = sample_rate_hz

    def transcribe(self, audio: NDArray[np.int16]) -> str:
        """Transcribe with the wrapped engine; logs "transcription" with the audio length and the time it took.
        In:  audio  NDArray[np.int16]  mono samples at the sample rate
        Out:        str                text, "" when nothing was understood
        """
        start_s = time.monotonic()
        text = self._stt.transcribe(audio)
        self._log.write(
            "conversations",
            {
                "event": "transcription",
                "audio_s": len(audio) / self._sample_rate_hz,
                "transcribe_ms": (time.monotonic() - start_s) * 1000,
                "text": text,
            },
        )
        return text
