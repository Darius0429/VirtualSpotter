import logging
import time
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.domains.common.voice.base import SpeechToText

logger = logging.getLogger(__name__)

SAMPLE_BYTES: int = 2


class RecordedSpeechToText(SpeechToText):
    """Wraps a speech-to-text engine and keeps every recording as a WAV file named after the time and what was
    heard, so a misheard question can be listened to and tried on another engine.
    _stt             SpeechToText  the wrapped engine
    _folder          Path          folder of the recordings
    _sample_rate_hz  int           Hz, of the audio given to transcribe
    """

    def __init__(self, stt: SpeechToText, folder: Path, sample_rate_hz: int) -> None:
        """Store the engine and where the recordings go.
        In:  stt             SpeechToText
             folder          Path          created on the first recording
             sample_rate_hz  int           Hz
        Out:                 None
        """
        self._stt: SpeechToText = stt
        self._folder: Path = folder
        self._sample_rate_hz: int = sample_rate_hz

    def transcribe(self, audio: NDArray[np.int16]) -> str:
        """Transcribe with the wrapped engine and keep the recording; a disk error is logged and ignored.
        In:  audio  NDArray[np.int16]  mono samples at the sample rate
        Out:        str                text, "" when nothing was understood
        """
        text = self._stt.transcribe(audio)
        heard = "".join(c if c.isalnum() or c == " " else "" for c in text)[:60].strip() or "nothing"
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
            with wave.open(str(self._folder / f"{time.strftime('%Y%m%d_%H%M%S')} {heard}.wav"), "wb") as file:
                file.setnchannels(1)
                file.setsampwidth(SAMPLE_BYTES)
                file.setframerate(self._sample_rate_hz)
                file.writeframes(audio.tobytes())
        except OSError as error:
            logger.warning("recording not saved: %s", error)
        return text
