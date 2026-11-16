import json
from pathlib import Path

import numpy as np
import vosk
from numpy.typing import NDArray

from ai_race_engineer.voice.base import SpeechToText, VoiceUnavailable


class VoskSpeechToText(SpeechToText):
    """Local speech-to-text with a Vosk model; transcribes a whole utterance at once.
    _model           vosk.Model  loaded once
    _sample_rate_hz  int         Hz, of the audio given to transcribe
    """

    def __init__(self, model_dir: Path, sample_rate_hz: int) -> None:
        """Load the model. Raises VoiceUnavailable with the download hint when the model folder is missing.
        In:  model_dir       Path  unpacked Vosk model folder
             sample_rate_hz  int   Hz
        Out:                 None
        """
        if not model_dir.is_dir():
            raise VoiceUnavailable(
                f"Vosk model folder {model_dir} is missing; download and unpack {model_dir.name} from "
                "https://alphacephei.com/vosk/models into models/"
            )
        vosk.SetLogLevel(-1)
        self._model: vosk.Model = vosk.Model(str(model_dir))
        self._sample_rate_hz: int = sample_rate_hz

    def transcribe(self, audio: NDArray[np.int16]) -> str:
        """Turn one recorded utterance into text.
        In:  audio  NDArray[np.int16]  mono samples at the sample rate
        Out:        str                text, "" when nothing was understood
        """
        recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate_hz)
        recognizer.AcceptWaveform(audio.tobytes())
        return json.loads(recognizer.FinalResult())["text"]
