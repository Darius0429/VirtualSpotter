from collections.abc import Callable

import numpy as np
from faster_whisper import WhisperModel
from numpy.typing import NDArray

from ai_race_engineer.domains.common.voice.base import SpeechToText, VoiceUnavailable

WHISPER_RATE_HZ: int = 16000
INT16_SCALE: float = 32768.0


class WhisperSpeechToText(SpeechToText):
    """Local speech-to-text with a Whisper model run by faster-whisper on the CPU; transcribes a whole utterance at
    once. A prompt with the words the driver uses steers it towards racing terms.
    _model   WhisperModel  loaded once
    _prompt  str                 text the model takes as what was said before; holds the racing vocabulary
    _names   Callable[[], str]   more words for the prompt at the moment of the question, e.g. driver names
    """

    def __init__(
        self, model: str, compute_type: str, prompt: str, sample_rate_hz: int, names: Callable[[], str]
    ) -> None:
        """Load the model; the first use downloads it. Raises VoiceUnavailable when the audio is not 16 kHz or the
        model cannot be loaded or downloaded.
        In:  model           str  model name, e.g. "base.en", or the folder of a downloaded model
             compute_type    str  number format of the weights, e.g. "int8"
             prompt          str  words and phrases the driver uses
             sample_rate_hz  int  Hz of the audio given to transcribe; Whisper needs 16000
             names           Callable[[], str]  more words for the prompt, asked for at every question
        Out:                 None
        """
        if sample_rate_hz != WHISPER_RATE_HZ:
            raise VoiceUnavailable(f"Whisper needs {WHISPER_RATE_HZ} Hz audio, not {sample_rate_hz} Hz")
        try:
            self._model: WhisperModel = WhisperModel(model, device="cpu", compute_type=compute_type)
        except Exception as error:
            raise VoiceUnavailable(f"Whisper model {model} could not be loaded: {error}") from error
        self._prompt: str = prompt
        self._names: Callable[[], str] = names

    def transcribe(self, audio: NDArray[np.int16]) -> str:
        """Turn one recorded utterance into text.
        In:  audio  NDArray[np.int16]  mono samples at 16 kHz
        Out:        str                text, "" when nothing was understood
        """
        segments, _ = self._model.transcribe(
            audio.astype(np.float32) / INT16_SCALE,
            language="en",
            initial_prompt=f"{self._prompt} {self._names()}".strip(),
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments)
