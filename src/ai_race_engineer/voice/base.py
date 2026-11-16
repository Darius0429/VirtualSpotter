from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class TextToSpeech(ABC):
    """Common interface for all text-to-speech engines."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Say a text to the driver; blocks until it is spoken.
        In:  text  str
        Out:       None
        """


class SpeechToText(ABC):
    """Common interface for all speech-to-text engines; the sample rate is a constructor argument of each engine."""

    @abstractmethod
    def transcribe(self, audio: NDArray[np.int16]) -> str:
        """Turn one recorded utterance into text.
        In:  audio  NDArray[np.int16]  mono samples at the engine's sample rate
        Out:        str                text, "" when nothing was understood
        """


class VoiceUnavailable(Exception):
    """A voice input device or model cannot be used; the message says what is available."""
