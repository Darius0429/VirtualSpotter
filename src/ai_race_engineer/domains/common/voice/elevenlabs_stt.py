import io
import logging
import wave

import numpy as np
from elevenlabs.client import ElevenLabs
from numpy.typing import NDArray

from ai_race_engineer.domains.common.voice.base import SpeechToText

logger = logging.getLogger(__name__)

SAMPLE_BYTES: int = 2


class ElevenLabsSpeechToText(SpeechToText):
    """Speech-to-text with the ElevenLabs API; transcribes a whole utterance at once. Needs the network.
    _client          ElevenLabs
    _model           str         ElevenLabs model, e.g. "scribe_v2"
    _language        str         language code of the speech, e.g. "en"
    _sample_rate_hz  int         Hz, of the audio given to transcribe
    """

    def __init__(self, client: ElevenLabs, model: str, language: str, sample_rate_hz: int) -> None:
        """Store the client, the model and the audio format.
        In:  client          ElevenLabs
             model           str
             language        str  language code, e.g. "en"
             sample_rate_hz  int  Hz
        Out:                 None
        """
        self._client: ElevenLabs = client
        self._model: str = model
        self._language: str = language
        self._sample_rate_hz: int = sample_rate_hz

    def transcribe(self, audio: NDArray[np.int16]) -> str:
        """Turn one recorded utterance into text.
        In:  audio  NDArray[np.int16]  mono samples at the sample rate
        Out:        str                text, "" when nothing was understood or the API call failed
        """
        recording = io.BytesIO()
        with wave.open(recording, "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(SAMPLE_BYTES)
            file.setframerate(self._sample_rate_hz)
            file.writeframes(audio.tobytes())
        recording.seek(0)
        try:
            result = self._client.speech_to_text.convert(
                file=("question.wav", recording, "audio/wav"), model_id=self._model, language_code=self._language
            )
        except Exception as error:
            logger.warning("speech-to-text call failed: %s", error)
            return ""
        return result.text.strip()
