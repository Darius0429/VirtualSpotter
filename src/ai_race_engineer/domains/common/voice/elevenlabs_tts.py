import logging

import numpy as np
import sounddevice
from elevenlabs.client import ElevenLabs
from numpy.typing import NDArray

from ai_race_engineer.domains.common.voice.base import TextToSpeech
from ai_race_engineer.domains.common.voice.radio_effect import RadioSound, radio_effect

logger = logging.getLogger(__name__)


class ElevenLabsTextToSpeech(TextToSpeech):
    """Speaks with an ElevenLabs voice made to sound like a team radio: the voice comes from the API as plain
    audio, the radio sound is added locally so it is the same every time, and the result is played on an output
    device. When the API fails the text is printed instead.
    _client          ElevenLabs
    _voice_id        str                  ElevenLabs voice
    _model           str                  ElevenLabs model, e.g. "eleven_flash_v2_5"
    _sample_rate_hz  int                  Hz of the audio asked for and played
    _device          int | str | None     output device; None = the Windows default
    _sound           RadioSound
    _rng             np.random.Generator  source of the radio noise
    """

    def __init__(
        self,
        client: ElevenLabs,
        voice_id: str,
        model: str,
        sample_rate_hz: int,
        device: int | str | None,
        sound: RadioSound,
    ) -> None:
        """Store the client, the voice and the radio sound.
        In:  client          ElevenLabs
             voice_id        str
             model           str
             sample_rate_hz  int               Hz; one of the PCM rates ElevenLabs gives (16000, 22050, 24000)
             device          int | str | None  output device index or name; None = the Windows default
             sound           RadioSound
        Out:                 None
        """
        self._client: ElevenLabs = client
        self._voice_id: str = voice_id
        self._model: str = model
        self._sample_rate_hz: int = sample_rate_hz
        self._device: int | str | None = device
        self._sound: RadioSound = sound
        self._rng: np.random.Generator = np.random.default_rng()

    def speak(self, text: str) -> None:
        """Say a text to the driver over the radio; blocks until it is spoken. A failed API call or a missing
        output device prints the text instead.
        In:  text  str
        Out:       None
        """
        try:
            sounddevice.play(self.synthesize(text), self._sample_rate_hz, device=self._device, blocking=True)
        except Exception as error:
            logger.warning("radio voice failed: %s", error)
            print(text, flush=True)

    def synthesize(self, text: str) -> NDArray[np.int16]:
        """The radio audio of a text; raises when the API call fails.
        In:  text  str
        Out:       NDArray[np.int16]  mono samples at the sample rate
        """
        chunks = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            model_id=self._model,
            text=text,
            output_format=f"pcm_{self._sample_rate_hz}",
        )
        voice = np.frombuffer(b"".join(chunks), dtype=np.int16)
        return radio_effect(voice, self._sample_rate_hz, self._sound, self._rng)
