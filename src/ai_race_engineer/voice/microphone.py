import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from ai_race_engineer.voice.base import VoiceUnavailable


class Microphone:
    """Records mono 16-bit audio from one input device; the stream stays open for the whole run.
    _blocks   list[NDArray[np.int16]]  audio blocks kept since start
    _keeping  bool                     whether incoming blocks are kept
    _stream   sd.InputStream
    """

    def __init__(self, device: int | str | None, sample_rate_hz: int) -> None:
        """Open and start the input stream.
        Raises VoiceUnavailable with the available devices when the device cannot be opened.
        In:  device          int | str | None  device index or name; None = Windows default input
             sample_rate_hz  int               Hz
        Out:                 None
        """
        self._blocks: list[NDArray[np.int16]] = []
        self._keeping: bool = False
        try:
            self._stream: sd.InputStream = sd.InputStream(
                samplerate=sample_rate_hz, channels=1, dtype="int16", device=device, callback=self._keep
            )
            self._stream.start()
        except (sd.PortAudioError, ValueError) as error:
            raise VoiceUnavailable(
                f"microphone {device!r} cannot be opened ({error}); devices:\n{input_devices()}"
            ) from error

    def start(self) -> None:
        """Empty the buffer and begin keeping audio.
        In:  -
        Out: None
        """
        self._blocks = []
        self._keeping = True

    def stop(self) -> NDArray[np.int16]:
        """Stop keeping audio and return what was recorded since start.
        In:  -
        Out: NDArray[np.int16]  mono samples at the sample rate
        """
        self._keeping = False
        return np.concatenate([np.zeros((0, 1), np.int16), *self._blocks])[:, 0]

    def _keep(self, block: NDArray[np.int16], frames: int, time: object, status: sd.CallbackFlags) -> None:
        """PortAudio callback: keep one audio block while recording.
        In:  block   NDArray[np.int16]  frames x 1 samples
             frames  int                samples in the block
             time    object             PortAudio time stamps
             status  sd.CallbackFlags   overflow flags
        Out:         None
        """
        if self._keeping:
            self._blocks.append(block.copy())


def input_devices() -> str:
    """The audio devices as sounddevice lists them, and the index of the default input.
    In:  -
    Out: str
    """
    return f"{sd.query_devices()}\ndefault input: {sd.default.device[0]}"
