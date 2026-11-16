from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

INT16_SCALE: float = 32768.0
CLICK_S: float = 0.01


@dataclass(frozen=True)
class RadioSound:
    """How a clean voice is made to sound like a team radio.
    low_hz         float  Hz, lower edge of the band the radio passes
    high_hz        float  Hz, upper edge of the band
    drive          float  gain into the soft clipper; higher is more distorted
    level          float  0-1 loudness of the voice after clipping
    hiss           float  0-1 loudness of the noise under the voice
    static_level   float  0-1 loudness of the static bursts
    static_in_s    float  s of static before the voice
    static_out_s   float  s of static after the voice
    """

    low_hz: float
    high_hz: float
    drive: float
    level: float
    hiss: float
    static_level: float
    static_in_s: float
    static_out_s: float


def radio_effect(
    audio: NDArray[np.int16], sample_rate_hz: int, sound: RadioSound, rng: np.random.Generator
) -> NDArray[np.int16]:
    """Make a clean voice sound like a team radio: narrow band, soft clipping, hiss under the voice, and a burst of
    static with a click before and after it.
    In:  audio           NDArray[np.int16]    mono voice
         sample_rate_hz  int                  Hz
         sound           RadioSound
         rng             np.random.Generator  source of the noise
    Out:                 NDArray[np.int16]    mono, longer than the input by the two bursts of static
    """
    band = butter(4, [sound.low_hz, sound.high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    voice = np.tanh(sosfilt(band, audio.astype(np.float64) / INT16_SCALE) * sound.drive) * sound.level
    voice += sosfilt(band, rng.normal(0.0, sound.hiss, voice.size))
    parts = [
        _static(sound.static_in_s, sample_rate_hz, sound.static_level, band, rng),
        voice,
        _static(sound.static_out_s, sample_rate_hz, sound.static_level, band, rng),
    ]
    return (np.clip(np.concatenate(parts), -1.0, 1.0) * (INT16_SCALE - 1)).astype(np.int16)


def _static(
    seconds: float, sample_rate_hz: int, level: float, band: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    """A fading burst of radio static that starts with a click.
    In:  seconds         float                s
         sample_rate_hz  int                  Hz
         level           float                0-1 loudness
         band            NDArray[np.float64]  band-pass filter sections
         rng             np.random.Generator
    Out:                 NDArray[np.float64]  samples in -1..1
    """
    samples = int(seconds * sample_rate_hz)
    burst = sosfilt(band, rng.normal(0.0, level, samples)) * np.linspace(1.0, 0.3, samples)
    burst[: int(CLICK_S * sample_rate_hz)] += 2 * level
    return burst
