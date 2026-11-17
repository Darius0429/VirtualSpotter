import numpy as np
import pytest

from ai_race_engineer.domains.common.voice.radio_effect import RadioSound, radio_effect

RATE_HZ: int = 22050
SOUND: RadioSound = RadioSound(
    low_hz=350.0,
    high_hz=3000.0,
    drive=6.0,
    level=0.6,
    hiss=0.012,
    static_level=0.25,
    static_in_s=0.22,
    static_out_s=0.15,
)


def _tone(frequency_hz: float) -> np.ndarray:
    """One second of a sine at a tenth of full scale.
    In:  frequency_hz  float
    Out:               np.ndarray  int16 samples
    """
    return (np.sin(2 * np.pi * frequency_hz * np.arange(RATE_HZ) / RATE_HZ) * 3276).astype(np.int16)


def _loudness(audio: np.ndarray) -> float:
    """Loudness of the voice part, without the static before and after it.
    In:  audio  np.ndarray  output of radio_effect for a one-second input
    Out:        float       RMS in int16 units
    """
    start = int(SOUND.static_in_s * RATE_HZ)
    return float(np.sqrt(np.mean(audio[start : start + RATE_HZ].astype(np.float64) ** 2)))


def test_output_is_the_voice_with_static_before_and_after() -> None:
    """The result is int16, longer than the input by the two bursts, and loud where the bursts are."""
    out = radio_effect(_tone(1000.0), RATE_HZ, SOUND, np.random.default_rng(1))
    assert out.dtype == np.int16
    assert out.size == RATE_HZ + int(0.22 * RATE_HZ) + int(0.15 * RATE_HZ)
    assert np.abs(out[: int(0.01 * RATE_HZ)]).max() > 5000


def test_only_the_radio_band_gets_through() -> None:
    """A tone inside the band stays loud; a deep and a high tone are cut to a fraction of it."""
    inside, deep, high = (
        _loudness(radio_effect(_tone(hz), RATE_HZ, SOUND, np.random.default_rng(1))) for hz in (1000.0, 60.0, 8000.0)
    )
    assert deep < 0.2 * inside
    assert high < 0.2 * inside
    assert inside == pytest.approx(0.6 * np.tanh(0.6) * 32767 / np.sqrt(2), rel=0.25)
