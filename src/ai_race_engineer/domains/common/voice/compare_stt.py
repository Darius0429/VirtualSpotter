import time
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.domains.common.voice.base import SpeechToText


class CompareSpeechToText(SpeechToText):
    """Runs several speech-to-text engines on the same utterance and prints what each heard and how long it took;
    for choosing an engine with the driver's own voice and microphone.
    _engines  Mapping[str, SpeechToText]  name -> engine, the first one gives the returned text
    """

    def __init__(self, engines: Mapping[str, SpeechToText]) -> None:
        """Store the engines.
        In:  engines  Mapping[str, SpeechToText]  name -> engine; the first one gives the returned text
        Out:          None
        """
        self._engines: Mapping[str, SpeechToText] = engines

    def transcribe(self, audio: NDArray[np.int16]) -> str:
        """Turn one recorded utterance into text with every engine and print each result.
        In:  audio  NDArray[np.int16]  mono samples at the engines' sample rate
        Out:        str                the text of the first engine
        """
        texts: list[str] = []
        for name, engine in self._engines.items():
            start_s = time.monotonic()
            texts.append(engine.transcribe(audio))
            print(f'  {name:12s} {(time.monotonic() - start_s) * 1000:5.0f} ms  "{texts[-1]}"', flush=True)
        return texts[0]
