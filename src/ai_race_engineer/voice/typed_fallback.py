from collections.abc import Callable

from ai_race_engineer.voice.base import VoiceUnavailable


def or_typed(build: Callable[[], Callable[[], str]]) -> Callable[[], str]:
    """Build the voice question source, or fall back to typed questions when the wheel, microphone or model is
    missing; the reason is printed.
    In:  build  Callable[[], Callable[[], str]]  builds the voice question source; may raise VoiceUnavailable
    Out:        Callable[[], str]                blocks until the next question
    """
    try:
        return build()
    except VoiceUnavailable as error:
        print(f"{error}\nvoice questions are off, type questions instead")
        return input
