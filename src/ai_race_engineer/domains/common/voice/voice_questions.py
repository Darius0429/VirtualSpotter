import time

from ai_race_engineer.domains.common.voice.base import SpeechToText
from ai_race_engineer.domains.common.voice.button import Button
from ai_race_engineer.domains.common.voice.microphone import Microphone
from ai_race_engineer.domains.common.voice.question_filter import drop_reason
from ai_race_engineer.domains.common.voice.talk_switch import TalkAction, TalkSwitch


class VoiceQuestions:
    """Push-to-talk question source: records while the talk button says so and returns the transcribed question.
    _button          Button
    _microphone      Microphone
    _stt             SpeechToText
    _switch          TalkSwitch
    _sample_rate_hz  int           Hz, of the microphone and the engine
    _min_audio_s     float         s, shorter recordings are dropped
    _min_words       int           fewer words are dropped
    _poll_hz         float         Hz, button polling
    """

    def __init__(
        self,
        button: Button,
        microphone: Microphone,
        stt: SpeechToText,
        switch: TalkSwitch,
        sample_rate_hz: int,
        min_audio_s: float,
        min_words: int,
        poll_hz: float,
    ) -> None:
        """Store the blocks to wire together.
        In:  button          Button
             microphone      Microphone
             stt             SpeechToText
             switch          TalkSwitch
             sample_rate_hz  int           Hz
             min_audio_s     float         s
             min_words       int           words
             poll_hz         float         Hz
        Out:                 None
        """
        self._button: Button = button
        self._microphone: Microphone = microphone
        self._stt: SpeechToText = stt
        self._switch: TalkSwitch = switch
        self._sample_rate_hz: int = sample_rate_hz
        self._min_audio_s: float = min_audio_s
        self._min_words: int = min_words
        self._poll_hz: float = poll_hz

    def __call__(self) -> str:
        """Block until the driver has asked a question that passes the filter; dropped recordings are printed.
        In:  -
        Out: str  the question
        """
        self._switch.reset(self._button.pressed())
        while True:
            action = self._switch.update(self._button.pressed(), time.monotonic())
            if action is TalkAction.START:
                self._microphone.start()
                print("[mic] listening...")
            elif action is TalkAction.STOP:
                audio = self._microphone.stop()
                audio_s = len(audio) / self._sample_rate_hz
                if audio_s < self._min_audio_s:
                    print(f"[mic] dropped: too short ({audio_s:.1f} s)")
                    continue
                start_s = time.monotonic()
                text = self._stt.transcribe(audio)
                print(f'[mic] "{text}" ({audio_s:.1f} s, {(time.monotonic() - start_s) * 1000:.0f} ms)')
                reason = drop_reason(audio_s, text, self._min_audio_s, self._min_words)
                if not reason:
                    return text
                print(f"[mic] dropped: {reason}")
            time.sleep(1 / self._poll_hz)
