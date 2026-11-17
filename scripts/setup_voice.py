import os
from collections.abc import Callable
from pathlib import Path

import typer
import yaml
from elevenlabs.client import ElevenLabs

from ai_race_engineer.domains.common.voice.base import SpeechToText
from ai_race_engineer.domains.common.voice.button import Button
from ai_race_engineer.domains.common.voice.compare_stt import CompareSpeechToText
from ai_race_engineer.domains.common.voice.elevenlabs_stt import ElevenLabsSpeechToText
from ai_race_engineer.domains.common.voice.joystick_button import JoystickButton, joystick_names, wait_for_press
from ai_race_engineer.domains.common.voice.keyboard_button import KeyboardButton
from ai_race_engineer.domains.common.voice.microphone import Microphone, input_devices
from ai_race_engineer.domains.common.voice.talk_switch import TalkMode, TalkSwitch
from ai_race_engineer.domains.common.voice.voice_questions import VoiceQuestions
from ai_race_engineer.domains.common.voice.vosk_stt import VoskSpeechToText
from ai_race_engineer.domains.common.voice.whisper_stt import WhisperSpeechToText

STT: dict[str, Callable[[dict[str, object], int, Callable[[], str]], SpeechToText]] = {
    "whisper": lambda cfg, rate, names: WhisperSpeechToText(**cfg["whisper"], sample_rate_hz=rate, names=names),
    "elevenlabs": lambda cfg, rate, names: ElevenLabsSpeechToText(
        ElevenLabs(api_key=os.environ[cfg["tts"]["api_key_env"]]), **cfg["elevenlabs_stt"], sample_rate_hz=rate
    ),
    "vosk": lambda cfg, rate, names: VoskSpeechToText(Path(cfg["vosk"]["model_dir"]), rate),
}

BUTTONS: dict[str, Callable[[dict[str, object]], Button]] = {
    "joystick": lambda cfg: JoystickButton(cfg["device_name"], cfg["button"]),
    "keyboard": lambda cfg: KeyboardButton(cfg["key_vk"]),
}

app = typer.Typer()


@app.command()
def microphones() -> None:
    """List the audio devices to choose the microphone from.
    In:  -
    Out: None
    """
    print(input_devices())
    print("under devices: in configs/lmu.yaml set microphone.device to the index or name (null = default)")


@app.command()
def button() -> None:
    """Find the device name and index of the talk button by pressing it.
    In:  -
    Out: None
    """
    print(joystick_names())
    print("press the button you want to talk with...")
    name, index = wait_for_press(50.0)
    print(f'under devices: button: in configs/lmu.yaml set\n  device_name: "{name}"\n  button: {index}')


@app.command()
def listen(config: Path = Path("configs/lmu.yaml"), compare: bool = False) -> None:
    """Test the talk button, microphone and speech-to-text: print every question heard until Ctrl+C.
    In:  config   Path  the domain's YAML (devices)
         compare  bool  run every speech-to-text engine on each question and print what each heard and how long
                        it took, instead of only the configured one
    Out: None
    """
    cfg = yaml.safe_load(config.read_text())["devices"]
    rate = cfg["sample_rate_hz"]
    questions = VoiceQuestions(
        BUTTONS[cfg["button"]["kind"]](cfg["button"]),
        Microphone(cfg["microphone"]["device"], rate),
        CompareSpeechToText({name: build(cfg, rate, str) for name, build in STT.items()})
        if compare
        else STT[cfg["stt"]["engine"]](cfg, rate, str),
        TalkSwitch(TalkMode[cfg["talk"]["mode"]], cfg["talk"]["max_s"]),
        rate,
        **cfg["filter"],
        poll_hz=cfg["poll_hz"],
    )
    print("ready: hold the talk button, ask a question, release (Ctrl+C to stop)")
    while True:
        print(f"question: {questions()}")


if __name__ == "__main__":
    app()
