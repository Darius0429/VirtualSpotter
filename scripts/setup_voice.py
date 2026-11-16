from collections.abc import Callable
from pathlib import Path

import typer
import yaml

from ai_race_engineer.voice.button import Button
from ai_race_engineer.voice.joystick_button import JoystickButton, joystick_names, wait_for_press
from ai_race_engineer.voice.keyboard_button import KeyboardButton
from ai_race_engineer.voice.microphone import Microphone, input_devices
from ai_race_engineer.voice.talk_switch import TalkMode, TalkSwitch
from ai_race_engineer.voice.voice_questions import VoiceQuestions
from ai_race_engineer.voice.vosk_stt import VoskSpeechToText

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
    print("set microphone.device in configs/voice.yaml to the index or name (null = default)")


@app.command()
def button() -> None:
    """Find the device name and index of the talk button by pressing it.
    In:  -
    Out: None
    """
    print(joystick_names())
    print("press the button you want to talk with...")
    name, index = wait_for_press(50.0)
    print(f'under button: in configs/voice.yaml set\n  device_name: "{name}"\n  button: {index}')


@app.command()
def listen(voice: Path = Path("configs/voice.yaml")) -> None:
    """Test the talk button, microphone and speech-to-text: print every question heard until Ctrl+C.
    In:  voice  Path  voice-input config YAML
    Out: None
    """
    cfg = yaml.safe_load(voice.read_text())
    rate = cfg["sample_rate_hz"]
    questions = VoiceQuestions(
        BUTTONS[cfg["button"]["kind"]](cfg["button"]),
        Microphone(cfg["microphone"]["device"], rate),
        VoskSpeechToText(Path(cfg["vosk"]["model_dir"]), rate),
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
