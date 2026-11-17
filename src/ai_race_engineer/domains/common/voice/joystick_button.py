import os
import time

import pygame

from ai_race_engineer.domains.common.voice.base import VoiceUnavailable
from ai_race_engineer.domains.common.voice.button import Button


class JoystickButton(Button):
    """One button of a wheel or controller, read through pygame (SDL2).
    _joystick  pygame.joystick.JoystickType  the opened device
    _button    int                           button index on the device
    """

    def __init__(self, device_name: str, button: int) -> None:
        """Open the first connected device whose name contains device_name (case-insensitive).
        Raises VoiceUnavailable with the connected device names when none matches or it has no such button.
        In:  device_name  str  part of the device name
             button       int  button index
        Out:              None
        """
        _start_joysticks()
        joysticks = [pygame.joystick.Joystick(index) for index in range(pygame.joystick.get_count())]
        matches = [joystick for joystick in joysticks if device_name.lower() in joystick.get_name().lower()]
        if not matches or button >= matches[0].get_numbuttons():
            raise VoiceUnavailable(
                f'no talk button {button} on a device named "{device_name}"; connected: {joystick_names()}'
            )
        self._joystick: pygame.joystick.JoystickType = matches[0]
        self._button: int = button

    def pressed(self) -> bool:
        """Whether the button is down now; False when the device was unplugged.
        In:  -
        Out: bool
        """
        try:
            pygame.event.pump()
            return bool(self._joystick.get_button(self._button))
        except pygame.error:
            return False


def joystick_names() -> list[str]:
    """Names of the connected wheels and controllers.
    In:  -
    Out: list[str]
    """
    _start_joysticks()
    return [pygame.joystick.Joystick(index).get_name() for index in range(pygame.joystick.get_count())]


def wait_for_press(poll_hz: float) -> tuple[str, int]:
    """Block until any button on any connected device goes down.
    In:  poll_hz  float            Hz
    Out:          tuple[str, int]  (device name, button index)
    """
    _start_joysticks()
    joysticks = [pygame.joystick.Joystick(index) for index in range(pygame.joystick.get_count())]
    while True:
        pygame.event.pump()
        for joystick in joysticks:
            for button in range(joystick.get_numbuttons()):
                if joystick.get_button(button):
                    return joystick.get_name(), button
        time.sleep(1 / poll_hz)


def _start_joysticks() -> None:
    """Start pygame's joystick system so that buttons are read while the game window has the focus; no window is
    created.
    In:  -
    Out: None
    """
    os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
    pygame.init()
    pygame.joystick.init()
