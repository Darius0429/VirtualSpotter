import ctypes

from ai_race_engineer.voice.button import Button


class KeyboardButton(Button):
    """A keyboard key as the talk button, read from Windows whatever window has the focus.
    _key_vk  int  Windows virtual-key code
    """

    def __init__(self, key_vk: int) -> None:
        """Store the key.
        In:  key_vk  int  Windows virtual-key code, e.g. 0x7B = F12
        Out:         None
        """
        self._key_vk: int = key_vk

    def pressed(self) -> bool:
        """Whether the key is down now.
        In:  -
        Out: bool
        """
        return ctypes.windll.user32.GetAsyncKeyState(self._key_vk) & 0x8000 != 0
