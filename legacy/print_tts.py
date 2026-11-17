from ai_race_engineer.voice.base import TextToSpeech


class PrintTextToSpeech(TextToSpeech):
    """Prints the text instead of speaking it."""

    def speak(self, text: str) -> None:
        """Print a text.
        In:  text  str
        Out:       None
        """
        print(text, flush=True)
