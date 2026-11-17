def drop_reason(audio_s: float, text: str, min_audio_s: float, min_words: int) -> str:
    """Why a recorded question is not passed to the agent (accidental press, noise).
    In:  audio_s      float  s, length of the recording
         text         str    transcription
         min_audio_s  float  s, shorter recordings are dropped
         min_words    int    fewer words are dropped
    Out:              str    "" = keep; else the reason to print
    """
    if audio_s < min_audio_s:
        return f"too short ({audio_s:.1f} s)"
    if not text:
        return "nothing understood"
    words = len(text.split())
    if words < min_words:
        return f"too few words ({words})"
    return ""
