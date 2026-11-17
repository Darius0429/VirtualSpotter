from ai_race_engineer.domains.common.voice.question_filter import drop_reason


def test_short_audio_is_dropped() -> None:
    """A recording below min_audio_s is an accidental press."""
    assert drop_reason(0.2, "how much fuel", 0.4, 2) == "too short (0.2 s)"


def test_empty_text_is_dropped() -> None:
    """Nothing understood is not a question."""
    assert drop_reason(2.0, "", 0.4, 2) == "nothing understood"


def test_too_few_words_is_dropped() -> None:
    """One word with min_words 2 is noise."""
    assert drop_reason(2.0, "the", 0.4, 2) == "too few words (1)"


def test_normal_question_is_kept() -> None:
    """A normal question gives no reason."""
    assert drop_reason(3.1, "how much fuel do i have", 0.4, 2) == ""


def test_exactly_at_both_limits_is_kept() -> None:
    """A recording of exactly min_audio_s with exactly min_words is kept."""
    assert drop_reason(0.4, "box now", 0.4, 2) == ""
