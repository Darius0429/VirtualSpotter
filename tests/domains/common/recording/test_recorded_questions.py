from ai_race_engineer.domains.common.recording.recorded_questions import questions_within

RECORDS: list[dict[str, object]] = [
    {"seq": 1, "wall_s": 100.0, "track": "Bahrain", "state": {}},
    {"seq": 2, "wall_s": 130.0, "reason": "question", "question": "how is my fuel", "state": {}},
    {"seq": 3, "wall_s": 110.0, "reason": "question", "question": "gap ahead", "state": {}},
    {"seq": 4, "wall_s": 400.0, "reason": "question", "question": "who won", "state": {}},
    {"seq": 5, "wall_s": 50.0, "reason": "question", "question": "radio check", "state": {}},
]


def test_only_questions_are_kept_oldest_first() -> None:
    """Records of the stream that are not questions are left out; the questions come in the order asked."""
    assert questions_within(RECORDS, 0.0, 1000.0) == [
        (50.0, "radio check"),
        (110.0, "gap ahead"),
        (130.0, "how is my fuel"),
        (400.0, "who won"),
    ]


def test_questions_outside_the_recording_are_left_out() -> None:
    """Questions asked before the recording's first row or after its last (another session of the run) are not asked
    again; the ends of the range count as inside."""
    assert questions_within(RECORDS, 100.0, 130.0) == [(110.0, "gap ahead"), (130.0, "how is my fuel")]


def test_no_questions_in_the_range() -> None:
    """A recording during which nothing was asked has no questions."""
    assert questions_within(RECORDS, 200.0, 300.0) == []
