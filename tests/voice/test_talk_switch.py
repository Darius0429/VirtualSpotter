from ai_race_engineer.voice.talk_switch import TalkAction, TalkMode, TalkSwitch


def _actions(switch: TalkSwitch, samples: list[tuple[bool, float]]) -> list[TalkAction]:
    """Feed button samples to a switch.
    In:  switch   TalkSwitch
         samples  list[tuple[bool, float]]  (pressed, time in s)
    Out:          list[TalkAction]          one per sample
    """
    return [switch.update(pressed, time_s) for pressed, time_s in samples]


def test_hold_press_starts_and_release_stops() -> None:
    """HOLD records from the press to the release."""
    switch = TalkSwitch(TalkMode.HOLD, 15.0)
    assert _actions(switch, [(True, 0.0), (True, 1.0), (False, 2.0)]) == [
        TalkAction.START,
        TalkAction.NONE,
        TalkAction.STOP,
    ]


def test_hold_past_max_stops_once_until_pressed_again() -> None:
    """HOLD held past max_s gives one STOP; a new recording needs a release and a new press."""
    switch = TalkSwitch(TalkMode.HOLD, 15.0)
    assert _actions(switch, [(True, 0.0), (True, 15.0), (True, 16.0), (False, 17.0), (True, 18.0)]) == [
        TalkAction.START,
        TalkAction.STOP,
        TalkAction.NONE,
        TalkAction.NONE,
        TalkAction.START,
    ]


def test_toggle_press_starts_and_next_press_stops() -> None:
    """TOGGLE keeps recording after the release and stops on the next press."""
    switch = TalkSwitch(TalkMode.TOGGLE, 15.0)
    assert _actions(switch, [(True, 0.0), (False, 0.5), (True, 3.0)]) == [
        TalkAction.START,
        TalkAction.NONE,
        TalkAction.STOP,
    ]


def test_toggle_never_pressed_again_stops_at_max() -> None:
    """TOGGLE without a second press is cut at max_s."""
    switch = TalkSwitch(TalkMode.TOGGLE, 15.0)
    assert _actions(switch, [(True, 0.0), (False, 0.5), (False, 14.9), (False, 15.0)]) == [
        TalkAction.START,
        TalkAction.NONE,
        TalkAction.NONE,
        TalkAction.STOP,
    ]


def test_steady_states_give_nothing() -> None:
    """A button that stays up, and one that stays down while recording, give no action."""
    switch = TalkSwitch(TalkMode.HOLD, 15.0)
    assert _actions(switch, [(False, 0.0), (False, 1.0)]) == [TalkAction.NONE, TalkAction.NONE]
    switch.update(True, 2.0)
    assert _actions(switch, [(True, 3.0), (True, 4.0)]) == [TalkAction.NONE, TalkAction.NONE]


def test_reset_with_button_down_does_not_start() -> None:
    """A button already down at the reset is not a new press."""
    switch = TalkSwitch(TalkMode.HOLD, 15.0)
    switch.reset(True)
    assert switch.update(True, 0.0) is TalkAction.NONE
