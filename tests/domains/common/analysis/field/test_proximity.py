import math

import numpy as np

from ai_race_engineer.domains.common.analysis.field.proximity import to_player_frame


def test_heading_zero_ahead_and_right() -> None:
    """At heading 0 forward is +z and right is +x."""
    out = to_player_frame((10.0, 20.0), 0.0, np.array([[10.0, 25.0], [13.0, 20.0]]))
    np.testing.assert_allclose(out, [[0.0, 5.0], [3.0, 0.0]], atol=1e-9)


def test_heading_quarter_turn() -> None:
    """At heading pi/2 forward is +x and right is -z."""
    out = to_player_frame((0.0, 0.0), math.pi / 2, np.array([[5.0, 0.0], [0.0, -3.0]]))
    np.testing.assert_allclose(out, [[0.0, 5.0], [3.0, 0.0]], atol=1e-9)


def test_car_behind_negative_longitudinal() -> None:
    """A car behind has negative longitudinal offset."""
    out = to_player_frame((0.0, 0.0), 0.0, np.array([[0.0, -8.0]]))
    np.testing.assert_allclose(out, [[0.0, -8.0]], atol=1e-9)


def test_player_itself_is_origin() -> None:
    """The player's own position maps to (0, 0)."""
    out = to_player_frame((4.0, -2.0), 1.1, np.array([[4.0, -2.0]]))
    np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-9)


def test_several_cars_one_call() -> None:
    """All rows are transformed in one call."""
    others = np.array([[0.0, 5.0], [-2.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    out = to_player_frame((0.0, 0.0), 0.0, others)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out, [[0.0, 5.0], [-2.0, 0.0], [0.0, -1.0], [1.0, 1.0]], atol=1e-9)
