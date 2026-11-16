from collections.abc import Callable

from ai_race_engineer.events.overall_best import OverallBest
from ai_race_engineer.events.personal_best import PersonalBest
from ai_race_engineer.telemetry.snapshot import Snapshot, VehicleState


def _kinds(
    event: OverallBest | PersonalBest,
    make_snapshot: Callable[..., Snapshot],
    make_vehicle: Callable[..., VehicleState],
    laps: list[tuple[float, float]],
) -> list[str | None]:
    """Callout kinds an event gives over a run of best laps.
    In:  event          OverallBest | PersonalBest
         make_snapshot  Callable[..., Snapshot]
         make_vehicle   Callable[..., VehicleState]
         laps           list[tuple[float, float]]    best lap (s) of the player and of another car at each check
    Out:                list[str | None]             kind of the callout of each check; None = no callout
    """
    kinds: list[str | None] = []
    for mine_s, other_s in laps:
        player = make_vehicle(slot_id=0, is_player=True, best_lap_s=mine_s)
        callout = event.check(make_snapshot(player=player, vehicles=[make_vehicle(slot_id=1, best_lap_s=other_s)]), 0.0)
        kinds.append(None if callout is None else callout.kind)
    return kinds


LAPS: list[tuple[float, float]] = [(100.0, 99.0), (100.0, 98.0), (99.5, 98.0), (97.0, 98.0), (97.0, 98.0)]


def test_overall_best_says_only_overall_fastest_laps(
    make_snapshot: Callable[..., Snapshot], make_vehicle: Callable[..., VehicleState]
) -> None:
    """The first check seeds; another car's and the player's overall best are said, a mere personal best is not."""
    assert _kinds(OverallBest(), make_snapshot, make_vehicle, LAPS) == [None, "fastest_lap", None, "fastest_lap", None]


def test_personal_best_says_only_the_players_bests(
    make_snapshot: Callable[..., Snapshot], make_vehicle: Callable[..., VehicleState]
) -> None:
    """Another car's overall best is not said; the player's best is, also when it is the overall best."""
    kinds = _kinds(PersonalBest(), make_snapshot, make_vehicle, LAPS)
    assert kinds == [None, None, "personal_best", "personal_best", None]
