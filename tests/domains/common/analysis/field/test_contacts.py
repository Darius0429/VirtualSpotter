from types import SimpleNamespace

from ai_race_engineer.domains.common.analysis.field.contacts import ContactTracker
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment

SEGMENTS: tuple[Segment, ...] = (Segment("turn 1", True, 0.0, 500.0, 250.0),)


def _snapshot(time_s: float, impacts: dict[int, tuple[float, float]], far: bool = False) -> SimpleNamespace:
    """Snapshot stand-in: the player (slot 0) with Ann (slot 1) 5 m away, or 200 m away, and Bob (slot 2) far off.
    In:  time_s   float                          s session time
         impacts  dict[int, tuple[float, float]]  slot id -> (impact time s, magnitude)
         far      bool                           Ann is 200 m from the player
    Out:          SimpleNamespace
    """

    def vehicle(slot_id: int, name: str, x_m: float) -> SimpleNamespace:
        return SimpleNamespace(
            slot_id=slot_id,
            driver_name=name,
            class_name="Hyper",
            is_player=slot_id == 0,
            in_garage=False,
            position_xz_m=(x_m, 0.0),
            lap_distance_m=100.0,
        )

    def resources(slot_id: int) -> SimpleNamespace:
        impact_s, magnitude = impacts.get(slot_id, (0.0, 0.0))
        return SimpleNamespace(slot_id=slot_id, last_impact_s=impact_s, last_impact_magnitude=magnitude)

    return SimpleNamespace(
        session=SimpleNamespace(elapsed_s=time_s),
        vehicles=[vehicle(0, "Me", 0.0), vehicle(1, "Ann", 200.0 if far else 5.0), vehicle(2, "Bob", 900.0)],
        resources=[resources(0), resources(1), resources(2)],
    )


def _tracker(recorded: list[tuple[str, str]]) -> ContactTracker:
    """Tracker with the config values that keeps what it records.
    In:  recorded  list[tuple[str, str]]  (kind, text) of every timeline entry added
    Out:           ContactTracker
    """
    return ContactTracker(
        lambda kind, text, fields: recorded.append((kind, text)),
        lambda: SEGMENTS,
        min_magnitude=100.0,
        hard_magnitude=500.0,
        near_m=12.0,
        hold_s=3.0,
    )


def test_hit_with_a_car_close_by_is_contact_between_the_two_and_is_recorded_once() -> None:
    """The first snapshot only seeds; a new impact with Ann 5 m away is contact with her, named at the track
    segment; her own impact from the same touch and a second impact right after add nothing."""
    recorded: list[tuple[str, str]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0, {0: (5.0, 900.0)}))
    assert tracker.contacts == ()
    tracker.update(_snapshot(20.0, {0: (20.0, 400.0)}))
    tracker.update(_snapshot(20.1, {0: (20.0, 400.0), 1: (20.1, 380.0)}))
    tracker.update(_snapshot(21.0, {0: (21.0, 300.0), 1: (20.1, 380.0)}))
    assert recorded == [("incident", "Contact between you and Ann, Hyper at turn 1")]
    (contact,) = tracker.contacts
    assert (contact.driver, contact.other_driver, contact.involves_player, contact.place) == (
        "Me",
        "Ann",
        True,
        "turn 1",
    )


def test_solo_hits_are_kept_for_the_player_and_only_hard_ones_for_others() -> None:
    """With no car near, the player's hit is kept, another car's light hit is dropped and its hard hit is kept;
    hits below the minimum never count."""
    recorded: list[tuple[str, str]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0, {}, far=True))
    tracker.update(_snapshot(20.0, {0: (20.0, 150.0), 2: (20.0, 200.0), 1: (20.0, 50.0)}, far=True))
    tracker.update(_snapshot(30.0, {0: (20.0, 150.0), 2: (30.0, 800.0), 1: (20.0, 50.0)}, far=True))
    assert [text for _, text in recorded] == ["You hit something alone at turn 1", "Bob hit something alone at turn 1"]


def test_new_session_starts_without_contacts() -> None:
    """A session clock that goes back drops the contacts and seeds again."""
    recorded: list[tuple[str, str]] = []
    tracker = _tracker(recorded)
    tracker.update(_snapshot(10.0, {}))
    tracker.update(_snapshot(20.0, {0: (20.0, 400.0)}))
    tracker.update(_snapshot(5.0, {0: (4.0, 400.0)}))
    assert tracker.contacts == ()
