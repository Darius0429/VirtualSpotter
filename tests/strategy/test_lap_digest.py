from types import SimpleNamespace

from ai_race_engineer.strategy.lap_digest import lap_digest
from ai_race_engineer.strategy.timeline import TimelineEntry


def _state() -> SimpleNamespace:
    """Race state stand-in: the player (slot 0) is P2 in class behind Ann and ahead of Bob, with a GT3 car between
    them on the overall order; Bob's last lap was 5 s off his pace and he has pitted once.
    In:  -
    Out: SimpleNamespace
    """

    def vehicle(
        slot_id: int, name: str, class_name: str, place: int, last_s: float, pitstops: int = 0
    ) -> SimpleNamespace:
        return SimpleNamespace(
            slot_id=slot_id,
            driver_name=name,
            class_name=class_name,
            place=place,
            last_lap_s=last_s,
            pitstops=pitstops,
            in_pits=False,
        )

    def car(slot_id: int, progress_laps: float) -> SimpleNamespace:
        return SimpleNamespace(slot_id=slot_id, progress_laps=progress_laps, lap_time_s=100.0)

    def projection(stops: int, laps: float) -> SimpleNamespace:
        return SimpleNamespace(stops_remaining=stops, laps_on_board=laps)

    return SimpleNamespace(
        player_slot_id=0,
        vehicles=[
            vehicle(1, "Ann", "Hyper", 1, 100.2),
            vehicle(0, "Me", "Hyper", 2, 100.0),
            vehicle(9, "Gus", "GT3", 3, 112.0),
            vehicle(2, "Bob", "Hyper", 4, 105.0, pitstops=1),
        ],
        cars=[car(1, 10.52), car(0, 10.50), car(9, 9.4), car(2, 10.45)],
        estimates={slot: SimpleNamespace(lap_time_s=100.0) for slot in (0, 1, 2)},
        projection=SimpleNamespace(cars={0: projection(1, 11.5), 1: projection(1, 12.0), 2: projection(0, 20.0)}),
    )


def _digests() -> list[dict[str, object]]:
    """The digest of lap 9 (Bob P2 and 1 s ahead, the player P3) and, built on it, of lap 10 from _state().
    In:  -
    Out: list[dict[str, object]]
    """
    before = _state()
    ann, me, gus, bob = before.vehicles
    before.vehicles = [ann, bob, gus, me]
    bob.place, me.place = 2, 4
    before.cars[3].progress_laps = 10.51
    lap_9 = lap_digest(before, [], [], 9, 3.0, 1)
    entries = [
        TimelineEntry(900.0, 9, "pit", "Bob out after 30.0 seconds stationary", {}),
        TimelineEntry(1000.0, 10, "position", "You gained P2 from Bob", {}),
        TimelineEntry(1001.0, 10, "question", "Asked: who is ahead", {}),
        TimelineEntry(1002.0, 10, "story", "Lap 9: Bob pitted", {}),
    ]
    return [lap_9, lap_digest(_state(), [lap_9], entries, 10, 3.0, 1)]


def test_class_places_count_only_the_players_class() -> None:
    """The GT3 car between two Hypercars does not take a class place."""
    digest = _digests()[1]
    assert [(car["driver"], car["place"]) for car in digest["cars"]] == [("Ann", 1), ("Me", 2), ("Bob", 3)]


def test_digest_lists_the_class_with_changes_gaps_losses_and_the_laps_events() -> None:
    """Each car of the class has its place now and a lap ago, the gap to the player now and before, a marked loss
    when its lap was far off its pace, and what its energy allows; only this lap's timeline entries that are not
    the model's own are passed on."""
    digest = _digests()[1]
    assert (digest["lap"], digest["my_place"], digest["my_last_laps_s"]) == (10, 2, [100.0, 100.0])
    assert digest["events"] == ["You gained P2 from Bob"]
    ann, me, bob = digest["cars"]
    assert (ann["driver"], ann["place"], ann["gap_to_me_s"], ann["lost_s"]) == ("Ann", 1, 2.0, None)
    assert (me["me"], me["place"], me["place_a_lap_ago"], me["gap_to_me_s"]) == (True, 2, 3, 0.0)
    assert (bob["place"], bob["place_a_lap_ago"], bob["lost_s"]) == (3, 2, 5.0)
    assert (bob["gap_to_me_s"], bob["gap_to_me_a_lap_ago_s"], bob["gap_to_me_trend_laps_ago_s"]) == (-5.0, 1.0, 1.0)
    assert (bob["pit_stops"], bob["stops_remaining"], bob["laps_of_energy"]) == (1, 0, 20.0)
