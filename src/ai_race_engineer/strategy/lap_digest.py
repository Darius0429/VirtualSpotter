from collections.abc import Sequence

from ai_race_engineer.strategy.race_state import RaceState
from ai_race_engineer.strategy.timeline import TimelineEntry

OWN_KINDS: tuple[str, ...] = ("question", "story")


def lap_digest(
    state: RaceState,
    history: Sequence[dict[str, object]],
    entries: Sequence[TimelineEntry],
    lap: int,
    loss_s: float,
    trend_laps: int,
) -> dict[str, object]:
    """What happened in the player's class on the lap the player just finished, as plain facts: every car of the
    class with its place now and a lap ago, the gap to the player now, a lap ago and trend_laps laps ago, its last
    lap against its pace, its pit stops and what its energy allows, and the timeline entries of the lap.
    In:  state       RaceState
         history     Sequence[dict[str, object]]  the digests of the laps before, oldest first
         entries     Sequence[TimelineEntry]      the session timeline
         lap         int                          the lap just finished, as the driver counts it
         loss_s      float                        s a last lap must be slower than the car's pace to be marked lost
         trend_laps  int                          laps back over which the gap trend is given
    Out:             dict[str, object]            {lap, my_place, my_last_laps_s (oldest first), trend_laps, cars:
                                                  [{slot_id, place, driver, me, place_a_lap_ago, gap_to_me_s
                                                  (positive = ahead of the player), gap_to_me_a_lap_ago_s,
                                                  gap_to_me_trend_laps_ago_s, last_lap_s, pace_s, lost_s, pit_stops,
                                                  in_pits, stops_remaining, laps_of_energy}], events: [text]}
    """
    player = next(v for v in state.vehicles if v.slot_id == state.player_slot_id)
    in_class = sorted((v for v in state.vehicles if v.class_name == player.class_name), key=lambda v: v.place)
    inputs = {car.slot_id: car for car in state.cars}
    mine = inputs[state.player_slot_id]
    lap_ago = _cars(history, 1)
    trend_ago = _cars(history, trend_laps)
    cars: list[dict[str, object]] = []
    for place, vehicle in enumerate(in_class, start=1):
        car = inputs.get(vehicle.slot_id)
        estimate = state.estimates.get(vehicle.slot_id)
        projection = state.projection.cars.get(vehicle.slot_id)
        pace_s = None if estimate is None else estimate.lap_time_s
        slow = pace_s is not None and vehicle.last_lap_s is not None and vehicle.last_lap_s - pace_s >= loss_s
        gap_s = None if car is None else round((car.progress_laps - mine.progress_laps) * mine.lap_time_s, 1)
        cars.append(
            {
                "slot_id": vehicle.slot_id,
                "place": place,
                "driver": vehicle.driver_name,
                "me": vehicle.slot_id == state.player_slot_id,
                "place_a_lap_ago": lap_ago.get(vehicle.slot_id, {}).get("place"),
                "gap_to_me_s": gap_s,
                "gap_to_me_a_lap_ago_s": lap_ago.get(vehicle.slot_id, {}).get("gap_to_me_s"),
                "gap_to_me_trend_laps_ago_s": trend_ago.get(vehicle.slot_id, {}).get("gap_to_me_s"),
                "last_lap_s": None if vehicle.last_lap_s is None else round(vehicle.last_lap_s, 1),
                "pace_s": None if pace_s is None else round(pace_s, 1),
                "lost_s": round(vehicle.last_lap_s - pace_s, 1) if slow else None,
                "pit_stops": vehicle.pitstops,
                "in_pits": vehicle.in_pits,
                "stops_remaining": None if projection is None else projection.stops_remaining,
                "laps_of_energy": None
                if projection is None or projection.laps_on_board is None
                else round(projection.laps_on_board, 1),
            }
        )
    my_laps = [_mine(digest)["last_lap_s"] for digest in history[-trend_laps:]]
    return {
        "lap": lap,
        "my_place": next(car["place"] for car in cars if car["me"]),
        "my_last_laps_s": [*my_laps, None if player.last_lap_s is None else round(player.last_lap_s, 1)],
        "trend_laps": trend_laps,
        "cars": cars,
        "events": [entry.text for entry in entries if entry.lap == lap and entry.kind not in OWN_KINDS],
    }


def _cars(history: Sequence[dict[str, object]], laps_back: int) -> dict[int, dict[str, object]]:
    """The cars of the digest a number of laps back.
    In:  history    Sequence[dict[str, object]]  digests, oldest first
         laps_back  int                          1 = the lap before
    Out:            dict[int, dict[str, object]]  slot id -> car; empty when the history is shorter
    """
    if len(history) < laps_back:
        return {}
    return {car["slot_id"]: car for car in history[-laps_back]["cars"]}


def _mine(digest: dict[str, object]) -> dict[str, object]:
    """The player's car in a digest.
    In:  digest  dict[str, object]
    Out:         dict[str, object]
    """
    return next(car for car in digest["cars"] if car["me"])
