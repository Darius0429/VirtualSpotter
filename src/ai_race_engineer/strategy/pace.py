import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from ai_race_engineer.telemetry.snapshot import VehicleState


@dataclass(frozen=True)
class PaceStats:
    """Pace numbers of one car.
    last_lap_s    float | None  s; None when the last lap has no valid time
    best_lap_s    float | None  s
    mean_lap_s    float | None  s, mean of the clean-lap window (the projection pace); None without clean laps
    median_lap_s  float | None  s, median of the same window
    clean_laps    int           laps in the window
    """

    last_lap_s: float | None
    best_lap_s: float | None
    mean_lap_s: float | None
    median_lap_s: float | None
    clean_laps: int


def pace_stats(vehicle: VehicleState, clean_laps_s: Sequence[float]) -> PaceStats:
    """Pace numbers of a car from its scoring state and its clean-lap window.
    In:  vehicle       VehicleState
         clean_laps_s  Sequence[float]  s, the clean laps in the pace window
    Out:               PaceStats
    """
    return PaceStats(
        last_lap_s=vehicle.last_lap_s,
        best_lap_s=vehicle.best_lap_s,
        mean_lap_s=statistics.fmean(clean_laps_s) if clean_laps_s else None,
        median_lap_s=statistics.median(clean_laps_s) if clean_laps_s else None,
        clean_laps=len(clean_laps_s),
    )


def class_references(player: VehicleState, vehicles: Sequence[VehicleState]) -> dict[str, VehicleState]:
    """The cars of the player's class its pace is compared with.
    In:  player    VehicleState
         vehicles  Sequence[VehicleState]  all cars, including the player
    Out:           dict[str, VehicleState]  "class_leader", "ahead", "behind"; a key is left out when there is no
                                            such car (the player leads its class, or is first or last in it)
    """
    in_class = sorted((v for v in vehicles if v.class_name == player.class_name), key=lambda v: v.place)
    index = in_class.index(player)
    references: dict[str, VehicleState] = {}
    if index > 0:
        references["class_leader"] = in_class[0]
        references["ahead"] = in_class[index - 1]
    if index < len(in_class) - 1:
        references["behind"] = in_class[index + 1]
    return references
