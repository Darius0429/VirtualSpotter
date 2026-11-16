from collections.abc import Sequence
from dataclasses import dataclass

from ai_race_engineer.strategy.pit_time import PitStopSample
from ai_race_engineer.telemetry.pit_estimate import PitEstimate
from ai_race_engineer.telemetry.race_snapshot import CarResources
from ai_race_engineer.telemetry.snapshot import VehicleState


class PitStopLog:
    """Measures every car's passes through the pit lane; a pass during which the car is in the garage is dropped.
    _stop_speed_mps  float               m/s below which a car in the pits counts as stationary
    _samples         list[PitStopSample]  finished passes
    _visits          dict[int, _Visit]    slot id -> pass in progress whose entry was seen
    _in_pits         dict[int, bool]      slot id -> in pits at the previous update
    _last_time_s     float                s session clock of the previous update
    _player_laps     int | None           player's completed laps at the previous update
    """

    def __init__(self, stop_speed_mps: float) -> None:
        """Set the stationary threshold.
        In:  stop_speed_mps  float  m/s
        Out:                 None
        """
        self._stop_speed_mps: float = stop_speed_mps
        self._samples: list[PitStopSample] = []
        self._visits: dict[int, _Visit] = {}
        self._in_pits: dict[int, bool] = {}
        self._last_time_s: float = 0.0
        self._player_laps: int | None = None

    def update(
        self,
        vehicles: Sequence[VehicleState],
        resources: Sequence[CarResources],
        track_length_m: float,
        time_s: float,
        estimate: PitEstimate | None,
    ) -> tuple[PitStopSample, ...]:
        """Feed the current state of all cars; the log is cleared when the player's lap count goes down.
        In:  vehicles        Sequence[VehicleState]  all cars, including the player
             resources       Sequence[CarResources]  cars with telemetry
             track_length_m  float                   m
             time_s          float                   s session clock
             estimate        PitEstimate | None      the game's latest estimate for the player
        Out:                 tuple[PitStopSample, ...]  all passes measured so far
        """
        player_laps = next(v.completed_laps for v in vehicles if v.is_player)
        if self._player_laps is not None and player_laps < self._player_laps:
            self._samples, self._visits, self._in_pits = [], {}, {}
        self._player_laps = player_laps
        resources_by_id = {r.slot_id: r for r in resources}
        for vehicle in vehicles:
            car = resources_by_id.get(vehicle.slot_id)
            visit = self._visits.get(vehicle.slot_id)
            if vehicle.in_garage:
                self._visits.pop(vehicle.slot_id, None)
            elif vehicle.in_pits and self._in_pits.get(vehicle.slot_id) is False:
                self._visits[vehicle.slot_id] = _Visit(
                    time_s, vehicle.lap_distance_m, _fuel_l(car), _energy(car), estimate if vehicle.is_player else None
                )
            elif vehicle.in_pits and visit is not None and vehicle.speed_mps < self._stop_speed_mps:
                visit.stationary_s += time_s - self._last_time_s
            elif not vehicle.in_pits and visit is not None:
                del self._visits[vehicle.slot_id]
                self._samples.append(
                    PitStopSample(
                        slot_id=vehicle.slot_id,
                        is_player=vehicle.is_player,
                        lane_time_s=time_s - visit.entry_time_s - visit.stationary_s,
                        lane_length_m=(vehicle.lap_distance_m - visit.entry_lap_distance_m) % track_length_m,
                        stationary_s=visit.stationary_s,
                        energy_added=_energy(car) - visit.entry_energy,
                        fuel_added_l=_fuel_l(car) - visit.entry_fuel_l,
                        estimate=visit.estimate,
                    )
                )
            self._in_pits[vehicle.slot_id] = vehicle.in_pits
        self._last_time_s = time_s
        return tuple(self._samples)

    def samples(self) -> tuple[PitStopSample, ...]:
        """The passes measured so far (read from other blocks and threads).
        In:  -
        Out: tuple[PitStopSample, ...]
        """
        return tuple(self._samples)


@dataclass
class _Visit:
    """A pass through the pit lane in progress.
    entry_time_s          float               s session clock at pit entry
    entry_lap_distance_m  float               m lap distance at pit entry
    entry_fuel_l          float               litres at pit entry
    entry_energy          float               fraction of virtual energy at pit entry
    estimate              PitEstimate | None  the game's estimate at pit entry (player only)
    stationary_s          float               s standing still so far
    """

    entry_time_s: float
    entry_lap_distance_m: float
    entry_fuel_l: float
    entry_energy: float
    estimate: PitEstimate | None
    stationary_s: float = 0.0


def _fuel_l(car: CarResources | None) -> float:
    """Fuel on board.
    In:  car  CarResources | None
    Out:      float  litres; 0 for a car without telemetry
    """
    return 0.0 if car is None else car.fuel_l


def _energy(car: CarResources | None) -> float:
    """Virtual energy on board.
    In:  car  CarResources | None
    Out:      float  fraction; 0 for a car without telemetry or without virtual energy
    """
    return 0.0 if car is None or car.virtual_energy is None else car.virtual_energy
