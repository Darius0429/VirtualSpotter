from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from ai_race_engineer.strategy.laps import reference_lap_time_s
from ai_race_engineer.telemetry.race_snapshot import GAME_PHASE_GREEN, CarResources
from ai_race_engineer.telemetry.snapshot import VehicleState


@dataclass(frozen=True)
class CarEstimate:
    """Current pace and usage of one car.
    slot_id          int           game slot id
    lap_time_s       float | None  s/lap: mean of the last clean laps, else best lap, else None
    fuel_l_per_lap   float | None  litres/lap
    energy_per_lap   float | None  fraction/lap of virtual energy
    fuel_per_energy  float | None  litres per 1 % energy = fuel_l_per_lap / (energy_per_lap * 100)
    clean_laps_s     tuple[float, ...]  s, the clean laps in the pace window, oldest first
    """

    slot_id: int
    lap_time_s: float | None
    fuel_l_per_lap: float | None
    energy_per_lap: float | None
    fuel_per_energy: float | None
    clean_laps_s: tuple[float, ...] = ()


class FieldTracker:
    """Tracks pace and fuel/energy use per lap of every car, from its first line crossing under green.
    _pace_window_laps         int                     clean laps to average the lap time over
    _consumption_window_laps  int                     clean laps to average the usage over
    _cars                     dict[int, _CarHistory]  slot id -> history
    """

    def __init__(self, pace_window_laps: int, consumption_window_laps: int) -> None:
        """Set the averaging windows.
        In:  pace_window_laps         int  laps
             consumption_window_laps  int  laps
        Out:                          None
        """
        self._pace_window_laps: int = pace_window_laps
        self._consumption_window_laps: int = consumption_window_laps
        self._cars: dict[int, _CarHistory] = {}

    def update(
        self, vehicles: Sequence[VehicleState], resources: Sequence[CarResources], game_phase: int
    ) -> dict[int, CarEstimate]:
        """Feed the current state of all cars.
        In:  vehicles    Sequence[VehicleState]  all cars
             resources   Sequence[CarResources]  cars with telemetry
             game_phase  int                     mGamePhase (5 = green)
        Out:             dict[int, CarEstimate]  slot id -> estimate, for every car in vehicles
        """
        resources_by_id = {r.slot_id: r for r in resources}
        estimates: dict[int, CarEstimate] = {}
        for vehicle in vehicles:
            if vehicle.slot_id not in self._cars:
                self._cars[vehicle.slot_id] = _CarHistory(self._pace_window_laps, self._consumption_window_laps)
            estimates[vehicle.slot_id] = self._cars[vehicle.slot_id].update(
                vehicle, resources_by_id.get(vehicle.slot_id), game_phase == GAME_PHASE_GREEN
            )
        return estimates


class _CarHistory:
    """Lap times and usage of one car over its clean laps: laps started at a line crossing under green and driven
    without the pits. A line crossing is a change of the lap start time together with a higher lap count: the game
    also sets the lap start time at the green without a crossing, so the lap of the start is not measured. The lap in
    progress at first sight and the formation lap are dropped; a lap count going down restarts the car.
    _lap_times_s     deque[float]  s, last clean laps
    _fuel_l          deque[float]  litres used on the last clean laps
    _energy          deque[float]  fraction used on the last clean laps
    _measuring       bool          the current lap started at a line crossing under green
    _pitted          bool          the car was in the pits during the current lap
    _lap_start_s     float | None  s session time the current lap started; None before the first update
    _completed_laps  int           completed laps at the previous update
    _start_fuel_l    float | None  litres at the start of the current lap
    _start_energy    float | None  fraction at the start of the current lap
    """

    def __init__(self, pace_window_laps: int, consumption_window_laps: int) -> None:
        """Set the averaging windows.
        In:  pace_window_laps         int  laps
             consumption_window_laps  int  laps
        Out:                          None
        """
        self._lap_times_s: deque[float] = deque(maxlen=pace_window_laps)
        self._fuel_l: deque[float] = deque(maxlen=consumption_window_laps)
        self._energy: deque[float] = deque(maxlen=consumption_window_laps)
        self._measuring: bool = False
        self._pitted: bool = False
        self._lap_start_s: float | None = None
        self._completed_laps: int = 0
        self._start_fuel_l: float | None = None
        self._start_energy: float | None = None

    def update(self, vehicle: VehicleState, resources: CarResources | None, is_green: bool) -> CarEstimate:
        """Feed the car's current state.
        In:  vehicle    VehicleState
             resources  CarResources | None  None for a car without telemetry
             is_green   bool                 the game phase is green
        Out:            CarEstimate
        """
        fuel_l = None if resources is None else resources.fuel_l
        energy = None if resources is None else resources.virtual_energy
        if vehicle.completed_laps < self._completed_laps:
            self._lap_times_s.clear()
            self._fuel_l.clear()
            self._energy.clear()
            self._measuring = False
        elif self._lap_start_s is not None and vehicle.lap_start_s != self._lap_start_s:
            crossed = vehicle.completed_laps > self._completed_laps
            if crossed and self._measuring and not self._pitted:
                self._finish_lap(vehicle, fuel_l, energy)
            self._measuring, self._pitted = is_green and crossed, False
            self._start_fuel_l, self._start_energy = fuel_l, energy
        self._pitted = self._pitted or vehicle.in_pits
        self._lap_start_s, self._completed_laps = vehicle.lap_start_s, vehicle.completed_laps
        fuel_l_per_lap = _mean(self._fuel_l)
        energy_per_lap = _mean(self._energy)
        return CarEstimate(
            slot_id=vehicle.slot_id,
            lap_time_s=_mean(self._lap_times_s) or reference_lap_time_s(vehicle),
            fuel_l_per_lap=fuel_l_per_lap,
            energy_per_lap=energy_per_lap,
            fuel_per_energy=fuel_l_per_lap / (energy_per_lap * 100) if fuel_l_per_lap and energy_per_lap else None,
            clean_laps_s=tuple(self._lap_times_s),
        )

    def _finish_lap(self, vehicle: VehicleState, fuel_l: float | None, energy: float | None) -> None:
        """Store the lap time and usage of the clean lap that just ended; an amount that rose is skipped, and so is
        the lap time of a lap the game gives no time for (invalidated lap).
        In:  vehicle  VehicleState
             fuel_l   float | None  litres on board now
             energy   float | None  fraction on board now
        Out:          None
        """
        if vehicle.last_lap_s is not None:
            self._lap_times_s.append(vehicle.last_lap_s)
        if fuel_l is not None and self._start_fuel_l is not None and fuel_l <= self._start_fuel_l:
            self._fuel_l.append(self._start_fuel_l - fuel_l)
        if energy is not None and self._start_energy is not None and energy <= self._start_energy:
            self._energy.append(self._start_energy - energy)


def _mean(values: Sequence[float]) -> float | None:
    """Mean of the stored laps.
    In:  values  Sequence[float]
    Out:         float | None  None without values
    """
    return sum(values) / len(values) if values else None
