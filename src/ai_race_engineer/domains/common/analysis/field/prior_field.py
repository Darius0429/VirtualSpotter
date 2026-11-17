from collections.abc import Callable, Sequence
from dataclasses import replace

from ai_race_engineer.domains.common.analysis.field.field import CarEstimate, FieldTracker
from ai_race_engineer.domains.common.analysis.field.priors import (
    EventPriors,
    PriorSource,
    class_usage,
    fill_estimate,
    history_pace_s,
    prior_energy_per_lap,
    prior_pace_s,
)
from ai_race_engineer.domains.common.data.garage import GarageInfo
from ai_race_engineer.domains.common.data.race_snapshot import CarResources
from ai_race_engineer.domains.common.data.snapshot import VehicleState

HISTORY_SOURCES: tuple[str, str] = ("history_car", "history_class")


class PriorFieldTracker(FieldTracker):
    """Field tracker whose estimates are filled with priors while a car has no measured pace or usage, so a race can
    be projected from the grid. Until the player has completed strategy_start_laps laps the priors are used for every
    car and the laps measured meanwhile are only collected; from then on measured values replace priors field by
    field.
    measured      dict[int, CarEstimate]             slot id -> what was measured this session, before priors
                                                     (replaced each update, read from other threads)
    sources       dict[int, PriorSource]             slot id -> where pace and usage came from (replaced each update)
    pre_race      bool                               the pre-race figures are in use for every car (read from other
                                                     threads)
    _pace_factor  float                              race pace = factor * a qualifying or best lap
    _priors       Callable[[], EventPriors | None]   the session's priors (non-blocking)
    _garage       Callable[[], GarageInfo | None]    the game's garage figures for the player (non-blocking)
    _strategy_start_laps  int                        laps the player completes before measured values take over
    """

    def __init__(
        self,
        pace_window_laps: int,
        consumption_window_laps: int,
        pace_factor: float,
        priors: Callable[[], EventPriors | None],
        garage: Callable[[], GarageInfo | None],
        strategy_start_laps: int,
    ) -> None:
        """Set the averaging windows and the sources of the priors.
        In:  pace_window_laps         int                               laps
             consumption_window_laps  int                               laps
             pace_factor              float                             race pace = factor * a qualifying or best lap
             priors                   Callable[[], EventPriors | None]  the session's priors (non-blocking)
             garage                   Callable[[], GarageInfo | None]   the player's garage figures (non-blocking)
             strategy_start_laps      int                               laps the player completes before the measured
                                                                        values replace the priors
        Out:                          None
        """
        super().__init__(pace_window_laps, consumption_window_laps)
        self.measured: dict[int, CarEstimate] = {}
        self.sources: dict[int, PriorSource] = {}
        self.pre_race: bool = True
        self._pace_factor: float = pace_factor
        self._priors: Callable[[], EventPriors | None] = priors
        self._garage: Callable[[], GarageInfo | None] = garage
        self._strategy_start_laps: int = strategy_start_laps

    def update(
        self, vehicles: Sequence[VehicleState], resources: Sequence[CarResources], game_phase: int
    ) -> dict[int, CarEstimate]:
        """Feed the current state of all cars.
        In:  vehicles    Sequence[VehicleState]  all cars
             resources   Sequence[CarResources]  cars with telemetry
             game_phase  int                     mGamePhase (5 = green)
        Out:             dict[int, CarEstimate]  slot id -> estimate with priors filled in, for every car in vehicles
        """
        measured = super().update(vehicles, resources, game_phase)
        priors, garage = self._priors(), self._garage()
        pre_race = any(v.is_player and v.completed_laps < self._strategy_start_laps for v in vehicles)
        used = {
            v.slot_id: CarEstimate(v.slot_id, None, None, None, None) if pre_race else measured[v.slot_id]
            for v in vehicles
        }
        best_lap_s = {v.slot_id: None if pre_race else v.best_lap_s for v in vehicles}
        own_pace = {v.slot_id: self._pace(v, used[v.slot_id], best_lap_s[v.slot_id], priors, None) for v in vehicles}
        own_usage = {v.slot_id: self._usage(v, used[v.slot_id], priors, garage, None) for v in vehicles}
        class_best: dict[str, float] = {}
        by_class: dict[str, list[tuple[float | None, str]]] = {}
        for vehicle in vehicles:
            pace_s, pace_source = own_pace[vehicle.slot_id]
            if pace_s is not None and pace_source not in HISTORY_SOURCES:
                class_best[vehicle.class_name] = min(pace_s, class_best.get(vehicle.class_name, pace_s))
            by_class.setdefault(vehicle.class_name, []).append(own_usage[vehicle.slot_id])
        usage_of_class = {name: class_usage(figures) for name, figures in by_class.items()}
        estimates: dict[int, CarEstimate] = {}
        sources: dict[int, PriorSource] = {}
        for vehicle in vehicles:
            estimate = used[vehicle.slot_id]
            pace_s, pace_source = self._pace(
                vehicle, estimate, best_lap_s[vehicle.slot_id], priors, class_best.get(vehicle.class_name)
            )
            energy, energy_source = self._usage(vehicle, estimate, priors, garage, usage_of_class[vehicle.class_name])
            filled = fill_estimate(estimate, pace_s, energy, self._ratio(vehicle, priors, garage))
            estimates[vehicle.slot_id] = replace(filled, clean_laps_s=measured[vehicle.slot_id].clean_laps_s)
            sources[vehicle.slot_id] = PriorSource(pace_source, energy_source)
        self.measured, self.sources, self.pre_race = measured, sources, pre_race
        return estimates

    def _pace(
        self,
        vehicle: VehicleState,
        estimate: CarEstimate,
        best_lap_s: float | None,
        priors: EventPriors | None,
        class_best_s: float | None,
    ) -> tuple[float | None, str]:
        """Pace of one car and where it comes from: measured once it has a clean lap in use, else the first prior.
        In:  vehicle       VehicleState
             estimate      CarEstimate         the measured values in use (empty before the strategy starts)
             best_lap_s    float | None        s, its best lap this session when that may be used
             priors        EventPriors | None  None while the priors are not known yet
             class_best_s  float | None        s/lap, the fastest pace known in its class
        Out:               tuple[float | None, str]  (s/lap, PriorSource.pace)
        """
        if estimate.clean_laps_s:
            return estimate.lap_time_s, "measured"
        if priors is None:
            return prior_pace_s(None, None, best_lap_s, class_best_s, None, None, self._pace_factor)
        known = priors.remembered.get((vehicle.driver_name, vehicle.class_name))
        car = priors.car_history if vehicle.is_player else None
        of_class = priors.class_history.get(vehicle.class_name)
        return prior_pace_s(
            priors.qualifying_best_s.get(vehicle.driver_name),
            None if known is None else known.lap_time_s,
            best_lap_s,
            class_best_s,
            None if car is None else history_pace_s(car.race_pace_s, car.best_qualifying_lap_s, self._pace_factor),
            None if of_class is None else history_pace_s(of_class.race_pace_s, of_class.pole_lap_s, self._pace_factor),
            self._pace_factor,
        )

    def _usage(
        self,
        vehicle: VehicleState,
        estimate: CarEstimate,
        priors: EventPriors | None,
        garage: GarageInfo | None,
        class_usage_per_lap: float | None,
    ) -> tuple[float | None, str]:
        """Energy use per lap of one car and where it comes from: measured once it has a clean lap in use, else the
        first prior; the history of the car and the garage figure count for the player only.
        In:  vehicle              VehicleState
             estimate             CarEstimate         the measured values in use (empty before the strategy starts)
             priors               EventPriors | None  None while the priors are not known yet
             garage               GarageInfo | None   the player's garage figures
             class_usage_per_lap  float | None        fraction/lap, the best-known figure of its class
        Out:                      tuple[float | None, str]  (fraction/lap, PriorSource.energy)
        """
        if estimate.energy_per_lap is not None:
            return estimate.energy_per_lap, "measured"
        known = None if priors is None else priors.remembered.get((vehicle.driver_name, vehicle.class_name))
        car = None if priors is None or not vehicle.is_player else priors.car_history
        of_class = None if priors is None else priors.class_history.get(vehicle.class_name)
        return prior_energy_per_lap(
            None if known is None else known.energy_per_lap,
            None if car is None else car.energy_per_lap,
            garage.energy_per_lap if garage is not None and vehicle.is_player else None,
            class_usage_per_lap,
            None if of_class is None else of_class.energy_per_lap,
        )

    def _ratio(self, vehicle: VehicleState, priors: EventPriors | None, garage: GarageInfo | None) -> float | None:
        """Fuel per energy of one car while it is not measured: remembered, then the history of the player's car,
        then the player's garage ratio, then the history of the class.
        In:  vehicle  VehicleState
             priors   EventPriors | None  None while the priors are not known yet
             garage   GarageInfo | None   the player's garage figures
        Out:          float | None        litres per 1 % energy
        """
        known = None if priors is None else priors.remembered.get((vehicle.driver_name, vehicle.class_name))
        car = None if priors is None or not vehicle.is_player else priors.car_history
        of_class = None if priors is None else priors.class_history.get(vehicle.class_name)
        candidates = (
            None if known is None else known.fuel_per_energy,
            None if car is None else car.fuel_per_energy,
            garage.fuel_ratio if garage is not None and vehicle.is_player else None,
            None if of_class is None else of_class.fuel_per_energy,
        )
        return next((ratio for ratio in candidates if ratio is not None), None)
