from dataclasses import dataclass

from ai_race_engineer.strategy.fuel import LapConsumption
from ai_race_engineer.telemetry.snapshot import PlayerState


@dataclass(frozen=True)
class ConsumptionEstimate:
    """Current per-lap consumption.
    fuel_l_per_lap  float | None  litres/lap
    energy_per_lap  float | None  fraction/lap of virtual energy (None when the game has none)
    """

    fuel_l_per_lap: float | None
    energy_per_lap: float | None


class ConsumptionTracker:
    """Tracks fuel and virtual energy use per lap from the player's state; shared by the monitor and events.
    _window_laps          int                  laps to average over
    _fuel_seed_l_per_lap  float | None         litres/lap
    _energy_seed_per_lap  float | None         fraction/lap
    _fuel                 LapConsumption
    _energy               LapConsumption
    estimate              ConsumptionEstimate  latest estimate (replaced, never mutated)
    """

    def __init__(
        self, window_laps: int, fuel_seed_l_per_lap: float | None = None, energy_seed_per_lap: float | None = None
    ) -> None:
        """Set the averaging window and optional seeds.
        In:  window_laps          int           laps
             fuel_seed_l_per_lap  float | None  litres/lap
             energy_seed_per_lap  float | None  fraction/lap
        Out:                      None
        """
        self._window_laps: int = window_laps
        self._fuel_seed_l_per_lap: float | None = fuel_seed_l_per_lap
        self._energy_seed_per_lap: float | None = energy_seed_per_lap
        self._fuel: LapConsumption = LapConsumption(window_laps, fuel_seed_l_per_lap)
        self._energy: LapConsumption = LapConsumption(window_laps, energy_seed_per_lap)
        self.estimate: ConsumptionEstimate = ConsumptionEstimate(None, None)

    def update(self, player: PlayerState) -> ConsumptionEstimate:
        """Feed the player's lap count, fuel and virtual energy.
        In:  player  PlayerState
        Out:         ConsumptionEstimate  also stored in estimate
        """
        laps = player.vehicle.completed_laps
        energy = None if player.virtual_energy is None else self._energy.update(laps, player.virtual_energy)
        self.estimate = ConsumptionEstimate(self._fuel.update(laps, player.fuel_l), energy)
        return self.estimate

    def reset(self) -> None:
        """Drop measured laps for a new session; seeds are kept.
        In:  -
        Out: None
        """
        self._fuel = LapConsumption(self._window_laps, self._fuel_seed_l_per_lap)
        self._energy = LapConsumption(self._window_laps, self._energy_seed_per_lap)
        self.estimate = ConsumptionEstimate(None, None)
