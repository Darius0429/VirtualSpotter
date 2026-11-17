from ai_race_engineer.domains.common.callouts.base import Callout, Event, Priority
from ai_race_engineer.domains.common.analysis.strategy.consumption import ConsumptionTracker
from ai_race_engineer.domains.common.analysis.strategy.fuel import amount_to_add, amount_to_finish
from ai_race_engineer.domains.common.analysis.pace.laps import lap_fraction, laps_remaining, reference_lap_time_s
from ai_race_engineer.domains.common.data.snapshot import Snapshot


class LowFuel(Event):
    """Warns once per stint when fuel or virtual energy on board covers few laps and will not reach the flag.
    _consumption  ConsumptionTracker  shared, updated by the monitor before events run
    _fuel         _LowAmount          fuel latch
    _energy       _LowAmount          virtual energy latch
    """

    name = "low_fuel"

    def __init__(self, consumption: ConsumptionTracker, low_fuel_laps: float, low_energy_laps: float) -> None:
        """Set the consumption source and warning thresholds.
        In:  consumption      ConsumptionTracker
             low_fuel_laps    float  laps of fuel on board at or below which to warn
             low_energy_laps  float  laps of virtual energy on board at or below which to warn
        Out:                  None
        """
        self._consumption: ConsumptionTracker = consumption
        self._fuel: _LowAmount = _LowAmount(low_fuel_laps)
        self._energy: _LowAmount = _LowAmount(low_energy_laps)

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Low fuel callout, else low energy callout; nothing while in the pits.
        In:  snapshot  Snapshot
             now_s     float     s monotonic clock (unused)
        Out:           Callout | None  kind "low_fuel" (litres) or "low_energy" (fraction); facts laps_on_board,
                                       and laps_remaining, to_add when laps remaining is known
        """
        player = snapshot.player
        if player.vehicle.in_pits:
            return None
        laps_left = self._laps_remaining(snapshot)
        estimate = self._consumption.estimate
        laps = self._fuel.check(player.fuel_l, estimate.fuel_l_per_lap, laps_left)
        if laps is not None:
            if laps_left is None:
                return Callout("low_fuel", f"Fuel for {laps:.1f} laps", Priority.STRATEGY, {"laps_on_board": laps})
            to_add = amount_to_add(amount_to_finish(estimate.fuel_l_per_lap, laps_left), player.fuel_l)
            return Callout(
                "low_fuel",
                f"Fuel for {laps:.1f} laps, {to_add:.0f} litres short",
                Priority.STRATEGY,
                {"laps_on_board": laps, "laps_remaining": laps_left, "to_add": to_add},
            )
        if player.virtual_energy is None:
            return None
        laps = self._energy.check(player.virtual_energy, estimate.energy_per_lap, laps_left)
        if laps is None:
            return None
        if laps_left is None:
            return Callout("low_energy", f"Energy for {laps:.1f} laps", Priority.STRATEGY, {"laps_on_board": laps})
        to_add = amount_to_add(amount_to_finish(estimate.energy_per_lap, laps_left), player.virtual_energy)
        return Callout(
            "low_energy",
            f"Energy for {laps:.1f} laps, {to_add * 100:.0f} percent short",
            Priority.STRATEGY,
            {"laps_on_board": laps, "laps_remaining": laps_left, "to_add": to_add},
        )

    def reset(self) -> None:
        """Re-arm both warnings for a new session.
        In:  -
        Out: None
        """
        self._fuel = _LowAmount(self._fuel.threshold_laps)
        self._energy = _LowAmount(self._energy.threshold_laps)

    def _laps_remaining(self, snapshot: Snapshot) -> float | None:
        """Laps the player still drives, based on the overall leader.
        In:  snapshot  Snapshot
        Out:           float | None  laps; None when no car is in P1 or the leader or player has no reference lap time
        """
        session = snapshot.session
        player = snapshot.player.vehicle
        leader = next((v for v in snapshot.vehicles if v.place == 1), None)
        if leader is None:
            return None
        leader_lap_s = reference_lap_time_s(leader)
        player_lap_s = reference_lap_time_s(player)
        if leader_lap_s is None or player_lap_s is None:
            return None
        return laps_remaining(
            session.remaining_s,
            session.max_laps,
            leader.completed_laps + lap_fraction(leader.lap_distance_m, session.track_length_m),
            leader_lap_s,
            player.completed_laps + lap_fraction(player.lap_distance_m, session.track_length_m),
            player_lap_s,
            player.place == 1,
        )


class _LowAmount:
    """Once-per-stint latch for a low amount (fuel or energy); a rising amount (refuel) re-arms it.
    threshold_laps  float         laps
    _warned         bool
    _last_amount    float | None  litres or fraction at the previous check
    """

    def __init__(self, threshold_laps: float) -> None:
        """Set the warning threshold.
        In:  threshold_laps  float  laps
        Out:                 None
        """
        self.threshold_laps: float = threshold_laps
        self._warned: bool = False
        self._last_amount: float | None = None

    def check(self, amount: float, per_lap: float | None, laps_remaining: float | None) -> float | None:
        """Laps on board when a warning is due.
        In:  amount          float         litres or fraction on board
             per_lap         float | None  litres/lap or fraction/lap
             laps_remaining  float | None  laps to the flag (None = unknown, warn on threshold alone)
        Out:                 float | None  laps on board; None when no warning is due
        """
        if self._last_amount is not None and amount > self._last_amount:
            self._warned = False
        self._last_amount = amount
        if self._warned or per_lap is None:
            return None
        laps_on_board = amount / per_lap
        if laps_on_board > self.threshold_laps or (laps_remaining is not None and laps_remaining <= laps_on_board):
            return None
        self._warned = True
        return laps_on_board
