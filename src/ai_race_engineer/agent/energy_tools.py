from collections.abc import Callable

from ai_race_engineer.strategy.race_state import RaceTracker

UNAVAILABLE: dict[str, object] = {"available": False}


class EnergyTools:
    """Agent tool for the energy to leave the pits with and how much lifting and coasting can make up.
    _tracker           RaceTracker           source of the latest race state
    _saving_per_lap    tuple[float, float]   percent of energy per lap that lifting and coasting saves on average,
                                             low and high
    """

    def __init__(self, tracker: RaceTracker, saving_percent_per_lap: tuple[float, float]) -> None:
        """Store the state source and the saving that is realistic.
        In:  tracker                 RaceTracker
             saving_percent_per_lap  tuple[float, float]  percent of energy per lap, low and high
        Out:                         None
        """
        self._tracker: RaceTracker = tracker
        self._saving_per_lap: tuple[float, float] = (saving_percent_per_lap[0], saving_percent_per_lap[1])

    def pit_energy(self) -> dict[str, object]:
        """The energy figure for the driver's pit stop and what saving can and cannot do. The driver sets in the
        car the energy to LEAVE the pits with, not the amount to add, so answer with leave_pits_with_percent. It
        covers the laps after the stop at the current usage. Lifting and coasting saves only a little per lap
        (realistic_saving_percent_per_lap, low and high), so over the laps after the stop the driver can make up at
        most can_save_after_stop_percent: a short fill below leave_pits_with_percent minus that amount does not
        reach the flag. Without a stop left, the same is given for the laps to the flag: short_by_percent is what
        is missing (negative = spare) and can_save_to_flag_percent what saving can make up. Use it for how much
        energy or fuel the driver needs, what to set at the stop, short fuelling and how much to save.
        In:  -
        Out: dict[str, object]  {usage_percent_per_lap, energy_on_board_percent, laps_remaining,
                                realistic_saving_percent_per_lap, and with a stop left: pit_at_end_of_lap,
                                energy_at_stop_percent, laps_after_stop, leave_pits_with_percent, add_percent,
                                can_save_after_stop_percent, shortest_fill_leave_with_percent; without one:
                                short_by_percent, can_save_to_flag_percent, reachable_by_saving};
                                {"available": False} without a race state or when the energy use is not known
        """
        state = self._tracker.state
        if state is None:
            return UNAVAILABLE
        car = next(c for c in state.cars if c.slot_id == state.player_slot_id)
        projection = state.projection.cars[state.player_slot_id]
        if car.virtual_energy is None or not car.energy_per_lap:
            return UNAVAILABLE
        low, high = self._saving_per_lap
        usage = car.energy_per_lap * 100
        on_board = car.virtual_energy * 100
        result: dict[str, object] = {
            "usage_percent_per_lap": round(usage, 2),
            "energy_on_board_percent": round(on_board, 1),
            "laps_remaining": round(projection.laps_remaining, 1),
            "realistic_saving_percent_per_lap": [low, high],
        }
        if projection.pit_lap is None:
            short = projection.laps_remaining * usage - on_board
            saving = [round(projection.laps_remaining * low, 2), round(projection.laps_remaining * high, 2)]
            return {
                **result,
                "short_by_percent": round(short, 2),
                "can_save_to_flag_percent": saving,
                "reachable_by_saving": short <= saving[1],
            }
        to_stop = max(projection.pit_lap - car.progress_laps, 0.0)
        after = max(projection.laps_remaining - to_stop, 0.0)
        at_stop = max(on_board - to_stop * usage, 0.0)
        leave_with = min(after * usage, 100.0)
        saving = [round(after * low, 2), round(after * high, 2)]
        return {
            **result,
            "pit_at_end_of_lap": projection.pit_lap,
            "energy_at_stop_percent": round(at_stop, 1),
            "laps_after_stop": round(after, 1),
            "leave_pits_with_percent": round(leave_with, 1),
            "add_percent": round(leave_with - at_stop, 1),
            "can_save_after_stop_percent": saving,
            "shortest_fill_leave_with_percent": round(leave_with - saving[1], 1),
        }

    def tools(self) -> list[Callable[..., object]]:
        """The functions an agent may call.
        In:  -
        Out: list[Callable[..., object]]  pit_energy
        """
        return [self.pit_energy]
