import dataclasses
from collections.abc import Callable

from ai_race_engineer.domains.common.callouts.rival_watch import RivalWatch
from ai_race_engineer.domains.common.callouts.text import lap_time_text
from ai_race_engineer.domains.common.analysis.pace.pace import class_references, pace_stats
from ai_race_engineer.domains.common.analysis.strategy.race_finish import strategy_resource
from ai_race_engineer.domains.common.analysis.field.race_state import RaceState, RaceTracker
from ai_race_engineer.domains.common.analysis.strategy.rejoin import predict_rejoin
from ai_race_engineer.domains.common.analysis.field.rivals import laps_to_catch
from ai_race_engineer.domains.common.analysis.pace.sector_delta import Splits, splits
from ai_race_engineer.domains.common.analysis.field.traffic import TrafficCar, find_traffic

UNAVAILABLE: dict[str, object] = {"available": False}


class RaceTools:
    """The race-state lookups and the rival-updates switch both agents use; the registration point for agent tools.
    _tracker               RaceTracker  source of the latest race state
    _rival_watch           RivalWatch   switched by set_rival_updates
    _traffic_horizon_laps  float        laps within which other-class cars are reported as traffic
    """

    def __init__(self, tracker: RaceTracker, rival_watch: RivalWatch, traffic_horizon_laps: float) -> None:
        """Store the state source, the rival watch and the traffic horizon.
        In:  tracker               RaceTracker
             rival_watch           RivalWatch
             traffic_horizon_laps  float        laps within which other-class cars are reported as traffic
        Out:                       None
        """
        self._tracker: RaceTracker = tracker
        self._rival_watch: RivalWatch = rival_watch
        self._traffic_horizon_laps: float = traffic_horizon_laps

    def race_finish(self) -> dict[str, object]:
        """How the race is projected to end: the net leader, the time until the chequered flag and the projected
        finishing order of the driver's class with each car's remaining stops. When the net leader's energy or fuel
        use per lap is not known yet, the leader is projected without stops and laps_approximate is true: the flag
        time and the lap counts can then be off by a lap.
        In:  -
        Out: dict[str, object]  net_leader, net_leader_place, flag_time_s, leader_laps_remaining, laps_approximate,
                                class_order ([{driver, projected_class_place, stops_remaining}]);
                                {"available": False} without a race state
        """
        state = self._tracker.state
        if state is None:
            return UNAVAILABLE
        projection = state.projection
        vehicles = {v.slot_id: v for v in state.vehicles}
        class_name = projection.cars[state.player_slot_id].class_name
        in_class = sorted(
            (c for c in projection.cars.values() if c.class_name == class_name), key=lambda c: c.class_place
        )
        return {
            "net_leader": vehicles[projection.leader_slot_id].driver_name,
            "net_leader_place": vehicles[projection.leader_slot_id].place,
            "flag_time_s": projection.flag_time_s,
            "leader_laps_remaining": projection.cars[projection.leader_slot_id].laps_remaining,
            "laps_approximate": strategy_resource(next(c for c in state.cars if c.slot_id == projection.leader_slot_id))
            is None,
            "class_order": [
                {
                    "driver": vehicles[c.slot_id].driver_name,
                    "projected_class_place": c.class_place,
                    "stops_remaining": c.stops_remaining,
                }
                for c in in_class
            ],
        }

    def my_race(self) -> dict[str, object]:
        """The driver's own race to the flag: the lap the driver is on, laps remaining, laps on board, the lap at
        the end of which to pit (lap numbers as the driver counts them: the current lap is completed laps + 1), stops
        remaining, energy and fuel still to add in total to reach the flag (negative = surplus), the fuel used per
        1 % of energy against the ratio set in the garage, and whether fuel runs out before energy. What to add at a
        stop now is in pit_stop.
        In:  -
        Out: dict[str, object]  the player's projection with units in the keys; {"available": False} without a
                                race state
        """
        state = self._tracker.state
        if state is None:
            return UNAVAILABLE
        mine = state.projection.cars[state.player_slot_id]
        car = next(c for c in state.cars if c.slot_id == state.player_slot_id)
        return {
            "current_lap": next(v for v in state.vehicles if v.slot_id == state.player_slot_id).completed_laps + 1,
            "laps_remaining": mine.laps_remaining,
            "finish_time_s": mine.finish_time_s,
            "projected_place": mine.place,
            "projected_class_place": mine.class_place,
            "strategy_resource": mine.resource,
            "stops_remaining": mine.stops_remaining,
            "laps_on_board": mine.laps_on_board,
            "pit_at_end_of_lap": mine.pit_lap,
            "energy_to_add_to_finish_percent": None if mine.energy_to_add is None else mine.energy_to_add * 100,
            "fuel_to_add_to_finish_l": mine.fuel_to_add_l,
            "fuel_per_energy_l_per_percent": car.fuel_per_energy,
            "set_fuel_ratio_l_per_percent": state.set_fuel_ratio,
            "fuel_laps_on_board": mine.fuel_laps_on_board,
            "fuel_margin_laps": mine.fuel_margin_laps,
            "fuel_runs_out_first": mine.fuel_runs_out_first,
        }

    def rivals(self) -> list[dict[str, object]]:
        """The driver's rivals in class: who they are, gap, pace, pit status, projected finish and how many laps
        until the gap closes.
        In:  -
        Out: list[dict[str, object]]  the Rival fields (lap times as "m:ss.s" text) plus laps_to_catch (laps; None
                                      when the gap is not closing);
                                      [{"available": False}] without a race state
        """
        state = self._tracker.state
        if state is None:
            return [UNAVAILABLE]
        return [
            {
                **dataclasses.asdict(r),
                "lap_time_s": _lap_text(r.lap_time_s),
                "best_lap_s": _lap_text(r.best_lap_s),
                "laps_to_catch": laps_to_catch(r.gap_s, r.pace_delta_s),
            }
            for r in state.rivals
        ]

    def pit_stop(self, driver: str = "") -> dict[str, object]:
        """What a pit stop now would cost and where the car rejoins: stop time, energy and fuel to add at this stop
        (what is needed to the flag, at most a full load), position, gaps and traffic. For the driver's own car leave
        driver empty; for another car give the driver's name.
        In:  driver  str  driver name or part of it; empty = the player
        Out:         dict[str, object]  the Rejoin fields, ahead and behind as driver names, and for the player the
                                        game's own estimate as game_estimate_total_s; {"available": False} without
                                        a race state or when no car matches
        """
        state = self._tracker.state
        if state is None:
            return UNAVAILABLE
        slot_id = _slot_id(state, driver)
        if slot_id is None:
            return UNAVAILABLE
        names = {v.slot_id: v.driver_name for v in state.vehicles}
        rejoin = predict_rejoin(
            next(c for c in state.cars if c.slot_id == slot_id),
            state.projection.cars[slot_id].laps_remaining,
            [c for c in state.cars if c.slot_id != slot_id],
            state.pit,
            state.traffic_window_s,
        )
        result: dict[str, object] = {
            "driver": names[slot_id],
            "lane_loss_s": rejoin.lane_loss_s,
            "service_s": rejoin.service_s,
            "stop_time_s": rejoin.stop_time_s,
            "energy_to_add_percent": rejoin.energy_to_add * 100,
            "fuel_to_add_l": rejoin.fuel_to_add_l,
            "rejoin_place": rejoin.place,
            "rejoin_class_place": rejoin.class_place,
            "class_car_ahead": names.get(rejoin.ahead_slot_id),
            "gap_ahead_s": rejoin.gap_ahead_s,
            "class_car_behind": names.get(rejoin.behind_slot_id),
            "gap_behind_s": rejoin.gap_behind_s,
            "traffic_cars": rejoin.traffic,
        }
        if slot_id == state.player_slot_id:
            result["game_estimate_total_s"] = None if state.estimate is None else state.estimate.total_s
        return result

    def my_pace(self) -> dict[str, object]:
        """The driver's pace: last lap, best lap, mean and median of the recent clean laps (laps without pits or an
        invalidated time) and how many there are, the sector times of the last lap and the best sectors, and the mean
        pace of the class leader and of the cars directly ahead and behind in class with their difference to the
        driver (negative = that car is faster).
        In:  -
        Out: dict[str, object]  lap times as "m:ss.s" text, differences in s/lap (negative = that car is faster);
                                sector lists are [S1, S2, S3] in s, None for a sector without a valid time;
                                {"available": False} without a race state
        """
        state = self._tracker.state
        if state is None:
            return UNAVAILABLE
        player = next(v for v in state.vehicles if v.slot_id == state.player_slot_id)
        estimate = state.estimates[player.slot_id]
        stats = pace_stats(player, estimate.clean_laps_s)
        result: dict[str, object] = {
            "average_race_pace": _lap_text(stats.mean_lap_s),
            "last_lap": _lap_text(stats.last_lap_s),
            "best_lap": _lap_text(stats.best_lap_s),
            "median_clean_lap": _lap_text(stats.median_lap_s),
            "clean_laps_in_window": stats.clean_laps,
            "last_lap_sectors_s": list(_last_splits(state, player.slot_id)),
            "best_sectors_s": list(_best_splits(state, player.slot_id)),
        }
        for key, car in class_references(player, state.vehicles).items():
            pace_s = state.estimates[car.slot_id].lap_time_s
            result[key] = {
                "driver": car.driver_name,
                "average_race_pace": _lap_text(pace_s),
                "s_per_lap_slower_than_driver": None if pace_s is None else round(pace_s - estimate.lap_time_s, 2),
            }
        return result

    def traffic(self) -> dict[str, object]:
        """Cars of other classes the driver will meet on the road in the next laps: slower cars ahead that the
        driver catches and faster cars behind that catch the driver, each with the gap on the road now, the closing
        rate, the time and laps until they meet, and the lap (as the driver counts laps) and place on the lap where
        that happens. Nearest first.
        In:  -
        Out: dict[str, object]  "catching" and "caught_by", lists with s, laps and m in the keys;
                                {"available": False} without a race state
        """
        state = self._tracker.state
        if state is None:
            return UNAVAILABLE
        names = {v.slot_id: v.driver_name for v in state.vehicles}
        catching, caught_by = find_traffic(
            next(c for c in state.cars if c.slot_id == state.player_slot_id),
            [c for c in state.cars if c.slot_id != state.player_slot_id],
            state.pit.track_length_m,
            self._traffic_horizon_laps,
        )
        return {
            "catching": [_traffic_entry(car, names) for car in catching],
            "caught_by": [_traffic_entry(car, names) for car in caught_by],
        }

    def set_rival_updates(self, enabled: bool) -> str:
        """Switch the spoken rival updates on or off.
        In:  enabled  bool  True = on
        Out:          str   confirmation sentence
        """
        self._rival_watch.set_enabled(enabled)
        return "Rival updates on." if enabled else "Rival updates off."

    def tools(self) -> list[Callable[..., object]]:
        """The functions an agent may call.
        In:  -
        Out: list[Callable[..., object]]  race_finish, my_race, rivals, pit_stop, my_pace, traffic,
                                          set_rival_updates
        """
        return [
            self.race_finish,
            self.my_race,
            self.rivals,
            self.pit_stop,
            self.my_pace,
            self.traffic,
            self.set_rival_updates,
        ]


def _lap_text(lap_s: float | None) -> str | None:
    """Lap time ready to be said, so the model does not convert seconds itself.
    In:  lap_s  float | None  s
    Out:        str | None    "m:ss.s"; None without a time
    """
    return None if lap_s is None else lap_time_text(lap_s)


def _slot_id(state: RaceState, driver: str) -> int | None:
    """Slot id of the projected car a driver name refers to.
    In:  state   RaceState
         driver  str        driver name or part of it, any case; empty = the player
    Out:         int | None  None when no projected car matches
    """
    if not driver:
        return state.player_slot_id
    projected = {c.slot_id for c in state.cars}
    return next(
        (v.slot_id for v in state.vehicles if v.slot_id in projected and driver.lower() in v.driver_name.lower()), None
    )


def _last_splits(state: RaceState, slot_id: int) -> Splits:
    """Sector times of a car's last completed lap.
    In:  state    RaceState
         slot_id  int
    Out:          Splits  s (S1, S2, S3); None for a sector without a valid time
    """
    sectors = state.sectors[slot_id]
    vehicle = next(v for v in state.vehicles if v.slot_id == slot_id)
    return splits(sectors.last_sector1_s, sectors.last_sector2_cum_s, vehicle.last_lap_s)


def _best_splits(state: RaceState, slot_id: int) -> Splits:
    """Best sector times of a car: its best sector 1, best sectors 1+2 and best lap, split up.
    In:  state    RaceState
         slot_id  int
    Out:          Splits  s (S1, S2, S3); None for a sector without a valid time
    """
    vehicle = next(v for v in state.vehicles if v.slot_id == slot_id)
    return splits(vehicle.best_sector1_s, vehicle.best_sector2_cum_s, vehicle.best_lap_s)


def _traffic_entry(car: TrafficCar, names: dict[int, str]) -> dict[str, object]:
    """One traffic car as tool output.
    In:  car    TrafficCar
         names  dict[int, str]  slot id -> driver name
    Out:        dict[str, object]  units in the keys
    """
    return {
        "driver": names[car.slot_id],
        "class": car.class_name,
        "gap_on_road_s": car.gap_s,
        "closing_s_per_lap": car.closing_s_per_lap,
        "laps_to_meet": car.laps_to_meet,
        "time_to_meet_s": car.time_to_meet_s,
        "meet_on_lap": car.meet_lap,
        "meet_at_lap_fraction": car.meet_lap_fraction,
        "meet_at_lap_distance_m": car.meet_lap_distance_m,
    }
