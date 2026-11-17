import dataclasses
from collections.abc import Callable, Sequence

from ai_race_engineer.domains.common.tools.race_tools import UNAVAILABLE, RaceTools
from ai_race_engineer.domains.common.callouts.grid_briefing import GridBriefing
from ai_race_engineer.domains.common.callouts.rival_watch import RivalWatch
from ai_race_engineer.domains.common.callouts.text import lap_time_text
from ai_race_engineer.domains.common.store.session_keeper import SessionKeeper
from ai_race_engineer.domains.common.analysis.strategy.pit_time import lane_loss_s, service_time_s
from ai_race_engineer.domains.common.analysis.field.prior_field import PriorFieldTracker
from ai_race_engineer.domains.common.analysis.field.priors import PriorSource
from ai_race_engineer.domains.common.analysis.strategy.race_finish import strategy_resource
from ai_race_engineer.domains.common.analysis.strategy.race_plan import race_plan
from ai_race_engineer.domains.common.analysis.field.race_state import RaceTracker
from ai_race_engineer.domains.common.analysis.strategy.saving import (
    SavingNeed,
    check_saving,
    saving_for_extra_lap,
    saving_to_reach,
    saving_to_shorten_stop,
    saving_to_skip_stop,
)
from ai_race_engineer.domains.common.analysis.session.session_log import SessionLog

NO_PACE: dict[str, object] = {"available": False, "reason": "no reference pace"}
NO_ENERGY: dict[str, object] = {"available": False, "reason": "no energy usage"}
UNKNOWN_SOURCE: PriorSource = PriorSource("none", "none")
SESSION_KINDS: dict[str, str] = {"practice": "practice", "qualifying": "qualify", "race": "race", "last_race": "race"}


class PlanTools(RaceTools):
    """The race tools plus the race plan (available from the grid), the energy-saving check, the grid briefing, the
    session summaries, lap lists, notes and the track history.
    _field                PriorFieldTracker  tells where the pace and usage of the plan come from
    _session_log          SessionLog         the player's laps and notes
    _keeper               SessionKeeper      summaries, results files and the history
    _briefing             GridBriefing       the grid comparison
    _max_saving_fraction  float              share of the usage per lap that is realistic to save
    _min_laps             int                clean laps needed before a lap-time cost is estimated
    _min_spread           float              spread in the usage needed before a lap-time cost is estimated
    """

    def __init__(
        self,
        tracker: RaceTracker,
        rival_watch: RivalWatch,
        traffic_horizon_laps: float,
        field: PriorFieldTracker,
        session_log: SessionLog,
        keeper: SessionKeeper,
        briefing: GridBriefing,
        max_saving_fraction: float,
        min_laps: int,
        min_spread: float,
    ) -> None:
        """Store the sources and the saving parameters.
        In:  tracker               RaceTracker
             rival_watch           RivalWatch
             traffic_horizon_laps  float              laps within which other-class cars are reported as traffic
             field                 PriorFieldTracker
             session_log           SessionLog
             keeper                SessionKeeper
             briefing              GridBriefing
             max_saving_fraction   float              share of the usage per lap that is realistic to save
             min_laps              int                clean laps needed before a cost is estimated
             min_spread            float              the lowest lap must use this share less than the mean before a
                                                      cost is estimated
        Out:                       None
        """
        super().__init__(tracker, rival_watch, traffic_horizon_laps)
        self._field: PriorFieldTracker = field
        self._session_log: SessionLog = session_log
        self._keeper: SessionKeeper = keeper
        self._briefing: GridBriefing = briefing
        self._max_saving_fraction: float = max_saving_fraction
        self._min_laps: int = min_laps
        self._min_spread: float = min_spread

    def race_plan(self) -> dict[str, object]:
        """The plan for the driver's whole race, available on the grid before the start and updated every lap: how
        many laps the race will have, the average lap time to the flag with stops included, how many laps a full
        load lasts, how many stops, the lap at the end of which to pit (as the driver counts laps), the energy on
        board, the energy to add in total and at the first stop, the fuel to add, how long the first stop takes, the
        fewest stops possible and the lowest starting energy that still needs only that many. It also says which
        pace and which usage per lap the plan is based on and where they come from, and whether it is still the
        pre-race plan (from qualifying, history and garage figures, used for the first laps of the race) or the plan
        from the laps measured in this race. When the net leader's energy or fuel use per lap is not known yet, the
        leader is projected without stops and laps_approximate is true: the race laps can then be off by one.
        In:  -
        Out: dict[str, object]  units in the keys (percent of energy, or litres for a car without virtual energy);
                                plan_basis "pre_race" | "measured"; without a known usage only the laps, pace and
                                sources with stops 0; {"available": False, "reason": "no reference pace"} without a
                                race state
        """
        state = self._tracker.state
        if state is None:
            return NO_PACE
        car = next(c for c in state.cars if c.slot_id == state.player_slot_id)
        mine = state.projection.cars[state.player_slot_id]
        source = self._field.sources.get(state.player_slot_id, UNKNOWN_SOURCE)
        leader = next(c for c in state.cars if c.slot_id == state.projection.leader_slot_id)
        plan = race_plan(car, mine)
        result: dict[str, object] = {
            "plan_basis": self._plan_basis(),
            "race_laps": round(car.progress_laps + mine.laps_remaining),
            "laps_remaining": mine.laps_remaining,
            "pace_used": lap_time_text(car.lap_time_s),
            "pace_source": source.pace,
            "usage_source": source.energy,
            "net_leader": next(v for v in state.vehicles if v.slot_id == state.projection.leader_slot_id).driver_name,
            "laps_approximate": strategy_resource(leader) is None,
            "stops": mine.stops_remaining,
        }
        if plan is None:
            return result
        name, _, per_lap, _ = strategy_resource(car)
        unit, scale = ("percent", 100.0) if name == "energy" else ("l", 1.0)
        energy_add, fuel_add_l = (
            (plan.first_stop_add, plan.first_stop_add * 100 * (car.fuel_per_energy or 0.0))
            if name == "energy"
            else (0.0, plan.first_stop_add)
        )
        result.update(
            {
                "average_lap_with_stops": lap_time_text(plan.average_lap_s),
                "stint_laps": plan.stint_laps,
                "pit_at_end_of_lap": plan.pit_lap,
                f"{name}_per_lap_{unit}": per_lap * scale,
                f"{name}_on_board_{unit}": plan.amount_on_board * scale,
                f"{name}_to_add_total_{unit}": plan.to_add_total * scale,
                f"first_stop_add_{unit}": plan.first_stop_add * scale,
                "fuel_to_add_total_l": plan.fuel_to_add_l,
                "first_stop_time_s": None
                if plan.pit_lap is None
                else lane_loss_s(state.pit, car.lap_time_s) + service_time_s(energy_add, fuel_add_l, state.pit.rates),
                "min_stops": plan.min_stops,
                f"min_start_{unit}_for_min_stops": plan.min_start_amount * scale,
            }
        )
        return result

    def my_race(self) -> dict[str, object]:
        """The driver's own race to the flag: the lap the driver is on, laps remaining, laps on board, the lap at
        the end of which to pit (lap numbers as the driver counts them: the current lap is completed laps + 1), stops
        remaining, energy and fuel still to add in total to reach the flag (negative = surplus), the fuel used per
        1 % of energy against the ratio set in the garage, whether fuel runs out before energy, and whether these
        numbers are still the pre-race plan or come from the laps measured in this race. What to add at a stop now
        is in pit_stop.
        In:  -
        Out: dict[str, object]  the player's projection with units in the keys and plan_basis "pre_race" |
                                "measured"; {"available": False} without a race state
        """
        result = super().my_race()
        return result if result is UNAVAILABLE else {"plan_basis": self._plan_basis(), **result}

    def pit_stop(self, driver: str = "") -> dict[str, object]:
        """What a pit stop now would cost and where the car rejoins: stop time, energy and fuel to add at this stop
        (what is needed to the flag, at most a full load), position, gaps and traffic, and whether the numbers are
        still from the pre-race plan or from the laps measured in this race. For the driver's own car leave driver
        empty; for another car give the driver's name.
        In:  driver  str  driver name or part of it; empty = the player
        Out:         dict[str, object]  the Rejoin fields, ahead and behind as driver names, plan_basis "pre_race" |
                                        "measured", and for the player the game's own estimate as
                                        game_estimate_total_s; {"available": False} without a race state or when no
                                        car matches
        """
        result = super().pit_stop(driver)
        return result if result is UNAVAILABLE else {"plan_basis": self._plan_basis(), **result}

    def saving(self, laps_needed: float = 0.0, shorten_stop_s: float = 0.0) -> dict[str, object]:
        """How much energy the driver would have to save per lap, whether that is realistic and what it costs in lap
        time. Without arguments it checks driving one more lap before the next stop and skipping the last stop.
        Give laps_needed to check making the energy on board last that many laps; give shorten_stop_s to check
        making the next stop that many seconds shorter by adding less energy.
        In:  laps_needed     float  laps the energy on board must last; 0 = not asked
             shorten_stop_s  float  s the stop should be shorter; 0 = not asked
        Out:                 dict[str, object]  up to four entries (one_more_lap_before_stop, skip_last_stop,
                                                reach_laps, shorten_stop), each with the saving in percent of the
                                                usage and in percent of energy per lap, the target usage, the laps
                                                to hold it, whether it is within the realistic limit and within the
                                                lowest usage the driver has shown, and the cost in s per lap and in
                                                total (None = unknown); plus plan_basis, the current and lowest
                                                usage, the limit and the clean laps measured; {"available": False}
                                                without a race state or for a car without virtual energy
        """
        state = self._tracker.state
        if state is None:
            return UNAVAILABLE
        car = next(c for c in state.cars if c.slot_id == state.player_slot_id)
        mine = state.projection.cars[state.player_slot_id]
        if mine.resource != "energy":
            return NO_ENERGY
        amount, per_lap, stops = car.virtual_energy, car.energy_per_lap, mine.stops_remaining
        needs: dict[str, SavingNeed | None] = {}
        if stops > 0:
            needs["one_more_lap_before_stop"] = saving_for_extra_lap(amount, per_lap, car.progress_laps % 1)
            needs["skip_last_stop"] = saving_to_skip_stop(amount, per_lap, 1.0, mine.laps_remaining, stops)
        if laps_needed > 0:
            needs["reach_laps"] = saving_to_reach(amount, per_lap, laps_needed)
        if shorten_stop_s > 0 and stops > 0:
            needs["shorten_stop"] = saving_to_shorten_stop(
                per_lap, mine.laps_remaining, shorten_stop_s, state.pit.rates
            )
        laps = self._session_log.clean_laps()
        result: dict[str, object] = {
            key: self._saving_entry(need, laps) for key, need in needs.items() if need is not None
        }
        result.update(
            {
                "plan_basis": self._plan_basis(),
                "current_usage_percent_per_lap": per_lap * 100,
                "lowest_usage_shown_percent_per_lap": min(used for _, used in laps) * 100 if laps else None,
                "max_saving_percent": self._max_saving_fraction * 100,
                "laps_measured": len(laps),
            }
        )
        return result

    def grid_briefing(self) -> list[dict[str, object]]:
        """Who the driver starts among, compared with the driver's own car: the car ahead and behind in class and
        the strategy rivals, each with the qualifying lap difference of this event (negative = that car was
        faster), tyre compounds, energy on board, laps on board and projected stops.
        In:  -
        Out: list[dict[str, object]]  the GridCar fields with energy in percent; empty without a race state
        """
        return [
            {
                **{k: v for k, v in dataclasses.asdict(car).items() if k != "energy_delta"},
                "energy_delta_percent": None if car.energy_delta is None else car.energy_delta * 100,
            }
            for car in self._briefing.cars()
        ]

    def session_summary(self, session: str = "current") -> dict[str, object]:
        """Summary of a session of the driver on this track: class place and places gained, laps, best lap,
        theoretical best, clean pace, best sectors, energy per lap, pit stops, incidents and penalties when known,
        the top of the class and the notes. Track limits are in it too: invalid_laps is the number of laps the game
        flagged for track limits and did not time (out laps and in laps are not invalid), and limits holds the
        track-limit points the driver has (points), the points at which a penalty comes (penalty_points), the
        penalties outstanding (penalties) and every offence with its lap and its place on the track (events). Use
        "current" for the running session, or "practice", "qualifying", "race" for the newest one of that kind, or
        "last_race" for the newest race before the running session.
        In:  session  str  "current" | "practice" | "qualifying" | "race" | "last_race"
        Out:          dict[str, object]  the SessionSummary fields without the lap list (lap times in s, energy as a
                                         fraction) and without yellow_flags when they are not counted;
                                         {"available": False} when there is no such session
        """
        if session == "current":
            current = self._keeper.summary()
            summary = None if current is None else dataclasses.asdict(current)
        else:
            summary = self._keeper.past_summary(SESSION_KINDS[session], session == "last_race")
        if summary is None:
            return UNAVAILABLE
        shown = {k: v for k, v in summary.items() if k != "yellow_flags" or v is not None}
        return {**shown, "player": {k: v for k, v in summary["player"].items() if k != "lap_list"}}

    def driver_laps(self, driver: str = "", session: str = "current") -> dict[str, object]:
        """Lap by lap data. For the driver's own car leave driver empty: every lap of the session with lap time,
        sectors, energy and fuel used, class place, tyres, whether it was a pit or a non-green lap, and lap_type:
        what the lap was ("lap", "first lap", "out lap", "in lap", "invalid lap", "aborted lap"). For another
        driver in the running session: the recent clean laps, best and last lap. For an earlier session ("practice", "qualifying",
        "race") the laps come from the game's results file.
        In:  driver   str  driver name or part of it; empty = the player
             session  str  "current" | "practice" | "qualifying" | "race" | "last_race"
        Out:          dict[str, object]  "driver" and "laps" (or the clean-lap window); {"available": False} when the
                                         session or the driver is not found
        """
        record = self._session_log.record()
        if record is None:
            return UNAVAILABLE
        if session != "current":
            results = self._keeper.past_results(SESSION_KINDS[session])
            wanted = (driver or record.player_name).lower()
            found = None if results is None else next((d for d in results.drivers if wanted in d.name.lower()), None)
            if found is None:
                return UNAVAILABLE
            return {"driver": found.name, "laps": [dataclasses.asdict(lap) for lap in found.lap_list]}
        if not driver:
            return {"driver": record.player_name, "laps": [dataclasses.asdict(lap) for lap in record.laps]}
        vehicle = next((v for v in record.vehicles if driver.lower() in v.driver_name.lower()), None)
        if vehicle is None:
            return UNAVAILABLE
        estimate = record.estimates.get(vehicle.slot_id)
        return {
            "driver": vehicle.driver_name,
            "recent_clean_laps_s": [] if estimate is None else list(estimate.clean_laps_s),
            "best_lap_s": vehicle.best_lap_s,
            "last_lap_s": vehicle.last_lap_s,
            "completed_laps": vehicle.completed_laps,
        }

    def add_note(self, text: str) -> str:
        """Store a note of the driver with the session time and the lap; it is kept in the session file.
        In:  text  str  the note
        Out:       str  confirmation sentence
        """
        note = self._session_log.add_note(text)
        return f"Noted on lap {note.lap}."

    def notes(self) -> list[dict[str, object]]:
        """The notes of the running session: the driver's own and the automatic ones (pit stops).
        In:  -
        Out: list[dict[str, object]]  time_s, lap, text, auto per note; empty without a session
        """
        record = self._session_log.record()
        return [] if record is None else [dataclasses.asdict(note) for note in record.notes]

    def track_history(self) -> dict[str, object]:
        """What earlier sessions on this track say about the driver's car and its class: best qualifying lap, best
        lap, typical race pace, best sectors, energy per lap, stint length, pit lane, the last results, and for the
        class the pole lap, the leader's race pace and the energy per lap.
        In:  -
        Out: dict[str, object]  track, car, car_history, class_history; {"available": False} before the first
                                session has been seen
        """
        priors, record = self._keeper.priors(), self._session_log.record()
        if priors is None or record is None:
            return UNAVAILABLE
        of_class = priors.class_history.get(record.class_name)
        return {
            "track": priors.track,
            "car": priors.car,
            "car_history": None if priors.car_history is None else dataclasses.asdict(priors.car_history),
            "class_history": None if of_class is None else dataclasses.asdict(of_class),
        }

    def tools(self) -> list[Callable[..., object]]:
        """The functions an agent may call.
        In:  -
        Out: list[Callable[..., object]]  the race tools, race_plan, saving, grid_briefing, session_summary,
                                          driver_laps, add_note, notes and track_history
        """
        return [
            *super().tools(),
            self.race_plan,
            self.saving,
            self.grid_briefing,
            self.session_summary,
            self.driver_laps,
            self.add_note,
            self.notes,
            self.track_history,
        ]

    def _plan_basis(self) -> str:
        """Which figures the projection runs on.
        In:  -
        Out: str  "pre_race" while the pre-race figures are held for the first laps, else "measured"
        """
        return "pre_race" if self._field.pre_race else "measured"

    def _saving_entry(self, need: SavingNeed, laps: Sequence[tuple[float, float]]) -> dict[str, object]:
        """One saving as tool output.
        In:  need  SavingNeed
             laps  Sequence[tuple[float, float]]  (lap time s, energy fraction used) per clean lap of the player
        Out:       dict[str, object]              units in the keys
        """
        check = check_saving(need, self._max_saving_fraction, laps, self._min_laps, self._min_spread)
        return {
            "saving_percent_of_usage": need.saving_fraction * 100,
            "saving_percent_energy_per_lap": need.saving_per_lap * 100,
            "target_usage_percent_per_lap": need.usage_per_lap * 100,
            "hold_for_laps": need.laps,
            "within_limit": check.within_limit,
            "within_lowest_shown": check.within_shown,
            "cost_s_per_lap": check.cost_s_per_lap,
            "total_cost_s": None if check.cost_s_per_lap is None else check.cost_s_per_lap * need.laps,
        }
