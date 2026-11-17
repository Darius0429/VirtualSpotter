from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ai_race_engineer.domains.common.analysis.field.field import CarEstimate, FieldTracker
from ai_race_engineer.domains.common.analysis.pace.laps import lap_fraction
from ai_race_engineer.domains.common.analysis.strategy.pit_log import PitStopLog
from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitLane, PitModel, PitRates, calibrate, measured_pit_lane
from ai_race_engineer.domains.common.analysis.strategy.race_finish import CarInput, RaceProjection, project_race
from ai_race_engineer.domains.common.analysis.field.rivals import Rival, find_rivals
from ai_race_engineer.domains.common.analysis.strategy.track_info import TrackInfo
from ai_race_engineer.domains.common.data.pit_estimate import PitEstimate
from ai_race_engineer.domains.common.data.race_snapshot import (
    GAME_PHASE_GREEN,
    GAME_PHASE_OVER,
    CarResources,
    LapSectors,
    RaceSnapshot,
)
from ai_race_engineer.domains.common.data.snapshot import SessionState, SessionType, VehicleState

UNSET_REMAINING_S: float = -1e9
UNSET_MAX_LAPS: int = 2147483647
FINISH_STATUS_FINISHED: int = 1
FINISH_STATUS_OUT: tuple[int, int] = (2, 3)
PRE_GREEN_PHASES: tuple[int, ...] = (0, 1, 2, 3, 4, 9)


@dataclass(frozen=True)
class RaceState:
    """One consistent computation of the race: inputs, pit model, projection and rivals.
    time_s            float                     s session clock of the computation
    player_slot_id    int
    vehicles          tuple[VehicleState, ...]  all cars
    cars              tuple[CarInput, ...]      projected cars
    estimates         dict[int, CarEstimate]    slot id -> pace and usage, all cars
    sectors           dict[int, LapSectors]     slot id -> last-lap sector times, all cars
    pit               PitModel                  calibrated rates + the lane in use
    estimate          PitEstimate | None        the game's latest, player only
    set_fuel_ratio    float | None              litres per 1 % energy set in the garage, when known
    projection        RaceProjection
    rivals            tuple[Rival, ...]
    traffic_window_s  float                     s around a rejoin point that count as traffic
    before_green      bool                      the race has not started: the game phase is before the green and the
                                                green has not been shown in this session
    """

    time_s: float
    player_slot_id: int
    vehicles: tuple[VehicleState, ...]
    cars: tuple[CarInput, ...]
    estimates: dict[int, CarEstimate]
    sectors: dict[int, LapSectors]
    pit: PitModel
    estimate: PitEstimate | None
    set_fuel_ratio: float | None
    projection: RaceProjection
    rivals: tuple[Rival, ...]
    traffic_window_s: float
    before_green: bool = False


class RaceTracker:
    """Keeps the race state up to date from every snapshot; shared by the rival watch and the agent tools.
    state              RaceState | None                       replaced, never mutated (read from other threads)
    finished           bool                                   the race is over for the player: the state is final
    green_seen         bool                                   the green has been shown in the running session (any
                                                              session type; read from other threads)
    _field             FieldTracker
    _pit_log           PitStopLog
    _tracks            Mapping[str, TrackInfo]                track name -> characteristics
    _default_rates     PitRates                               used until calibrated
    _default_lane      PitLane                                used for an unknown track with no pass measured
    _estimate          Callable[[], PitEstimate | None]       the game's latest pit estimate
    _pace_window_s     float                                  s/lap
    _finish_window_s   float                                  s
    _finish_window_exit_s  float                              s
    _fuel_margin_laps  float                                  laps
    _min_rival_s       float                                  s
    _strategy_since_s  dict[int, float]                       slot id -> s session time since when a strategy rival
    _traffic_window_s  float                                  s
    _period_s          float                                  session s between recomputations
    _set_fuel_ratio    Callable[[], float | None] | None      garage fuel ratio, litres per 1 % energy
    _session           SessionState | None                    session of the previous snapshot; None before the first
    """

    def __init__(
        self,
        field: FieldTracker,
        pit_log: PitStopLog,
        tracks: Mapping[str, TrackInfo],
        default_rates: PitRates,
        default_lane: PitLane,
        estimate: Callable[[], PitEstimate | None],
        pace_window_s: float,
        finish_window_s: float,
        finish_window_exit_s: float,
        min_rival_s: float,
        traffic_window_s: float,
        period_s: float,
        fuel_margin_laps: float,
        set_fuel_ratio: Callable[[], float | None] | None = None,
    ) -> None:
        """Store the blocks and parameters.
        In:  field             FieldTracker
             pit_log           PitStopLog
             tracks            Mapping[str, TrackInfo]            track name -> characteristics
             default_rates     PitRates                           used until calibrated in the session
             default_lane      PitLane                            used for an unknown track with no pass measured
             estimate          Callable[[], PitEstimate | None]   the game's latest pit estimate (non-blocking)
             pace_window_s     float                              s/lap, rival pace window
             finish_window_s   float                              s, strategy-rival window at the flag
             finish_window_exit_s  float                          s, window a strategy rival must leave to be dropped
             min_rival_s       float                              s a strategy rival stays one whatever its gap
             traffic_window_s  float                              s around a rejoin point that count as traffic
             period_s          float                              session s between recomputations
             fuel_margin_laps  float                              laps the fuel may be short of the energy before
                                                                  "fuel runs out first"
             set_fuel_ratio    Callable[[], float | None] | None  garage fuel ratio, litres per 1 % energy
        Out:                   None
        """
        self.state: RaceState | None = None
        self.finished: bool = False
        self.green_seen: bool = False
        self._field: FieldTracker = field
        self._pit_log: PitStopLog = pit_log
        self._tracks: Mapping[str, TrackInfo] = tracks
        self._default_rates: PitRates = default_rates
        self._default_lane: PitLane = default_lane
        self._estimate: Callable[[], PitEstimate | None] = estimate
        self._pace_window_s: float = pace_window_s
        self._finish_window_s: float = finish_window_s
        self._finish_window_exit_s: float = finish_window_exit_s
        self._fuel_margin_laps: float = fuel_margin_laps
        self._min_rival_s: float = min_rival_s
        self._strategy_since_s: dict[int, float] = {}
        self._traffic_window_s: float = traffic_window_s
        self._period_s: float = period_s
        self._set_fuel_ratio: Callable[[], float | None] | None = set_fuel_ratio
        self._session: SessionState | None = None

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot; every period_s of session time recompute the state. The state is None outside a race,
        while the player has no lap time, while the race has neither a time nor a lap limit and once the player is
        out of the race. When the session is over or the player has finished, the state is computed once more and
        then kept. Once the green has been shown the state is never before the green again, whatever the game phase,
        until a new session starts (another session type or track, or the session clock going backwards).
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        session = snapshot.session
        if (
            self._session is None
            or session.session_type != self._session.session_type
            or session.track_name != self._session.track_name
            or session.elapsed_s < self._session.elapsed_s
        ):
            self.green_seen = False
        self._session = session
        self.green_seen = self.green_seen or snapshot.flags.game_phase == GAME_PHASE_GREEN
        estimate = self._estimate()
        estimates = self._field.update(snapshot.vehicles, snapshot.resources, snapshot.flags.game_phase)
        samples = self._pit_log.update(
            snapshot.vehicles, snapshot.resources, session.track_length_m, session.elapsed_s, estimate
        )
        player = snapshot.player.vehicle
        before_green = snapshot.flags.game_phase in PRE_GREEN_PHASES and not self.green_seen
        if before_green and snapshot.time_remaining_s is not None:
            remaining_s = snapshot.time_remaining_s if snapshot.time_remaining_s > 0 else None
        else:
            remaining_s = None if session.remaining_s < UNSET_REMAINING_S else session.remaining_s
        max_laps = None if session.max_laps == UNSET_MAX_LAPS else session.max_laps
        resources = {r.slot_id: r for r in snapshot.resources}
        if (
            session.session_type != SessionType.RACE
            or estimates[player.slot_id].lap_time_s is None
            or (remaining_s is None and max_laps is None)
            or _is_out(resources.get(player.slot_id))
        ):
            self.state, self.finished, self._strategy_since_s = None, False, {}
            return
        is_over = snapshot.flags.game_phase == GAME_PHASE_OVER or _is_finished(resources.get(player.slot_id))
        if is_over and self.finished:
            return
        if not is_over and self.state is not None and 0 <= session.elapsed_s - self.state.time_s < self._period_s:
            return
        track = self._tracks.get(session.track_name)
        lane = track.pit_lane if track is not None and track.pit_lane is not None else measured_pit_lane(samples)
        pit = PitModel(calibrate(samples, self._default_rates), lane or self._default_lane, session.track_length_m)
        cars = tuple(
            _car_input(v, estimates[v.slot_id], resources.get(v.slot_id), session.track_length_m, before_green)
            for v in snapshot.vehicles
            if estimates[v.slot_id].lap_time_s is not None and not _is_out(resources.get(v.slot_id))
        )
        projection = project_race(cars, remaining_s, max_laps, pit, self._fuel_margin_laps)
        rivals = find_rivals(
            player,
            snapshot.vehicles,
            {c.slot_id: c.lap_time_s for c in cars},
            projection,
            session.track_length_m,
            self._pace_window_s,
            self._finish_window_s,
            self._finish_window_exit_s,
            self._strategy_since_s,
            session.elapsed_s,
            self._min_rival_s,
            before_green,
        )
        self._strategy_since_s = {
            r.slot_id: self._strategy_since_s.get(r.slot_id, session.elapsed_s)
            for r in rivals
            if "strategy" in r.groups
        }
        self.state = RaceState(
            time_s=session.elapsed_s,
            player_slot_id=player.slot_id,
            vehicles=snapshot.vehicles,
            cars=cars,
            estimates=estimates,
            sectors={s.slot_id: s for s in snapshot.sectors},
            pit=pit,
            estimate=estimate,
            set_fuel_ratio=None if self._set_fuel_ratio is None else self._set_fuel_ratio(),
            projection=projection,
            rivals=rivals,
            traffic_window_s=self._traffic_window_s,
            before_green=before_green,
        )
        self.finished = is_over


def _car_input(
    vehicle: VehicleState,
    estimate: CarEstimate,
    resources: CarResources | None,
    track_length_m: float,
    before_green: bool,
) -> CarInput:
    """Projection input of one car. Before the green a car without a completed lap is at 0 laps, wherever it stands
    on the grid or the formation lap.
    In:  vehicle         VehicleState
         estimate        CarEstimate          with a lap time
         resources       CarResources | None  None for a car without telemetry
         track_length_m  float                m
         before_green    bool                 the race has not started
    Out:                 CarInput
    """
    on_grid = before_green and vehicle.completed_laps == 0
    return CarInput(
        slot_id=vehicle.slot_id,
        class_name=vehicle.class_name,
        progress_laps=0.0 if on_grid else vehicle.completed_laps + lap_fraction(vehicle.lap_distance_m, track_length_m),
        lap_time_s=estimate.lap_time_s,
        finished=_is_finished(resources),
        in_pits=vehicle.in_pits,
        fuel_l=None if resources is None else resources.fuel_l,
        fuel_capacity_l=None if resources is None else resources.fuel_capacity_l,
        virtual_energy=None if resources is None else resources.virtual_energy,
        fuel_l_per_lap=estimate.fuel_l_per_lap,
        energy_per_lap=estimate.energy_per_lap,
        fuel_per_energy=estimate.fuel_per_energy,
    )


def _is_finished(resources: CarResources | None) -> bool:
    """Whether a car has taken the chequered flag.
    In:  resources  CarResources | None  None for a car without telemetry
    Out:            bool
    """
    return resources is not None and resources.finish_status == FINISH_STATUS_FINISHED


def _is_out(resources: CarResources | None) -> bool:
    """Whether a car is out of the race.
    In:  resources  CarResources | None  None for a car without telemetry
    Out:            bool                 True for a car that did not finish or was disqualified
    """
    return resources is not None and resources.finish_status in FINISH_STATUS_OUT
