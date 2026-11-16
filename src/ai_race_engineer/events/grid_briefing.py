from collections.abc import Callable, Mapping

from ai_race_engineer.events.base import Callout, Event, Priority
from ai_race_engineer.strategy.grid_briefing import GridCar, briefing_text, grid_briefing
from ai_race_engineer.strategy.priors import EventPriors, PriorSource
from ai_race_engineer.strategy.race_state import RaceTracker
from ai_race_engineer.telemetry.race_snapshot import CarTyres, RaceSnapshot
from ai_race_engineer.telemetry.snapshot import Snapshot


class GridBriefing(Event):
    """Tells the driver once before the start who it starts among: qualifying pace, tyres, energy and stops of the
    class neighbours and the strategy rivals; the only rival-related callout before the green.
    _tracker             RaceTracker                        source of the race state
    _priors              Callable[[], EventPriors | None]   the session's priors (qualifying times)
    _sources             Callable[[], Mapping[int, PriorSource]]  slot id -> where pace and usage come from
    _max_cars            int                                cars named in the spoken briefing
    _standing_speed_mps  float                              m/s below which the driver is standing on the grid
    _min_energy_delta    float                              fraction; energy differences below this are not mentioned
    _max_wait_s          float                              session s the briefing waits for the qualifying times
                                                            and the player's energy figure
    _tyres               dict[int, CarTyres]                slot id -> tyres, stored by update
    _speed_mps           float                              the player's speed, stored by update
    _given               bool                               the briefing of this race start has been given
    _waiting_since_s     float | None                       s session time the briefing could first be given; None
                                                            before that
    """

    name = "grid_briefing"

    def __init__(
        self,
        tracker: RaceTracker,
        priors: Callable[[], EventPriors | None],
        sources: Callable[[], Mapping[int, PriorSource]],
        max_cars: int,
        standing_speed_mps: float,
        min_energy_delta: float,
        max_wait_s: float,
    ) -> None:
        """Set the sources and thresholds.
        In:  tracker             RaceTracker
             priors              Callable[[], EventPriors | None]  the session's priors (non-blocking)
             sources             Callable[[], Mapping[int, PriorSource]]  slot id -> where pace and usage come from
                                                                   (non-blocking)
             max_cars            int                               cars named in the spoken briefing
             standing_speed_mps  float                             m/s below which the driver is standing still
             min_energy_delta    float                             fraction; energy differences below this are not
                                                                   mentioned
             max_wait_s          float                             session s the briefing waits for the qualifying
                                                                   times and the player's energy figure
        Out:                     None
        """
        self._tracker: RaceTracker = tracker
        self._priors: Callable[[], EventPriors | None] = priors
        self._sources: Callable[[], Mapping[int, PriorSource]] = sources
        self._max_cars: int = max_cars
        self._standing_speed_mps: float = standing_speed_mps
        self._min_energy_delta: float = min_energy_delta
        self._max_wait_s: float = max_wait_s
        self._tyres: dict[int, CarTyres] = {}
        self._speed_mps: float = 0.0
        self._given: bool = False
        self._waiting_since_s: float | None = None

    def update(self, snapshot: RaceSnapshot) -> None:
        """Store every car's tyres and the player's speed (source observer, runs before check).
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        self._tyres = {t.slot_id: t for t in snapshot.tyres}
        self._speed_mps = snapshot.player.vehicle.speed_mps

    def cars(self) -> tuple[GridCar, ...]:
        """The grid comparison now (also asked for by the agent).
        In:  -
        Out: tuple[GridCar, ...]  empty without a race state
        """
        state, priors = self._tracker.state, self._priors()
        if state is None:
            return ()
        return grid_briefing(
            state.player_slot_id,
            state.vehicles,
            state.cars,
            state.projection,
            state.rivals,
            self._tyres,
            {} if priors is None else priors.qualifying_best_s,
        )

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Give the briefing once per race start, before the green: as soon as the qualifying times and an energy
        figure of the player are known, or max_wait_s of session time after a race state and the priors first
        existed, whichever is first. Standing on the grid it is spoken at once; with the car already rolling it
        waits for a straight.
        In:  snapshot  Snapshot  (unused: the data arrives through update and the tracker)
             now_s     float     s monotonic clock (unused)
        Out:           Callout | None  kind "grid_briefing"
        """
        state, priors = self._tracker.state, self._priors()
        if self._given or state is None or not state.before_green or priors is None:
            return None
        if self._waiting_since_s is None:
            self._waiting_since_s = state.time_s
        source = self._sources().get(state.player_slot_id)
        ready = bool(priors.qualifying_best_s) and source is not None and source.energy != "none"
        if not ready and state.time_s - self._waiting_since_s < self._max_wait_s:
            return None
        text = briefing_text(self.cars(), self._max_cars, self._min_energy_delta)
        if not text:
            return None
        self._given = True
        standing = self._speed_mps < self._standing_speed_mps
        return Callout("grid_briefing", text, Priority.SPOTTER if standing else Priority.STRATEGY, {})

    def reset(self) -> None:
        """Re-arm for the next race start.
        In:  -
        Out: None
        """
        self._given, self._waiting_since_s = False, None
