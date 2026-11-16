import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ai_race_engineer.strategy.track_map import Segment
from ai_race_engineer.telemetry.race_snapshot import RaceSnapshot
from ai_race_engineer.telemetry.snapshot import VehicleState

INCIDENT: str = "incident"


@dataclass(frozen=True)
class Contact:
    """One hit a car took: against another car, or on its own against a wall or a kerb.
    time_s          float       s session time
    driver          str         driver of the car that took the hit
    other_driver    str | None  driver of the car it touched; None when no car was near
    involves_player bool        the player's car is one of the two
    magnitude       float       the game's impact magnitude
    lap_distance_m  float       m from the start/finish line
    place           str         name of the track segment, "" when the track has no map yet
    """

    time_s: float
    driver: str
    other_driver: str | None
    involves_player: bool
    magnitude: float
    lap_distance_m: float
    place: str


class ContactTracker:
    """Finds contact from the impacts the game reports for every car: a hit with another car close by is contact
    between the two, a hard hit with no car near is a car going off on its own. Each one is kept and written to the
    session timeline as an "incident". Light touches of the player and solo hits of other cars below the hard limit
    are left out.
    contacts         tuple[Contact, ...]  the contacts of the session, oldest first (replaced, never mutated)
    _record          Callable[[str, str, Mapping[str, object]], object]  adds a timeline entry (kind, text, fields)
    _segments        Callable[[], Sequence[Segment]]  the track map, empty until known
    _min_magnitude   float            impact magnitude from which a hit counts
    _hard_magnitude  float            impact magnitude from which a hit without a car near is kept for other cars
    _near_m          float            m within which another car counts as the one that was touched
    _hold_s          float            s in which further impacts of the same car are the same contact
    _seen_s          dict[int, float]  slot id -> impact time already looked at
    _last_s          dict[int, float]  slot id -> session time of its last kept contact
    _time_s          float            s session time of the last snapshot
    """

    def __init__(
        self,
        record: Callable[[str, str, Mapping[str, object]], object],
        segments: Callable[[], Sequence[Segment]],
        min_magnitude: float,
        hard_magnitude: float,
        near_m: float,
        hold_s: float,
    ) -> None:
        """Store where contacts go and what counts as one.
        In:  record          Callable[[str, str, Mapping[str, object]], object]  adds a timeline entry
             segments        Callable[[], Sequence[Segment]]  the track map, empty until known
             min_magnitude   float  impact magnitude from which a hit counts
             hard_magnitude  float  impact magnitude from which a solo hit of another car is kept
             near_m          float  m within which another car counts as the one that was touched
             hold_s          float  s in which further impacts of the same car are the same contact
        Out:                 None
        """
        self.contacts: tuple[Contact, ...] = ()
        self._record: Callable[[str, str, Mapping[str, object]], object] = record
        self._segments: Callable[[], Sequence[Segment]] = segments
        self._min_magnitude: float = min_magnitude
        self._hard_magnitude: float = hard_magnitude
        self._near_m: float = near_m
        self._hold_s: float = hold_s
        self._seen_s: dict[int, float] = {}
        self._last_s: dict[int, float] = {}
        self._time_s: float = -1.0

    def update(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer): keep and record the contacts of the cars whose impact time moved on.
        The first snapshot of a session only notes the impact times.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        now_s = snapshot.session.elapsed_s
        fresh = now_s < self._time_s or not self._seen_s
        if now_s < self._time_s:
            self.contacts, self._last_s = (), {}
        self._time_s = now_s
        vehicles = {vehicle.slot_id: vehicle for vehicle in snapshot.vehicles}
        for car in snapshot.resources:
            new = car.last_impact_s > self._seen_s.get(car.slot_id, 0.0)
            self._seen_s[car.slot_id] = car.last_impact_s
            vehicle = vehicles.get(car.slot_id)
            if fresh or not new or vehicle is None or car.last_impact_magnitude < self._min_magnitude:
                continue
            if now_s - self._last_s.get(car.slot_id, -math.inf) < self._hold_s:
                continue
            other = _nearest(vehicle, vehicles.values(), self._near_m)
            if other is None and not vehicle.is_player and car.last_impact_magnitude < self._hard_magnitude:
                continue
            self._last_s[car.slot_id] = now_s
            if other is not None and now_s - self._last_s.get(other.slot_id, -math.inf) < self._hold_s:
                continue
            if other is not None:
                self._last_s[other.slot_id] = now_s
            self._keep(vehicle, other, car.last_impact_magnitude, now_s)

    def _keep(self, vehicle: VehicleState, other: VehicleState | None, magnitude: float, now_s: float) -> None:
        """Keep one contact and add it to the timeline.
        In:  vehicle    VehicleState         the car that took the hit
             other      VehicleState | None  the car it touched
             magnitude  float                the game's impact magnitude
             now_s      float                s session time
        Out:            None
        """
        place = next(
            (s.name for s in self._segments() if s.start_m <= vehicle.lap_distance_m < s.end_m),
            "",
        )
        contact = Contact(
            time_s=now_s,
            driver=vehicle.driver_name,
            other_driver=None if other is None else other.driver_name,
            involves_player=vehicle.is_player or (other is not None and other.is_player),
            magnitude=magnitude,
            lap_distance_m=vehicle.lap_distance_m,
            place=place,
        )
        self.contacts = (*self.contacts, contact)
        self._record(INCIDENT, contact_text(vehicle, other, place), dict(contact.__dict__))


def contact_text(vehicle: VehicleState, other: VehicleState | None, place: str) -> str:
    """Sentence for a contact.
    In:  vehicle  VehicleState         the car that took the hit
         other    VehicleState | None  the car it touched; None when no car was near
         place    str                  name of the track segment, "" when unknown
    Out:          str
    """
    where = f" at {place}" if place else ""
    if other is None:
        return f"{'You' if vehicle.is_player else vehicle.driver_name} hit something alone{where}"
    if vehicle.is_player or other.is_player:
        rival = other if vehicle.is_player else vehicle
        return f"Contact between you and {rival.driver_name}, {rival.class_name}{where}"
    return f"Contact between {vehicle.driver_name} and {other.driver_name}{where}"


def _nearest(vehicle: VehicleState, vehicles: Sequence[VehicleState], near_m: float) -> VehicleState | None:
    """The closest other car on track within a distance.
    In:  vehicle   VehicleState
         vehicles  Sequence[VehicleState]  all cars
         near_m    float                   m
    Out:           VehicleState | None     None when no car is that close
    """
    others = [
        (math.dist(vehicle.position_xz_m, other.position_xz_m), other)
        for other in vehicles
        if other.slot_id != vehicle.slot_id and not other.in_garage
    ]
    distance_m, other = min(others, key=lambda pair: pair[0], default=(math.inf, None))
    return other if distance_m <= near_m else None
