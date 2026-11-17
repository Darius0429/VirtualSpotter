from collections.abc import Sequence
from dataclasses import dataclass

from ai_race_engineer.domains.common.data.pit_estimate import PitEstimate


@dataclass(frozen=True)
class PitRates:
    """How fast a stop fills the car.
    fuel_rate_l_per_s  float  litres/s
    energy_rate_per_s  float  fraction/s of virtual energy
    scale              float  measured stationary time / unscaled model time
    base_s             float  s added once per stop that adds energy or fuel (unscaled)
    """

    fuel_rate_l_per_s: float
    energy_rate_per_s: float
    scale: float
    base_s: float = 0.0


@dataclass(frozen=True)
class PitLane:
    """The pit lane of a track.
    time_s    float  s from the pit entry line to the exit line, excluding standing still
    length_m  float  m of lap covered between those lines
    """

    time_s: float
    length_m: float


@dataclass(frozen=True)
class PitModel:
    """The one pit-stop model used for every car.
    rates           PitRates
    lane            PitLane
    track_length_m  float  m
    """

    rates: PitRates
    lane: PitLane
    track_length_m: float


@dataclass(frozen=True)
class PitStopSample:
    """One measured pass through the pit lane.
    slot_id        int                 game slot id
    is_player      bool
    lane_time_s    float               s in the pit lane excluding stationary
    lane_length_m  float               m of lap between the car's entry and exit
    stationary_s   float               s standing still
    energy_added   float               fraction of virtual energy
    fuel_added_l   float               litres
    estimate       PitEstimate | None  the game's estimate at pit entry (player only)
    """

    slot_id: int
    is_player: bool
    lane_time_s: float
    lane_length_m: float
    stationary_s: float
    energy_added: float
    fuel_added_l: float
    estimate: PitEstimate | None


def lane_loss_s(pit: PitModel, lap_time_s: float) -> float:
    """Time a car loses by driving through the pit lane: lane time minus its race-pace time over the same stretch.
    In:  pit         PitModel
         lap_time_s  float  s/lap of the car
    Out:             float  s
    """
    return pit.lane.time_s - pit.lane.length_m / pit.track_length_m * lap_time_s


def fill_time_s(energy_to_add: float, fuel_to_add_l: float, rates: PitRates) -> float:
    """Time the filling itself takes, without the fixed time per stop; both fill at once, so the slower one counts.
    In:  energy_to_add  float     fraction of virtual energy
         fuel_to_add_l  float     litres
         rates          PitRates
    Out:                float     s
    """
    return rates.scale * max(energy_to_add / rates.energy_rate_per_s, fuel_to_add_l / rates.fuel_rate_l_per_s, 0.0)


def service_time_s(energy_to_add: float, fuel_to_add_l: float, rates: PitRates) -> float:
    """Time standing still at one stop to add energy and fuel: the filling time plus the fixed time per stop.
    In:  energy_to_add  float     fraction of virtual energy
         fuel_to_add_l  float     litres
         rates          PitRates
    Out:                float     s; 0 when nothing is added
    """
    fill_s = fill_time_s(energy_to_add, fuel_to_add_l, rates)
    return fill_s + rates.scale * rates.base_s if fill_s > 0 else 0.0


def calibrate(samples: Sequence[PitStopSample], default: PitRates) -> PitRates:
    """Pit rates fitted to the player's measured stops; each term keeps its default without data. The fixed time per
    stop is not fitted: it is taken off the game's fill times and kept as given.
    In:  samples  Sequence[PitStopSample]  measured passes of any car (only the player's are used)
         default  PitRates
    Out:          PitRates
    """
    player = [s for s in samples if s.is_player]
    with_estimate = [s for s in player if s.estimate is not None]
    energy_rate = _rate(
        [
            (s.energy_added, s.estimate.energy_s - default.base_s)
            for s in with_estimate
            if s.estimate.energy_s > default.base_s and s.energy_added > 0
        ],
        default.energy_rate_per_s,
    )
    fuel_rate = _rate(
        [
            (s.fuel_added_l, s.estimate.fuel_s - default.base_s)
            for s in with_estimate
            if s.estimate.fuel_s > default.base_s and s.fuel_added_l > 0
        ],
        default.fuel_rate_l_per_s,
    )
    unscaled = PitRates(fuel_rate, energy_rate, 1.0, default.base_s)
    scale = _rate(
        [
            (s.stationary_s, service_time_s(s.energy_added, s.fuel_added_l, unscaled))
            for s in player
            if s.stationary_s > 0 and _fill_only(s.estimate)
        ],
        default.scale,
    )
    return PitRates(fuel_rate, energy_rate, scale, default.base_s)


def measured_pit_lane(samples: Sequence[PitStopSample]) -> PitLane | None:
    """Pit lane averaged over the measured passes of any car.
    In:  samples  Sequence[PitStopSample]
    Out:          PitLane | None  None without samples
    """
    if not samples:
        return None
    return PitLane(
        sum(s.lane_time_s for s in samples) / len(samples), sum(s.lane_length_m for s in samples) / len(samples)
    )


def _rate(pairs: Sequence[tuple[float, float]], default: float) -> float:
    """Ratio of two summed quantities.
    In:  pairs    Sequence[tuple[float, float]]  (numerator, denominator) per sample
         default  float                          returned when the denominators sum to zero
    Out:          float
    """
    denominator = sum(d for _, d in pairs)
    return sum(n for n, _ in pairs) / denominator if denominator > 0 else default


def _fill_only(estimate: PitEstimate | None) -> bool:
    """Whether a stop was only filling: no tyres, damage, driver swap or penalty time in the game's estimate.
    In:  estimate  PitEstimate | None
    Out:           bool  True without an estimate
    """
    return estimate is None or estimate.total_s == max(estimate.fuel_s, estimate.energy_s)
