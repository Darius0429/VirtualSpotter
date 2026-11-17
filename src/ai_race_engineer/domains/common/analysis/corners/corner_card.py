from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ai_race_engineer.domains.common.analysis.corners.lap_segments import LapSamples
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment

MPS_TO_KPH: float = 3.6
FRONT_WHEELS: slice = slice(0, 2)
REAR_WHEELS: slice = slice(2, 4)


@dataclass(frozen=True)
class CardLimits:
    """Thresholds that turn the driver's inputs into corner events.
    brake_on       float  0-1 brake above which the driver is braking
    throttle_on    float  0-1 throttle above which the driver is on the throttle
    full_throttle  float  0-1 throttle from which the driver is at full throttle
    throttle_lift  float  0-1 drop of the throttle between two samples that counts as a lift
    turn_in_g      float  g lateral acceleration from which the car is turning in
    lock_ratio     float  wheel ratio below which a braked wheel is locking
    spin_ratio     float  wheel ratio above which a driven wheel is spinning
    search_m       float  m before a corner in which its braking is looked for
    after_exit_m   float  m after a corner where the speed is taken again
    """

    brake_on: float
    throttle_on: float
    full_throttle: float
    throttle_lift: float
    turn_in_g: float
    lock_ratio: float
    spin_ratio: float
    search_m: float
    after_exit_m: float


@dataclass(frozen=True)
class CornerEntry:
    """The braking for one corner; distances "to apex" are to the apex of the track map, the same on every lap.
    brake_point_m             float  m lap distance where the brake went on
    brake_to_apex_m           float  m from the brake point to the apex
    speed_at_brake_kph        float  km/h at the brake point
    lift_coast_m              float  m rolled off the throttle before the brake went on
    peak_decel_g              float  g strongest deceleration
    time_to_peak_decel_s      float  s from the brake point to the strongest deceleration
    braking_m                 float  m from the brake point to the last brake release
    release_to_apex_m         float  m from the last brake release to the apex; negative = released after the apex
    trail_brake_m             float  m braked while turning into this corner, up to the last brake release;
                                     the load of the corner before, the other way or let go in between, is
                                     not counted
    release_roughness_per_s   float  1/s mean change of the brake pedal speed while releasing; lower is smoother
    min_combined_g            float  g lowest combined acceleration between the strongest braking and the highest
                                     cornering load; a dip far below both means grip left unused
    downshifts                int
    lockup_m                  float  m braked with a locking wheel
    lockup_axle               str    "front" | "rear" | "" (none)
    """

    brake_point_m: float
    brake_to_apex_m: float
    speed_at_brake_kph: float
    lift_coast_m: float
    peak_decel_g: float
    time_to_peak_decel_s: float
    braking_m: float
    release_to_apex_m: float
    trail_brake_m: float
    release_roughness_per_s: float
    min_combined_g: float
    downshifts: int
    lockup_m: float
    lockup_axle: str


@dataclass(frozen=True)
class CornerMid:
    """The middle of one corner.
    min_speed_kph         float  km/h
    min_speed_m           float  m lap distance of the minimum speed
    min_speed_to_apex_m   float  m from the minimum speed to the apex; negative = slowest after the apex
    peak_lat_g            float  g
    coast_s               float  s off the brake and off the throttle
    steer_rate_per_s      float  1/s mean speed of the steering wheel, in lock fractions; lower is smoother
    countersteer_s        float  s steered against the corner
    """

    min_speed_kph: float
    min_speed_m: float
    min_speed_to_apex_m: float
    peak_lat_g: float
    coast_s: float
    steer_rate_per_s: float
    countersteer_s: float


@dataclass(frozen=True)
class CornerExit:
    """The exit of one corner.
    throttle_on_after_apex_m    float | None  m from the apex to the throttle pick-up; negative = before the apex;
                                              None when the throttle never came off
    full_throttle_after_apex_m  float | None  m from the apex to where the throttle stayed full; None when it never
                                              came off or was not full by after_exit_m
    throttle_acceptance         float | None  lateral g at the full-throttle point over the corner's peak lateral g
    part_throttle_s             float | None  s from the pick-up to full throttle
    throttle_lifts              int           lifts between the pick-up and full throttle
    wheelspin_s                 float         s with a spinning rear wheel after the minimum speed
    after_exit_kph              float         km/h after_exit_m after the corner
    """

    throttle_on_after_apex_m: float | None
    full_throttle_after_apex_m: float | None
    throttle_acceptance: float | None
    part_throttle_s: float | None
    throttle_lifts: int
    wheelspin_s: float
    after_exit_kph: float


@dataclass(frozen=True)
class SegmentCard:
    """What the player did in one segment on one lap.
    name            str
    corner          bool
    time_s          float                s in the segment
    time_to_apex_s  float | None         s from the start of the segment to the apex; None for a straight
    entry_kph       float                km/h at the start of the segment
    min_kph         float                km/h
    exit_kph        float                km/h at the end of the segment
    top_kph         float                km/h
    lowest_gear     int                  lowest forward gear; 0 when none was engaged
    fuel_l          float                litres used
    energy          float | None         virtual energy used, fraction 0-1; None when the game has none
    entry           CornerEntry | None   None for a straight or a corner taken without braking
    mid             CornerMid | None     None for a straight
    exit            CornerExit | None    None for a straight
    """

    name: str
    corner: bool
    time_s: float
    time_to_apex_s: float | None
    entry_kph: float
    min_kph: float
    exit_kph: float
    top_kph: float
    lowest_gear: int
    fuel_l: float
    energy: float | None
    entry: CornerEntry | None
    mid: CornerMid | None
    exit: CornerExit | None


def lap_cards(samples: LapSamples, segments: Sequence[Segment], limits: CardLimits) -> tuple[SegmentCard, ...]:
    """Cards of every segment of one lap. The braking of a corner is looked for from search_m before it, but not
    before the apex of the corner in front of it.
    In:  samples   LapSamples
         segments  Sequence[Segment]
         limits    CardLimits
    Out:           tuple[SegmentCard, ...]  one per segment, in the same order
    """
    cards: list[SegmentCard] = []
    floor_m = 0.0
    for segment in segments:
        cards.append(segment_card(samples, segment, max(floor_m, segment.start_m - limits.search_m), limits))
        floor_m = segment.apex_m if segment.corner else floor_m
    return tuple(cards)


def segment_card(samples: LapSamples, segment: Segment, search_from_m: float, limits: CardLimits) -> SegmentCard:
    """Card of one segment of one lap.
    In:  samples        LapSamples
         segment        Segment
         search_from_m  float       m lap distance from which the braking of a corner is looked for
         limits         CardLimits
    Out:                SegmentCard
    """
    edges_m = (segment.start_m, segment.end_m)
    first, last = (int(i) for i in np.searchsorted(samples.distance_m, edges_m))
    last = max(last, first + 1)
    inside = slice(first, last)
    start_s, end_s = np.interp(edges_m, samples.distance_m, samples.time_s)
    entry_mps, exit_mps = np.interp(edges_m, samples.distance_m, samples.speed_mps)
    fuel_in_l, fuel_out_l = np.interp(edges_m, samples.distance_m, samples.fuel_l)
    energy_in, energy_out = np.interp(edges_m, samples.distance_m, samples.energy)
    forward = samples.gear[inside][samples.gear[inside] > 0]
    slowest = first + int(np.argmin(samples.speed_mps[inside]))
    search = int(np.searchsorted(samples.distance_m, search_from_m))
    return SegmentCard(
        name=segment.name,
        corner=segment.corner,
        time_s=float(end_s - start_s),
        time_to_apex_s=float(_at(samples, samples.time_s, segment.apex_m) - start_s) if segment.corner else None,
        entry_kph=float(entry_mps) * MPS_TO_KPH,
        min_kph=float(samples.speed_mps[slowest]) * MPS_TO_KPH,
        exit_kph=float(exit_mps) * MPS_TO_KPH,
        top_kph=float(samples.speed_mps[inside].max()) * MPS_TO_KPH,
        lowest_gear=int(forward.min()) if forward.size else 0,
        fuel_l=float(fuel_in_l - fuel_out_l),
        energy=None if np.isnan(energy_in) else float(energy_in - energy_out),
        entry=_entry(samples, segment, search, slowest, first, last, limits) if segment.corner else None,
        mid=_mid(samples, segment, slowest, inside, limits) if segment.corner else None,
        exit=_exit(samples, segment, slowest, inside, limits) if segment.corner else None,
    )


def _entry(
    samples: LapSamples, segment: Segment, search: int, slowest: int, first: int, last: int, limits: CardLimits
) -> CornerEntry | None:
    """The braking for a corner: the braking that leads to its minimum speed, back to where the driver was last at
    full throttle. A brake application for the corner before, followed by full throttle, is not part of it; a
    second application after coasting is.
    In:  samples  LapSamples
         segment  Segment     a corner
         search   int         sample from which the braking is looked for
         slowest  int         sample of the minimum speed in the corner
         first    int         first sample of the corner
         last     int         sample after the last of the corner
         limits   CardLimits
    Out:          CornerEntry | None  None when the driver did not brake
    """
    distance_m, time_s = samples.distance_m, samples.time_s
    braked = np.flatnonzero(samples.brake[search : slowest + 1] > limits.brake_on) + search
    if not braked.size:
        return None
    on, off = int(braked[-1]), int(braked[-1])
    for sample in braked[::-1]:
        if samples.throttle[sample:on].max(initial=0.0) >= limits.full_throttle:
            break
        on = int(sample)
    lift = on
    while lift > search and samples.throttle[lift - 1] < limits.full_throttle:
        lift -= 1
    hardest = on + int(np.argmin(samples.lon_g[on : off + 1]))
    loaded = first + int(np.argmax(np.abs(samples.lat_g[first:last])))
    lat_g_peak = samples.lat_g[loaded]
    braked_g = samples.lat_g[on : off + 1]
    not_turning = np.flatnonzero((np.abs(braked_g) <= limits.turn_in_g) | (np.sign(braked_g) != np.sign(lat_g_peak)))
    turn_in = min(on + int(not_turning[-1]) + 1, off) if not_turning.size else on
    release = samples.brake[int(on + np.argmax(samples.brake[on : off + 1])) : off + 1]
    release_s = time_s[off + 1 - release.size : off + 1]
    locking = (samples.wheel_ratio[on : off + 1] < limits.lock_ratio) & (samples.brake[on : off + 1, None] > 0)
    locked_m = np.gradient(distance_m)[on : off + 1][locking.any(axis=1)].sum()
    front, rear = locking[:, FRONT_WHEELS].sum(), locking[:, REAR_WHEELS].sum()
    return CornerEntry(
        brake_point_m=float(distance_m[on]),
        brake_to_apex_m=float(segment.apex_m - distance_m[on]),
        speed_at_brake_kph=float(samples.speed_mps[on]) * MPS_TO_KPH,
        lift_coast_m=float(distance_m[on] - distance_m[lift]),
        peak_decel_g=float(-samples.lon_g[hardest]),
        time_to_peak_decel_s=float(time_s[hardest] - time_s[on]),
        braking_m=float(distance_m[off] - distance_m[on]),
        release_to_apex_m=float(segment.apex_m - distance_m[off]),
        trail_brake_m=float(max(distance_m[off] - distance_m[turn_in], 0.0)),
        release_roughness_per_s=_roughness(release, release_s),
        min_combined_g=float(_combined_g(samples)[hardest : max(min(off, loaded), hardest) + 1].min()),
        downshifts=int((np.diff(samples.gear[on : slowest + 1]) < 0).sum()),
        lockup_m=float(locked_m),
        lockup_axle="" if not locking.any() else "front" if front >= rear else "rear",
    )


def _mid(samples: LapSamples, segment: Segment, slowest: int, inside: slice, limits: CardLimits) -> CornerMid:
    """The middle of a corner.
    In:  samples  LapSamples
         segment  Segment     a corner
         slowest  int         sample of the minimum speed in the corner
         inside   slice       samples of the corner
         limits   CardLimits
    Out:          CornerMid
    """
    step_s = np.gradient(samples.time_s)[inside]
    steering = samples.steering[inside]
    lat_g = samples.lat_g[inside]
    coasting = (samples.brake[inside] <= limits.brake_on) & (samples.throttle[inside] <= limits.throttle_on)
    into_corner = np.sign(steering[int(np.argmax(np.abs(lat_g)))])
    against = (np.sign(steering) == -into_corner) & (np.abs(lat_g) > limits.turn_in_g)
    return CornerMid(
        min_speed_kph=float(samples.speed_mps[slowest]) * MPS_TO_KPH,
        min_speed_m=float(samples.distance_m[slowest]),
        min_speed_to_apex_m=float(segment.apex_m - samples.distance_m[slowest]),
        peak_lat_g=float(np.abs(lat_g).max()),
        coast_s=float(step_s[coasting].sum()),
        steer_rate_per_s=float(np.abs(np.gradient(samples.steering, samples.time_s)[inside]).mean()),
        countersteer_s=float(step_s[against].sum()),
    )


def _exit(samples: LapSamples, segment: Segment, slowest: int, inside: slice, limits: CardLimits) -> CornerExit:
    """The exit of a corner.
    In:  samples  LapSamples
         segment  Segment     a corner
         slowest  int         sample of the minimum speed in the corner
         inside   slice       samples of the corner
         limits   CardLimits
    Out:          CornerExit
    """
    distance_m, time_s, throttle = samples.distance_m, samples.time_s, samples.throttle
    after_m = min(segment.end_m + limits.after_exit_m, float(distance_m[-1]))
    end = int(np.searchsorted(distance_m, after_m))
    off = np.flatnonzero(throttle[inside.start : end] <= limits.throttle_on) + inside.start
    partial = np.flatnonzero(throttle[inside.start : end] < limits.full_throttle) + inside.start
    pick_up = int(off[-1]) + 1 if off.size and off[-1] + 1 < end else None
    full = int(partial[-1]) + 1 if pick_up is not None and partial[-1] + 1 < end else None
    peak_lat_g = float(np.abs(samples.lat_g[inside]).max())
    spinning = (samples.wheel_ratio[slowest:end, REAR_WHEELS] > limits.spin_ratio).any(axis=1)
    return CornerExit(
        throttle_on_after_apex_m=None if pick_up is None else float(distance_m[pick_up] - segment.apex_m),
        full_throttle_after_apex_m=None if full is None else float(distance_m[full] - segment.apex_m),
        throttle_acceptance=None if full is None else float(abs(samples.lat_g[full]) / peak_lat_g),
        part_throttle_s=None if full is None else float(time_s[full] - time_s[pick_up]),
        throttle_lifts=0
        if full is None
        else int((np.diff(throttle[pick_up : full + 1]) < -limits.throttle_lift).sum()),
        wheelspin_s=float(np.gradient(time_s)[slowest:end][spinning].sum()),
        after_exit_kph=_at(samples, samples.speed_mps, after_m) * MPS_TO_KPH,
    )


def _at(samples: LapSamples, values: np.ndarray, distance_m: float) -> float:
    """Value of a channel at a lap distance.
    In:  samples     LapSamples
         values      np.ndarray  the channel, one value per sample
         distance_m  float       m
    Out:             float
    """
    return float(np.interp(distance_m, samples.distance_m, values))


def _combined_g(samples: LapSamples) -> np.ndarray:
    """Combined lateral and longitudinal acceleration.
    In:  samples  LapSamples
    Out:          np.ndarray  g, one value per sample
    """
    return np.hypot(samples.lat_g, samples.lon_g)


def _roughness(pedal: np.ndarray, time_s: np.ndarray) -> float:
    """How unevenly a pedal moves: the mean change of its speed from one sample to the next.
    In:  pedal   np.ndarray  0-1, one value per sample
         time_s  np.ndarray  s of each sample
    Out:         float       1/s; 0 with fewer than 3 samples
    """
    if pedal.size < 3:
        return 0.0
    return float(np.abs(np.diff(np.gradient(pedal, time_s))).mean())
