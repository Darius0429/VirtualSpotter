import dataclasses
from dataclasses import dataclass

import numpy as np

from ai_race_engineer.strategy.corner_card import SegmentCard
from ai_race_engineer.strategy.lap_segments import LineSamples

LINE_STEP_M: float = 5.0


@dataclass(frozen=True)
class FindingLimits:
    """Differences to the reference lap from which something is worth saying.
    distance_m  float  m of a brake, throttle or line difference
    speed_kph   float  km/h
    decel_g     float  g
    time_s      float  s of coasting or countersteering
    """

    distance_m: float
    speed_kph: float
    decel_g: float
    time_s: float


@dataclass(frozen=True)
class SegmentDelta:
    """One segment of a lap against the same segment of a reference lap; every difference is mine minus reference.
    name           str
    time_s         float              s lost (negative = gained)
    to_apex_s      float | None       s lost from the start of the segment to the apex; None for a straight
    from_apex_s    float | None       s lost from the apex to the end of the segment; None for a straight
    line_wider_m   float | None       m the car was further from the inside at the apex; None for a straight
    deltas         dict[str, float]   difference of every figure both cards have, keyed by its field name
    findings       tuple[str, ...]    the differences worth saying, in driving order
    """

    name: str
    time_s: float
    to_apex_s: float | None
    from_apex_s: float | None
    line_wider_m: float | None
    deltas: dict[str, float]
    findings: tuple[str, ...]


def compare(
    mine: SegmentCard, reference: SegmentCard, line_wider_m: float | None, limits: FindingLimits
) -> SegmentDelta:
    """Compare one segment of a lap with the same segment of a reference lap.
    In:  mine          SegmentCard
         reference     SegmentCard
         line_wider_m  float | None   m the car was further from the inside at the apex; None for a straight
         limits        FindingLimits
    Out:               SegmentDelta
    """
    time_s = mine.time_s - reference.time_s
    to_apex_s = None if mine.time_to_apex_s is None else mine.time_to_apex_s - reference.time_to_apex_s
    deltas = {
        **_deltas(mine, reference),
        **_deltas(mine.entry, reference.entry),
        **_deltas(mine.mid, reference.mid),
        **_deltas(mine.exit, reference.exit),
    }
    return SegmentDelta(
        name=mine.name,
        time_s=time_s,
        to_apex_s=to_apex_s,
        from_apex_s=None if to_apex_s is None else time_s - to_apex_s,
        line_wider_m=line_wider_m,
        deltas=deltas,
        findings=_findings(mine, reference, deltas, line_wider_m, limits),
    )


def line_wider_m(mine: LineSamples, reference: LineSamples, apex_m: float) -> float:
    """How much further from the inside of a corner one lap's line was than another's at the apex.
    In:  mine       LineSamples
         reference  LineSamples
         apex_m     float        m lap distance of the apex
    Out:            float       m; positive = wider than the reference, negative = tighter
    """
    before, at, after = (_position(reference, apex_m + offset) for offset in (-LINE_STEP_M, 0.0, LINE_STEP_M))
    heading = after - before
    turn = _cross(at - before, after - at)
    side = _cross(heading, _position(mine, apex_m) - at) / float(np.hypot(*heading))
    return float(-np.sign(turn) * side)


def _deltas(mine: object, reference: object) -> dict[str, float]:
    """Difference of every number two dataclasses of the same kind both have.
    In:  mine       object  dataclass or None
         reference  object  dataclass of the same kind or None
    Out:            dict[str, float]  field name -> mine minus reference; empty when either is None
    """
    if mine is None or reference is None:
        return {}
    pairs = ((f.name, getattr(mine, f.name), getattr(reference, f.name)) for f in dataclasses.fields(mine))
    return {
        name: a - b
        for name, a, b in pairs
        if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool)
    }


def _findings(
    mine: SegmentCard, reference: SegmentCard, deltas: dict[str, float], wider_m: float | None, limits: FindingLimits
) -> tuple[str, ...]:
    """The differences worth saying, as short phrases in driving order.
    In:  mine       SegmentCard
         reference  SegmentCard
         deltas     dict[str, float]  field name -> mine minus reference
         wider_m    float | None      m wider at the apex
         limits     FindingLimits
    Out:            tuple[str, ...]
    """
    found: list[str] = []
    brake_m = deltas.get("brake_point_m", 0.0)
    if abs(brake_m) >= limits.distance_m:
        found.append(f"braked {abs(brake_m):.0f} m {'earlier' if brake_m < 0 else 'later'}")
    if mine.entry is not None and mine.entry.lift_coast_m >= limits.distance_m:
        found.append(f"coasted {mine.entry.lift_coast_m:.0f} m between lifting and braking")
    if deltas.get("peak_decel_g", 0.0) <= -limits.decel_g:
        found.append(f"braked softer, {mine.entry.peak_decel_g:.1f} g against {reference.entry.peak_decel_g:.1f} g")
    if mine.entry is not None and mine.entry.lockup_m >= limits.distance_m:
        found.append(f"locked a {mine.entry.lockup_axle} wheel for {mine.entry.lockup_m:.0f} m")
    if wider_m is not None and abs(wider_m) >= limits.distance_m / 2:
        found.append(f"{abs(wider_m):.1f} m {'wider' if wider_m > 0 else 'tighter'} at the apex")
    if abs(deltas.get("min_speed_kph", 0.0)) >= limits.speed_kph:
        slower = deltas["min_speed_kph"] < 0
        found.append(f"minimum speed {abs(deltas['min_speed_kph']):.0f} kph {'lower' if slower else 'higher'}")
    if deltas.get("coast_s", 0.0) >= limits.time_s:
        found.append(f"coasted {deltas['coast_s']:.1f} s longer in the corner")
    if deltas.get("countersteer_s", 0.0) >= limits.time_s:
        found.append("had to countersteer")
    full_m = deltas.get("full_throttle_after_apex_m", 0.0)
    if abs(full_m) >= limits.distance_m:
        found.append(f"full throttle {abs(full_m):.0f} m {'later' if full_m > 0 else 'earlier'}")
    if deltas.get("throttle_lifts", 0) > 0:
        found.append("lifted on the exit")
    if deltas.get("wheelspin_s", 0.0) >= limits.time_s:
        found.append(f"wheelspin for {mine.exit.wheelspin_s:.1f} s on the exit")
    if abs(deltas.get("exit_kph", 0.0)) >= limits.speed_kph:
        found.append(f"exit speed {abs(deltas['exit_kph']):.0f} kph {'lower' if deltas['exit_kph'] < 0 else 'higher'}")
    return tuple(found)


def _position(samples: LineSamples, distance_m: float) -> np.ndarray:
    """World position at a lap distance.
    In:  samples     LineSamples
         distance_m  float        m
    Out:             np.ndarray   m (x, z)
    """
    return np.array(
        [np.interp(distance_m, samples.distance_m, samples.x_m), np.interp(distance_m, samples.distance_m, samples.z_m)]
    )


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    """z component of the cross product of two vectors in the plane.
    In:  a  np.ndarray  (x, z)
         b  np.ndarray  (x, z)
    Out:    float
    """
    return float(a[0] * b[1] - a[1] * b[0])
