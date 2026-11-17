from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.domains.common.data.snapshot import PlayerState

GRAVITY_MPS2: float = 9.81


@dataclass(frozen=True)
class LineSamples:
    """The line and the speed of one car on one full lap, one value per sample.
    distance_m  NDArray[np.float64]  m lap distance as the game measures it, filled in between its updates from the
                                     driven line: the same place on the track has the same distance on every lap
    time_s      NDArray[np.float64]  s from the start of the lap; on a finished lap scaled to end at the lap time
    x_m         NDArray[np.float64]  m world x
    z_m         NDArray[np.float64]  m world z
    speed_mps   NDArray[np.float64]  m/s
    """

    distance_m: NDArray[np.float64]
    time_s: NDArray[np.float64]
    x_m: NDArray[np.float64]
    z_m: NDArray[np.float64]
    speed_mps: NDArray[np.float64]


@dataclass(frozen=True)
class LapSamples(LineSamples):
    """The player's telemetry of one full lap, one value per sample: the line and the speed, and
    throttle     NDArray[np.float64]  0-1
    brake        NDArray[np.float64]  0-1 pedal
    steering     NDArray[np.float64]  -1 (left) to 1 (right)
    gear         NDArray[np.int64]    -1 reverse, 0 neutral, 1+ forward
    lat_g        NDArray[np.float64]  g lateral acceleration
    lon_g        NDArray[np.float64]  g longitudinal acceleration, braking negative
    wheel_ratio  NDArray[np.float64]  rolling speed of each wheel over the car's speed, 1 = rolling freely (the lap's
                                      median); one row per sample, columns FL, FR, RL, RR
    fuel_l       NDArray[np.float64]  litres on board
    energy       NDArray[np.float64]  virtual energy on board, fraction 0-1; NaN when the game has none
    """

    throttle: NDArray[np.float64]
    brake: NDArray[np.float64]
    steering: NDArray[np.float64]
    gear: NDArray[np.int64]
    lat_g: NDArray[np.float64]
    lon_g: NDArray[np.float64]
    wheel_ratio: NDArray[np.float64]
    fuel_l: NDArray[np.float64]
    energy: NDArray[np.float64]


def lap_samples(players: Sequence[PlayerState], lap_time_s: float | None) -> LapSamples:
    """Arrays of one lap from the player states polled from the line on, for a finished lap or for the lap in
    progress; states polled twice from the same telemetry tick are dropped.
    In:  players     Sequence[PlayerState]  states of one lap in order, from the line
         lap_time_s  float | None           s the game timed the lap; None = the lap is still being driven
    Out:             LapSamples
    """
    ticks = [players[0], *(b for a, b in zip(players, players[1:], strict=False) if _moved(a, b))]
    x_m, z_m = np.array([p.vehicle.position_xz_m for p in ticks], dtype=np.float64).T
    speed_mps = np.array([p.vehicle.speed_mps for p in ticks], dtype=np.float64)
    lap_distance_m = np.array([p.vehicle.lap_distance_m for p in ticks], dtype=np.float64)
    line = line_samples(x_m, z_m, speed_mps, lap_distance_m, lap_time_s)
    wheel_mps = np.array([[w.speed_mps for w in p.wheels] or [p.vehicle.speed_mps] * 4 for p in ticks])
    ratio = np.divide(wheel_mps, speed_mps[:, None], out=np.ones_like(wheel_mps), where=speed_mps[:, None] > 0)
    rolling = np.median(ratio, axis=0)
    return LapSamples(
        distance_m=line.distance_m,
        time_s=line.time_s,
        x_m=x_m,
        z_m=z_m,
        speed_mps=speed_mps,
        throttle=np.array([p.throttle for p in ticks], dtype=np.float64),
        brake=np.array([p.brake for p in ticks], dtype=np.float64),
        steering=np.array([p.steering for p in ticks], dtype=np.float64),
        gear=np.array([p.gear for p in ticks], dtype=np.int64),
        lat_g=np.array([p.accel_lateral_mps2 for p in ticks], dtype=np.float64) / GRAVITY_MPS2,
        lon_g=-np.array([p.accel_longitudinal_mps2 for p in ticks], dtype=np.float64) / GRAVITY_MPS2,
        wheel_ratio=ratio / np.where(rolling > 0, rolling, 1.0),
        fuel_l=np.array([p.fuel_l for p in ticks], dtype=np.float64),
        energy=np.array([p.virtual_energy for p in ticks], dtype=np.float64),
    )


def line_samples(
    x_m: NDArray[np.float64],
    z_m: NDArray[np.float64],
    speed_mps: NDArray[np.float64],
    lap_distance_m: NDArray[np.float64],
    lap_time_s: float | None,
) -> LineSamples:
    """Line of one lap from the positions and speeds polled from the line on. The game's own lap distance and clock
    step at the scoring rate, so the distance is the driven line pinned to the game's lap distance at each of its
    updates, and the time is the driven line over the speed.
    In:  x_m             NDArray[np.float64]  m world x of each sample
         z_m             NDArray[np.float64]  m world z of each sample
         speed_mps       NDArray[np.float64]  m/s of each sample
         lap_distance_m  NDArray[np.float64]  m lap distance the game showed at each sample
         lap_time_s      float | None         s the time is scaled to end at; None = not scaled
    Out:                 LineSamples
    """
    chord_m = np.hypot(np.diff(x_m), np.diff(z_m))
    mean_mps = (speed_mps[1:] + speed_mps[:-1]) / 2
    step_s = np.divide(chord_m, mean_mps, out=np.zeros_like(chord_m), where=mean_mps > 0)
    driven_m = np.concatenate(([0.0], np.cumsum(chord_m)))
    time_s = np.concatenate(([0.0], np.cumsum(step_s)))
    time_scale = 1.0 if lap_time_s is None else lap_time_s / time_s[-1]
    return LineSamples(_anchored_m(driven_m, lap_distance_m), time_s * time_scale, x_m, z_m, speed_mps)


def _anchored_m(driven_m: NDArray[np.float64], lap_distance_m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lap distance of every sample: the game's lap distance where the game has just updated it and further than
    ever before on the lap, the driven line between those samples and beyond the first and the last of them.
    In:  driven_m        NDArray[np.float64]  m driven from the first sample
         lap_distance_m  NDArray[np.float64]  m lap distance the game showed at each sample
    Out:                 NDArray[np.float64]  m
    """
    fresh = np.flatnonzero(np.diff(lap_distance_m) > 0) + 1
    before_m = np.concatenate(([-np.inf], np.maximum.accumulate(lap_distance_m[fresh])[:-1]))
    fresh = fresh[lap_distance_m[fresh] > before_m]
    if fresh.size < 2:
        return lap_distance_m[0] + driven_m
    at_m, shown_m = driven_m[fresh], lap_distance_m[fresh]
    slope = (shown_m[-1] - shown_m[0]) / (at_m[-1] - at_m[0])
    ahead_m = np.clip(driven_m - at_m[-1], 0.0, None) - np.clip(at_m[0] - driven_m, 0.0, None)
    return np.interp(driven_m, at_m, shown_m) + ahead_m * slope


def _moved(before: PlayerState, after: PlayerState) -> bool:
    """Whether the car's position changed between two states.
    In:  before  PlayerState
         after   PlayerState
    Out:         bool
    """
    return before.vehicle.position_xz_m != after.vehicle.position_xz_m
