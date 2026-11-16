from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ai_race_engineer.telemetry.snapshot import PlayerState

GRAVITY_MPS2: float = 9.81


@dataclass(frozen=True)
class LineSamples:
    """The line and the speed of one car on one full lap, one value per sample.
    distance_m  NDArray[np.float64]  m driven from the start/finish line, scaled to end at the track length
    time_s      NDArray[np.float64]  s from the start of the lap, scaled to end at the lap time
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


def lap_samples(players: Sequence[PlayerState], track_length_m: float, lap_time_s: float) -> LapSamples:
    """Arrays of one lap from the player states polled from line to line; states polled twice from the same
    telemetry tick are dropped. Distance is the length of the driven line and time is that length over the speed,
    because the game's own lap distance and clock step at the scoring rate.
    In:  players         Sequence[PlayerState]  states of one lap in order
         track_length_m  float                  m
         lap_time_s      float                  s the game timed the lap
    Out:                 LapSamples
    """
    ticks = [players[0], *(b for a, b in zip(players, players[1:], strict=False) if _moved(a, b))]
    x_m, z_m = np.array([p.vehicle.position_xz_m for p in ticks], dtype=np.float64).T
    speed_mps = np.array([p.vehicle.speed_mps for p in ticks], dtype=np.float64)
    line = line_samples(x_m, z_m, speed_mps, track_length_m, lap_time_s)
    wheel_mps = np.array([[w.speed_mps for w in p.wheels] or [p.vehicle.speed_mps] * 4 for p in ticks])
    ratio = np.divide(wheel_mps, speed_mps[:, None], out=np.ones_like(wheel_mps), where=speed_mps[:, None] > 0)
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
        wheel_ratio=ratio / np.median(ratio, axis=0),
        fuel_l=np.array([p.fuel_l for p in ticks], dtype=np.float64),
        energy=np.array([p.virtual_energy for p in ticks], dtype=np.float64),
    )


def line_samples(
    x_m: NDArray[np.float64],
    z_m: NDArray[np.float64],
    speed_mps: NDArray[np.float64],
    track_length_m: float,
    lap_time_s: float,
) -> LineSamples:
    """Line of one lap from the positions and speeds polled from line to line. Distance is the length of the driven
    line and time is that length over the speed, each scaled to the track length and the timed lap.
    In:  x_m             NDArray[np.float64]  m world x of each sample
         z_m             NDArray[np.float64]  m world z of each sample
         speed_mps       NDArray[np.float64]  m/s of each sample
         track_length_m  float                m
         lap_time_s      float                s the game timed the lap
    Out:                 LineSamples
    """
    chord_m = np.hypot(np.diff(x_m), np.diff(z_m))
    mean_mps = (speed_mps[1:] + speed_mps[:-1]) / 2
    step_s = np.divide(chord_m, mean_mps, out=np.zeros_like(chord_m), where=mean_mps > 0)
    distance_m = np.concatenate(([0.0], np.cumsum(chord_m)))
    time_s = np.concatenate(([0.0], np.cumsum(step_s)))
    return LineSamples(
        distance_m * track_length_m / distance_m[-1], time_s * lap_time_s / time_s[-1], x_m, z_m, speed_mps
    )


def _moved(before: PlayerState, after: PlayerState) -> bool:
    """Whether the car's position changed between two states.
    In:  before  PlayerState
         after   PlayerState
    Out:         bool
    """
    return before.vehicle.position_xz_m != after.vehicle.position_xz_m
