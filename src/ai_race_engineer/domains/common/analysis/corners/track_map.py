from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class Segment:
    """One corner or straight of a track; the segments cover the lap from the start/finish line in driving order.
    name     str           "turn 3" | "straight 2", counted from the start/finish line
    corner   bool          False = straight
    start_m  float         m lap distance
    end_m    float         m lap distance
    apex_m   float | None  m lap distance of the tightest point; None for a straight
    """

    name: str
    corner: bool
    start_m: float
    end_m: float
    apex_m: float | None


def track_map(
    distance_m: NDArray[np.float64],
    x_m: NDArray[np.float64],
    z_m: NDArray[np.float64],
    track_length_m: float,
    step_m: float,
    smooth_m: float,
    max_radius_m: float,
    min_turn_deg: float,
    entry_pad_m: float,
    exit_pad_m: float,
    merge_gap_m: float,
) -> tuple[Segment, ...]:
    """Corners and straights of a track from the driven line of one lap. A corner is where the path radius is below
    max_radius_m and the line turns by at least min_turn_deg (less is a kink or a weave on a straight); corners
    closer than merge_gap_m become one (chicanes, esses). Each corner is then widened by the pads to hold the
    braking and the exit, as far as the neighbouring corners leave room. The start/finish line is taken to lie on a
    straight.
    In:  distance_m      NDArray[np.float64]  m lap distance of each sample, increasing
         x_m             NDArray[np.float64]  m world x of each sample
         z_m             NDArray[np.float64]  m world z of each sample
         track_length_m  float                m
         step_m          float                m spacing the line is resampled to
         smooth_m        float                m window of the smoothing filter
         max_radius_m    float                m path radius below which the line is a corner
         min_turn_deg    float                deg the line must turn in a corner
         entry_pad_m     float                m added before a corner
         exit_pad_m      float                m added after a corner
         merge_gap_m     float                m; corners closer than this are merged
    Out:                 tuple[Segment, ...]  in driving order
    """
    grid = np.arange(0.0, track_length_m, step_m)
    signed = curvature_per_m(np.interp(grid, distance_m, x_m), np.interp(grid, distance_m, z_m), step_m, smooth_m)
    kappa = np.abs(signed)
    padded = np.concatenate(([False], kappa > 1.0 / max_radius_m, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    corners: list[tuple[float, float, float, float]] = []
    for start, end in zip(starts, ends, strict=True):
        if abs(signed[start:end].sum()) * step_m < np.radians(min_turn_deg):
            continue
        apex = start + int(np.argmax(kappa[start:end]))
        run = (float(grid[start]), float(grid[end - 1]), float(grid[apex]), float(kappa[apex]))
        if corners and run[0] - corners[-1][1] < merge_gap_m:
            last = corners[-1]
            corners[-1] = (last[0], run[1], *(run[2:] if run[3] > last[3] else last[2:]))
        else:
            corners.append(run)
    return _segments(corners, track_length_m, entry_pad_m, exit_pad_m)


def curvature_per_m(
    x_m: NDArray[np.float64], z_m: NDArray[np.float64], step_m: float, smooth_m: float
) -> NDArray[np.float64]:
    """Signed curvature of a closed path sampled at equal spacing: (x'z'' - z'x'') / (x'^2 + z'^2)^1.5, with the
    derivatives from a Savitzky-Golay filter.
    In:  x_m       NDArray[np.float64]  m world x
         z_m       NDArray[np.float64]  m world z
         step_m    float                m between two samples
         smooth_m  float                m filter window
    Out:           NDArray[np.float64]  1/m
    """
    window = int(smooth_m / step_m) // 2 * 2 + 1
    dx, ddx, dz, ddz = (
        savgol_filter(values, window, 3, deriv=order, delta=step_m, mode="wrap")
        for values in (x_m, z_m)
        for order in (1, 2)
    )
    return (dx * ddz - dz * ddx) / (dx**2 + dz**2) ** 1.5


def _segments(
    corners: list[tuple[float, float, float, float]], track_length_m: float, entry_pad_m: float, exit_pad_m: float
) -> tuple[Segment, ...]:
    """Named, padded corners with the straights between them. A corner's exit pad ends at the next corner at the
    latest; its entry pad starts where the corner before it ends at the earliest.
    In:  corners         list[tuple[float, float, float, float]]  (start_m, end_m, apex_m, peak curvature 1/m)
         track_length_m  float                                    m
         entry_pad_m     float                                    m added before a corner
         exit_pad_m      float                                    m added after a corner
    Out:                 tuple[Segment, ...]                      in driving order
    """
    segments: list[Segment] = []
    position_m = 0.0
    straights = 0
    for number, (turn_in_m, turn_out_m, apex_m, _) in enumerate(corners, start=1):
        start_m = max(turn_in_m - entry_pad_m, position_m)
        end_m = min(turn_out_m + exit_pad_m, corners[number][0] if number < len(corners) else track_length_m)
        if start_m > position_m:
            straights += 1
            segments.append(Segment(f"straight {straights}", False, position_m, start_m, None))
        segments.append(Segment(f"turn {number}", True, start_m, end_m, apex_m))
        position_m = end_m
    if position_m < track_length_m:
        segments.append(Segment(f"straight {straights + 1}", False, position_m, track_length_m, None))
    return tuple(segments)
