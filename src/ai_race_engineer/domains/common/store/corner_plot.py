from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ai_race_engineer.domains.common.analysis.corners.lap_segments import LapSamples, LineSamples
from ai_race_engineer.domains.common.analysis.corners.track_map import Segment

BRAKING: str = "#d62728"
COASTING: str = "#8c8c8c"
THROTTLE: str = "#2ca02c"
MPS_TO_KPH: float = 3.6


class CornerPlotter:
    """Draws one corner of a lap against a reference lap and saves it as a PNG: the driven line and the speed over
    the lap distance, both coloured by what the driver did (braking red, coasting grey, throttle green), with the
    reference lap dashed and the time lost to it.
    _folder       Path   folder of the pictures
    _margin_m     float  m shown before and after the corner
    _brake_on     float  0-1 brake above which the driver is braking
    _throttle_on  float  0-1 throttle above which the driver is on the throttle
    """

    def __init__(self, folder: Path, margin_m: float, brake_on: float, throttle_on: float) -> None:
        """Set the folder and what counts as braking and throttle.
        In:  folder       Path   created on the first save
             margin_m     float  m shown before and after the corner
             brake_on     float  0-1
             throttle_on  float  0-1
        Out:              None
        """
        self._folder: Path = folder
        self._margin_m: float = margin_m
        self._brake_on: float = brake_on
        self._throttle_on: float = throttle_on

    def save(self, mine: LapSamples, reference: LineSamples, segment: Segment, lap: int, reference_name: str) -> Path:
        """Draw a corner of one lap against a reference lap of the driver or of another car; a disk error raises
        OSError.
        In:  mine            LapSamples
             reference       LineSamples  the lap drawn dashed
             segment         Segment      the corner
             lap             int          lap number of mine
             reference_name  str          what the reference is called in the picture, e.g. "lap 4" or a driver
        Out:                 Path         the PNG file
        """
        low_m, high_m = segment.start_m - self._margin_m, segment.end_m + self._margin_m
        own = np.flatnonzero((mine.distance_m >= low_m) & (mine.distance_m <= high_m))
        ref = np.flatnonzero((reference.distance_m >= low_m) & (reference.distance_m <= high_m))
        colours = np.where(
            mine.brake[own] > self._brake_on,
            BRAKING,
            np.where(mine.throttle[own] > self._throttle_on, THROTTLE, COASTING),
        )
        figure = Figure(figsize=(11, 4.6), dpi=90)
        FigureCanvasAgg(figure)
        line, speed = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.5]})
        line.plot(reference.x_m[ref], reference.z_m[ref], color="black", lw=1, ls="--", label=reference_name)
        _coloured(line, mine.x_m[own], mine.z_m[own], colours, 4.0)
        slowest = own[int(np.argmin(mine.speed_mps[own]))]
        line.plot(mine.x_m[slowest], mine.z_m[slowest], "o", color="black", ms=6)
        line.annotate(
            f"min {mine.speed_mps[slowest] * MPS_TO_KPH:.0f} kph",
            (mine.x_m[slowest], mine.z_m[slowest]),
            textcoords="offset points",
            xytext=(8, 8),
        )
        line.annotate("entry", (mine.x_m[own[0]], mine.z_m[own[0]]), textcoords="offset points", xytext=(6, 6))
        line.set_aspect("equal")
        line.margins(0.1)
        line.set_xticks([])
        line.set_yticks([])
        line.set_title("driven line")
        line.legend(loc="best", fontsize=8)
        speed.plot(
            reference.distance_m[ref],
            reference.speed_mps[ref] * MPS_TO_KPH,
            color="black",
            lw=1,
            ls="--",
            label=reference_name,
        )
        _coloured(speed, mine.distance_m[own], mine.speed_mps[own] * MPS_TO_KPH, colours, 3.0)
        for label, colour in (("braking", BRAKING), ("coasting", COASTING), ("throttle", THROTTLE)):
            speed.plot([], [], color=colour, lw=3, label=label)
        speed.axvline(segment.apex_m, color="orange", lw=1, label="apex")
        speed.set_xlabel("lap distance (m)")
        speed.set_ylabel("speed (kph)")
        speed.grid(alpha=0.3)
        speed.legend(loc="lower left", fontsize=8)
        speed.set_title(f"{segment.name}: lap {lap} against {reference_name}")
        lost_s = (mine.time_s[own] - mine.time_s[own[0]]) - (
            np.interp(mine.distance_m[own], reference.distance_m, reference.time_s)
            - np.interp(mine.distance_m[own[0]], reference.distance_m, reference.time_s)
        )
        loss = speed.twinx()
        loss.plot(mine.distance_m[own], lost_s, color="tab:purple", lw=1.2)
        loss.axhline(0.0, color="tab:purple", lw=0.4)
        loss.set_ylabel(f"time lost to {reference_name} (s)", color="tab:purple")
        figure.tight_layout()
        self._folder.mkdir(parents=True, exist_ok=True)
        path = self._folder / f"lap {lap} {segment.name} against {reference_name}.png"
        figure.savefig(path)
        return path


def _coloured(
    axes: Axes, x: NDArray[np.float64], y: NDArray[np.float64], colours: NDArray[np.str_], width: float
) -> None:
    """Draw a line whose pieces each have their own colour.
    In:  axes     Axes
         x        NDArray[np.float64]  one value per point
         y        NDArray[np.float64]  one value per point
         colours  NDArray[np.str_]     colour of the piece that starts at each point
         width    float                line width in points
    Out:          None
    """
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    pieces = np.concatenate([points[:-1], points[1:]], axis=1)
    axes.add_collection(LineCollection(pieces, colors=list(colours[:-1]), linewidths=width, capstyle="round"))
    axes.autoscale()
