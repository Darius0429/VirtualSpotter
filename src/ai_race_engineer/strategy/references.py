from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ai_race_engineer.strategy.corner_card import SegmentCard
from ai_race_engineer.strategy.lap_segments import LineSamples
from ai_race_engineer.strategy.rival_lines import RivalLap
from ai_race_engineer.strategy.segment_tracker import AnalysedLap


@dataclass(frozen=True)
class Reference:
    """A lap the driver's segments, sectors and laps are compared with.
    name        str                             what the lap is called: "lap 9", "stored best", a driver's name
    lap_time_s  float                           s
    line        LineSamples                     the line and the speed of the lap
    cards       tuple[SegmentCard, ...] | None  what the driver did in every segment; None for another car's lap
    """

    name: str
    lap_time_s: float
    line: LineSamples
    cards: tuple[SegmentCard, ...] | None


def own_reference(lap: AnalysedLap, session_laps: Sequence[AnalysedLap]) -> Reference:
    """A lap of the driver as a reference, named after its lap number when it was driven in this session.
    In:  lap           AnalysedLap
         session_laps  Sequence[AnalysedLap]  the analysed laps of the session
    Out:               Reference
    """
    number = next((entry.lap for entry in session_laps if entry.lap_time_s == lap.lap_time_s), 0)
    return Reference(f"lap {number}" if number else "stored best", lap.lap_time_s, lap.samples, lap.cards)


def rival_reference(lap: RivalLap) -> Reference:
    """A lap of another car as a reference.
    In:  lap  RivalLap
    Out:      Reference
    """
    return Reference(lap.driver_name, lap.lap_time_s, lap.line, None)


def fastest(candidates: Iterable[Reference | None]) -> Reference | None:
    """The fastest of some laps.
    In:  candidates  Iterable[Reference | None]
    Out:             Reference | None  None when there is none
    """
    return min((lap for lap in candidates if lap is not None), key=lambda lap: lap.lap_time_s, default=None)
