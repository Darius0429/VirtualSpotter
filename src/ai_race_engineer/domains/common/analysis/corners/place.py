from collections.abc import Sequence

from ai_race_engineer.domains.common.analysis.corners.track_map import Segment


def place(segments: Sequence[Segment], lap_distance_m: float, apex_window_m: float) -> str:
    """Where on the track a lap distance is, in words.
    In:  segments        Sequence[Segment]  the track map; empty until known
         lap_distance_m  float              m from the start/finish line
         apex_window_m   float              m either side of the apex that reads "apex of"
    Out:                 str  "entry of turn 2" | "apex of turn 2" | "exit of turn 2" | "straight 3"; "" without a map
    """
    segment = next((s for s in segments if s.start_m <= lap_distance_m < s.end_m), None)
    if segment is None:
        return ""
    if not segment.corner:
        return segment.name
    if abs(lap_distance_m - segment.apex_m) <= apex_window_m:
        return f"apex of {segment.name}"
    return f"{'entry' if lap_distance_m < segment.apex_m else 'exit'} of {segment.name}"
