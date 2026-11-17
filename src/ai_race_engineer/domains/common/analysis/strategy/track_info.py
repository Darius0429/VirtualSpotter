from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitLane


@dataclass(frozen=True)
class TrackInfo:
    """Characteristics of one track.
    length_m   float                             m
    pit_lane   PitLane | None                    None when the track entry has no pit keys
    straights  tuple[tuple[float, float], ...]   (start_m, end_m) lap distances; empty when not given
    """

    length_m: float
    pit_lane: PitLane | None
    straights: tuple[tuple[float, float], ...]


def track_infos(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, TrackInfo]:
    """Track characteristics from the parsed tracks.yaml mapping.
    In:  raw  Mapping[str, Mapping[str, Any]]  track name -> keys length_m (m), and optionally pit_lane_time_s (s),
                                               pit_entry_m, pit_exit_m (m), straights ([[start_m, end_m], ...])
    Out:      dict[str, TrackInfo]             track name -> TrackInfo
    """
    return {name: _track_info(entry) for name, entry in raw.items()}


def _track_info(entry: Mapping[str, Any]) -> TrackInfo:
    """Characteristics of one track from its tracks.yaml entry.
    In:  entry  Mapping[str, Any]  keys as in track_infos
    Out:        TrackInfo
    """
    length_m = entry["length_m"]
    pit_lane = None
    if "pit_lane_time_s" in entry:
        pit_lane = PitLane(entry["pit_lane_time_s"], (entry["pit_exit_m"] - entry["pit_entry_m"]) % length_m)
    return TrackInfo(length_m, pit_lane, tuple((start_m, end_m) for start_m, end_m in entry.get("straights", ())))
