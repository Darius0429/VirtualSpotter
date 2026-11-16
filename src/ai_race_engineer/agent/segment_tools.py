import dataclasses
from collections.abc import Callable
from pathlib import Path

from ai_race_engineer.recorders.corner_plot import CornerPlotter
from ai_race_engineer.strategy.corner_compare import FindingLimits, SegmentDelta, compare, line_wider_m
from ai_race_engineer.strategy.lap_summary import theoretical_best_s
from ai_race_engineer.strategy.rival_lines import RivalLap, RivalLineTracker, segment_line
from ai_race_engineer.strategy.segment_tracker import AnalysedLap, SegmentTracker
from ai_race_engineer.strategy.track_map import Segment
from ai_race_engineer.telemetry.snapshot import WheelState

UNAVAILABLE: dict[str, object] = {"available": False}
WHEELS: tuple[str, ...] = ("front_left", "front_right", "rear_left", "rear_right")
LEFT_WHEELS: tuple[int, ...] = (0, 2)


class SegmentTools:
    """Agent tools on the driver's own telemetry: where a lap lost time, the analysis and the plot of one corner,
    the comparison with another car's line and speed, and the state of the car. Unless another car is named, every
    comparison is with the driver's fastest analysed lap of the session.
    _tracker   SegmentTracker    the analysed laps and the latest snapshot
    _rivals    RivalLineTracker  the fastest lap of every other car
    _search_m  float             m before a corner in which a car's slowing point is looked for
    _plotter   CornerPlotter   draws the corner plots
    _findings  FindingLimits   differences from which something is worth saying
    """

    def __init__(
        self,
        tracker: SegmentTracker,
        rivals: RivalLineTracker,
        plotter: CornerPlotter,
        findings: FindingLimits,
        search_m: float,
    ) -> None:
        """Store the trackers, the plotter and the limits.
        In:  tracker   SegmentTracker
             rivals    RivalLineTracker
             plotter   CornerPlotter
             findings  FindingLimits
             search_m  float             m before a corner in which a car's slowing point is looked for
        Out:           None
        """
        self._tracker: SegmentTracker = tracker
        self._rivals: RivalLineTracker = rivals
        self._search_m: float = search_m
        self._plotter: CornerPlotter = plotter
        self._findings: FindingLimits = findings

    def lap_losses(self, lap: int) -> dict[str, object]:
        """Where one lap lost time against the driver's best lap of the session: every corner and straight with the
        time lost in it, most lost first, split into the part before and after the apex, and what the driver did
        differently there in plain words. Also the theoretical best lap (the best pass of every segment added up)
        and the shares of the lap at full throttle, on the brake and coasting. Start here for any question about
        where time is lost or how to go faster; then look at the worst corners with corner_analysis. Only full laps
        outside the pits are analysed.
        In:  lap  int  lap number as the driver counts it; 0 = the last analysed lap
        Out:      dict[str, object]  {lap, lap_time_s, best_lap, best_lap_time_s, theoretical_best_s, analysed_laps,
                                     full_throttle_percent, braking_percent, coasting_percent, segments: [{name,
                                     lost_s, lost_to_apex_s, lost_from_apex_s, findings}]}; {"available": False}
                                     when no lap or not this lap is analysed
        """
        laps = self._tracker.laps
        chosen = _chosen(laps, lap)
        if chosen is None:
            return UNAVAILABLE
        best = _best(laps)
        deltas = [self._delta(chosen, best, index) for index in range(len(chosen.cards))]
        return _rounded(
            {
                "lap": chosen.lap,
                "lap_time_s": chosen.lap_time_s,
                "best_lap": best.lap,
                "best_lap_time_s": best.lap_time_s,
                "theoretical_best_s": theoretical_best_s([entry.cards for entry in laps]),
                "analysed_laps": [entry.lap for entry in laps],
                "full_throttle_percent": chosen.shares.full_throttle * 100,
                "braking_percent": chosen.shares.braking * 100,
                "coasting_percent": chosen.shares.coasting * 100,
                "segments": [
                    {
                        "name": delta.name,
                        "lost_s": delta.time_s,
                        "lost_to_apex_s": delta.to_apex_s,
                        "lost_from_apex_s": delta.from_apex_s,
                        "findings": list(delta.findings),
                    }
                    for delta in sorted(deltas, key=lambda delta: delta.time_s, reverse=True)
                ],
            }
        )

    def corner_analysis(self, corner: str, lap: int) -> dict[str, object]:
        """Everything the driver did in one corner or on one straight on one lap, and the difference to the same
        segment on the driver's best lap. For a corner: the entry (brake point, speed there, coasting before the
        brake, strongest deceleration and how fast it was reached, braking distance, where the brake was released,
        trail braking, a locking wheel), the middle (minimum speed and where, cornering load, coasting, steering
        smoothness, countersteering) and the exit (throttle pick-up, where full throttle was reached, cornering
        load still on the car at full throttle, lifts, wheelspin, exit speed), with fuel and energy used. Distances
        "to apex" and "after apex" are to the apex of the track map, the same point on every lap. In vs_best every
        number is this lap minus the best lap: a negative brake_point_m is an earlier brake point, a positive
        line_wider_m a wider line at the apex. Segments carry their real-life turn numbers or names ("turn 9 and
        10" is one arc of two official turns); straights are "straight N" from the start/finish line.
        In:  corner  str  name of the segment, e.g. "turn 3", "Arnage" or "straight 2"
             lap     int  lap number as the driver counts it; 0 = the last analysed lap
        Out:         dict[str, object]  {lap, best_lap, name, time_s, ..., entry: {...}, mid: {...}, exit: {...},
                                        session_best_time_s, vs_best: {lost_s, lost_to_apex_s, lost_from_apex_s,
                                        line_wider_m, findings, <figure>: difference}}; {"available": False} when
                                        the lap is not analysed or the track has no such segment
        """
        laps = self._tracker.laps
        chosen = _chosen(laps, lap)
        index = _index(self._tracker.segments, corner)
        if chosen is None or index is None:
            return UNAVAILABLE
        best = _best(laps)
        delta = self._delta(chosen, best, index)
        card = dataclasses.asdict(chosen.cards[index])
        energy = card.pop("energy")
        return _rounded(
            {
                "lap": chosen.lap,
                "best_lap": best.lap,
                **card,
                "energy_percent": None if energy is None else energy * 100,
                "session_best_time_s": min(entry.cards[index].time_s for entry in laps),
                "vs_best": {
                    "lost_s": delta.time_s,
                    "lost_to_apex_s": delta.to_apex_s,
                    "lost_from_apex_s": delta.from_apex_s,
                    "line_wider_m": delta.line_wider_m,
                    "findings": list(delta.findings),
                    **delta.deltas,
                },
            }
        )

    def corner_plot(self, corner: str, lap: int, against: str, their_last_lap: bool) -> Path | dict[str, object]:
        """Draw one corner of one lap against the driver's best lap, or against a lap of another car,
        and look at the picture: on the left the driven line, on the right the speed over the lap distance, both
        coloured red where the driver braked, grey where the car coasted and green on the throttle. The lap it is
        compared with is the dashed black line, the orange line marks the apex and the purple line is the time lost
        to that lap as it builds up through the corner. Use it after corner_analysis or rival_comparison shows time
        lost in a corner, to see where on the line and on the pedals it went. The picture is saved for the driver.
        In:  corner   str  name of the corner, e.g. "turn 3" or "Arnage"
             lap      int  lap number as the driver counts it; 0 = the last analysed lap
             against  str  driver of another car, or part of the name; empty = the driver's own best lap
             their_last_lap  bool  with another car: true = its last full lap, false = its fastest lap
        Out:          Path | dict[str, object]  the PNG file; {"available": False} when the lap is not analysed, the
                                                track has no such segment or no lap of that car was seen
        """
        laps = self._tracker.laps
        chosen = _chosen(laps, lap)
        index = _index(self._tracker.segments, corner)
        rival = self._rival(against, their_last_lap) if against else None
        if chosen is None or index is None or (against and rival is None):
            return UNAVAILABLE
        best = _best(laps)
        which = "last lap" if their_last_lap else "best lap"
        reference, name = (
            (best.samples, f"lap {best.lap}") if rival is None else (rival.line, f"{rival.driver_name} {which}")
        )
        return self._plotter.save(chosen.samples, reference, self._tracker.segments[index], chosen.lap, name)

    def rival_comparison(self, driver: str, corner: str, last_lap: bool) -> dict[str, object]:
        """The driver against another car, from the line and the speed of both (the game gives no pedals of other
        cars): either the best lap of each, or the last analysed lap of the driver against the last full lap of
        the other car, which shows what is happening in the race now. The two last laps need not be the same race
        lap: laps through the pits are skipped. Without a corner: every corner and straight with the time lost to
        that car, most lost first, and the difference in minimum and exit speed. With a corner: for both cars the
        time, the entry, minimum, exit and top speed, where the car starts slowing and where it is slowest, and how
        much wider the driver's line is at the apex. Every difference is the driver minus the other car. Use it to
        see where a rival is faster and what they do differently; draw it with corner_plot and against.
        In:  driver    str   driver of the other car, or part of the name; empty = the fastest in the driver's class
             corner    str   name of a segment, e.g. "turn 3"; empty = all segments
             last_lap  bool  true = the last lap of both, false = the best lap of both
        Out:         dict[str, object]  {driver, class, their_lap_time_s, my_lap, my_lap_time_s, segments: [{name,
                                        lost_s, min_kph, exit_kph}]} or, with a corner, {..., name, me: {...},
                                        them: {...}, difference: {...}, line_wider_m}; {"available": False, "drivers":
                                        [names with a lap]} when the driver has no analysed lap, no lap of that
                                        car was seen or the track has no such segment
        """
        rival = self._rival(driver, last_lap)
        index = _index(self._tracker.segments, corner) if corner else None
        if not self._tracker.laps or rival is None or (corner and index is None):
            return {**UNAVAILABLE, "drivers": sorted(lap.driver_name for lap in self._rivals.best.values())}
        best = self._tracker.laps[-1] if last_lap else _best(self._tracker.laps)
        segments = self._tracker.segments
        mine = [segment_line(best.samples, segment, self._search_m) for segment in segments]
        theirs = [segment_line(rival.line, segment, self._search_m) for segment in segments]
        head = {
            "laps_compared": "last" if last_lap else "best",
            "driver": rival.driver_name,
            "class": rival.class_name,
            "their_lap_time_s": rival.lap_time_s,
            "my_lap": best.lap,
            "my_lap_time_s": best.lap_time_s,
        }
        if index is None:
            rows = [
                {
                    "name": segment.name,
                    "lost_s": me.time_s - them.time_s,
                    "min_kph": me.min_kph - them.min_kph,
                    "exit_kph": me.exit_kph - them.exit_kph,
                }
                for segment, me, them in zip(segments, mine, theirs, strict=True)
            ]
            return _rounded({**head, "segments": sorted(rows, key=lambda row: row["lost_s"], reverse=True)})
        me, them = dataclasses.asdict(mine[index]), dataclasses.asdict(theirs[index])
        apex_m = segments[index].apex_m
        return _rounded(
            {
                **head,
                "name": segments[index].name,
                "me": me,
                "them": them,
                "difference": {key: me[key] - them[key] for key in me if me[key] is not None},
                "line_wider_m": None if apex_m is None else line_wider_m(best.samples, rival.line, apex_m),
            }
        )

    def car_health(self) -> dict[str, object]:
        """The state of the driver's car right now: fuel and energy on board, tyre compounds, and per wheel the
        tread temperature on the inner, middle and outer part, the carcass temperature, the pressure, the brake
        temperature and how much of the tyre is left; also the track temperature.
        In:  -
        Out: dict[str, object]  {fuel_l, energy_percent, compound_front, compound_rear, track_temp_c, wheels:
                                {wheel: {tread_inner_c, tread_middle_c, tread_outer_c, carcass_c, pressure_kpa,
                                brake_c, tyre_left_percent}}}; {"available": False} without telemetry
        """
        snapshot = self._tracker.latest
        if snapshot is None:
            return UNAVAILABLE
        player = snapshot.player
        tyres = next((t for t in snapshot.tyres if t.slot_id == player.vehicle.slot_id), None)
        return _rounded(
            {
                "fuel_l": player.fuel_l,
                "energy_percent": None if player.virtual_energy is None else player.virtual_energy * 100,
                "compound_front": player.tyre_compound_front,
                "compound_rear": player.tyre_compound_rear,
                "track_temp_c": snapshot.session.track_temp_c,
                "wheels": {
                    name: {**_wheel(wheel, index), "tyre_left_percent": tyres and tyres.wear[index] * 100}
                    for index, (name, wheel) in enumerate(zip(WHEELS, player.wheels, strict=False))
                },
            },
            digits=1,
        )

    def tools(self) -> list[Callable[..., object]]:
        """The functions an agent may call.
        In:  -
        Out: list[Callable[..., object]]  lap_losses, corner_analysis, corner_plot, rival_comparison, car_health
        """
        return [self.lap_losses, self.corner_analysis, self.corner_plot, self.rival_comparison, self.car_health]

    def _rival(self, driver: str, last_lap: bool) -> RivalLap | None:
        """A lap of another car.
        In:  driver    str   driver name or part of it, any case; empty = the car with the fastest lap in the
                             player's class
             last_lap  bool  True = the car's last full lap, False = its fastest
        Out:           RivalLap | None  None when no such lap was seen
        """
        wanted = driver.strip().lower()
        slots = [
            slot
            for slot, lap in self._rivals.best.items()
            if (wanted in lap.driver_name.lower() if wanted else lap.class_name == self._rivals.player_class)
        ]
        slot = min(slots, key=lambda slot: self._rivals.best[slot].lap_time_s, default=None)
        return None if slot is None else (self._rivals.last if last_lap else self._rivals.best)[slot]

    def _delta(self, chosen: AnalysedLap, best: AnalysedLap, index: int) -> SegmentDelta:
        """One segment of a lap against the best lap.
        In:  chosen  AnalysedLap
             best    AnalysedLap
             index   int          position of the segment in the track map
        Out:         SegmentDelta
        """
        segment = self._tracker.segments[index]
        wider_m = line_wider_m(chosen.samples, best.samples, segment.apex_m) if segment.corner else None
        return compare(chosen.cards[index], best.cards[index], wider_m, self._findings)


def _chosen(laps: tuple[AnalysedLap, ...], lap: int) -> AnalysedLap | None:
    """The lap asked for.
    In:  laps  tuple[AnalysedLap, ...]  oldest first
         lap   int                      lap number; 0 = the last one
    Out:       AnalysedLap | None       None when it is not analysed
    """
    return next((entry for entry in reversed(laps) if lap in (0, entry.lap)), None)


def _best(laps: tuple[AnalysedLap, ...]) -> AnalysedLap:
    """The fastest lap.
    In:  laps  tuple[AnalysedLap, ...]
    Out:       AnalysedLap
    """
    return min(laps, key=lambda entry: entry.lap_time_s)


def _index(segments: tuple[Segment, ...], name: str) -> int | None:
    """Position of a segment in the track map, whatever the case of its name.
    In:  segments  tuple[Segment, ...]
         name      str
    Out:           int | None  None when the track has no such segment
    """
    return next((i for i, segment in enumerate(segments) if segment.name.lower() == name.strip().lower()), None)


def _wheel(wheel: WheelState, index: int) -> dict[str, object]:
    """One wheel as tool output, with the tread temperatures named from the car's centre line outwards.
    In:  wheel  WheelState
         index  int         0 FL, 1 FR, 2 RL, 3 RR
    Out:        dict[str, object]
    """
    left, middle, right = wheel.surface_temp_c
    inner, outer = (right, left) if index in LEFT_WHEELS else (left, right)
    return {
        "tread_inner_c": inner,
        "tread_middle_c": middle,
        "tread_outer_c": outer,
        "carcass_c": wheel.carcass_temp_c,
        "pressure_kpa": wheel.pressure_kpa,
        "brake_c": wheel.brake_temp_c,
    }


def _rounded(value: object, digits: int = 2) -> object:
    """Tool output with every number rounded, so the model reads short figures.
    In:  value   object  mappings, lists, numbers, text or None
         digits  int     decimals kept
    Out:         object  the same structure
    """
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(item, digits) for item in value]
    return value
