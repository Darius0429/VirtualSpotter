import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from ai_race_engineer.domains.base import Domain, Tool
from ai_race_engineer.domains.common.callouts.base import Event
from ai_race_engineer.domains.common.data.garage import GarageInfo, GarageSource
from ai_race_engineer.domains.common.data.pit_estimate import PitEstimate, PitEstimateSource
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot
from ai_race_engineer.domains.common.data.race_source import RaceSource
from ai_race_engineer.domains.lmu.energy_tools import EnergyTools
from ai_race_engineer.domains.lmu.garage_poller import GaragePoller
from ai_race_engineer.domains.lmu.instructions import LMU_INSTRUCTIONS
from ai_race_engineer.domains.lmu.lmu_garage import LMUGarageSource
from ai_race_engineer.domains.lmu.lmu_pit_estimate import LMUPitEstimateSource
from ai_race_engineer.domains.lmu.lmu_race import LMURaceSource
from ai_race_engineer.domains.lmu.pit_estimate_poller import PitEstimatePoller
from ai_race_engineer.domains.lmu.plan_briefing import PlanBriefing
from ai_race_engineer.domains.lmu.replay import ReplaySource
from ai_race_engineer.domains.lmu.replay_garage import ReplayGarage
from ai_race_engineer.domains.lmu.replay_pit_estimate import ReplayPitEstimate
from ai_race_engineer.domains.lmu.replay_wall_clock import ReplayWallClock
from ai_race_engineer.domains.lmu.timed_pit_estimate import TimedPitEstimate


class LMUDomain(Domain):
    """Le Mans Ultimate, live or from a raw dump: the shared pipeline plus the game's pit estimate, garage figures,
    virtual energy tool and plan briefing.
    _poller         PitEstimatePoller  the game's pit-stop estimate, read in its own thread
    _garage_poller  GaragePoller       the game's garage figures, read in its own thread
    _energy_tools   EnergyTools
    _plan_briefing  PlanBriefing
    _recorded_s     tuple[float, float]  s unix time of the first and last row of the recording; all time when live
    """

    def __init__(
        self,
        config: Mapping[str, object],
        recording: Path | None = None,
        speed: float = 1.0,
        log: bool = True,
        questions: bool = False,
    ) -> None:
        """Pick the live game or a recording and build the pipeline around it.
        In:  config       Mapping[str, object]  configs/lmu.yaml
             recording    Path | None           raw-dump folder to replay; None = live game
             speed        float                 replay s per wall s
             log          bool                  write the run log
             questions    bool                  in a replay, ask the questions of the recorded run again
        Out:              None
        """
        est_cfg, garage_cfg = config["source"]["pit_estimate"], config["source"]["garage"]
        if recording:
            replay = ReplaySource(recording, speed)
            source: RaceSource = replay
            estimates: PitEstimateSource = ReplayPitEstimate(recording, replay.time_s)
            garage_source: GarageSource = ReplayGarage(recording, replay.time_s)
            wall_clock = ReplayWallClock(recording, replay.time_s)
            now_unix = wall_clock.unix_s
            self._recorded_s: tuple[float, float] = wall_clock.span_s
            time_scale = speed
        else:
            source = LMURaceSource()
            estimates = LMUPitEstimateSource(est_cfg["url"], est_cfg["timeout_s"])
            garage_source = LMUGarageSource(garage_cfg["url"], garage_cfg["timeout_s"])
            now_unix = time.time
            self._recorded_s = (-math.inf, math.inf)
            time_scale = 1.0
        super().__init__(config, source, now_unix, time_scale, log, recording, questions)
        self._poller: PitEstimatePoller = PitEstimatePoller(
            TimedPitEstimate(estimates, self._health.timed), est_cfg["period_s"] / time_scale
        )
        self._garage_poller: GaragePoller = GaragePoller(
            self._health.timed("garage_read", garage_source.read), garage_cfg["period_s"] / time_scale
        )
        race_cfg = config["race"]
        self._energy_tools: EnergyTools = EnergyTools(
            self._tracker, race_cfg["lift_coast_saving_percent_per_lap"], race_cfg["lift_coast_cost_s_per_lap"]
        )
        self._plan_briefing: PlanBriefing = PlanBriefing(
            self._plan_tools.race_plan, self._energy_tools.pit_energy, config["callouts"]["plan_briefing_hold_s"]
        )

    def instructions(self) -> str:
        """What the LMU data holds and how to use the tools.
        In:  -
        Out: str
        """
        return LMU_INSTRUCTIONS

    def _own_observers(self) -> list[Callable[[RaceSnapshot], None]]:
        """The garage poller follows the session, the car and the pits.
        In:  -
        Out: list[Callable[[RaceSnapshot], None]]
        """
        return [self._garage_poller.update]

    def _own_callouts(self) -> list[Event]:
        """The race plan read once before the start.
        In:  -
        Out: list[Event]
        """
        return [self._plan_briefing]

    def _own_tools(self) -> list[Tool]:
        """The virtual energy tools.
        In:  -
        Out: list[Tool]
        """
        return self._energy_tools.tools()

    def _own_loops(self) -> list[Callable[[], None]]:
        """The two pollers of the game's REST API.
        In:  -
        Out: list[Callable[[], None]]
        """
        return [self._poller.run, self._garage_poller.run]

    def _recorded_wall_s(self) -> tuple[float, float]:
        """The time the replayed recording covers.
        In:  -
        Out: tuple[float, float]  s unix time of its first and last row
        """
        return self._recorded_s

    def _pit_estimate(self) -> PitEstimate | None:
        """The game's latest pit-stop estimate for the player (non-blocking).
        In:  -
        Out: PitEstimate | None  None before the first read or when the game gave none
        """
        return self._poller.latest()

    def _garage(self) -> GarageInfo | None:
        """The game's latest garage figures for the player (non-blocking).
        In:  -
        Out: GarageInfo | None  None until a read succeeds
        """
        return self._garage_poller.latest()
