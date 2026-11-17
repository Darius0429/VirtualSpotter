import logging
import math
import os
import queue
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import yaml
from elevenlabs.client import ElevenLabs

from ai_race_engineer.domains.common.analysis.corners.corner_card import CardLimits
from ai_race_engineer.domains.common.analysis.corners.corner_compare import FindingLimits
from ai_race_engineer.domains.common.analysis.corners.lap_types import LapTypeTracker
from ai_race_engineer.domains.common.analysis.corners.live_segments import LiveSegmentTracker
from ai_race_engineer.domains.common.analysis.corners.rival_lines import RivalLineTracker
from ai_race_engineer.domains.common.analysis.corners.segment_tracker import SegmentTracker
from ai_race_engineer.domains.common.analysis.field.contacts import ContactTracker
from ai_race_engineer.domains.common.analysis.field.prior_field import PriorFieldTracker
from ai_race_engineer.domains.common.analysis.field.race_state import RaceTracker
from ai_race_engineer.domains.common.analysis.pace.straight_learned import LearnedStraightDetector
from ai_race_engineer.domains.common.analysis.pace.straight_live import LiveStraightDetector
from ai_race_engineer.domains.common.analysis.session.lap_history import LapHistory
from ai_race_engineer.domains.common.analysis.session.session_log import SessionLog
from ai_race_engineer.domains.common.analysis.session.timeline_builder import TimelineBuilder
from ai_race_engineer.domains.common.analysis.strategy.consumption import ConsumptionTracker
from ai_race_engineer.domains.common.analysis.strategy.pit_log import PitStopLog
from ai_race_engineer.domains.common.analysis.strategy.pit_time import PitLane, PitRates
from ai_race_engineer.domains.common.analysis.strategy.track_info import track_infos
from ai_race_engineer.domains.common.callouts.base import Callout, Event, Priority
from ai_race_engineer.domains.common.callouts.fastest_lap import FastestLap
from ai_race_engineer.domains.common.callouts.flags import Flags
from ai_race_engineer.domains.common.callouts.grid_briefing import GridBriefing
from ai_race_engineer.domains.common.callouts.logged_queue import LoggedCalloutQueue
from ai_race_engineer.domains.common.callouts.low_fuel import LowFuel
from ai_race_engineer.domains.common.callouts.monitor import EventMonitor
from ai_race_engineer.domains.common.callouts.overall_best import OverallBest
from ai_race_engineer.domains.common.callouts.personal_best import PersonalBest
from ai_race_engineer.domains.common.callouts.rival_watch import RivalWatch
from ai_race_engineer.domains.common.callouts.sector_comparison import SectorComparison
from ai_race_engineer.domains.common.callouts.spotter import Spotter
from ai_race_engineer.domains.common.callouts.traffic_ahead import TrafficAhead
from ai_race_engineer.domains.common.data.garage import GarageInfo
from ai_race_engineer.domains.common.data.observed import ObservedSource
from ai_race_engineer.domains.common.data.pit_estimate import PitEstimate
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot
from ai_race_engineer.domains.common.data.race_source import RaceSource
from ai_race_engineer.domains.common.data.session_clock import SessionClock
from ai_race_engineer.domains.common.recording.health_log import HealthLog
from ai_race_engineer.domains.common.recording.keep_running import keep_running
from ai_race_engineer.domains.common.recording.log_handler import RunLogHandler, log_thread_errors
from ai_race_engineer.domains.common.recording.recorded_questions import RecordedQuestions
from ai_race_engineer.domains.common.recording.run_folder import RunFolder
from ai_race_engineer.domains.common.recording.run_info import run_info
from ai_race_engineer.domains.common.recording.run_log import RunLog
from ai_race_engineer.domains.common.recording.state_log import StateLog
from ai_race_engineer.domains.common.store.best_lap_store import BestLapStore
from ai_race_engineer.domains.common.store.corner_plot import CornerPlotter
from ai_race_engineer.domains.common.store.history_store import HistoryStore
from ai_race_engineer.domains.common.store.segment_recorder import SegmentRecorder
from ai_race_engineer.domains.common.store.segment_store import SegmentStore
from ai_race_engineer.domains.common.store.session_keeper import SessionKeeper
from ai_race_engineer.domains.common.store.session_store import SessionStore
from ai_race_engineer.domains.common.store.timeline_recorder import TimelineRecorder
from ai_race_engineer.domains.common.store.timeline_store import TimelineStore
from ai_race_engineer.domains.common.store.track_map_store import TrackMapStore
from ai_race_engineer.domains.common.tools.history_tools import HistoryTools
from ai_race_engineer.domains.common.tools.race_context import RaceContext
from ai_race_engineer.domains.common.tools.store_tools import StoreTools
from ai_race_engineer.domains.common.tools.timeline_tools import TimelineTools
from ai_race_engineer.domains.common.tools.traffic_tools import TrafficTools
from ai_race_engineer.domains.common.voice.base import SpeechToText, TextToSpeech
from ai_race_engineer.domains.common.voice.button import Button
from ai_race_engineer.domains.common.voice.elevenlabs_stt import ElevenLabsSpeechToText
from ai_race_engineer.domains.common.voice.elevenlabs_tts import ElevenLabsTextToSpeech
from ai_race_engineer.domains.common.voice.joystick_button import JoystickButton
from ai_race_engineer.domains.common.voice.keyboard_button import KeyboardButton
from ai_race_engineer.domains.common.voice.logged_gate import LoggedSpeechGate
from ai_race_engineer.domains.common.voice.logged_stt import LoggedSpeechToText
from ai_race_engineer.domains.common.voice.logged_tts import LoggedTextToSpeech
from ai_race_engineer.domains.common.voice.microphone import Microphone
from ai_race_engineer.domains.common.voice.radio_effect import RadioSound
from ai_race_engineer.domains.common.voice.recorded_stt import RecordedSpeechToText
from ai_race_engineer.domains.common.voice.speech_loop import SpeechLoop
from ai_race_engineer.domains.common.voice.talk_switch import TalkMode, TalkSwitch
from ai_race_engineer.domains.common.voice.timed_print_tts import TimedPrintTextToSpeech
from ai_race_engineer.domains.common.voice.typed_fallback import or_typed
from ai_race_engineer.domains.common.voice.voice_questions import VoiceQuestions
from ai_race_engineer.domains.common.voice.vosk_stt import VoskSpeechToText
from ai_race_engineer.domains.common.voice.whisper_stt import WhisperSpeechToText

__all__ = ["FINISH", "LAP", "Domain", "Happening", "RunLog", "Tool"]

type Tool = Callable[..., object]

LAP: str = "lap"
FINISH: str = "finish"

STT: dict[str, Callable[[Mapping[str, object], int, Callable[[], str]], SpeechToText]] = {
    "whisper": lambda cfg, rate, names: WhisperSpeechToText(**cfg["whisper"], sample_rate_hz=rate, names=names),
    "elevenlabs": lambda cfg, rate, names: ElevenLabsSpeechToText(
        ElevenLabs(api_key=os.environ[cfg["tts"]["api_key_env"]]), **cfg["elevenlabs_stt"], sample_rate_hz=rate
    ),
    "vosk": lambda cfg, rate, names: VoskSpeechToText(Path(cfg["vosk"]["model_dir"]), rate),
}
BUTTONS: dict[str, Callable[[Mapping[str, object]], Button]] = {
    "joystick": lambda cfg: JoystickButton(cfg["device_name"], cfg["button"]),
    "keyboard": lambda cfg: KeyboardButton(cfg["key_vk"]),
}


@dataclass(frozen=True)
class Happening:
    """One thing that occurred in the session.
    time_s  float                 s session time
    lap     int                   the driver's lap in progress, 1-based
    kind    str                   timeline kinds ("flag", "pit", "incident", ...) plus "lap" and "finish"
    text    str                   short sentence; "" when only the facts matter
    facts   Mapping[str, object]  structured values, units in the key suffix; of a "lap" the facts of the lap just
                                  finished (lap_digest), of a "finish" {"laps": the facts of every finished race lap}
    """

    time_s: float
    lap: int
    kind: str
    text: str
    facts: Mapping[str, object]


class Domain(ABC):
    """One data source with everything the engineer may know about it: the shared pipeline of analysis, callouts,
    stores, tools and voice around a race source, with hooks for what only one game has.
    run_log           RunLog                                 debug log of the run
    _config           Mapping[str, object]                   the domain's YAML
    _run              RunFolder                              folder of everything this run writes; shared with a
                                                             recorder that is running
    _replay           Path | None                            the recording that is replayed; None = live
    _ask_recorded     bool                                   in a replay, ask the questions of the recorded run again
    _now_unix         Callable[[], float]                    s unix time (of the recording in a replay)
    _stopping         threading.Event                        set when the program shuts down
    _callouts         queue.Queue[Callout]                   everything waiting to be spoken
    _health           HealthLog                              timing health; its timer wraps a subclass's reads too
    _keeper           SessionKeeper
    _tracker          RaceTracker                            the race state
    _timeline         TimelineRecorder                       the session timeline
    _segment_store    SegmentStore                           the segments, sectors and laps of this run
    _lap_history      LapHistory                             the facts of every finished race lap
    _context          RaceContext                            where the driver is now
    _plan_tools       TimelineTools                          shared tools: race, plan, history files, timeline
    _segment_tools    StoreTools                             shared tools: laps, corners, car health
    _traffic_tools    TrafficTools
    _history_tools    HistoryTools
    _source           RaceSource
    _first_observers  list[Callable[[RaceSnapshot], None]]   shared observers before the subclass's own
    _last_observers   list[Callable[[RaceSnapshot], None]]   shared observers after the subclass's own
    _first_events     list[Event]                            shared callouts before the subclass's own
    _last_events      list[Event]                            shared callouts after the subclass's own
    _consumption      ConsumptionTracker
    _gate             LoggedSpeechGate
    _tts              TextToSpeech
    _straight         LearnedStraightDetector                tells when the driver is on a straight
    _waiting          queue.Queue[Happening]                 "lap" and "finish" waiting for the engineer
    _queued           int                                    laps of the history already queued
    _summed           bool                                   the finish of this race was queued
    _questions        Callable[[], str]                      blocks until the next question; set by listen
    _on_exit          ExitStack                              what is closed and stopped when the run ends
    _monitor          EventMonitor                           runs the callouts on the observed source (set on entry)
    """

    def __init__(
        self,
        config: Mapping[str, object],
        source: RaceSource,
        now_unix: Callable[[], float],
        time_scale: float,
        log: bool,
        replay: Path | None,
        questions: bool,
    ) -> None:
        """Build the shared pipeline around a source.
        In:  config       Mapping[str, object]  the domain's YAML
             source       RaceSource            live or recorded
             now_unix     Callable[[], float]   s unix time (of the recording in a replay)
             time_scale   float                 replay s per wall s; 1.0 live
             log          bool                  write the run log
             replay       Path | None           raw-dump folder that is replayed, stored in the run log; None = live
             questions    bool                  in a replay, ask the questions of the recorded run again; ignored live
        Out:              None
        """
        strat_cfg, race_cfg, speech_cfg = config["analysis"], config["race"], config["speech"]
        hist_cfg, log_cfg = config["history"], config["log"]
        pit_cfg, prior_cfg = race_cfg["pit"], race_cfg["priors"]
        track_table = track_infos(yaml.safe_load(Path(config["tracks"]).read_text()))
        clock = SessionClock()
        run = RunFolder(Path(config["runs"]), "engineer", None if replay else "recorder")
        self._run: RunFolder = run
        self._replay: Path | None = replay
        self._ask_recorded: bool = questions
        self._now_unix: Callable[[], float] = now_unix
        self.run_log: RunLog = RunLog(run.path / "engineer", log_cfg["enabled"] and log, clock.time_s)
        self.run_log.write("run", run_info(sys.argv, {"domain": config}, replay, time_scale))
        logging.getLogger().addHandler(RunLogHandler(self.run_log, log_cfg["level"]))
        log_thread_errors()
        self._config: Mapping[str, object] = config
        self._stopping: threading.Event = threading.Event()
        self._callouts: queue.Queue[Callout] = LoggedCalloutQueue(self.run_log)
        self._health: HealthLog = HealthLog(
            self.run_log,
            log_cfg["health_period_s"],
            {"log_queue": self.run_log.queue_size, "callouts": self._callouts.qsize},
        )
        pit_log = PitStopLog(pit_cfg["stop_speed_mps"])
        session_log = SessionLog(
            pit_log.samples,
            lambda: field.measured,
            lambda: straight.segments(),
            race_cfg["flags"]["sector_yellow_value"],
            now_unix,
        )
        self._keeper: SessionKeeper = SessionKeeper(
            session_log,
            SessionStore(Path(hist_cfg["sessions_root"])),
            HistoryStore(Path(hist_cfg["history_file"])),
            source.results_path,
            PitRates(**pit_cfg["rates"]),
            self.run_log,
            hist_cfg["refresh_s"] / time_scale,
            hist_cfg["class_top"],
            hist_cfg["track_field"],
            prior_cfg["history_sessions"],
            prior_cfg["max_qualifying_age_h"],
            now_unix,
            hist_cfg["period_s"] / time_scale,
            hist_cfg["results_margin_s"],
            lambda: tracker.green_seen,
        )
        field = PriorFieldTracker(
            race_cfg["pace_window_laps"],
            strat_cfg["consumption_window_laps"],
            prior_cfg["pace_factor"],
            self._keeper.priors,
            self._garage,
            race_cfg["strategy_start_laps"],
        )
        tracker = RaceTracker(
            field,
            pit_log,
            track_table,
            PitRates(**pit_cfg["rates"]),
            PitLane(**pit_cfg["default_lane"]),
            self._pit_estimate,
            **race_cfg["rivals"],
            traffic_window_s=pit_cfg["traffic_window_s"],
            period_s=race_cfg["projection_period_s"],
            fuel_margin_laps=race_cfg["fuel_margin_laps"],
            set_fuel_ratio=self._fuel_ratio,
        )
        self._tracker: RaceTracker = tracker
        live = LiveStraightDetector(**speech_cfg["straight"])
        straight = LearnedStraightDetector(live, track_table, **speech_cfg["straight"], **speech_cfg["learned"])
        self._straight: LearnedStraightDetector = straight
        rival_watch = RivalWatch(
            tracker,
            **race_cfg["rival_watch"],
            min_rival_s=race_cfg["rivals"]["min_rival_s"],
            start_laps=race_cfg["strategy_start_laps"],
        )
        briefing = GridBriefing(tracker, self._keeper.priors, lambda: field.sources, **race_cfg["grid_briefing"])
        flags = Flags(**race_cfg["flags"])
        state_log = StateLog(self.run_log, tracker, log_cfg["state_period_s"])
        tl_cfg = hist_cfg["timeline"]
        self._timeline: TimelineRecorder = TimelineRecorder(
            TimelineBuilder(
                race_cfg["flags"]["sector_yellow_value"], race_cfg["flags"]["sector_of_index"], **tl_cfg["builder"]
            ),
            TimelineStore(Path(hist_cfg["sessions_root"]), tl_cfg["resume_margin_s"]),
            tracker,
            pit_log.samples,
            lambda: field.pre_race,
            now_unix,
        )
        map_store = TrackMapStore(Path(strat_cfg["track_maps"]))
        seg_cfg = strat_cfg["segments"]
        best_laps = BestLapStore(Path(strat_cfg["best_laps"]))
        segments = SegmentTracker(
            map_store.load,
            map_store.save,
            best_laps.load,
            best_laps.save,
            seg_cfg["keep_laps"],
            CardLimits(**seg_cfg["card"]),
            **seg_cfg["map"],
        )
        self._context: RaceContext = RaceContext(tracker, segments)
        rival_lines = RivalLineTracker()
        live_segments = LiveSegmentTracker(lambda: segments.segments)
        lap_types = LapTypeTracker()
        segment_store = SegmentStore(run.path / "segments.sqlite")
        self._segment_store: SegmentStore = segment_store
        segment_recorder = SegmentRecorder(
            segments,
            rival_lines,
            live_segments,
            lap_types,
            segment_store,
            best_laps.load_class,
            CardLimits(**seg_cfg["card"]),
            FindingLimits(**seg_cfg["findings"]),
            seg_cfg["plot_margin_m"],
        )
        self._segment_tools: StoreTools = StoreTools(
            segment_store,
            segments,
            CornerPlotter(
                run.path / "plots",
                seg_cfg["plot_margin_m"],
                seg_cfg["card"]["brake_on"],
                seg_cfg["card"]["throttle_on"],
            ),
            seg_cfg["card"]["search_m"],
        )
        laps_cfg = race_cfg["lap_history"]
        self._lap_history: LapHistory = LapHistory(
            tracker, self._timeline.entries, laps_cfg["loss_s"], laps_cfg["trend_laps"], self.run_log
        )
        contacts = ContactTracker(self._timeline.add, lambda: segments.segments, **race_cfg["contacts"])
        self._source: RaceSource = source
        self._first_observers: list[Callable[[RaceSnapshot], None]] = [
            clock.update,
            segments.update,
            rival_lines.update,
            live_segments.update,
            lap_types.update,
            segment_recorder.update,
        ]
        self._last_observers: list[Callable[[RaceSnapshot], None]] = [
            session_log.update,
            tracker.update,
            self._timeline.update,
            contacts.update,
            self._lap_history.update,
            self._feed_happenings,
            straight.update,
            flags.update,
            briefing.update,
            state_log.update,
        ]
        self._consumption: ConsumptionTracker = ConsumptionTracker(window_laps=strat_cfg["consumption_window_laps"])
        self._first_events: list[Event] = [
            flags,
            Spotter(**config["callouts"]["spotter"]),
            FastestLap(),
            OverallBest(),
            PersonalBest(),
        ]
        self._last_events: list[Event] = [
            TrafficAhead(tracker, race_cfg["traffic_callout_laps"], race_cfg["traffic_pack_gap_m"]),
            SectorComparison(),
            LowFuel(self._consumption, config["callouts"]["low_fuel_laps"], config["callouts"]["low_energy_laps"]),
            rival_watch,
            briefing,
        ]
        self._gate: LoggedSpeechGate = LoggedSpeechGate(
            self.run_log, Priority[speech_cfg["gate"]["urgent_priority"]], speech_cfg["gate"]["max_age_s"]
        )
        tts_cfg = config["devices"]["tts"]
        self._tts: TextToSpeech = LoggedTextToSpeech(
            ElevenLabsTextToSpeech(
                ElevenLabs(api_key=os.environ[tts_cfg["api_key_env"]]),
                tts_cfg["voice_id"],
                tts_cfg["model"],
                tts_cfg["sample_rate_hz"],
                tts_cfg["device"],
                RadioSound(**tts_cfg["radio"]),
            )
            if tts_cfg["engine"] == "radio"
            else TimedPrintTextToSpeech(clock.time_s, self._gate.waited_s, time_scale),
            self.run_log,
        )
        self._plan_tools: TimelineTools = TimelineTools(
            tracker,
            rival_watch,
            race_cfg["traffic_horizon_laps"],
            field,
            session_log,
            self._keeper,
            briefing,
            self._timeline,
            tl_cfg["max_entries"],
            **race_cfg["saving"],
        )
        self._history_tools: HistoryTools = HistoryTools(self._lap_history)
        self._traffic_tools: TrafficTools = TrafficTools(
            tracker, race_cfg["traffic_horizon_laps"], race_cfg["traffic_pack_gap_m"]
        )
        self._waiting: queue.Queue[Happening] = queue.Queue()
        self._queued: int = 0
        self._summed: bool = False
        self._on_exit: ExitStack = ExitStack()

    @abstractmethod
    def instructions(self) -> str:
        """What the data of this domain holds and how to use the tools, for the engineer's system prompt.
        In:  -
        Out: str
        """

    def tools(self) -> list[Tool]:
        """Everything the engineer can look up: the shared tools, then the subclass's own, in a fixed order.
        In:  -
        Out: list[Tool]  typed functions with a docstring
        """
        return [
            *self._plan_tools.tools(),
            *self._segment_tools.tools(),
            *self._traffic_tools.tools(),
            *self._own_tools(),
            *self._history_tools.tools(),
        ]

    def context(self) -> str:
        """Where the driver is now.
        In:  -
        Out: str  one line, e.g. "Now: race, on lap 3, 3.4 laps to go, P2 in class, at turn 7, 61 percent energy."
        """
        return self._context()

    def happenings(self) -> tuple[Happening, ...]:
        """The running session so far: the entries of the session timeline.
        In:  -
        Out: tuple[Happening, ...]  oldest first
        """
        return tuple(Happening(e.time_s, e.lap, e.kind, e.text, e.fields) for e in self._timeline.entries())

    def wait_happening(self) -> Happening:
        """The next happening the engineer acts on; blocks until there is one.
        In:  -
        Out: Happening  kind "lap" (a race lap the driver finished) or "finish" (the race is over for the driver)
        """
        return self._waiting.get()

    def listen(self) -> None:
        """Open the driver's input: the button, the microphone and the speech-to-text engine of the config, or typed
        questions when a device or model is missing; in a replay with questions, the questions of the recorded run.
        Call once, in the main thread, before the first ask; an input that cannot be set up at all raises here, at
        start-up.
        In:  -
        Out: None
        """
        self._questions: Callable[[], str] = self._recorded_questions() or self._question_source()

    def ask(self) -> str:
        """The driver's next question; blocks until there is one (call listen first). The race state at the question
        is written to the run log (stream "state").
        In:  -
        Out: str  the question; raises EOFError when the input is closed
        """
        question = self._questions()
        self.run_log.write("state", {"reason": "question", "question": question, "state": self._tracker.state})
        return question

    def say(self, kind: str, text: str, urgent: bool) -> None:
        """Queue a text of the engineer to be spoken to the driver; never blocks.
        In:  kind    str   what it is, e.g. "answer", "race_summary"
             text    str   final speech text
             urgent  bool  True = at once, False = on a straight
        Out:         None
        """
        self._callouts.put(Callout(kind, text, Priority.SPOTTER if urgent else Priority.STRATEGY, {}))

    def record(self, kind: str, text: str, facts: Mapping[str, str | float | int | bool | None]) -> None:
        """Keep an entry of the engineer (note, question, story) in the session timeline; never blocks.
        In:  kind   str  e.g. "question", "story"
             text   str  short sentence
             facts  Mapping[str, str | float | int | bool | None]  structured values
        Out:        None
        """
        self._timeline.add(kind, text, facts)

    def spawn(self, loop: Callable[[], None]) -> None:
        """Run a loop in a supervised daemon thread: it is started again after a failure until the run ends.
        In:  loop  Callable[[], None]  runs until the process ends
        Out:       None
        """
        wait_s = self._config["log"]["restart_wait_s"]
        threading.Thread(target=keep_running(loop, wait_s, self._stopping), daemon=True).start()

    def run(self) -> None:
        """Start the domain's threads and speak until the process ends (blocking; call inside the with block).
        In:  -
        Out: None
        """
        for loop in (*self._own_loops(), self._keeper.run, self._health.run, self._monitor.run):
            self.spawn(loop)
        poll_hz = self._config["speech"]["poll_hz"]
        SpeechLoop(self._callouts, self._gate, self._tts, self._straight.is_straight, poll_hz).run()

    def __enter__(self) -> "Domain":
        """Put the subclass's observers and callouts at their place among the shared ones and open the source.
        In:  -
        Out: Domain
        """
        observers = [*self._first_observers, *self._own_observers(), *self._last_observers]
        timed = (self._health.timed(f"observer.{o.__self__.__class__.__name__}", o) for o in observers)
        feed = ObservedSource(self._source, [self._health.gap("loop"), *timed])
        self._monitor: EventMonitor = EventMonitor(
            feed,
            [*self._first_events, *self._own_callouts(), *self._last_events],
            self._config["callouts"]["enabled"],
            self._consumption,
            self._callouts,
            self._config["source"]["poll_hz"],
        )
        with ExitStack() as opened:
            opened.enter_context(self._run)
            opened.enter_context(self.run_log)
            opened.enter_context(self._segment_store)
            opened.enter_context(feed)
            opened.callback(self._keeper.close)
            opened.callback(self._timeline.close)
            opened.callback(self._stopping.set)
            self._on_exit = opened.pop_all()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Stop the threads, write what is queued, close the source, the stores and the run log and release the
        run folder.
        In:  exc_type   type[BaseException] | None
             exc        BaseException | None
             traceback  TracebackType | None
        Out:            None
        """
        self._on_exit.__exit__(exc_type, exc, traceback)

    def _own_observers(self) -> list[Callable[[RaceSnapshot], None]]:
        """Snapshot observers only this domain has, called after the segment recorder.
        In:  -
        Out: list[Callable[[RaceSnapshot], None]]  empty = this domain has none
        """
        return []

    def _own_callouts(self) -> list[Event]:
        """Callouts only this domain has, checked after the personal best.
        In:  -
        Out: list[Event]  empty = this domain has none
        """
        return []

    def _own_tools(self) -> list[Tool]:
        """Tools only this domain has, listed between the traffic tools and the history tools.
        In:  -
        Out: list[Tool]  empty = this domain has none
        """
        return []

    def _own_loops(self) -> list[Callable[[], None]]:
        """Loops only this domain has, each run in a supervised thread.
        In:  -
        Out: list[Callable[[], None]]  empty = this domain has none
        """
        return []

    def _pit_estimate(self) -> PitEstimate | None:
        """The game's latest pit-stop estimate for the player (non-blocking).
        In:  -
        Out: PitEstimate | None  None = this domain has none
        """
        return None

    def _garage(self) -> GarageInfo | None:
        """The game's latest garage figures for the player (non-blocking).
        In:  -
        Out: GarageInfo | None  None = this domain has none
        """
        return None

    def _fuel_ratio(self) -> float | None:
        """Fuel ratio set in the garage.
        In:  -
        Out: float | None  litres per 1 % energy; None until known
        """
        garage = self._garage()
        return None if garage is None else garage.fuel_ratio

    def _recorded_wall_s(self) -> tuple[float, float]:
        """The time the replayed recording covers; recorded questions outside it are not asked again.
        In:  -
        Out: tuple[float, float]  s unix time of its first and last row; all time = this domain does not know
        """
        return -math.inf, math.inf

    def _feed_happenings(self, snapshot: RaceSnapshot) -> None:
        """Feed one snapshot (source observer, runs after the lap history): queue the facts of a lap that was just
        added to the history, and the finish once the race is over for the player. Only queues.
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        laps = self._lap_history.laps
        time_s, lap = snapshot.session.elapsed_s, snapshot.player.vehicle.completed_laps + 1
        if len(laps) < self._queued:
            self._queued, self._summed = 0, False
        for digest in laps[self._queued :]:
            self._waiting.put(Happening(time_s, lap, LAP, "", digest))
        self._queued = len(laps)
        if laps and self._tracker.finished and not self._summed:
            self._summed = True
            self._waiting.put(Happening(time_s, lap, FINISH, "", {"laps": laps}))

    def _recorded_questions(self) -> Callable[[], str] | None:
        """The questions of the recorded run a replay asks again, read from the engineer's log of the run the
        recording lies in (<run>/game/<session> -> <run>/engineer/state.jsonl).
        In:  -
        Out: Callable[[], str] | None  blocks until the next question is due; None when live, when they are not
                                       asked for, or when the recording has no such log (said in one line)
        """
        if self._replay is None or not self._ask_recorded:
            return None
        state_file = self._replay.absolute().parent.parent / "engineer" / "state.jsonl"
        if not state_file.is_file():
            print(f"no recorded questions found ({state_file}), ask your own")
            return None
        poll_s = 1 / self._config["devices"]["poll_hz"]
        return RecordedQuestions(state_file, *self._recorded_wall_s(), self._now_unix, poll_s)

    def _question_source(self) -> Callable[[], str]:
        """Open the button, the microphone and the speech-to-text engine of the config.
        In:  -
        Out: Callable[[], str]  blocks until the next question; typed questions when a device or model is missing
        """
        voice_cfg = self._config["devices"]
        rate = voice_cfg["sample_rate_hz"]
        return or_typed(
            lambda: VoiceQuestions(
                BUTTONS[voice_cfg["button"]["kind"]](voice_cfg["button"]),
                Microphone(voice_cfg["microphone"]["device"], rate),
                LoggedSpeechToText(
                    RecordedSpeechToText(
                        STT[voice_cfg["stt"]["engine"]](voice_cfg, rate, self._context.driver_names),
                        self._run.path / "voice",
                        rate,
                    ),
                    self.run_log,
                    rate,
                ),
                TalkSwitch(TalkMode[voice_cfg["talk"]["mode"]], voice_cfg["talk"]["max_s"]),
                rate,
                **voice_cfg["filter"],
                poll_hz=voice_cfg["poll_hz"],
            )
        )
