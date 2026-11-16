import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import typer
import yaml
from anthropic import Anthropic

from ai_race_engineer.agent.base import Agent, configure_retries
from ai_race_engineer.agent.claude_agent import ClaudeAgent
from ai_race_engineer.agent.energy_tools import EnergyTools
from ai_race_engineer.agent.history_tools import HistoryTools
from ai_race_engineer.agent.logged_agent import LoggedAgent
from ai_race_engineer.agent.race_notes import RaceNoteWriter
from ai_race_engineer.agent.race_story import RaceStory
from ai_race_engineer.agent.segment_tools import SegmentTools
from ai_race_engineer.agent.timeline_agent import TimelineAgent
from ai_race_engineer.agent.timeline_tools import TimelineTools
from ai_race_engineer.agent.traffic_tools import TrafficTools
from ai_race_engineer.events.base import Callout, Priority
from ai_race_engineer.events.fastest_lap import FastestLap
from ai_race_engineer.events.flags import Flags
from ai_race_engineer.events.grid_briefing import GridBriefing
from ai_race_engineer.events.logged_queue import LoggedCalloutQueue
from ai_race_engineer.events.low_fuel import LowFuel
from ai_race_engineer.events.monitor import EventMonitor
from ai_race_engineer.events.overall_best import OverallBest
from ai_race_engineer.events.personal_best import PersonalBest
from ai_race_engineer.events.rival_watch import RivalWatch
from ai_race_engineer.events.sector_comparison import SectorComparison
from ai_race_engineer.events.spotter import Spotter
from ai_race_engineer.events.timeline_builder import TimelineBuilder
from ai_race_engineer.events.traffic_ahead import TrafficAhead
from ai_race_engineer.recorders.corner_plot import CornerPlotter
from ai_race_engineer.recorders.health_log import HealthLog
from ai_race_engineer.recorders.history_store import HistoryStore
from ai_race_engineer.recorders.keep_running import keep_running
from ai_race_engineer.recorders.log_handler import RunLogHandler, log_thread_errors
from ai_race_engineer.recorders.run_info import run_info
from ai_race_engineer.recorders.run_log import RunLog
from ai_race_engineer.recorders.session_keeper import SessionKeeper
from ai_race_engineer.recorders.session_store import SessionStore
from ai_race_engineer.recorders.state_log import StateLog
from ai_race_engineer.recorders.timeline_recorder import TimelineRecorder
from ai_race_engineer.recorders.timeline_store import TimelineStore
from ai_race_engineer.recorders.track_map_store import TrackMapStore
from ai_race_engineer.strategy.consumption import ConsumptionTracker
from ai_race_engineer.strategy.corner_card import CardLimits
from ai_race_engineer.strategy.corner_compare import FindingLimits
from ai_race_engineer.strategy.lap_history import LapHistory
from ai_race_engineer.strategy.pit_log import PitStopLog
from ai_race_engineer.strategy.pit_time import PitLane, PitRates
from ai_race_engineer.strategy.prior_field import PriorFieldTracker
from ai_race_engineer.strategy.race_state import RaceTracker
from ai_race_engineer.strategy.rival_lines import RivalLineTracker
from ai_race_engineer.strategy.segment_tracker import SegmentTracker
from ai_race_engineer.strategy.session_log import SessionLog
from ai_race_engineer.strategy.straight_learned import LearnedStraightDetector
from ai_race_engineer.strategy.straight_live import LiveStraightDetector
from ai_race_engineer.strategy.track_info import track_infos
from ai_race_engineer.telemetry.garage import GarageSource
from ai_race_engineer.telemetry.garage_poller import GaragePoller
from ai_race_engineer.telemetry.lmu_garage import LMUGarageSource
from ai_race_engineer.telemetry.lmu_pit_estimate import LMUPitEstimateSource
from ai_race_engineer.telemetry.lmu_race import LMURaceSource
from ai_race_engineer.telemetry.observed import ObservedSource
from ai_race_engineer.telemetry.pit_estimate import PitEstimateSource
from ai_race_engineer.telemetry.pit_estimate_poller import PitEstimatePoller
from ai_race_engineer.telemetry.race_source import RaceSource
from ai_race_engineer.telemetry.replay import ReplaySource
from ai_race_engineer.telemetry.replay_garage import ReplayGarage
from ai_race_engineer.telemetry.replay_pit_estimate import ReplayPitEstimate
from ai_race_engineer.telemetry.replay_wall_clock import ReplayWallClock
from ai_race_engineer.telemetry.session_clock import SessionClock
from ai_race_engineer.telemetry.timed_pit_estimate import TimedPitEstimate
from ai_race_engineer.voice.ask_loop import AskLoop
from ai_race_engineer.voice.base import TextToSpeech
from ai_race_engineer.voice.button import Button
from ai_race_engineer.voice.joystick_button import JoystickButton
from ai_race_engineer.voice.keyboard_button import KeyboardButton
from ai_race_engineer.voice.logged_gate import LoggedSpeechGate
from ai_race_engineer.voice.logged_stt import LoggedSpeechToText
from ai_race_engineer.voice.logged_tts import LoggedTextToSpeech
from ai_race_engineer.voice.microphone import Microphone
from ai_race_engineer.voice.speech_loop import SpeechLoop
from ai_race_engineer.voice.talk_switch import TalkMode, TalkSwitch
from ai_race_engineer.voice.timed_print_tts import TimedPrintTextToSpeech
from ai_race_engineer.voice.typed_fallback import or_typed
from ai_race_engineer.voice.voice_questions import VoiceQuestions
from ai_race_engineer.voice.whisper_stt import WhisperSpeechToText

BUTTONS: dict[str, Callable[[dict[str, object]], Button]] = {
    "joystick": lambda cfg: JoystickButton(cfg["device_name"], cfg["button"]),
    "keyboard": lambda cfg: KeyboardButton(cfg["key_vk"]),
}


def main(
    game: Path = Path("configs/game.yaml"),
    strategy: Path = Path("configs/strategy.yaml"),
    events: Path = Path("configs/events.yaml"),
    spotter: Path = Path("configs/spotter.yaml"),
    race: Path = Path("configs/race.yaml"),
    speech: Path = Path("configs/speech.yaml"),
    agent: Path = Path("configs/agent.yaml"),
    log: Path = Path("configs/log.yaml"),
    history: Path = Path("configs/history.yaml"),
    tracks: Path = Path("configs/tracks.yaml"),
    voice: Path = Path("configs/voice.yaml"),
    session: Path | None = None,
    speed: float = 1.0,
    no_log: bool = False,
    recording: str = "",
) -> None:
    """Run the race engineer: spoken callouts (urgent at once, the rest on straights) and answers to spoken or typed
    questions, with a debug log of the run under data/logs.
    In:  game       Path         game config YAML (poll_hz)
         strategy   Path         strategy config YAML (consumption_window_laps, track maps, corner plots, segments)
         events     Path         events config YAML (thresholds, enabled per session type)
         spotter    Path         spotter config YAML (Spotter constructor args)
         race       Path         race config YAML (pace window, pit model, pit estimate, rivals, rival watch, flags,
                                 priors, garage, saving)
         speech     Path         speech config YAML (gate, straight detectors)
         agent      Path         agent config YAML (retries, Claude models)
         log        Path         run-log config YAML (root folder, level, periods)
         history    Path         history config YAML (session files, history file, refresh period, timeline)
         tracks     Path         track characteristics YAML
         voice      Path         voice-input config YAML (button, microphone, speech-to-text)
         session    Path | None  raw-dump folder to replay; None = live game
         speed      float        replay s per wall s
         no_log     bool         write no run log, whatever the config says
         recording  str          name of the raw recording made at the same time, stored in the run log
    Out:            None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    paths = (game, strategy, events, spotter, race, speech, agent, log, history, tracks, voice)
    configs = {str(p): yaml.safe_load(p.read_text()) for p in paths}
    (
        game_cfg,
        strat_cfg,
        ev_cfg,
        spot_cfg,
        race_cfg,
        speech_cfg,
        agent_cfg,
        log_cfg,
        hist_cfg,
        track_cfg,
        voice_cfg,
    ) = configs.values()
    pit_cfg, est_cfg, garage_cfg, prior_cfg = (race_cfg[key] for key in ("pit", "pit_estimate", "garage", "priors"))
    track_table = track_infos(track_cfg)
    clock = SessionClock()
    run_log = RunLog(Path(log_cfg["root"]), log_cfg["enabled"] and not no_log, clock.time_s)
    run_log.write("run", run_info(sys.argv, configs, session, speed, recording))
    logging.getLogger().addHandler(RunLogHandler(run_log, log_cfg["level"]))
    log_thread_errors()
    callouts: queue.Queue[Callout] = LoggedCalloutQueue(run_log)
    health = HealthLog(
        run_log, log_cfg["health_period_s"], {"log_queue": run_log.queue_size, "callouts": callouts.qsize}
    )
    if session:
        replay = ReplaySource(session, speed)
        source: RaceSource = replay
        estimates: PitEstimateSource = ReplayPitEstimate(session, replay.time_s)
        garage_source: GarageSource = ReplayGarage(session, replay.time_s)
        now_unix = ReplayWallClock(session, replay.time_s).unix_s
        time_scale = speed
    else:
        source = LMURaceSource()
        estimates = LMUPitEstimateSource(est_cfg["url"], est_cfg["timeout_s"])
        garage_source = LMUGarageSource(garage_cfg["url"], garage_cfg["timeout_s"])
        now_unix = time.time
        time_scale = 1.0
    poller = PitEstimatePoller(TimedPitEstimate(estimates, health.timed), est_cfg["period_s"] / time_scale)
    garage = GaragePoller(health.timed("garage_read", garage_source.read), garage_cfg["period_s"] / time_scale)
    pit_log = PitStopLog(pit_cfg["stop_speed_mps"])
    session_log = SessionLog(
        pit_log.samples,
        lambda: field.measured,
        lambda: straight.segments(),
        race_cfg["flags"]["sector_yellow_value"],
        now_unix,
    )
    keeper = SessionKeeper(
        session_log,
        SessionStore(Path(hist_cfg["sessions_root"])),
        HistoryStore(Path(hist_cfg["history_file"])),
        source.results_path,
        PitRates(**pit_cfg["rates"]),
        run_log,
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
        keeper.priors,
        garage.latest,
        race_cfg["strategy_start_laps"],
    )  # or: FieldTracker(race_cfg["pace_window_laps"], strat_cfg["consumption_window_laps"])
    tracker = RaceTracker(
        field,
        pit_log,
        track_table,
        PitRates(**pit_cfg["rates"]),
        PitLane(**pit_cfg["default_lane"]),
        poller.latest,
        **race_cfg["rivals"],
        traffic_window_s=pit_cfg["traffic_window_s"],
        period_s=race_cfg["projection_period_s"],
        fuel_margin_laps=race_cfg["fuel_margin_laps"],
        set_fuel_ratio=garage.fuel_ratio,
    )
    live = LiveStraightDetector(**speech_cfg["straight"])
    straight = LearnedStraightDetector(live, track_table, **speech_cfg["straight"], **speech_cfg["learned"])
    rival_watch = RivalWatch(
        tracker,
        **race_cfg["rival_watch"],
        min_rival_s=race_cfg["rivals"]["min_rival_s"],
        start_laps=race_cfg["strategy_start_laps"],
    )
    briefing = GridBriefing(tracker, keeper.priors, lambda: field.sources, **race_cfg["grid_briefing"])
    flags = Flags(**race_cfg["flags"])
    state_log = StateLog(run_log, tracker, log_cfg["state_period_s"])
    tl_cfg = hist_cfg["timeline"]
    timeline = TimelineRecorder(
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
    segments = SegmentTracker(
        map_store.load, map_store.save, seg_cfg["keep_laps"], CardLimits(**seg_cfg["card"]), **seg_cfg["map"]
    )
    rival_lines = RivalLineTracker()
    segment_tools = SegmentTools(
        segments,
        rival_lines,
        CornerPlotter(
            Path(strat_cfg["corner_plots"]),
            seg_cfg["plot_margin_m"],
            seg_cfg["card"]["brake_on"],
            seg_cfg["card"]["throttle_on"],
        ),
        FindingLimits(**seg_cfg["findings"]),
        seg_cfg["card"]["search_m"],
    )
    story_cfg = agent_cfg["story"]
    lap_history = LapHistory(tracker, timeline.entries, story_cfg["loss_s"], story_cfg["trend_laps"])
    story = (
        RaceStory(
            lap_history,
            tracker,
            timeline,
            RaceNoteWriter(Anthropic(), story_cfg["notes_model"], story_cfg["summary_model"], run_log),
            callouts,
        )
        if story_cfg["enabled"] and agent_cfg["api_key_env"] in os.environ
        else None
    )
    observers = [
        clock.update,
        segments.update,
        rival_lines.update,
        garage.update,
        session_log.update,
        tracker.update,
        timeline.update,
        lap_history.update,
        *([story.update] if story else []),
        straight.update,
        flags.update,
        briefing.update,
        state_log.update,
    ]
    feed = ObservedSource(
        source,
        [health.gap("loop"), *(health.timed(f"observer.{o.__self__.__class__.__name__}", o) for o in observers)],
    )
    consumption = ConsumptionTracker(window_laps=strat_cfg["consumption_window_laps"])
    monitor = EventMonitor(
        feed,
        [
            flags,
            Spotter(**spot_cfg),
            FastestLap(),
            OverallBest(),
            PersonalBest(),
            TrafficAhead(tracker, race_cfg["traffic_callout_laps"], race_cfg["traffic_pack_gap_m"]),
            SectorComparison(),
            LowFuel(consumption, ev_cfg["low_fuel_laps"], ev_cfg["low_energy_laps"]),
            rival_watch,
            briefing,
        ],
        ev_cfg["enabled"],
        consumption,
        callouts,
        game_cfg["poll_hz"],
    )
    gate = LoggedSpeechGate(run_log, Priority[speech_cfg["gate"]["urgent_priority"]], speech_cfg["gate"]["max_age_s"])
    tts: TextToSpeech = LoggedTextToSpeech(
        TimedPrintTextToSpeech(clock.time_s, gate.waited_s, speed), run_log
    )  # or: PrintTextToSpeech()
    tools = TimelineTools(
        tracker,
        rival_watch,
        race_cfg["traffic_horizon_laps"],
        field,
        session_log,
        keeper,
        briefing,
        timeline,
        tl_cfg["max_entries"],
        **race_cfg["saving"],
    )  # or: PlanTools(...) without timeline and tl_cfg["max_entries"]
    history_tools = HistoryTools(lap_history)
    energy_tools = EnergyTools(tracker, race_cfg["lift_coast_saving_percent_per_lap"])
    traffic_tools = TrafficTools(tracker, race_cfg["traffic_horizon_laps"], race_cfg["traffic_pack_gap_m"])
    configure_retries(agent_cfg["attempts"], agent_cfg["retry_wait_s"], agent_cfg["error_reply"])
    stopping = threading.Event()
    with run_log, feed, ExitStack() as on_exit:
        on_exit.callback(keeper.close)
        on_exit.callback(timeline.close)
        on_exit.callback(stopping.set)
        for loop in (poller.run, garage.run, keeper.run, health.run, monitor.run, *([story.run] if story else [])):
            threading.Thread(target=keep_running(loop, log_cfg["restart_wait_s"], stopping), daemon=True).start()
        if agent_cfg["api_key_env"] in os.environ:
            engineer: Agent = TimelineAgent(
                LoggedAgent(
                    ClaudeAgent(
                        Anthropic(),
                        [
                            *tools.tools(),
                            *segment_tools.tools(),
                            *traffic_tools.tools(),
                            *energy_tools.tools(),
                            *history_tools.tools(),
                        ],
                        **agent_cfg["claude"],
                        log=run_log,
                    ),
                    run_log,
                    tracker,
                    "claude",
                    agent_cfg["claude"]["model"],
                ),
                timeline,
                tl_cfg["answer_chars"],
            )  # or: LoggedAgent(...)
            rate = voice_cfg["sample_rate_hz"]
            questions = or_typed(
                lambda: VoiceQuestions(
                    BUTTONS[voice_cfg["button"]["kind"]](voice_cfg["button"]),
                    Microphone(voice_cfg["microphone"]["device"], rate),
                    LoggedSpeechToText(
                        WhisperSpeechToText(**voice_cfg["whisper"], sample_rate_hz=rate), run_log, rate
                    ),  # or: VoskSpeechToText(Path(voice_cfg["vosk"]["model_dir"]), rate)
                    TalkSwitch(TalkMode[voice_cfg["talk"]["mode"]], voice_cfg["talk"]["max_s"]),
                    rate,
                    **voice_cfg["filter"],
                    poll_hz=voice_cfg["poll_hz"],
                )
            )  # or: input
            threading.Thread(target=AskLoop(questions, engineer, callouts).run, daemon=True).start()
        else:
            print(f"{agent_cfg['api_key_env']} is not set: questions are disabled")
        SpeechLoop(callouts, gate, tts, straight.is_straight, speech_cfg["poll_hz"]).run()


if __name__ == "__main__":
    typer.run(main)
