import logging
import queue
import threading
from pathlib import Path

import typer
import yaml

from ai_race_engineer.events.base import Callout
from ai_race_engineer.events.fastest_lap import FastestLap
from ai_race_engineer.events.low_fuel import LowFuel
from ai_race_engineer.events.monitor import EventMonitor
from ai_race_engineer.events.sector_comparison import SectorComparison
from ai_race_engineer.events.spotter import Spotter
from ai_race_engineer.strategy.consumption import ConsumptionTracker
from ai_race_engineer.telemetry.base import TelemetrySource
from ai_race_engineer.telemetry.lmu import LMUSource
from ai_race_engineer.telemetry.rf2 import RF2Source

SOURCES: dict[str, type[TelemetrySource]] = {"lmu": LMUSource, "rf2": RF2Source}


def main(
    game: Path = Path("configs/game.yaml"),
    strategy: Path = Path("configs/strategy.yaml"),
    events: Path = Path("configs/events.yaml"),
    spotter: Path = Path("configs/spotter.yaml"),
) -> None:
    """Print live race callouts (fastest lap, sectors, low fuel/energy, spotter).
    In:  game      Path  game config YAML (game, poll_hz)
         strategy  Path  strategy config YAML (consumption_window_laps)
         events    Path  events config YAML (thresholds, enabled per session type)
         spotter   Path  spotter config YAML (Spotter constructor args)
    Out:           None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    game_cfg, strat_cfg, ev_cfg, spot_cfg = (yaml.safe_load(p.read_text()) for p in (game, strategy, events, spotter))
    source: TelemetrySource = SOURCES[game_cfg["game"]]()
    consumption = ConsumptionTracker(window_laps=strat_cfg["consumption_window_laps"])
    callouts: queue.Queue[Callout] = queue.Queue()
    monitor = EventMonitor(
        source,
        [
            FastestLap(),
            SectorComparison(),
            LowFuel(consumption, ev_cfg["low_fuel_laps"], ev_cfg["low_energy_laps"]),
            Spotter(**spot_cfg),
        ],
        ev_cfg["enabled"],
        consumption,
        callouts,
        game_cfg["poll_hz"],
    )
    with source:
        threading.Thread(target=monitor.run, daemon=True).start()
        while True:
            callout = callouts.get()
            print(f"[{callout.kind}] {callout.text}")


if __name__ == "__main__":
    typer.run(main)
