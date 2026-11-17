import logging
import queue
import threading
from pathlib import Path

import typer
import yaml

from ai_race_engineer.domains.common.callouts.base import Callout
from ai_race_engineer.domains.common.callouts.fastest_lap import FastestLap
from ai_race_engineer.domains.common.callouts.low_fuel import LowFuel
from ai_race_engineer.domains.common.callouts.monitor import EventMonitor
from ai_race_engineer.domains.common.callouts.sector_comparison import SectorComparison
from ai_race_engineer.domains.common.callouts.spotter import Spotter
from ai_race_engineer.domains.common.analysis.strategy.consumption import ConsumptionTracker
from ai_race_engineer.domains.common.data.base import TelemetrySource
from ai_race_engineer.domains.lmu.lmu import LMUSource
from ai_race_engineer.domains.rf2.rf2 import RF2Source

SOURCES: dict[str, type[TelemetrySource]] = {"lmu": LMUSource, "rf2": RF2Source}


def main(domain: str = "lmu", config: Path | None = None) -> None:
    """Print live race callouts (fastest lap, sectors, low fuel/energy, spotter).
    In:  domain  str          key of SOURCES, e.g. "lmu"
         config  Path | None  the domain's YAML (source, analysis, callouts); None = configs/<domain>.yaml
    Out:         None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load((config or Path(f"configs/{domain}.yaml")).read_text())
    ev_cfg = cfg["callouts"]
    source: TelemetrySource = SOURCES[domain]()
    consumption = ConsumptionTracker(window_laps=cfg["analysis"]["consumption_window_laps"])
    callouts: queue.Queue[Callout] = queue.Queue()
    monitor = EventMonitor(
        source,
        [
            FastestLap(),
            SectorComparison(),
            LowFuel(consumption, ev_cfg["low_fuel_laps"], ev_cfg["low_energy_laps"]),
            Spotter(**ev_cfg["spotter"]),
        ],
        ev_cfg["enabled"],
        consumption,
        callouts,
        cfg["source"]["poll_hz"],
    )
    with source:
        threading.Thread(target=monitor.run, daemon=True).start()
        while True:
            callout = callouts.get()
            print(f"[{callout.kind}] {callout.text}")


if __name__ == "__main__":
    typer.run(main)
