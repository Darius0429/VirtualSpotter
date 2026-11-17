import logging
import time
from pathlib import Path

import numpy as np
import typer
import yaml

from ai_race_engineer.domains.common.analysis.strategy.fuel import LapConsumption
from ai_race_engineer.domains.common.analysis.field.proximity import to_player_frame
from ai_race_engineer.domains.common.analysis.field.standings import standings
from ai_race_engineer.domains.common.data.base import TelemetrySource
from ai_race_engineer.domains.lmu.lmu import LMUSource
from ai_race_engineer.domains.rf2.rf2 import RF2Source

SOURCES: dict[str, type[TelemetrySource]] = {"lmu": LMUSource, "rf2": RF2Source}


def main(game: Path = Path("configs/game.yaml"), strategy: Path = Path("configs/strategy.yaml")) -> None:
    """Print live telemetry, standings, consumption and nearby-car offsets.
    In:  game      Path  game config YAML (game, poll_hz)
         strategy  Path  strategy config YAML (consumption_window_laps)
    Out:           None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    game_cfg = yaml.safe_load(game.read_text())
    strat_cfg = yaml.safe_load(strategy.read_text())
    source: TelemetrySource = SOURCES[game_cfg["game"]]()
    fuel = LapConsumption(window_laps=strat_cfg["consumption_window_laps"])
    energy = LapConsumption(window_laps=strat_cfg["consumption_window_laps"])
    with source:
        while True:
            snap = source.read()
            if snap is not None:
                me = snap.player.vehicle
                offsets = to_player_frame(
                    me.position_xz_m, me.heading_rad, np.array([v.position_xz_m for v in snap.vehicles])
                )
                print(
                    snap.session.session_type.name,
                    round(snap.session.remaining_s),
                    me.place,
                    standings(snap.vehicles).at[me.slot_id, "position_in_class"],
                    snap.player.fuel_l,
                    fuel.update(me.completed_laps, snap.player.fuel_l),
                    snap.player.virtual_energy,
                    None
                    if snap.player.virtual_energy is None
                    else energy.update(me.completed_laps, snap.player.virtual_energy),
                    offsets.round(1),
                )
            time.sleep(1 / game_cfg["poll_hz"])


if __name__ == "__main__":
    typer.run(main)
