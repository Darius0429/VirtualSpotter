import logging
from collections.abc import Callable
from pathlib import Path

import typer
import yaml

from ai_race_engineer.domains.base import Domain
from ai_race_engineer.domains.lmu.domain import LMUDomain
from ai_race_engineer.engineer.engineer import Engineer

DOMAINS: dict[str, Callable[..., Domain]] = {"lmu": LMUDomain}


def main(
    domain: str = "lmu",
    recording: Path | None = None,
    speed: float = 1.0,
    no_log: bool = False,
    questions: bool = False,
    config: Path | None = None,
    engineer_config: Path = Path("configs/engineer.yaml"),
) -> None:
    """Run the race engineer on a domain: spoken callouts (urgent at once, the rest on straights) and answers to
    spoken or typed questions; what the run writes goes into its run folder (config key runs).
    In:  domain           str          key of DOMAINS, e.g. "lmu"
         recording        Path | None  raw-dump folder to replay; None = live game
         speed            float        replay s per wall s
         no_log           bool         write no run log, whatever the config says
         questions        bool         in a replay, ask the questions the driver asked in the recorded run again
         config           Path | None  the domain's YAML; None = configs/<domain>.yaml
         engineer_config  Path         the engineer's YAML (model, notes)
    Out:                  None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    domain_cfg = yaml.safe_load((config or Path(f"configs/{domain}.yaml")).read_text())
    engineer_cfg = yaml.safe_load(engineer_config.read_text())
    wrapper: Domain = DOMAINS[domain](domain_cfg, recording, speed, not no_log, questions)
    engineer = Engineer(engineer_cfg)
    with wrapper:
        engineer.start(wrapper)
        wrapper.run()


if __name__ == "__main__":
    typer.run(main)
