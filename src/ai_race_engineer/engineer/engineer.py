import os
import threading
from collections.abc import Mapping

from anthropic import Anthropic

from ai_race_engineer.domains.base import Domain
from ai_race_engineer.engineer.ask_loop import AskLoop
from ai_race_engineer.engineer.base import Agent, configure_retries
from ai_race_engineer.engineer.claude_agent import ClaudeAgent
from ai_race_engineer.engineer.logged_agent import LoggedAgent
from ai_race_engineer.engineer.prompts import ROLE_PROMPT
from ai_race_engineer.engineer.race_notes import RaceNoteWriter
from ai_race_engineer.engineer.race_story import RaceStory
from ai_race_engineer.engineer.timeline_agent import TimelineAgent


class Engineer:
    """The race engineer on top of any domain: answers the driver's questions with the domain's tools and keeps
    the notes of the race. Every model call runs in the engineer's own threads.
    _config  Mapping[str, object]  configs/engineer.yaml
    """

    def __init__(self, config: Mapping[str, object]) -> None:
        """Store the settings.
        In:  config  Mapping[str, object]  configs/engineer.yaml (model, notes)
        Out:         None
        """
        self._config: Mapping[str, object] = config

    def start(self, domain: Domain) -> None:
        """Start answering questions and writing notes in background threads; returns at once. Without the API key
        in the environment nothing is started and questions are disabled.
        In:  domain  Domain  entered (inside its with block)
        Out:         None
        """
        model, notes = self._config["model"], self._config["notes"]
        domain.run_log.write("run", {"configs": {"engineer": self._config}})
        configure_retries(model["attempts"], model["retry_wait_s"], model["error_reply"])
        if model["api_key_env"] not in os.environ:
            print(f"{model['api_key_env']} is not set: questions are disabled")
            return
        claude = ClaudeAgent(
            Anthropic(),
            domain.tools(),
            **model["claude"],
            log=domain.run_log,
            context=domain.context,
            system=ROLE_PROMPT + domain.instructions(),
        )
        threading.Thread(target=claude.warm, daemon=True).start()
        agent: Agent = TimelineAgent(
            LoggedAgent(claude, domain.run_log, "claude", model["claude"]["model"]),
            domain.record,
            notes["answer_chars"],
        )
        domain.listen()
        threading.Thread(target=AskLoop(domain.ask, agent, domain.say).run, daemon=True).start()
        if notes["enabled"]:
            writer = RaceNoteWriter(Anthropic(), notes["notes_model"], notes["summary_model"], domain.run_log)
            domain.spawn(RaceStory(domain, writer).run)
