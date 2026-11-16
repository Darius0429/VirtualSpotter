import time

from ai_race_engineer.agent.base import Agent
from ai_race_engineer.recorders.run_log import RunLog
from ai_race_engineer.strategy.race_state import RaceTracker


class LoggedAgent(Agent):
    """Wraps an agent and writes each question, the race state it was asked in and the answer to the run log
    (streams "conversations" and "state").
    _agent       Agent            the wrapped agent
    _log         RunLog
    _tracker     RaceTracker      source of the race state at the question
    _agent_name  str              agent kind from the config
    _model_name  str              model name from the config
    """

    def __init__(self, agent: Agent, log: RunLog, tracker: RaceTracker, agent_name: str, model_name: str) -> None:
        """Store the agent, the log and what identifies the agent in the records.
        In:  agent       Agent
             log         RunLog
             tracker     RaceTracker
             agent_name  str          agent kind, e.g. "tool"
             model_name  str
        Out:             None
        """
        self._agent: Agent = agent
        self._log: RunLog = log
        self._tracker: RaceTracker = tracker
        self._agent_name: str = agent_name
        self._model_name: str = model_name

    def answer(self, question: str) -> str:
        """Answer one question with the wrapped agent, logging the question, the state and the answer with the
        total time.
        In:  question  str
        Out:           str  short speech text
        """
        start_s = time.monotonic()
        self._log.write(
            "conversations",
            {"event": "question", "agent": self._agent_name, "model": self._model_name, "question": question},
        )
        self._log.write("state", {"reason": "question", "question": question, "state": self._tracker.state})
        text = self._agent.answer(question)
        self._log.write("conversations", {"event": "answer", "text": text, "total_s": time.monotonic() - start_s})
        return text
