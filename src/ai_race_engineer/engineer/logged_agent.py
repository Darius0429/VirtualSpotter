import time

from ai_race_engineer.domains.base import RunLog
from ai_race_engineer.engineer.base import Agent


class LoggedAgent(Agent):
    """Wraps an agent and writes each question and the answer to the run log (stream "conversations").
    _agent       Agent            the wrapped agent
    _log         RunLog
    _agent_name  str              agent kind from the config
    _model_name  str              model name from the config
    """

    def __init__(self, agent: Agent, log: RunLog, agent_name: str, model_name: str) -> None:
        """Store the agent, the log and what identifies the agent in the records.
        In:  agent       Agent
             log         RunLog
             agent_name  str          agent kind, e.g. "tool"
             model_name  str
        Out:             None
        """
        self._agent: Agent = agent
        self._log: RunLog = log
        self._agent_name: str = agent_name
        self._model_name: str = model_name

    def answer(self, question: str) -> str:
        """Answer one question with the wrapped agent, logging the question and the answer with the total time.
        In:  question  str
        Out:           str  short speech text
        """
        start_s = time.monotonic()
        self._log.write(
            "conversations",
            {"event": "question", "agent": self._agent_name, "model": self._model_name, "question": question},
        )
        text = self._agent.answer(question)
        self._log.write("conversations", {"event": "answer", "text": text, "total_s": time.monotonic() - start_s})
        return text
