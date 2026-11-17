import queue
from collections.abc import Callable

from ai_race_engineer.engineer.base import Agent
from ai_race_engineer.domains.common.callouts.base import Callout, Priority


class AskLoop:
    """Takes the driver's questions and queues the agent's answers to be spoken at once.
    _questions  Callable[..., str]    blocks until the next question (typed now; push-to-talk later)
    _agent      Agent
    _callouts   queue.Queue[Callout]  output queue
    """

    def __init__(self, questions: Callable[..., str], agent: Agent, callouts: queue.Queue[Callout]) -> None:
        """Store the blocks to wire together.
        In:  questions  Callable[..., str]    blocks until the next question
             agent      Agent
             callouts   queue.Queue[Callout]  output queue
        Out:            None
        """
        self._questions: Callable[..., str] = questions
        self._agent: Agent = agent
        self._callouts: queue.Queue[Callout] = callouts

    def run(self) -> None:
        """Answer questions until the question source is closed (run in a daemon thread).
        In:  -
        Out: None
        """
        while True:
            try:
                question = self._questions()
            except EOFError:
                return
            self._callouts.put(Callout("answer", self._agent.answer(question), Priority.SPOTTER, {}))
