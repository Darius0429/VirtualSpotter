from collections.abc import Callable

from ai_race_engineer.engineer.base import Agent


class AskLoop:
    """Takes the driver's questions and has the agent's answers spoken at once.
    _questions  Callable[..., str]                 blocks until the next question (typed now; push-to-talk later)
    _agent      Agent
    _say        Callable[[str, str, bool], None]   speaks a text: (kind, text, urgent)
    """

    def __init__(self, questions: Callable[..., str], agent: Agent, say: Callable[[str, str, bool], None]) -> None:
        """Store the blocks to wire together.
        In:  questions  Callable[..., str]                blocks until the next question
             agent      Agent
             say        Callable[[str, str, bool], None]  speaks a text: (kind, text, urgent)
        Out:            None
        """
        self._questions: Callable[..., str] = questions
        self._agent: Agent = agent
        self._say: Callable[[str, str, bool], None] = say

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
            self._say("answer", self._agent.answer(question), True)
