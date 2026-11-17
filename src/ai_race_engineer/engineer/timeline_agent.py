from collections.abc import Callable, Mapping

from ai_race_engineer.engineer.base import Agent


class TimelineAgent(Agent):
    """Wraps an agent and records every question with its answer in the session timeline.
    _agent         Agent             the wrapped agent
    _record        Callable[[str, str, Mapping[str, str | float | int | bool | None]], None]
                                     keeps an entry in the session timeline: (kind, text, facts)
    _answer_chars  int               characters of an answer kept in the entry's text
    """

    def __init__(
        self,
        agent: Agent,
        record: Callable[[str, str, Mapping[str, str | float | int | bool | None]], None],
        answer_chars: int,
    ) -> None:
        """Store the agent and the way into the timeline.
        In:  agent         Agent
             record        Callable[[str, str, Mapping[str, str | float | int | bool | None]], None]
                                             keeps an entry in the session timeline: (kind, text, facts)
             answer_chars  int               characters of an answer kept in the entry's text
        Out:               None
        """
        self._agent: Agent = agent
        self._record: Callable[[str, str, Mapping[str, str | float | int | bool | None]], None] = record
        self._answer_chars: int = answer_chars

    def answer(self, question: str) -> str:
        """Answer one question with the wrapped agent and add a "question" entry to the timeline.
        In:  question  str
        Out:           str  short speech text
        """
        text = self._agent.answer(question)
        self._record(
            "question",
            f'Asked: "{question}" Answer: {text[: self._answer_chars]}',
            {"question": question, "answer": text},
        )
        return text
