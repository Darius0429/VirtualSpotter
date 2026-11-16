from ai_race_engineer.agent.base import Agent
from ai_race_engineer.recorders.timeline_recorder import TimelineRecorder


class TimelineAgent(Agent):
    """Wraps an agent and records every question with its answer in the session timeline.
    _agent         Agent             the wrapped agent
    _timeline      TimelineRecorder
    _answer_chars  int               characters of an answer kept in the entry's text
    """

    def __init__(self, agent: Agent, timeline: TimelineRecorder, answer_chars: int) -> None:
        """Store the agent and the timeline.
        In:  agent         Agent
             timeline      TimelineRecorder
             answer_chars  int               characters of an answer kept in the entry's text
        Out:               None
        """
        self._agent: Agent = agent
        self._timeline: TimelineRecorder = timeline
        self._answer_chars: int = answer_chars

    def answer(self, question: str) -> str:
        """Answer one question with the wrapped agent and add a "question" entry to the timeline.
        In:  question  str
        Out:           str  short speech text
        """
        text = self._agent.answer(question)
        self._timeline.add(
            "question",
            f'Asked: "{question}" Answer: {text[: self._answer_chars]}',
            {"question": question, "answer": text},
        )
        return text
