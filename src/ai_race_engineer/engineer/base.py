import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Common interface for all race-engineer agents."""

    @abstractmethod
    def answer(self, question: str) -> str:
        """Answer one question of the driver; blocking, never called from the telemetry thread.
        In:  question  str
        Out:           str  short speech text
        """


@dataclass(frozen=True)
class Retries:
    """How a failed model round trip is tried again.
    attempts      int    tries per question
    retry_wait_s  float  s to wait between two tries
    error_reply   str    spoken when every try failed
    """

    attempts: int
    retry_wait_s: float
    error_reply: str


_retries: Retries = Retries(attempts=1, retry_wait_s=0.0, error_reply="")


def configure_retries(attempts: int, retry_wait_s: float, error_reply: str) -> None:
    """Set how every agent tries a failed model round trip again (from configs/agent.yaml, once at start; until then
    a question is tried once and a failure gives an empty reply).
    In:  attempts      int    tries per question
         retry_wait_s  float  s to wait between two tries
         error_reply   str    spoken when every try failed
    Out:               None
    """
    global _retries
    _retries = Retries(attempts, retry_wait_s, error_reply)


def answer_with_retries(ask: Callable[[], str]) -> str:
    """Run one model round trip, trying again after a failed call (rate limit, network); there is no wait after the
    last failed try.
    In:  ask  Callable[[], str]  the model round trip; raises when the model or network fails
    Out:      str                its answer; the configured error reply when every attempt fails
    """
    retries = _retries
    for attempt in range(retries.attempts):
        try:
            return ask()
        except Exception as error:
            logger.warning("agent call failed (attempt %d of %d): %s", attempt + 1, retries.attempts, error)
            if attempt + 1 < retries.attempts:
                time.sleep(retries.retry_wait_s)
    return retries.error_reply
