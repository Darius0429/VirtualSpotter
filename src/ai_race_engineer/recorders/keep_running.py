import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def keep_running(target: Callable[[], None], wait_s: float, stopping: threading.Event) -> Callable[[], None]:
    """Thread target that starts a loop again after it failed, so one exception does not end the thread silently.
    Once the program is shutting down a failure is expected (the source is closed under the loop): it is not logged
    and the loop is not started again.
    In:  target    Callable[[], None]  the loop to run
         wait_s    float               s to wait before it is started again
         stopping  threading.Event     set when the program shuts down
    Out:           Callable[[], None]  runs target until it returns; every exception is logged with its traceback
    """

    def run() -> None:
        """Run the target, again after every failure.
        In:  -
        Out: None
        """
        while True:
            try:
                target()
                return
            except Exception:
                if stopping.is_set():
                    return
                logger.exception(
                    "%s failed, starting it again in %.1f s", getattr(target, "__qualname__", target), wait_s
                )
                time.sleep(wait_s)

    return run
