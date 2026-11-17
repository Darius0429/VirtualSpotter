import logging
import threading

from ai_race_engineer.domains.common.recording.run_log import RunLog

logger = logging.getLogger(__name__)


class RunLogHandler(logging.Handler):
    """Copies stdlib log records into the run log (stream "log").
    _log  RunLog
    """

    def __init__(self, log: RunLog, level: str) -> None:
        """Set the run log and the lowest level that is copied.
        In:  log    RunLog
             level  str     stdlib level name, e.g. "INFO"
        Out:        None
        """
        super().__init__(level)
        self._log: RunLog = log

    def emit(self, record: logging.LogRecord) -> None:
        """Queue one log record with its logger, thread and traceback.
        In:  record  logging.LogRecord
        Out:         None
        """
        self._log.write(
            "log",
            {
                "level": record.levelname,
                "logger": record.name,
                "thread": record.threadName,
                "message": record.getMessage(),
                "traceback": logging.Formatter().formatException(record.exc_info) if record.exc_info else None,
            },
        )


def log_thread_errors() -> None:
    """Send the uncaught exception of any thread to the stdlib log with its traceback, instead of only to stderr.
    In:  -
    Out: None
    """
    threading.excepthook = _log_thread_error


def _log_thread_error(args: threading.ExceptHookArgs) -> None:
    """Log one uncaught thread exception.
    In:  args  threading.ExceptHookArgs  exception type, value, traceback and the thread
    Out:       None
    """
    name = "?" if args.thread is None else args.thread.name
    logger.error("uncaught exception in thread %s", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
