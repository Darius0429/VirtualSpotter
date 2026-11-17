import importlib.metadata
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

PACKAGES: tuple[str, ...] = (
    "anthropic",
    "matplotlib",
    "scipy",
    "numpy",
    "pandas",
    "pyarrow",
    "PyYAML",
    "typer",
)
GIT_TIMEOUT_S: float = 5.0


def run_info(
    argv: Sequence[str], configs: Mapping[str, object], replay: Path | None, speed: float
) -> dict[str, object]:
    """What is needed to reproduce a run: command line, configuration, code version and data source. API keys are
    never read.
    In:  argv       Sequence[str]         command line
         configs    Mapping[str, object]  config file path -> its loaded content
         replay     Path | None           raw-dump folder that is replayed; None = live game
         speed      float                 replay s per wall s
    Out:            dict[str, object]     argv, configs, git_commit, git_dirty, python, packages, source, replay, speed
    """
    status = _git("status", "--porcelain")
    return {
        "argv": list(argv),
        "configs": dict(configs),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in PACKAGES},
        "source": "live" if replay is None else "replay",
        "replay": replay,
        "speed": speed,
    }


def _git(*arguments: str) -> str | None:
    """Output of one git command in the working directory.
    In:  arguments  str         git arguments
    Out:            str | None  stripped output; None when git is missing, fails or takes too long
    """
    try:
        done = subprocess.run(["git", *arguments], capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def _version(package: str) -> str | None:
    """Installed version of a package.
    In:  package  str         distribution name
    Out:          str | None  None when it is not installed
    """
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
