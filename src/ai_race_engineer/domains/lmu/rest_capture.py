import json
import threading
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path

IDLE_S: float = 0.2


class RestCapture:
    """Polls pages of the game's local REST API in a background thread and appends each changed answer to
    <folder>/rest/<name>.jsonl as one line: poll_s, wall_s, session_et, data. A page marked pits_only (the garage)
    is read once per session folder and then only while the player is in the pits (reading it on track makes the
    game stutter).
    folder      Path | None  session folder written into; None = nothing is polled (set by the recorder)
    session_et  float        s session time now (set by the recorder)
    in_pits     bool         whether the player is in the pits (set by the recorder)
    _url        str          base address of the API
    _timeout_s  float        s a read may take
    _pages      Mapping[str, Mapping[str, object]]  name -> path (str, added to the address), period_s (float, s
                                                    between two reads), pits_only (bool)
    _t0         float        s monotonic clock at which the recorder started
    """

    def __init__(self, url: str, timeout_s: float, pages: Mapping[str, Mapping[str, object]]) -> None:
        """Store what is polled.
        In:  url        str    base address, e.g. http://localhost:6397
             timeout_s  float  s a read may take
             pages      Mapping[str, Mapping[str, object]]  name -> path (str), period_s (float, s), pits_only (bool)
        Out:            None
        """
        self.folder: Path | None = None
        self.session_et: float = 0.0
        self.in_pits: bool = True
        self._url: str = url
        self._timeout_s: float = timeout_s
        self._pages: Mapping[str, Mapping[str, object]] = pages
        self._t0: float = 0.0

    def start(self, t0: float) -> None:
        """Start the polling thread (daemon: it ends with the program).
        In:  t0  float  s monotonic clock (time.perf_counter) at which the recorder started, the zero of poll_s
        Out:     None
        """
        self._t0 = t0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        """Polling thread: read every page that is due and store the answers that changed; a new folder starts
        over, so its files begin with the current answer of every page.
        In:  -
        Out: None
        """
        last: dict[str, bytes] = {}
        due = dict.fromkeys(self._pages, 0.0)
        current: Path | None = None
        while True:
            folder = self.folder
            if folder != current:
                current, last = folder, {}
                due = dict.fromkeys(self._pages, 0.0)
            now = time.perf_counter()
            for name, page in self._pages.items():
                if folder is None or now < due[name]:
                    continue
                if page["pits_only"] and not self.in_pits and name in last:
                    continue
                due[name] = now + page["period_s"]
                answer = self._read(page["path"])
                if answer is None or answer[0] == last.get(name):
                    continue
                row = {"poll_s": now - self._t0, "wall_s": time.time(), "session_et": self.session_et}
                if _append(folder / "rest" / f"{name}.jsonl", {**row, "data": answer[1]}):
                    last[name] = answer[0]
            time.sleep(IDLE_S)

    def _read(self, path: str) -> tuple[bytes, object] | None:
        """Read one page.
        In:  path  str                          path added to the address
        Out:       tuple[bytes, object] | None  (the answer as sent, its JSON content); None when the game does not
                                                answer or the answer is not JSON
        """
        try:
            with urllib.request.urlopen(self._url + path, timeout=self._timeout_s) as response:
                body = response.read()
            return body, json.loads(body)
        except (OSError, ValueError):
            return None


def _append(file: Path, row: Mapping[str, object]) -> bool:
    """Append one row to a JSON-lines file; a disk error is reported on the console.
    In:  file  Path                  its folder is created
         row   Mapping[str, object]  JSON-ready values
    Out:       bool                  whether the row was written
    """
    try:
        file.parent.mkdir(exist_ok=True)
        with file.open("a", encoding="utf-8") as lines:
            lines.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as error:
        print(f"rest answer not stored: {error}")
        return False
    return True
