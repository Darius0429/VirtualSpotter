import time
from collections.abc import Callable

from ai_race_engineer.domains.common.data.garage import GarageInfo, merge_garage
from ai_race_engineer.domains.common.data.snapshot import SessionType, Snapshot


class GaragePoller:
    """Reads the garage figures in its own thread: once at start and once after a change of session or car, else
    only while the player is in the pits or the garage (polling this endpoint on track makes the game lag). The
    figures kept are dropped when the session or the player's car changes; a read in flight at that moment is
    discarded.
    _read      Callable[[], GarageInfo | None]  one read of the garage source
    _period_s  float                            s between reads in the pits
    _in_pits   bool                             the player is in the pits or garage (set by the telemetry thread)
    _latest    GarageInfo | None                the last good value of each figure; replaced, never mutated
    _started   bool                             the one read at start has been made (once per process)
    _due       bool                             one read is owed after a change of session or car
    _key       tuple[SessionType, str, str] | None  session type, track and the player's vehicle name at the previous
                                                snapshot; None before the first
    _elapsed_s  float                           s session time of the previous snapshot
    _changes   int                              changes of session or car seen so far
    """

    def __init__(self, read: Callable[[], GarageInfo | None], period_s: float) -> None:
        """Set the read call and the polling period.
        In:  read      Callable[[], GarageInfo | None]  one read of the garage source (blocking)
             period_s  float                            s
        Out:           None
        """
        self._read: Callable[[], GarageInfo | None] = read
        self._period_s: float = period_s
        self._in_pits: bool = False
        self._latest: GarageInfo | None = None
        self._started: bool = False
        self._due: bool = False
        self._key: tuple[SessionType, str, str] | None = None
        self._elapsed_s: float = 0.0
        self._changes: int = 0

    def update(self, snapshot: Snapshot) -> None:
        """Store whether the player is in the pits or garage (source observer). On a new session (another session
        type or track, or the session clock going backwards) or another car of the player the figures kept are
        dropped and one read is asked for.
        In:  snapshot  Snapshot
        Out:           None
        """
        session, vehicle = snapshot.session, snapshot.player.vehicle
        key = (session.session_type, session.track_name, vehicle.vehicle_name)
        if self._key is not None and (key != self._key or session.elapsed_s < self._elapsed_s):
            self._latest, self._due, self._changes = None, True, self._changes + 1
        self._key, self._elapsed_s = key, session.elapsed_s
        self._in_pits = vehicle.in_pits or vehicle.in_garage

    def run(self) -> None:
        """Read once, then every period_s while the player is in the pits or a read is owed after a change of
        session or car, forever (run in a daemon thread). The read at start is made once per process: a run that is
        started again after a failure does not repeat it, so a failure that persists never turns into polling on
        track.
        In:  -
        Out: None
        """
        if not self._started:
            self._started = True
            self._poll()
        while True:
            time.sleep(self._period_s)
            if self._in_pits or self._due:
                self._due = False
                self._poll()

    def latest(self) -> GarageInfo | None:
        """The last good value of each garage figure.
        In:  -
        Out: GarageInfo | None  None until a read succeeds
        """
        return self._latest

    def fuel_ratio(self) -> float | None:
        """Fuel ratio set in the garage.
        In:  -
        Out: float | None  litres per 1 % energy; None until known
        """
        return None if self._latest is None else self._latest.fuel_ratio

    def _poll(self) -> None:
        """One read, merged into the figures kept; discarded when the session or the car changed while it was in
        flight (the read owed for the change then follows).
        In:  -
        Out: None
        """
        changes = self._changes
        new = self._read()
        if changes == self._changes:
            self._latest = merge_garage(self._latest, new)
