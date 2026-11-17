from collections import deque
from collections.abc import Sequence

from ai_race_engineer.domains.common.callouts.base import Callout, Event, Priority
from ai_race_engineer.domains.common.data.race_snapshot import (
    FLAG_BLUE,
    GAME_PHASE_FULL_COURSE_YELLOW,
    GAME_PHASE_GREEN,
    SECTOR_FLAG_CLEAR,
    RaceSnapshot,
    TrackFlags,
)
from ai_race_engineer.domains.common.data.snapshot import Snapshot


class Flags(Event):
    """Calls sector yellows and their clearing, full-course yellow and green, and blue flags for the player. Sector
    flags are only called while the game phase is green; in any other phase they are followed silently.
    _sector_yellow_value  int                  mSectorFlag value that means yellow
    _sector_of_index      Sequence[int]        sector number (1-3) of mSectorFlag[0], [1], [2]
    _clear_hold_s         float                s a sector must stay clear before the clear is called
    _flags                TrackFlags           latest flags, stored by update
    _player_sector        int                  sector number (1-3) the player is in, stored by update
    _seeded               bool                 False until the first check after a reset
    _yellow               list[bool]           per flag index: the driver has been told the sector is yellow
    _clear_since_s        list[float | None]   per flag index: s monotonic clock since a called yellow reads clear
    _phase                int                  mGamePhase at the last check
    _player_flag          int                  player's mFlag at the last check
    _pending              deque[Callout]       callouts not yet returned
    """

    name = "flags"

    def __init__(self, sector_yellow_value: int, sector_of_index: Sequence[int], clear_hold_s: float) -> None:
        """Set how the game's sector flags are read.
        In:  sector_yellow_value  int            mSectorFlag value that means yellow
             sector_of_index      Sequence[int]  sector number (1-3) of mSectorFlag[0], [1], [2]
             clear_hold_s         float          s a sector must stay clear before "Sector n clear"
        Out:                      None
        """
        self._sector_yellow_value: int = sector_yellow_value
        self._sector_of_index: Sequence[int] = sector_of_index
        self._clear_hold_s: float = clear_hold_s
        self._flags: TrackFlags = TrackFlags(0, 0, (SECTOR_FLAG_CLEAR, SECTOR_FLAG_CLEAR, SECTOR_FLAG_CLEAR), 0)
        self._player_sector: int = 0
        self._seeded: bool = False
        self._yellow: list[bool] = [False, False, False]
        self._clear_since_s: list[float | None] = [None, None, None]
        self._phase: int = 0
        self._player_flag: int = 0
        self._pending: deque[Callout] = deque()

    def update(self, snapshot: RaceSnapshot) -> None:
        """Store the latest flags and the player's sector (source observer, runs before check).
        In:  snapshot  RaceSnapshot
        Out:           None
        """
        self._flags = snapshot.flags
        self._player_sector = snapshot.player.vehicle.sector_index + 1

    def check(self, snapshot: Snapshot, now_s: float) -> Callout | None:
        """Queue a callout for every flag change since the last check and return one; the first check is silent.
        In:  snapshot  Snapshot  (unused: the flags arrive through update)
             now_s     float     s monotonic clock
        Out:           Callout | None  kind "yellow_sector_<n>", "full_course_yellow" or "blue_flag"
        """
        flags = self._flags
        if self._seeded and flags.game_phase == GAME_PHASE_GREEN:
            self._pending.extend(self._sector_callouts(flags.sector_flags, now_s))
        else:
            self._yellow = [value == self._sector_yellow_value for value in flags.sector_flags]
            self._clear_since_s = [None, None, None]
        if self._seeded:
            self._pending.extend(_phase_callouts(self._phase, flags.game_phase))
            if flags.player_flag == FLAG_BLUE != self._player_flag:
                self._pending.append(Callout("blue_flag", "Blue flag", Priority.SPOTTER, {}))
        self._seeded, self._phase, self._player_flag = True, flags.game_phase, flags.player_flag
        return self._pending.popleft() if self._pending else None

    def reset(self) -> None:
        """Forget what was announced; the next check seeds silently.
        In:  -
        Out: None
        """
        self._seeded = False
        self._pending.clear()

    def _sector_callouts(self, sector_flags: tuple[int, int, int], now_s: float) -> list[Callout]:
        """Callouts for sector yellows raised, and cleared for clear_hold_s. Only the clear value clears a yellow
        aloud; any other value ends it silently. A yellow raised again within the hold is not called again.
        In:  sector_flags  tuple[int, int, int]  latest mSectorFlag[0..2]
             now_s         float                 s monotonic clock
        Out:               list[Callout]
        """
        callouts: list[Callout] = []
        for index, (sector, value) in enumerate(zip(self._sector_of_index, sector_flags, strict=True)):
            since_s = self._clear_since_s[index]
            self._clear_since_s[index] = None
            if value == self._sector_yellow_value:
                if not self._yellow[index]:
                    self._yellow[index] = True
                    callouts.append(_yellow(sector, sector == self._player_sector))
            elif value != SECTOR_FLAG_CLEAR:
                self._yellow[index] = False
            elif self._yellow[index] and since_s is not None and now_s - since_s >= self._clear_hold_s:
                self._yellow[index] = False
                callouts.append(_clear(sector))
            elif self._yellow[index]:
                self._clear_since_s[index] = now_s if since_s is None else since_s
        return callouts


def _yellow(sector: int, is_player_sector: bool) -> Callout:
    """Callout for a yellow flag raised in a sector.
    In:  sector            int   sector number 1-3
         is_player_sector  bool  the player is in that sector
    Out:                   Callout
    """
    text = f"Yellow flag, sector {sector}" + (", your sector" if is_player_sector else "")
    return Callout(f"yellow_sector_{sector}", text, Priority.SPOTTER, {"sector": sector})


def _clear(sector: int) -> Callout:
    """Callout for a sector that is clear again; it reuses the kind of the yellow.
    In:  sector  int  sector number 1-3
    Out:         Callout
    """
    return Callout(f"yellow_sector_{sector}", f"Sector {sector} clear", Priority.SPOTTER, {"sector": sector})


def _phase_callouts(old_phase: int, new_phase: int) -> list[Callout]:
    """Callouts for a full-course yellow starting or ending.
    In:  old_phase  int  mGamePhase at the last check
         new_phase  int  latest mGamePhase
    Out:            list[Callout]
    """
    if new_phase == GAME_PHASE_FULL_COURSE_YELLOW != old_phase:
        return [Callout("full_course_yellow", "Full course yellow", Priority.SPOTTER, {})]
    if old_phase == GAME_PHASE_FULL_COURSE_YELLOW and new_phase == GAME_PHASE_GREEN:
        return [Callout("full_course_yellow", "Green flag", Priority.SPOTTER, {})]
    return []
