from collections.abc import Callable

import pytest

from ai_race_engineer.domains.common.callouts.base import Callout, Priority
from ai_race_engineer.domains.common.callouts.flags import Flags
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot, TrackFlags
from ai_race_engineer.domains.common.data.snapshot import Snapshot, VehicleState

CLEAR: tuple[int, int, int] = (11, 11, 11)


class _Track:
    """A Flags event (yellow = 1, clear hold 3 s) fed as the source observer and the monitor do: update, then check.
    event     Flags
    _vehicle  Callable[..., VehicleState]  vehicle factory
    _snap     Callable[..., Snapshot]      snapshot factory
    _now_s    float                        s monotonic clock of the last step
    """

    def __init__(self, make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
        """Build the event.
        In:  make_vehicle   Callable[..., VehicleState]
             make_snapshot  Callable[..., Snapshot]
        Out:                None
        """
        self.event: Flags = Flags(sector_yellow_value=1, sector_of_index=[1, 2, 3], clear_hold_s=3.0)
        self._vehicle: Callable[..., VehicleState] = make_vehicle
        self._snap: Callable[..., Snapshot] = make_snapshot
        self._now_s: float = 0.0

    def step(
        self,
        sector_flags: tuple[int, int, int] = CLEAR,
        game_phase: int = 5,
        player_flag: int = 0,
        sector_index: int = 0,
        dt_s: float = 1.0,
    ) -> Callout | None:
        """One tick with the given flags.
        In:  sector_flags  tuple[int, int, int]  mSectorFlag[0..2]
             game_phase    int                   mGamePhase
             player_flag   int                   player's mFlag
             sector_index  int                   player's sector, 0-2
             dt_s          float                 s since the previous tick
        Out:               Callout | None
        """
        self._now_s += dt_s
        base = self._snap(self._vehicle(slot_id=0, is_player=True, sector_index=sector_index))
        flags = TrackFlags(game_phase, 0, sector_flags, player_flag)
        snapshot = RaceSnapshot(base.session, base.player, base.vehicles, (), flags)
        self.event.update(snapshot)
        return self.event.check(snapshot, self._now_s)


@pytest.fixture
def track(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> _Track:
    """A flags event ready to be fed.
    In:  make_vehicle   Callable[..., VehicleState]
         make_snapshot  Callable[..., Snapshot]
    Out:                _Track
    """
    return _Track(make_vehicle, make_snapshot)


def test_first_check_seeds_silently(track: _Track) -> None:
    """Flags already out when joining are not announced."""
    assert track.step((1, 11, 11), game_phase=6, player_flag=6) is None
    assert track.step((1, 11, 11), game_phase=6, player_flag=6) is None


def test_yellow_in_a_sector(track: _Track) -> None:
    """A sector flag turning yellow is called with its sector number, once."""
    track.step()
    callout = track.step((11, 1, 11))
    assert (callout.kind, callout.text, callout.priority) == (
        "yellow_sector_2",
        "Yellow flag, sector 2",
        Priority.SPOTTER,
    )
    assert callout.facts == {"sector": 2}
    assert track.step((11, 1, 11)) is None


def test_yellow_in_the_players_sector(track: _Track) -> None:
    """A yellow in the sector the player is in says so."""
    track.step(sector_index=1)
    assert track.step((11, 1, 11), sector_index=1).text == "Yellow flag, sector 2, your sector"


def test_sector_order_and_yellow_value_come_from_the_config(make_snapshot: Callable[..., Snapshot]) -> None:
    """The sector number of each flag index and the yellow value are parameters."""
    event = Flags(sector_yellow_value=2, sector_of_index=[3, 1, 2], clear_hold_s=3.0)
    base = make_snapshot()
    for sector_flags in (CLEAR, (2, 11, 11)):
        snapshot = RaceSnapshot(base.session, base.player, base.vehicles, (), TrackFlags(5, 0, sector_flags, 0))
        event.update(snapshot)
        callout = event.check(snapshot, 0.0)
    assert callout.text == "Yellow flag, sector 3"


def test_sector_clear_after_the_hold_reuses_the_kind(track: _Track) -> None:
    """A yellow going away is called under the kind of its raise once the sector has stayed clear for the hold."""
    track.step()
    track.step((11, 11, 1))
    assert track.step(CLEAR) is None
    assert track.step(CLEAR, dt_s=2.9) is None
    callout = track.step(CLEAR, dt_s=0.1)
    assert (callout.kind, callout.text, callout.priority) == ("yellow_sector_3", "Sector 3 clear", Priority.SPOTTER)
    assert track.step(CLEAR, dt_s=5.0) is None


def test_flicker_within_the_hold_gives_a_single_yellow(track: _Track) -> None:
    """A yellow that drops out and returns within the hold is neither cleared nor called again."""
    track.step()
    callouts = [track.step(flags, dt_s=0.5) for flags in ((1, 11, 11), CLEAR, (1, 11, 11), CLEAR, (1, 11, 11))]
    assert [c.text for c in callouts if c is not None] == ["Yellow flag, sector 1, your sector"]


def test_yellow_after_an_announced_clear_is_called_again(track: _Track) -> None:
    """Once the clear was called a new yellow in that sector is news."""
    track.step()
    track.step((1, 11, 11))
    track.step(CLEAR)
    assert track.step(CLEAR, dt_s=3.0).text == "Sector 1 clear"
    assert track.step((1, 11, 11)).kind == "yellow_sector_1"


@pytest.mark.parametrize("game_phase", [0, 3, 8])
def test_sector_flags_outside_green_running_are_silent(track: _Track, game_phase: int) -> None:
    """Before the start, on the formation lap and after the session nothing about sectors is announced."""
    track.step(game_phase=game_phase)
    assert track.step((1, 11, 3), game_phase=game_phase) is None
    assert track.step(CLEAR, game_phase=game_phase, dt_s=5.0) is None


def test_nothing_at_the_transition_into_and_out_of_green(track: _Track) -> None:
    """Flags that changed while not green are known at the green, and a yellow standing at the end is not cleared."""
    track.step(game_phase=3)
    track.step((1, 11, 11), game_phase=3)
    assert track.step((1, 11, 11), game_phase=5) is None
    assert track.step((1, 11, 3), game_phase=8) is None
    assert track.step(CLEAR, game_phase=8, dt_s=5.0) is None


def test_other_values_do_not_clear_aloud(track: _Track) -> None:
    """A yellow turning into the end-of-clock value 3 is not a clear; only the value 11 is."""
    track.step()
    track.step((11, 11, 1))
    assert track.step((11, 11, 3)) is None
    assert track.step((11, 11, 3), dt_s=5.0) is None
    assert track.step(CLEAR, dt_s=5.0) is None


def test_full_course_yellow_and_green(track: _Track) -> None:
    """Game phase 6 is a full-course yellow; back to 5 is green, under the same kind."""
    track.step()
    callout = track.step(game_phase=6)
    assert (callout.kind, callout.text, callout.priority) == (
        "full_course_yellow",
        "Full course yellow",
        Priority.SPOTTER,
    )
    assert track.step(game_phase=6) is None
    callout = track.step(game_phase=5)
    assert (callout.kind, callout.text, callout.priority) == ("full_course_yellow", "Green flag", Priority.SPOTTER)


def test_green_start_is_not_called(track: _Track) -> None:
    """Only a green after a full-course yellow is called, not the start or the end of the race."""
    track.step(game_phase=3)
    assert track.step(game_phase=5) is None
    assert track.step(game_phase=8) is None


def test_blue_flag_once_per_raise(track: _Track) -> None:
    """A blue flag is called when it comes out, not while it stays or when it goes."""
    track.step()
    callout = track.step(player_flag=6)
    assert (callout.kind, callout.text, callout.priority) == ("blue_flag", "Blue flag", Priority.SPOTTER)
    assert track.step(player_flag=6) is None
    assert track.step(player_flag=0) is None
    assert track.step(player_flag=6).kind == "blue_flag"


def test_several_changes_come_one_per_check(track: _Track) -> None:
    """Changes in the same tick are queued and returned one per check."""
    track.step()
    texts = [track.step((1, 11, 1), player_flag=6).text for _ in range(3)]
    assert texts == ["Yellow flag, sector 1, your sector", "Yellow flag, sector 3", "Blue flag"]
    assert track.step((1, 11, 1), player_flag=6) is None


def test_reset_seeds_again(track: _Track) -> None:
    """After a reset the flags that are out are taken as known and pending callouts are dropped."""
    track.step()
    assert track.step((1, 1, 11)).kind == "yellow_sector_1"
    track.event.reset()
    assert track.step((1, 1, 1)) is None
    assert track.step((1, 1, 1)) is None
