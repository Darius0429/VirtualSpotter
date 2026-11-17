from collections.abc import Callable

import pytest

from ai_race_engineer.domains.common.callouts.sector_comparison import SectorComparison
from ai_race_engineer.domains.common.analysis.pace.sectors import sector_deltas_s, sector_splits_s
from ai_race_engineer.domains.common.data.snapshot import Snapshot, VehicleState


def _timed(make_vehicle: Callable[..., VehicleState], slot_id: int, lap_s: float | None, **kw: object) -> VehicleState:
    """Vehicle with a best lap and best sectors (S1 30 s, S2 cumulative 65 s) unless overridden.
    In:  make_vehicle  Callable[..., VehicleState]
         slot_id       int
         lap_s         float | None  s best lap
         kw            object        further VehicleState overrides
    Out:               VehicleState
    """
    times = {"best_lap_s": lap_s, "best_sector1_s": 30.0, "best_sector2_cum_s": 65.0} if lap_s else {}
    return make_vehicle(slot_id=slot_id, **{**times, **kw})


def _seeded(make_snapshot: Callable[..., Snapshot], me: VehicleState, rivals: list[VehicleState]) -> SectorComparison:
    """SectorComparison after a silent seeding check.
    In:  make_snapshot  Callable[..., Snapshot]
         me             VehicleState        player
         rivals         list[VehicleState]
    Out:                SectorComparison
    """
    event = SectorComparison()
    assert event.check(make_snapshot(me, rivals), 0.0) is None
    return event


def test_rival_pb_beats_player(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """Deltas are reported per sector, rival minus player."""
    me = _timed(make_vehicle, 0, 100.0, is_player=True)
    event = _seeded(make_snapshot, me, [_timed(make_vehicle, 1, 101.0)])
    rival = _timed(make_vehicle, 1, 99.5, driver_name="Ann", best_sector1_s=29.8, best_sector2_cum_s=64.6)
    callout = event.check(make_snapshot(me, [rival]), 1.0)
    expected = sector_deltas_s(sector_splits_s(29.8, 64.6, 99.5), sector_splits_s(30.0, 65.0, 100.0))
    assert callout is not None
    assert callout.kind == "sector_comparison"
    assert (callout.facts["d1_s"], callout.facts["d2_s"], callout.facts["d3_s"]) == pytest.approx(expected)
    assert callout.text == "Ann is 0.50 quicker: S1 -0.20, S2 -0.20, S3 -0.10"


def test_rival_pb_still_slower(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """A rival improvement that is still slower than the player is not reported."""
    me = _timed(make_vehicle, 0, 100.0, is_player=True)
    event = _seeded(make_snapshot, me, [_timed(make_vehicle, 1, 102.0)])
    assert event.check(make_snapshot(me, [_timed(make_vehicle, 1, 100.5)]), 1.0) is None


def test_player_no_best(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """Without a player best lap there is nothing to compare."""
    me = make_vehicle(slot_id=0, is_player=True)
    event = _seeded(make_snapshot, me, [_timed(make_vehicle, 1, 102.0)])
    assert event.check(make_snapshot(me, [_timed(make_vehicle, 1, 99.0)]), 1.0) is None


def test_rival_missing_sectors(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """A rival without sector times is not reported."""
    me = _timed(make_vehicle, 0, 100.0, is_player=True)
    event = _seeded(make_snapshot, me, [_timed(make_vehicle, 1, 102.0)])
    rival = _timed(make_vehicle, 1, 99.0, best_sector1_s=None)
    assert event.check(make_snapshot(me, [rival]), 1.0) is None


def test_two_improvers_fastest_reported(
    make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]
) -> None:
    """With several improvers in one tick the fastest is reported."""
    me = _timed(make_vehicle, 0, 100.0, is_player=True)
    event = _seeded(make_snapshot, me, [_timed(make_vehicle, 1, 102.0), _timed(make_vehicle, 2, 102.0)])
    callout = event.check(make_snapshot(me, [_timed(make_vehicle, 1, 99.5), _timed(make_vehicle, 2, 99.0)]), 1.0)
    assert callout is not None
    assert callout.facts["driver"] == "driver2"


def test_other_class_ignored(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """Cars of another class are not compared."""
    me = _timed(make_vehicle, 0, 100.0, is_player=True)
    event = _seeded(make_snapshot, me, [_timed(make_vehicle, 1, 95.0, class_name="Hypercar")])
    assert event.check(make_snapshot(me, [_timed(make_vehicle, 1, 90.0, class_name="Hypercar")]), 1.0) is None


def test_first_check_silent(make_vehicle: Callable[..., VehicleState], make_snapshot: Callable[..., Snapshot]) -> None:
    """Existing rival times are not reported on the first check."""
    me = _timed(make_vehicle, 0, 100.0, is_player=True)
    assert SectorComparison().check(make_snapshot(me, [_timed(make_vehicle, 1, 99.0)]), 0.0) is None
