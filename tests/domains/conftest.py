import dataclasses
from collections.abc import Callable, Sequence

import pytest

from ai_race_engineer.domains.common.data.snapshot import PlayerState, SessionState, SessionType, Snapshot, VehicleState


@pytest.fixture
def make_vehicle() -> Callable[..., VehicleState]:
    """Factory for vehicles with neutral defaults (GT3, place 1, at the origin, heading 0, 30 m/s, no times).
    In:  -
    Out: Callable[..., VehicleState]  keyword overrides of any VehicleState field
    """
    return _vehicle


@pytest.fixture
def make_snapshot() -> Callable[..., Snapshot]:
    """Factory for snapshots; the player's vehicle comes first in vehicles.
    In:  -
    Out: Callable[..., Snapshot]  see _snapshot
    """
    return _snapshot


def _vehicle(**overrides: object) -> VehicleState:
    """Vehicle with neutral defaults and the given overrides.
    In:  overrides  object  VehicleState field values
    Out:            VehicleState
    """
    base: dict[str, object] = dict.fromkeys((f.name for f in dataclasses.fields(VehicleState)), 0)
    base.update(
        is_player=False,
        driver_name=f"driver{overrides.get('slot_id', 0)}",
        vehicle_name="",
        class_name="GT3",
        place=1,
        lap_distance_m=0.0,
        best_lap_s=None,
        last_lap_s=None,
        best_sector1_s=None,
        best_sector2_cum_s=None,
        in_pits=False,
        in_garage=False,
        position_xz_m=(0.0, 0.0),
        position_y_m=0.0,
        heading_rad=0.0,
        speed_mps=30.0,
    )
    base.update(overrides)
    return VehicleState(**base)


def _snapshot(
    player: VehicleState | None = None,
    vehicles: Sequence[VehicleState] = (),
    session_type: SessionType = SessionType.RACE,
    elapsed_s: float = 100.0,
    remaining_s: float = 3600.0,
    max_laps: int = 1000,
    track_name: str = "Track",
    track_length_m: float = 1000.0,
    fuel_l: float = 50.0,
    virtual_energy: float | None = None,
) -> Snapshot:
    """Snapshot of the player plus other cars.
    In:  player          VehicleState | None      player's vehicle (default: slot 0, is_player)
         vehicles        Sequence[VehicleState]   other cars
         session_type    SessionType
         elapsed_s       float                    s
         remaining_s     float                    s
         max_laps        int                      laps
         track_name      str
         track_length_m  float                    m
         fuel_l          float                    litres
         virtual_energy  float | None             fraction 0-1
    Out:                 Snapshot
    """
    me = player if player is not None else _vehicle(slot_id=0, is_player=True)
    session = SessionState(session_type, elapsed_s, remaining_s, max_laps, track_name, track_length_m, 20.0, 20.0)
    state = PlayerState(
        me, fuel_l, virtual_energy, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, "", "", (0.0, 0.0, 0.0, 0.0)
    )
    return Snapshot(session, state, (me, *vehicles))
