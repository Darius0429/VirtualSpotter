from dataclasses import fields

from ai_race_engineer.domains.lmu.convert import (
    ScoringInfo,
    VehicleScoring,
    VehicleTelemetry,
    player_state,
    session_state,
    vehicle_state,
)
from ai_race_engineer.domains.common.data.frame import CarSample, CarStatus, PlayerSample, SessionInfo, StateSample
from ai_race_engineer.domains.common.data.snapshot import VehicleState

_CORNERS: tuple[str, ...] = ("fl", "fr", "rl", "rr")
_STATUS_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(VehicleState) if f.name not in {"position_xz_m", "position_y_m", "heading_rad", "speed_mps"}
)


def session_info(info: ScoringInfo, player: VehicleScoring, game: str) -> SessionInfo:
    """Session identity from the scoring info and the player's scoring entry.
    In:  info    LMUScoringInfo | rF2ScoringInfo
         player  LMUVehicleScoring | rF2VehicleScoring  player's scoring entry
         game    str                                    'lmu' or 'rf2'
    Out:         SessionInfo
    """
    session, vehicle = session_state(info), vehicle_state(player, player)
    return SessionInfo(
        game=game,
        track_name=session.track_name,
        track_length_m=session.track_length_m,
        vehicle_name=vehicle.vehicle_name,
        class_name=vehicle.class_name,
        session_type=session.session_type,
        session_number=info.mSession,
        max_laps=session.max_laps,
    )


def player_sample(telem: VehicleTelemetry, player: VehicleScoring) -> PlayerSample:
    """Player telemetry row at the telemetry rate.
    In:  telem   LMUVehicleTelemetry | rF2VehicleTelemetry  player's telemetry
         player  LMUVehicleScoring | rF2VehicleScoring      player's scoring entry (lap distance)
    Out:         PlayerSample
    """
    vehicle = vehicle_state(player, telem)
    state = player_state(telem, vehicle, None, None)
    return PlayerSample(
        time_s=telem.mElapsedTime,
        throttle=state.throttle,
        brake=state.brake,
        clutch=state.clutch,
        steering=state.steering,
        gear=state.gear,
        rpm=state.rpm,
        speed_mps=vehicle.speed_mps,
        pos_x_m=vehicle.position_xz_m[0],
        pos_y_m=vehicle.position_y_m,
        pos_z_m=vehicle.position_xz_m[1],
        heading_rad=vehicle.heading_rad,
        accel_x_mps2=state.accel_lateral_mps2,
        accel_y_mps2=state.accel_vertical_mps2,
        accel_z_mps2=state.accel_longitudinal_mps2,
        lap_number=telem.mLapNumber,
        lap_distance_m=vehicle.lap_distance_m,
    )


def car_sample(time_s: float, scoring: VehicleScoring, motion: VehicleTelemetry | VehicleScoring) -> CarSample:
    """Position row of one car at the telemetry rate.
    In:  time_s   float                     s player telemetry clock
         scoring  LMUVehicleScoring | rF2VehicleScoring  the car's scoring entry
         motion   telemetry or scoring struct            supplies mPos, mOri, mLocalVel (telemetry preferred)
    Out:          CarSample
    """
    vehicle = vehicle_state(scoring, motion)
    return CarSample(
        time_s=time_s,
        slot_id=vehicle.slot_id,
        pos_x_m=vehicle.position_xz_m[0],
        pos_y_m=vehicle.position_y_m,
        pos_z_m=vehicle.position_xz_m[1],
        heading_rad=vehicle.heading_rad,
        speed_mps=vehicle.speed_mps,
        lap_distance_m=vehicle.lap_distance_m,
    )


def state_sample(
    time_s: float,
    info: ScoringInfo,
    telem: VehicleTelemetry,
    player: VehicleScoring,
    energy_fraction: float | None,
    lap_invalidated: bool | None,
) -> StateSample:
    """Player and session state row at the scoring rate.
    In:  time_s           float         s player telemetry clock when the scoring update was seen
         info             LMUScoringInfo | rF2ScoringInfo
         telem            LMUVehicleTelemetry | rF2VehicleTelemetry  player's telemetry
         player           LMUVehicleScoring | rF2VehicleScoring      player's scoring entry
         energy_fraction  float | None  virtual energy, fraction 0-1 (None when the game has none)
         lap_invalidated  bool | None   current lap invalid (None when the game has none)
    Out:                  StateSample
    """
    session = session_state(info)
    state = player_state(telem, vehicle_state(player, telem), energy_fraction, lap_invalidated)
    temps = dict(zip(_CORNERS, state.tyre_surface_temp_c, strict=True))
    return StateSample(
        time_s=time_s,
        session_time_s=info.mCurrentET,
        remaining_s=session.remaining_s,
        game_phase=info.mGamePhase,
        yellow_flag_state=int.from_bytes(info.mYellowFlagState, signed=True),
        sector_flag_1=info.mSectorFlag[0],
        sector_flag_2=info.mSectorFlag[1],
        sector_flag_3=info.mSectorFlag[2],
        player_flag=player.mFlag,
        fuel_l=state.fuel_l,
        virtual_energy=state.virtual_energy,
        lap_invalidated=state.lap_invalidated,
        tyre_compound_front=state.tyre_compound_front,
        tyre_compound_rear=state.tyre_compound_rear,
        **{f"tyre_temp_c_{c}": temps[c] for c in _CORNERS},
        **{f"tyre_wear_{c}": w.mWear for c, w in zip(_CORNERS, telem.mWheels, strict=True)},
        **{f"tyre_pressure_kpa_{c}": w.mPressure for c, w in zip(_CORNERS, telem.mWheels, strict=True)},
        **{f"dent_{i}": dent for i, dent in enumerate(telem.mDentSeverity)},
        overheating=bool(telem.mOverheating),
        detached=bool(telem.mDetached),
        track_temp_c=session.track_temp_c,
        ambient_temp_c=session.ambient_temp_c,
        raining=info.mRaining,
        path_wetness_avg=info.mAvgPathWetness,
    )


def car_status(time_s: float, scoring: VehicleScoring) -> CarStatus:
    """Scoring row of one car at the scoring rate.
    In:  time_s   float  s player telemetry clock when the scoring update was seen
         scoring  LMUVehicleScoring | rF2VehicleScoring  the car's scoring entry
    Out:          CarStatus
    """
    vehicle = vehicle_state(scoring, scoring)
    return CarStatus(
        time_s=time_s,
        **{name: getattr(vehicle, name) for name in _STATUS_FIELDS},
        last_sector1_s=scoring.mLastSector1 if scoring.mLastSector1 > 0 else None,
        last_sector2_cum_s=scoring.mLastSector2 if scoring.mLastSector2 > 0 else None,
    )
