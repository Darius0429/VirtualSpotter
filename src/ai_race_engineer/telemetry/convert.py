import math
from collections.abc import Sequence

from pyLMUSharedMemory.lmu_data import LMUScoringInfo, LMUVect3, LMUVehicleScoring, LMUVehicleTelemetry, LMUWheel
from pyRfactor2SharedMemory.rf2_data import rF2ScoringInfo, rF2Vec3, rF2VehicleScoring, rF2VehicleTelemetry, rF2Wheel

from ai_race_engineer.telemetry.snapshot import PlayerState, SessionState, SessionType, VehicleState, WheelState

type ScoringInfo = LMUScoringInfo | rF2ScoringInfo
type VehicleScoring = LMUVehicleScoring | rF2VehicleScoring
type VehicleTelemetry = LMUVehicleTelemetry | rF2VehicleTelemetry
type Vect3 = LMUVect3 | rF2Vec3
type Wheel = LMUWheel | rF2Wheel

KELVIN: float = 273.15


def session_state(info: ScoringInfo) -> SessionState:
    """Session state from the scoring info struct.
    In:  info  LMUScoringInfo | rF2ScoringInfo
    Out:       SessionState
    """
    return SessionState(
        session_type=_session_type(info.mSession),
        elapsed_s=info.mCurrentET,
        remaining_s=info.mEndET - info.mCurrentET,
        max_laps=info.mMaxLaps,
        track_name=_text(info.mTrackName),
        track_length_m=info.mLapDist,
        track_temp_c=info.mTrackTemp,
        ambient_temp_c=info.mAmbientTemp,
    )


def vehicle_state(scoring: VehicleScoring, motion: VehicleTelemetry | VehicleScoring) -> VehicleState:
    """Vehicle state from its scoring struct, with position/heading/speed from the motion struct.
    In:  scoring  LMUVehicleScoring | rF2VehicleScoring  scoring entry of the car
         motion   telemetry or scoring struct            supplies mPos, mOri, mLocalVel (telemetry preferred)
    Out:          VehicleState
    """
    return VehicleState(
        slot_id=scoring.mID,
        is_player=bool(scoring.mIsPlayer),
        driver_name=_text(scoring.mDriverName),
        vehicle_name=_text(scoring.mVehicleName),
        class_name=_text(scoring.mVehicleClass),
        place=scoring.mPlace,
        completed_laps=scoring.mTotalLaps,
        lap_distance_m=scoring.mLapDist,
        lap_start_s=scoring.mLapStartET,
        sector_index=(scoring.mSector + 2) % 3,
        best_lap_s=_lap_time(scoring.mBestLapTime),
        last_lap_s=_lap_time(scoring.mLastLapTime),
        best_sector1_s=_lap_time(scoring.mBestSector1),
        best_sector2_cum_s=_lap_time(scoring.mBestSector2),
        time_behind_next_s=scoring.mTimeBehindNext,
        laps_behind_next=scoring.mLapsBehindNext,
        time_behind_leader_s=scoring.mTimeBehindLeader,
        laps_behind_leader=scoring.mLapsBehindLeader,
        in_pits=bool(scoring.mInPits),
        in_garage=bool(scoring.mInGarageStall),
        pit_state=scoring.mPitState,
        pitstops=scoring.mNumPitstops,
        position_xz_m=(motion.mPos.x, motion.mPos.z),
        position_y_m=motion.mPos.y,
        heading_rad=_heading_rad(motion.mOri),
        speed_mps=_speed_mps(motion.mLocalVel),
    )


def player_state(
    telem: VehicleTelemetry, vehicle: VehicleState, energy_fraction: float | None, lap_invalidated: bool | None
) -> PlayerState:
    """Player state from the player's telemetry struct.
    In:  telem            LMUVehicleTelemetry | rF2VehicleTelemetry
         vehicle          VehicleState  the player's vehicle state
         energy_fraction  float | None  virtual energy, fraction 0-1 (None when the game has none)
         lap_invalidated  bool | None   current lap invalid (None when the game has none)
    Out:                  PlayerState
    """
    return PlayerState(
        vehicle=vehicle,
        fuel_l=telem.mFuel,
        virtual_energy=energy_fraction,
        lap_invalidated=lap_invalidated,
        throttle=telem.mUnfilteredThrottle,
        brake=telem.mUnfilteredBrake,
        clutch=telem.mUnfilteredClutch,
        steering=telem.mUnfilteredSteering,
        rpm=telem.mEngineRPM,
        gear=telem.mGear,
        accel_lateral_mps2=telem.mLocalAccel.x,
        accel_longitudinal_mps2=telem.mLocalAccel.z,
        accel_vertical_mps2=telem.mLocalAccel.y,
        tyre_compound_front=_text(telem.mFrontTireCompoundName),
        tyre_compound_rear=_text(telem.mRearTireCompoundName),
        tyre_surface_temp_c=(
            _tyre_temp_c(telem.mWheels[0]),
            _tyre_temp_c(telem.mWheels[1]),
            _tyre_temp_c(telem.mWheels[2]),
            _tyre_temp_c(telem.mWheels[3]),
        ),
        wheels=tuple(_wheel_state(telem.mWheels[index]) for index in range(4)),
    )


def _wheel_state(wheel: Wheel) -> WheelState:
    """Wheel state from one wheel of the telemetry struct.
    In:  wheel  LMUWheel | rF2Wheel  mRotation in rad/s (negative forward), mStaticUndeflectedRadius in cm,
                                     temperatures in K, mPressure in kPa
    Out:        WheelState
    """
    return WheelState(
        speed_mps=-wheel.mRotation * wheel.mStaticUndeflectedRadius / 100,
        grip_fraction=wheel.mGripFract,
        surface_temp_c=(
            wheel.mTemperature[0] - KELVIN,
            wheel.mTemperature[1] - KELVIN,
            wheel.mTemperature[2] - KELVIN,
        ),
        carcass_temp_c=wheel.mTireCarcassTemperature - KELVIN,
        pressure_kpa=wheel.mPressure,
        brake_temp_c=wheel.mBrakeTemp - KELVIN,
    )


def _text(raw: bytes) -> str:
    """Decode a fixed-size game string.
    In:  raw  bytes  null-padded bytes
    Out:      str
    """
    return raw.decode("utf-8", errors="replace").rstrip("\x00").strip()


def _lap_time(raw_s: float) -> float | None:
    """Lap or sector time, None when not set.
    In:  raw_s  float         s (game uses <= 0 for no time)
    Out:        float | None  s
    """
    return raw_s if raw_s > 0 else None


def _speed_mps(vel: Vect3) -> float:
    """Speed from a local velocity vector.
    In:  vel  LMUVect3 | rF2Vec3  m/s per axis
    Out:      float               m/s
    """
    return math.hypot(vel.x, vel.y, vel.z)


def _heading_rad(ori: Sequence[Vect3]) -> float:
    """Heading from the orientation matrix rows; forward = -(local z axis) = (sin h, cos h) in world (x, z).
    In:  ori  Sequence[LMUVect3 | rF2Vec3]  3 rows of the orientation matrix
    Out:      float                         rad
    """
    return math.atan2(-ori[0].z, -ori[2].z)


def _session_type(m_session: int) -> SessionType:
    """Map the game's session number to a session type.
    In:  m_session  int  0 test day, 1-4 practice, 5-8 qualify, 9 warmup, 10-13 race
    Out:            SessionType
    """
    if m_session == 0:
        return SessionType.TEST_DAY
    if m_session <= 4:
        return SessionType.PRACTICE
    if m_session <= 8:
        return SessionType.QUALIFY
    if m_session == 9:
        return SessionType.WARMUP
    return SessionType.RACE


def _tyre_temp_c(wheel: Wheel) -> float:
    """Mean tyre surface temperature of one wheel.
    In:  wheel  LMUWheel | rF2Wheel  mTemperature in K (left/centre/right)
    Out:        float                degC
    """
    return sum(wheel.mTemperature) / 3 - 273.15
