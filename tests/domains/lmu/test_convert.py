import math

import numpy as np
import pytest
from pyLMUSharedMemory.lmu_data import LMUScoringInfo, LMUVect3, LMUVehicleScoring, LMUVehicleTelemetry, LMUWheel
from pyRfactor2SharedMemory.rf2_data import rF2ScoringInfo, rF2VehicleScoring, rF2VehicleTelemetry

from ai_race_engineer.domains.common.analysis.field.proximity import to_player_frame
from ai_race_engineer.domains.lmu.convert import (
    _heading_rad,
    _lap_time,
    _session_type,
    _speed_mps,
    _text,
    _tyre_temp_c,
    player_state,
    session_state,
    vehicle_state,
)
from ai_race_engineer.domains.common.data.snapshot import SessionType

ScoringStruct = LMUVehicleScoring | rF2VehicleScoring
TelemetryStruct = LMUVehicleTelemetry | rF2VehicleTelemetry


def _set_yaw(ori: object, heading_rad: float) -> None:
    """Write a yaw-only orientation matrix whose forward (-local z) is (sin h, cos h) in world (x, z).
    In:  ori          object  mOri array of 3 vectors
         heading_rad  float   rad
    Out:              None
    """
    s, c = math.sin(heading_rad), math.cos(heading_rad)
    rows = ((-c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, -c))
    for row, (x, y, z) in zip(ori, rows, strict=True):
        row.x, row.y, row.z = x, y, z


def _fill_scoring(scoring: ScoringStruct) -> ScoringStruct:
    """Fill a vehicle scoring struct with fixed test values.
    In:  scoring  LMUVehicleScoring | rF2VehicleScoring
    Out:          same struct, filled
    """
    scoring.mID = 7
    scoring.mIsPlayer = True
    scoring.mDriverName = b"Jane Doe"
    scoring.mVehicleName = b"Car #7"
    scoring.mVehicleClass = b"Hypercar"
    scoring.mPlace = 3
    scoring.mTotalLaps = 12
    scoring.mLapDist = 1234.5
    scoring.mLapStartET = 1500.0
    scoring.mSector = 1
    scoring.mBestLapTime = 210.5
    scoring.mLastLapTime = -1.0
    scoring.mBestSector1 = 70.0
    scoring.mBestSector2 = 0.0
    scoring.mTimeBehindNext = 1.5
    scoring.mLapsBehindNext = 0
    scoring.mTimeBehindLeader = 12.25
    scoring.mLapsBehindLeader = 1
    scoring.mInPits = False
    scoring.mInGarageStall = False
    scoring.mPitState = 0
    scoring.mNumPitstops = 2
    scoring.mPos.x, scoring.mPos.y, scoring.mPos.z = 1.0, 2.0, 3.0
    scoring.mLocalVel.x, scoring.mLocalVel.y, scoring.mLocalVel.z = 0.0, 0.0, -50.0
    _set_yaw(scoring.mOri, 0.3)
    return scoring


def _fill_telemetry(telem: TelemetryStruct) -> TelemetryStruct:
    """Fill a vehicle telemetry struct with fixed test values.
    In:  telem  LMUVehicleTelemetry | rF2VehicleTelemetry
    Out:        same struct, filled
    """
    telem.mID = 7
    telem.mPos.x, telem.mPos.y, telem.mPos.z = 100.0, 5.0, -200.0
    telem.mLocalVel.x, telem.mLocalVel.y, telem.mLocalVel.z = 3.0, 0.0, -4.0
    _set_yaw(telem.mOri, 1.2)
    telem.mFuel = 55.5
    telem.mUnfilteredThrottle = 0.8
    telem.mUnfilteredBrake = 0.1
    telem.mUnfilteredClutch = 0.0
    telem.mUnfilteredSteering = -0.25
    telem.mEngineRPM = 7500.0
    telem.mGear = 4
    telem.mLocalAccel.x, telem.mLocalAccel.y, telem.mLocalAccel.z = 9.0, 1.0, -3.0
    telem.mFrontTireCompoundName = b"Medium"
    telem.mRearTireCompoundName = b"Hard"
    for i, wheel in enumerate(telem.mWheels):
        wheel.mTemperature[:] = [353.15 + i, 353.15 + i, 353.15 + i]
    return telem


def _fill_info(info: LMUScoringInfo | rF2ScoringInfo) -> LMUScoringInfo | rF2ScoringInfo:
    """Fill a scoring info struct with fixed test values.
    In:  info  LMUScoringInfo | rF2ScoringInfo
    Out:       same struct, filled
    """
    info.mSession = 10
    info.mCurrentET = 600.0
    info.mEndET = 3600.0
    info.mMaxLaps = 2147483647
    info.mTrackName = b"Le Mans"
    info.mLapDist = 13626.0
    info.mTrackTemp = 32.0
    info.mAmbientTemp = 21.0
    return info


@pytest.mark.parametrize(
    ("m_session", "expected"),
    [
        (0, SessionType.TEST_DAY),
        (1, SessionType.PRACTICE),
        (4, SessionType.PRACTICE),
        (5, SessionType.QUALIFY),
        (8, SessionType.QUALIFY),
        (9, SessionType.WARMUP),
        (10, SessionType.RACE),
        (13, SessionType.RACE),
    ],
)
def test_session_type_ranges(m_session: int, expected: SessionType) -> None:
    """Game session numbers map to session types."""
    assert _session_type(m_session) == expected


@pytest.mark.parametrize(("m_sector", "expected"), [(1, 0), (2, 1), (0, 2)])
def test_sector_remap(m_sector: int, expected: int) -> None:
    """Game sector numbering (0 = sector 3) maps to 0, 1, 2."""
    scoring = _fill_scoring(LMUVehicleScoring())
    scoring.mSector = m_sector
    assert vehicle_state(scoring, scoring).sector_index == expected


def test_speed() -> None:
    """Speed is the magnitude of the local velocity."""
    vel = LMUVect3()
    vel.x, vel.y, vel.z = 3.0, 0.0, -4.0
    assert _speed_mps(vel) == pytest.approx(5.0)


def test_kelvin_to_celsius() -> None:
    """Tyre temperature is the mean of three readings in degC."""
    wheel = LMUWheel()
    wheel.mTemperature[:] = [353.15, 363.15, 373.15]
    assert _tyre_temp_c(wheel) == pytest.approx(90.0)


@pytest.mark.parametrize("raw", [0.0, -1.0])
def test_lap_time_unset_is_none(raw: float) -> None:
    """Lap times <= 0 mean no time."""
    assert _lap_time(raw) is None


def test_lap_time_set() -> None:
    """Positive lap times pass through."""
    assert _lap_time(210.5) == 210.5


def test_name_decoding() -> None:
    """Fixed-size bytes decode to trimmed strings."""
    assert _text(b"Le Mans\x00\x00") == "Le Mans"


def test_heading_consistent_with_player_frame() -> None:
    """A yaw-only orientation gives the heading that puts a car ahead-right of the player ahead-right in its frame."""
    telem = LMUVehicleTelemetry()
    heading = 0.7
    _set_yaw(telem.mOri, heading)
    assert _heading_rad(telem.mOri) == pytest.approx(heading)
    forward = np.array([math.sin(heading), math.cos(heading)])
    right = -np.array([telem.mOri[0].x, telem.mOri[2].x])
    other = np.array([[10.0, 20.0]]) + 3.0 * forward + 2.0 * right
    np.testing.assert_allclose(to_player_frame((10.0, 20.0), heading, other), [[2.0, 3.0]], atol=1e-9)


def test_scoring_struct_as_motion_fallback() -> None:
    """The scoring struct supplies motion when a car has no telemetry entry."""
    scoring = _fill_scoring(LMUVehicleScoring())
    vehicle = vehicle_state(scoring, scoring)
    assert vehicle.position_xz_m == (1.0, 3.0)
    assert vehicle.position_y_m == 2.0
    assert vehicle.heading_rad == pytest.approx(0.3)
    assert vehicle.speed_mps == pytest.approx(50.0)


def test_motion_from_telemetry() -> None:
    """Position, heading and speed come from the telemetry struct when given."""
    vehicle = vehicle_state(_fill_scoring(LMUVehicleScoring()), _fill_telemetry(LMUVehicleTelemetry()))
    assert vehicle.position_xz_m == (100.0, -200.0)
    assert vehicle.heading_rad == pytest.approx(1.2)
    assert vehicle.speed_mps == pytest.approx(5.0)
    assert vehicle.best_lap_s == 210.5
    assert vehicle.last_lap_s is None
    assert vehicle.best_sector2_cum_s is None
    assert vehicle.class_name == "Hypercar"


def test_lmu_and_rf2_identical() -> None:
    """Identical LMU and rF2 struct contents give identical snapshot parts."""
    lmu_vehicle = vehicle_state(_fill_scoring(LMUVehicleScoring()), _fill_telemetry(LMUVehicleTelemetry()))
    rf2_vehicle = vehicle_state(_fill_scoring(rF2VehicleScoring()), _fill_telemetry(rF2VehicleTelemetry()))
    assert lmu_vehicle == rf2_vehicle
    assert session_state(_fill_info(LMUScoringInfo())) == session_state(_fill_info(rF2ScoringInfo()))
    lmu_player = player_state(_fill_telemetry(LMUVehicleTelemetry()), lmu_vehicle, None, None)
    rf2_player = player_state(_fill_telemetry(rF2VehicleTelemetry()), rf2_vehicle, None, None)
    assert lmu_player == rf2_player
    assert lmu_player.tyre_surface_temp_c == pytest.approx((80.0, 81.0, 82.0, 83.0))
    assert session_state(_fill_info(LMUScoringInfo())).remaining_s == 3000.0
