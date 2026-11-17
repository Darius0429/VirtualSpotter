from ai_race_engineer.telemetry.convert import ScoringInfo, VehicleScoring, VehicleTelemetry
from ai_race_engineer.telemetry.race_snapshot import CarResources, CarTyres, LapSectors, TrackFlags


def car_resources(scoring: VehicleScoring, telem: VehicleTelemetry, energy_fraction: float | None) -> CarResources:
    """Resources of one car from its scoring and telemetry structs.
    In:  scoring          LMUVehicleScoring | rF2VehicleScoring      scoring entry of the car
         telem            LMUVehicleTelemetry | rF2VehicleTelemetry  telemetry entry of the car
         energy_fraction  float | None                               virtual energy, fraction 0-1 (None = game has none)
    Out:                  CarResources
    """
    return CarResources(
        slot_id=scoring.mID,
        fuel_l=telem.mFuel,
        fuel_capacity_l=telem.mFuelCapacity,
        virtual_energy=energy_fraction,
        finish_status=scoring.mFinishStatus,
        last_impact_s=telem.mLastImpactET,
        last_impact_magnitude=telem.mLastImpactMagnitude,
    )


def track_flags(info: ScoringInfo, player: VehicleScoring) -> TrackFlags:
    """Flag state from the scoring info struct and the player's scoring entry.
    In:  info    LMUScoringInfo | rF2ScoringInfo
         player  LMUVehicleScoring | rF2VehicleScoring  the player's scoring entry
    Out:         TrackFlags
    """
    return TrackFlags(
        game_phase=info.mGamePhase,
        yellow_flag_state=int.from_bytes(info.mYellowFlagState, signed=True),
        sector_flags=(info.mSectorFlag[0], info.mSectorFlag[1], info.mSectorFlag[2]),
        player_flag=player.mFlag,
    )


def lap_sectors(scoring: VehicleScoring) -> LapSectors:
    """Sector times of a car's last completed lap from its scoring struct.
    In:  scoring  LMUVehicleScoring | rF2VehicleScoring  scoring entry of the car
    Out:          LapSectors
    """
    return LapSectors(
        slot_id=scoring.mID,
        last_sector1_s=scoring.mLastSector1 if scoring.mLastSector1 > 0 else None,
        last_sector2_cum_s=scoring.mLastSector2 if scoring.mLastSector2 > 0 else None,
        sector1_s=scoring.mCurSector1 if scoring.mCurSector1 > 0 else None,
        sector2_cum_s=scoring.mCurSector2 if scoring.mCurSector2 > 0 else None,
    )


def car_tyres(scoring: VehicleScoring, telem: VehicleTelemetry) -> CarTyres:
    """Tyres of one car from its scoring and telemetry structs.
    In:  scoring  LMUVehicleScoring | rF2VehicleScoring      scoring entry of the car
         telem    LMUVehicleTelemetry | rF2VehicleTelemetry  telemetry entry of the car
    Out:          CarTyres
    """
    return CarTyres(
        slot_id=scoring.mID,
        compound_front=telem.mFrontTireCompoundName.decode("utf-8", errors="replace").rstrip("\x00").strip(),
        compound_rear=telem.mRearTireCompoundName.decode("utf-8", errors="replace").rstrip("\x00").strip(),
        wear=(telem.mWheels[0].mWear, telem.mWheels[1].mWear, telem.mWheels[2].mWear, telem.mWheels[3].mWear),
    )
