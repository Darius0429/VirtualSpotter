import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ai_race_engineer.telemetry.convert import player_state, session_state, vehicle_state
from ai_race_engineer.telemetry.race_convert import car_resources, car_tyres, lap_sectors, track_flags
from ai_race_engineer.telemetry.race_snapshot import RaceSnapshot
from ai_race_engineer.telemetry.race_source import RaceSource
from ai_race_engineer.telemetry.raw_row import RawRow

type Table = dict[str, NDArray[Any]]

MOTION_COLUMNS: tuple[str, ...] = (
    "mPos.x",
    "mPos.y",
    "mPos.z",
    "mOri.z.0",
    "mOri.z.2",
    "mLocalVel.x",
    "mLocalVel.y",
    "mLocalVel.z",
)
INFO_COLUMNS: tuple[str, ...] = (
    "poll_s",
    "mSession",
    "mCurrentET",
    "mEndET",
    "mMaxLaps",
    "mTrackName",
    "mLapDist",
    "mTrackTemp",
    "mAmbientTemp",
    "mGamePhase",
    "mYellowFlagState",
    "mSectorFlag.0",
    "mSectorFlag.1",
    "mSectorFlag.2",
    "mSessionTimeRemaining",
)
SCORING_COLUMNS: tuple[str, ...] = (
    "poll_s",
    "mID",
    "mIsPlayer",
    "mDriverName",
    "mVehicleName",
    "mVehicleClass",
    "mPlace",
    "mTotalLaps",
    "mLapDist",
    "mLapStartET",
    "mSector",
    "mBestLapTime",
    "mLastLapTime",
    "mBestSector1",
    "mBestSector2",
    "mLastSector1",
    "mLastSector2",
    "mTimeBehindNext",
    "mLapsBehindNext",
    "mTimeBehindLeader",
    "mLapsBehindLeader",
    "mInPits",
    "mInGarageStall",
    "mPitState",
    "mNumPitstops",
    "mFinishStatus",
    "mFlag",
    *MOTION_COLUMNS,
)
CAR_COLUMNS: tuple[str, ...] = (
    "poll_s",
    "mID",
    "mFuel",
    "mFuelCapacity",
    "mVirtualEnergy",
    "mFrontTireCompoundName",
    "mRearTireCompoundName",
    *(f"mWheels.mWear.{wheel}" for wheel in range(4)),
    *MOTION_COLUMNS,
)
WHEEL_FIELDS: tuple[str, ...] = (
    "mRotation",
    "mStaticUndeflectedRadius",
    "mGripFract",
    "mTireCarcassTemperature",
    "mPressure",
    "mBrakeTemp",
)
PLAYER_COLUMNS: tuple[str, ...] = (
    *CAR_COLUMNS,
    "mLapInvalidated",
    "mUnfilteredThrottle",
    "mUnfilteredBrake",
    "mUnfilteredClutch",
    "mUnfilteredSteering",
    "mEngineRPM",
    "mGear",
    "mLocalAccel.x",
    "mLocalAccel.y",
    "mLocalAccel.z",
    *(f"mWheels.mTemperature.{wheel}.{part}" for wheel in range(4) for part in range(3)),
    *(f"mWheels.{field}.{wheel}" for field in WHEEL_FIELDS for wheel in range(4)),
)


class ReplaySource(RaceSource):
    """Replays a raw-dump folder (flattened game structs in Parquet tables) as if it were the live game.
    _folder        Path                                                  raw-dump folder
    _speed         float                                                 replay s per wall s
    _max_age_s     float                                                 s after which a car's telemetry is stale
    _info          Table                                                 scoring info rows, sorted by poll_s
    _scoring       Table                                                 car scoring rows, sorted by poll_s
    _player        Table                                                 player telemetry rows, sorted by poll_s
    _cars          Table                                                 other cars' telemetry rows, sorted by poll_s
    _car_rows      dict[int, tuple[NDArray[np.float64], NDArray[np.intp]]]  slot id -> (poll_s, row in _cars)
    _first_poll_s  float                                                 s, poll time of the first scoring info row
    _start_s       float                                                 s monotonic clock at open
    """

    def __init__(self, folder: Path, speed: float = 1.0, max_telemetry_age_s: float = 2.0) -> None:
        """Set the dump to replay and its speed.
        In:  folder               Path   raw-dump folder
             speed                float  replay s per wall s
             max_telemetry_age_s  float  replay s after which another car's last telemetry row is no longer used
        Out:                      None
        """
        self._folder: Path = folder
        self._speed: float = speed
        self._max_age_s: float = max_telemetry_age_s
        self._info: Table = {}
        self._scoring: Table = {}
        self._player: Table = {}
        self._cars: Table = {}
        self._car_rows: dict[int, tuple[NDArray[np.float64], NDArray[np.intp]]] = {}
        self._first_poll_s: float = 0.0
        self._start_s: float = time.monotonic()

    def open(self) -> None:
        """Load the four tables and start the replay clock; a missing folder or table raises OSError.
        In:  -
        Out: None
        """
        self._info = _table(self._folder / "raw_scoring_info_5hz", INFO_COLUMNS)
        self._scoring = _table(self._folder / "raw_cars_scoring_5hz", SCORING_COLUMNS)
        self._player = _table(self._folder / "raw_player_telemetry_100hz", PLAYER_COLUMNS)
        self._cars = _table(self._folder / "raw_cars_telemetry_50hz", CAR_COLUMNS)
        rows = pd.Series(np.arange(len(self._cars["mID"]))).groupby(self._cars["mID"]).indices
        self._car_rows = {int(slot_id): (self._cars["poll_s"][index], index) for slot_id, index in rows.items()}
        self._first_poll_s = float(self._info["poll_s"][0])
        self._start_s = time.monotonic()

    def read(self) -> RaceSnapshot | None:
        """Snapshot of the latest recorded rows at or before the replay time, cars matched by slot id.
        In:  -
        Out: RaceSnapshot | None  None before the first row, after the last, or without a player with telemetry
        """
        now_s = self.time_s()
        info_row = _latest(self._info["poll_s"], now_s)
        player_row = _latest(self._player["poll_s"], now_s)
        if info_row < 0 or player_row < 0 or now_s > self._info["poll_s"][-1]:
            return None
        info = RawRow(self._info, info_row)
        tick_s = self._info["poll_s"][info_row]
        first = int(np.searchsorted(self._scoring["poll_s"], tick_s, side="left"))
        last = int(np.searchsorted(self._scoring["poll_s"], tick_s, side="right"))
        entries = [RawRow(self._scoring, row) for row in range(first, last)]
        telem_by_id = self._car_telemetry(now_s)
        telem = RawRow(self._player, player_row)
        telem_by_id[telem.mID] = telem
        vehicles = tuple(vehicle_state(s, telem_by_id.get(s.mID, s)) for s in entries)
        player = next((v for v in vehicles if v.is_player), None)
        if player is None or player.slot_id != telem.mID:
            return None
        return RaceSnapshot(
            session=session_state(info),
            player=player_state(
                telem, player, energy_fraction=telem.mVirtualEnergy, lap_invalidated=telem.mLapInvalidated
            ),
            vehicles=vehicles,
            resources=tuple(
                car_resources(s, telem_by_id[s.mID], telem_by_id[s.mID].mVirtualEnergy)
                for s in entries
                if s.mID in telem_by_id
            ),
            flags=track_flags(info, next(s for s in entries if s.mIsPlayer)),
            sectors=tuple(lap_sectors(s) for s in entries),
            time_remaining_s=info.mSessionTimeRemaining,
            tyres=tuple(car_tyres(s, telem_by_id[s.mID]) for s in entries if s.mID in telem_by_id),
        )

    def results_path(self) -> Path | None:
        """Results file the game pointed to while the dump was recorded (stewardResults in paths.json).
        In:  -
        Out: Path | None  None for a dump without a readable paths.json or without the key
        """
        try:
            return Path(json.loads((self._folder / "paths.json").read_text())["stewardResults"])
        except (OSError, ValueError, KeyError):
            return None

    def close(self) -> None:
        """Drop the loaded tables.
        In:  -
        Out: None
        """
        self._info, self._scoring, self._player, self._cars, self._car_rows = {}, {}, {}, {}, {}

    def time_s(self) -> float:
        """Current replay time.
        In:  -
        Out: float  s on the dump's poll_s axis
        """
        return self._first_poll_s + (time.monotonic() - self._start_s) * self._speed

    def _car_telemetry(self, now_s: float) -> dict[int, RawRow]:
        """Latest telemetry row of every other car at or before a time.
        In:  now_s  float              s on the poll_s axis
        Out:        dict[int, RawRow]  slot id -> telemetry row; cars without a fresh enough row are left out
        """
        telem_by_id: dict[int, RawRow] = {}
        for slot_id, (poll_s, rows) in self._car_rows.items():
            index = _latest(poll_s, now_s)
            if index >= 0 and now_s - poll_s[index] <= self._max_age_s:
                telem_by_id[slot_id] = RawRow(self._cars, int(rows[index]))
        return telem_by_id


def _table(folder: Path, columns: tuple[str, ...]) -> Table:
    """Load the needed columns of one raw-dump table.
    In:  folder   Path             folder of Parquet part files
         columns  tuple[str, ...]  column names to load
    Out:          Table            column name -> values, rows sorted by poll_s
    """
    frame = pd.read_parquet(folder, columns=list(columns)).sort_values("poll_s", kind="stable")
    return {name: frame[name].to_numpy() for name in columns}


def _latest(poll_s: NDArray[np.float64], now_s: float) -> int:
    """Index of the latest row at or before a time.
    In:  poll_s  NDArray[np.float64]  s, sorted
         now_s   float                s
    Out:         int                  row index; -1 when every row is later
    """
    return int(np.searchsorted(poll_s, now_s, side="right")) - 1
