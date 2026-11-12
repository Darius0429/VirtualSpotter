import logging

from pyLMUSharedMemory.lmu_data import LMUConstants, LMUObjectOut, LMUScoringData, LMUTelemetryData
from pyLMUSharedMemory.lmu_mmap import MMapControl

from ai_race_engineer.telemetry.base import TelemetrySource
from ai_race_engineer.telemetry.convert import player_state, session_state, vehicle_state
from ai_race_engineer.telemetry.snapshot import Snapshot

logger = logging.getLogger(__name__)


class LMUSource(TelemetrySource):
    """Le Mans Ultimate telemetry from the game's built-in shared memory.
    _mmap       MMapControl  LMU shared memory map (copy access)
    _available  bool | None  whether the last read found a live session (None before the first read)
    """

    def __init__(self) -> None:
        """Prepare the LMU shared memory map.
        In:  -
        Out: None
        """
        self._mmap: MMapControl = MMapControl(LMUConstants.LMU_SHARED_MEMORY_FILE, LMUObjectOut)
        self._available: bool | None = None

    def open(self) -> None:
        """Map the LMU shared memory.
        In:  -
        Out: None
        """
        self._mmap.create(access_mode=0)

    def read(self) -> Snapshot | None:
        """Latest LMU snapshot; logs once when the game connects or disconnects.
        In:  -
        Out: Snapshot | None  None when there are no vehicles or no player telemetry
        """
        self._mmap.update()
        snapshot = self._snapshot(self._mmap.data.scoring, self._mmap.data.telemetry)
        if (snapshot is not None) != self._available:
            self._available = snapshot is not None
            logger.info("game connected" if self._available else "waiting for game")
        return snapshot

    def close(self) -> None:
        """Release the LMU shared memory.
        In:  -
        Out: None
        """
        self._mmap.close()

    def _snapshot(self, scoring: LMUScoringData, telemetry: LMUTelemetryData) -> Snapshot | None:
        """Build a snapshot from LMU scoring and telemetry, matching cars by slot id.
        In:  scoring    LMUScoringData    scoring block
             telemetry  LMUTelemetryData  telemetry block
        Out:            Snapshot | None   None when there is no player with telemetry
        """
        telem_by_id = {t.mID: t for t in telemetry.telemInfo[: telemetry.activeVehicles]}
        vehicles = tuple(
            vehicle_state(s, telem_by_id.get(s.mID, s))
            for s in scoring.vehScoringInfo[: scoring.scoringInfo.mNumVehicles]
        )
        player = next((v for v in vehicles if v.is_player), None)
        if player is None or player.slot_id not in telem_by_id:
            return None
        telem = telem_by_id[player.slot_id]
        return Snapshot(
            session=session_state(scoring.scoringInfo),
            player=player_state(
                telem, player, energy_fraction=telem.mVirtualEnergy, lap_invalidated=telem.mLapInvalidated
            ),
            vehicles=vehicles,
        )
