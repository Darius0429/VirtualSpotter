import logging

from pyRfactor2SharedMemory.rf2_data import rF2Scoring, rF2Telemetry, rFactor2Constants
from pyRfactor2SharedMemory.rf2_mmap import MMapControl

from ai_race_engineer.domains.common.data.base import TelemetrySource
from ai_race_engineer.domains.lmu.convert import player_state, session_state, vehicle_state
from ai_race_engineer.domains.common.data.snapshot import Snapshot

logger = logging.getLogger(__name__)


class RF2Source(TelemetrySource):
    """rFactor 2 telemetry from The Iron Wolf's shared memory plugin.
    _scoring    MMapControl  scoring map (copy access)
    _telemetry  MMapControl  telemetry map (copy access)
    _available  bool | None  whether the last read found a live session (None before the first read)
    """

    def __init__(self) -> None:
        """Prepare the rF2 scoring and telemetry maps.
        In:  -
        Out: None
        """
        self._scoring: MMapControl = MMapControl(rFactor2Constants.MM_SCORING_FILE_NAME, rF2Scoring)
        self._telemetry: MMapControl = MMapControl(rFactor2Constants.MM_TELEMETRY_FILE_NAME, rF2Telemetry)
        self._available: bool | None = None

    def open(self) -> None:
        """Map the rF2 scoring and telemetry shared memory.
        In:  -
        Out: None
        """
        self._scoring.create(access_mode=0)
        self._telemetry.create(access_mode=0)

    def read(self) -> Snapshot | None:
        """Latest rF2 snapshot; logs once when the game connects or disconnects.
        In:  -
        Out: Snapshot | None  None when there are no vehicles or no player telemetry (also when the plugin is missing)
        """
        self._scoring.update()
        self._telemetry.update()
        snapshot = self._snapshot(self._scoring.data, self._telemetry.data)
        if (snapshot is not None) != self._available:
            self._available = snapshot is not None
            logger.info("game connected" if self._available else "waiting for game")
        return snapshot

    def close(self) -> None:
        """Release the rF2 shared memory.
        In:  -
        Out: None
        """
        self._scoring.close()
        self._telemetry.close()

    def _snapshot(self, scoring: rF2Scoring, telemetry: rF2Telemetry) -> Snapshot | None:
        """Build a snapshot from rF2 scoring and telemetry, matching cars by slot id.
        In:  scoring    rF2Scoring       scoring buffer
             telemetry  rF2Telemetry     telemetry buffer
        Out:            Snapshot | None  None when there is no player with telemetry
        """
        telem_by_id = {t.mID: t for t in telemetry.mVehicles[: telemetry.mNumVehicles]}
        vehicles = tuple(
            vehicle_state(s, telem_by_id.get(s.mID, s)) for s in scoring.mVehicles[: scoring.mScoringInfo.mNumVehicles]
        )
        player = next((v for v in vehicles if v.is_player), None)
        if player is None or player.slot_id not in telem_by_id:
            return None
        return Snapshot(
            session=session_state(scoring.mScoringInfo),
            player=player_state(telem_by_id[player.slot_id], player, energy_fraction=None, lap_invalidated=None),
            vehicles=vehicles,
        )
