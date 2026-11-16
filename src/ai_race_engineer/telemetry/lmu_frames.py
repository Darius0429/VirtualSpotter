import logging

from pyLMUSharedMemory.lmu_data import LMUConstants, LMUObjectOut, LMUScoringData, LMUTelemetryData
from pyLMUSharedMemory.lmu_mmap import MMapControl

from ai_race_engineer.telemetry.frame import Frame, Scoring
from ai_race_engineer.telemetry.frame_convert import car_sample, car_status, player_sample, session_info, state_sample
from ai_race_engineer.telemetry.frame_source import FrameSource

logger = logging.getLogger(__name__)


class LMUFrameSource(FrameSource):
    """Le Mans Ultimate frames from the game's built-in shared memory, one per player telemetry tick.
    _mmap            MMapControl   LMU shared memory map (copy access, independent of the live source)
    _available       bool | None   whether the last read found a live session (None before the first read)
    _last_telem_s    float | None  s player telemetry clock of the last frame
    _last_scoring_s  float | None  s scoring clock of the last scoring update
    """

    def __init__(self) -> None:
        """Prepare the LMU shared memory map.
        In:  -
        Out: None
        """
        self._mmap: MMapControl = MMapControl(LMUConstants.LMU_SHARED_MEMORY_FILE, LMUObjectOut)
        self._available: bool | None = None
        self._last_telem_s: float | None = None
        self._last_scoring_s: float | None = None

    def open(self) -> None:
        """Map the LMU shared memory.
        In:  -
        Out: None
        """
        self._mmap.create(access_mode=0)

    def read_new(self) -> Frame | None:
        """Frame of a new LMU telemetry tick; logs once when the game connects or disconnects.
        In:  -
        Out: Frame | None  None when the clock did not advance or there is no player with telemetry
        """
        self._mmap.update()
        return self._frame(self._mmap.data.scoring, self._mmap.data.telemetry)

    def close(self) -> None:
        """Release the LMU shared memory.
        In:  -
        Out: None
        """
        self._mmap.close()

    def _frame(self, scoring: LMUScoringData, telemetry: LMUTelemetryData) -> Frame | None:
        """Build a frame from LMU scoring and telemetry, matching cars by slot id.
        In:  scoring    LMUScoringData    scoring block
             telemetry  LMUTelemetryData  telemetry block
        Out:            Frame | None      None when the clock did not advance or there is no player with telemetry
        """
        telem_by_id = {t.mID: t for t in telemetry.telemInfo[: telemetry.activeVehicles]}
        cars = scoring.vehScoringInfo[: scoring.scoringInfo.mNumVehicles]
        player = next((s for s in cars if s.mIsPlayer and s.mID in telem_by_id), None)
        if (player is not None) != self._available:
            self._available = player is not None
            logger.info("game connected" if self._available else "waiting for game")
        if player is None or telem_by_id[player.mID].mElapsedTime == self._last_telem_s:
            return None
        telem, info = telem_by_id[player.mID], scoring.scoringInfo
        time_s = self._last_telem_s = telem.mElapsedTime
        new_scoring = info.mCurrentET != self._last_scoring_s
        self._last_scoring_s = info.mCurrentET
        return Frame(
            session=session_info(info, player, "lmu"),
            player=player_sample(telem, player),
            cars=tuple(car_sample(time_s, s, telem_by_id.get(s.mID, s)) for s in cars),
            scoring=Scoring(
                state=state_sample(time_s, info, telem, player, telem.mVirtualEnergy, telem.mLapInvalidated),
                cars=tuple(car_status(time_s, s) for s in cars),
            )
            if new_scoring
            else None,
        )
