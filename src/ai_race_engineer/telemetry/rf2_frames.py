import logging

from pyRfactor2SharedMemory.rf2_data import rF2Scoring, rF2Telemetry, rFactor2Constants
from pyRfactor2SharedMemory.rf2_mmap import MMapControl

from ai_race_engineer.telemetry.frame import Frame, Scoring
from ai_race_engineer.telemetry.frame_convert import car_sample, car_status, player_sample, session_info, state_sample
from ai_race_engineer.telemetry.frame_source import FrameSource

logger = logging.getLogger(__name__)


class RF2FrameSource(FrameSource):
    """rFactor 2 frames from The Iron Wolf's shared memory plugin, one per player telemetry tick.
    _scoring         MMapControl   scoring map (copy access, independent of the live source)
    _telemetry       MMapControl   telemetry map (copy access)
    _available       bool | None   whether the last read found a live session (None before the first read)
    _last_telem_s    float | None  s player telemetry clock of the last frame
    _last_scoring_s  float | None  s scoring clock of the last scoring update
    """

    def __init__(self) -> None:
        """Prepare the rF2 scoring and telemetry maps.
        In:  -
        Out: None
        """
        self._scoring: MMapControl = MMapControl(rFactor2Constants.MM_SCORING_FILE_NAME, rF2Scoring)
        self._telemetry: MMapControl = MMapControl(rFactor2Constants.MM_TELEMETRY_FILE_NAME, rF2Telemetry)
        self._available: bool | None = None
        self._last_telem_s: float | None = None
        self._last_scoring_s: float | None = None

    def open(self) -> None:
        """Map the rF2 scoring and telemetry shared memory.
        In:  -
        Out: None
        """
        self._scoring.create(access_mode=0)
        self._telemetry.create(access_mode=0)

    def read_new(self) -> Frame | None:
        """Frame of a new rF2 telemetry tick; logs once when the game connects or disconnects.
        In:  -
        Out: Frame | None  None when the clock did not advance or there is no player with telemetry
        """
        self._scoring.update()
        self._telemetry.update()
        return self._frame(self._scoring.data, self._telemetry.data)

    def close(self) -> None:
        """Release the rF2 shared memory.
        In:  -
        Out: None
        """
        self._scoring.close()
        self._telemetry.close()

    def _frame(self, scoring: rF2Scoring, telemetry: rF2Telemetry) -> Frame | None:
        """Build a frame from rF2 scoring and telemetry, matching cars by slot id.
        In:  scoring    rF2Scoring    scoring buffer
             telemetry  rF2Telemetry  telemetry buffer
        Out:            Frame | None  None when the clock did not advance or there is no player with telemetry
        """
        telem_by_id = {t.mID: t for t in telemetry.mVehicles[: telemetry.mNumVehicles]}
        cars = scoring.mVehicles[: scoring.mScoringInfo.mNumVehicles]
        player = next((s for s in cars if s.mIsPlayer and s.mID in telem_by_id), None)
        if (player is not None) != self._available:
            self._available = player is not None
            logger.info("game connected" if self._available else "waiting for game")
        if player is None or telem_by_id[player.mID].mElapsedTime == self._last_telem_s:
            return None
        telem, info = telem_by_id[player.mID], scoring.mScoringInfo
        time_s = self._last_telem_s = telem.mElapsedTime
        new_scoring = info.mCurrentET != self._last_scoring_s
        self._last_scoring_s = info.mCurrentET
        return Frame(
            session=session_info(info, player, "rf2"),
            player=player_sample(telem, player),
            cars=tuple(car_sample(time_s, s, telem_by_id.get(s.mID, s)) for s in cars),
            scoring=Scoring(
                state=state_sample(time_s, info, telem, player, None, None),
                cars=tuple(car_status(time_s, s) for s in cars),
            )
            if new_scoring
            else None,
        )
