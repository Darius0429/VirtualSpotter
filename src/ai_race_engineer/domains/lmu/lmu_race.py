from pathlib import Path

from ai_race_engineer.domains.lmu.lmu import LMUSource
from ai_race_engineer.domains.lmu.race_convert import (
    car_resources,
    car_tyres,
    lap_sectors,
    penalty_state,
    track_flags,
)
from ai_race_engineer.domains.common.data.race_snapshot import RaceSnapshot
from ai_race_engineer.domains.common.data.race_source import RaceSource


class LMURaceSource(LMUSource, RaceSource):
    """Le Mans Ultimate telemetry with every car's resources and the flags, from the same shared memory map."""

    def read(self) -> RaceSnapshot | None:
        """Latest LMU race snapshot.
        In:  -
        Out: RaceSnapshot | None  None when there are no vehicles or no player telemetry
        """
        snapshot = super().read()
        if snapshot is None:
            return None
        scoring, telemetry = self._mmap.data.scoring, self._mmap.data.telemetry
        telem_by_id = {t.mID: t for t in telemetry.telemInfo[: telemetry.activeVehicles]}
        entries = scoring.vehScoringInfo[: scoring.scoringInfo.mNumVehicles]
        player = next(s for s in entries if s.mIsPlayer)
        return RaceSnapshot(
            session=snapshot.session,
            player=snapshot.player,
            vehicles=snapshot.vehicles,
            resources=tuple(
                car_resources(s, telem_by_id[s.mID], telem_by_id[s.mID].mVirtualEnergy)
                for s in entries
                if s.mID in telem_by_id
            ),
            flags=track_flags(scoring.scoringInfo, player),
            sectors=tuple(lap_sectors(s) for s in entries),
            time_remaining_s=scoring.scoringInfo.mSessionTimeRemaining,
            tyres=tuple(car_tyres(s, telem_by_id[s.mID]) for s in entries if s.mID in telem_by_id),
            penalties=penalty_state(scoring.scoringInfo, player, telem_by_id[player.mID]),
        )

    def results_path(self) -> Path | None:
        """Results file the game points to in shared memory (stewardResults).
        In:  -
        Out: Path | None  None while the game gives no path
        """
        text = self._mmap.data.paths.stewardResults.decode("utf-8", errors="replace").rstrip("\x00").strip()
        return Path(text) if text else None
