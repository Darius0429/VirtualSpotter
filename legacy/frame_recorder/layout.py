import re
from datetime import datetime
from pathlib import Path

from ai_race_engineer.domains.common.analysis.pace.lap_table import LapRecord
from ai_race_engineer.domains.common.data.frame import CarSample, CarStatus, PlayerSample, SessionInfo, StateSample

TABLES: dict[str, type] = {
    "player_telemetry_100hz": PlayerSample,
    "cars_positions_50hz": CarSample,
    "player_state_5hz": StateSample,
    "cars_state_5hz": CarStatus,
    "laps": LapRecord,
}
SESSION_FILE: str = "session.yaml"


def session_dir(root: Path, started: datetime, info: SessionInfo) -> Path:
    """Folder of one recorded session, with filesystem-unsafe characters replaced by '_'.
    In:  root     Path         root of the session folders
         started  datetime     local wall-clock start of the recording
         info     SessionInfo
    Out:          Path         root/<YYYYmmdd_HHMMSS>_<track>_<car>_<session type>
    """
    name = f"{started:%Y%m%d_%H%M%S}_{info.track_name}_{info.vehicle_name}_{info.session_type.name.lower()}"
    return root / re.sub(r"[^\w.\-#]+", "_", name)
