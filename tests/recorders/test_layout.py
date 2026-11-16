from datetime import datetime
from pathlib import Path

from ai_race_engineer.recorders.layout import session_dir
from ai_race_engineer.telemetry.frame import SessionInfo
from ai_race_engineer.telemetry.snapshot import SessionType


def _info(track_name: str, vehicle_name: str) -> SessionInfo:
    """Race session info with the given names.
    In:  track_name    str
         vehicle_name  str
    Out:               SessionInfo
    """
    return SessionInfo("lmu", track_name, 13626.0, vehicle_name, "Hyper", SessionType.RACE, 10, 0)


def test_folder_name_format() -> None:
    """Folder is root/<YYYYmmdd_HHMMSS>_<track>_<car>_<session type>."""
    folder = session_dir(Path("root"), datetime(2025, 3, 2, 16, 34, 34), _info("Sebring", "Car7"))
    assert folder == Path("root") / "20250302_163434_Sebring_Car7_race"


def test_unsafe_characters_replaced() -> None:
    """Slashes, colons, quotes and spaces are replaced."""
    folder = session_dir(Path("root"), datetime(2025, 3, 2), _info("Circuit de la Sarthe", 'D\'St "A" #7:LM/x'))
    assert folder.name == "20250302_000000_Circuit_de_la_Sarthe_D_St_A_#7_LM_x_race"
