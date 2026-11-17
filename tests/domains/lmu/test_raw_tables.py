import numpy as np
import pyarrow as pa
import pytest
from numpy.typing import NDArray
from pyLMUSharedMemory.lmu_data import LMUObjectOut

from ai_race_engineer.domains.lmu.raw_recorder import HEADER
from ai_race_engineer.domains.lmu.raw_tables import flatten_one, safe, text, to_columns
from ai_race_engineer.domains.lmu.replay import CAR_COLUMNS, INFO_COLUMNS, PLAYER_COLUMNS, SCORING_COLUMNS

STAMP: dict[str, float] = {"poll_s": 1.5, "wall_s": 1700000000.0}


@pytest.fixture
def game() -> np.void:
    """The game's whole shared-memory struct, zeroed.
    In:  -
    Out: np.void
    """
    return np.zeros(1, dtype=np.dtype(LMUObjectOut))[0]


def test_player_table_has_every_column_the_replay_reads(game: np.void) -> None:
    """The player's telemetry keeps its field names; the header and the generic block get their key in front."""
    row = {
        **STAMP,
        "header": np.zeros(1, dtype=HEADER),
        "telem": game["telemetry"]["telemInfo"][:1].copy(),
        "generic": game["generic"].reshape(1).copy(),
    }
    columns = to_columns("raw_player_telemetry_100hz", [row])
    assert set(PLAYER_COLUMNS) <= set(columns)
    assert {"header.activeVehicles", "header.playerVehicleIdx", "header.playerHasVehicle"} <= set(columns)
    assert any(name.startswith("generic.") for name in columns)
    assert pa.Table.from_pydict(columns).num_rows == 1


def test_cars_table_has_every_column_the_replay_reads(game: np.void) -> None:
    """Every car of a poll is one row with the stamps of its poll."""
    rows = [
        {**STAMP, "telem": game["telemetry"]["telemInfo"][:3].copy()},
        {"poll_s": 2.5, "wall_s": 1700000001.0, "telem": game["telemetry"]["telemInfo"][:2].copy()},
    ]
    columns = to_columns("raw_cars_telemetry_50hz", rows)
    assert set(CAR_COLUMNS) <= set(columns)
    assert columns["poll_s"].tolist() == [1.5, 1.5, 1.5, 2.5, 2.5]
    assert pa.Table.from_pydict(columns).num_rows == 5


def test_scoring_tables_have_every_column_the_replay_reads(game: np.void) -> None:
    """The scoring info and the cars' scoring keep their field names."""
    info = to_columns("raw_scoring_info_5hz", [{**STAMP, "info": game["scoring"]["scoringInfo"].reshape(1).copy()}])
    cars = to_columns("raw_cars_scoring_5hz", [{**STAMP, "veh": game["scoring"]["vehScoringInfo"][:4].copy()}])
    assert set(INFO_COLUMNS) <= set(info)
    assert set(SCORING_COLUMNS) <= set(cars)
    assert pa.Table.from_pydict(info).num_rows == 1
    assert pa.Table.from_pydict(cars).num_rows == 4


def test_scoring_stream_is_one_text_per_row() -> None:
    """The scoring stream table holds the stamps and the text."""
    columns = to_columns("raw_scoring_stream", [{**STAMP, "text": "a"}, {"poll_s": 2.0, "wall_s": 3.0, "text": "b"}])
    assert columns == {"poll_s": [1.5, 2.0], "wall_s": [1700000000.0, 3.0], "text": ["a", "b"]}


def test_char_arrays_are_bytes_and_long_byte_arrays_one_binary_column(game: np.void) -> None:
    """Text fields are stored as null-padded bytes, number arrays as one column per element, padding as one fixed
    binary column."""
    telem = game["telemetry"]["telemInfo"][:1].copy()
    telem["mTrackName"] = _chars(b"Bahrain", 64)
    telem["mWheels"]["mTemperature"][0, 1, 2] = 351.0
    table = pa.Table.from_pydict(to_columns("raw_cars_telemetry_50hz", [{**STAMP, "telem": telem}]))
    assert table["mTrackName"][0].as_py().rstrip(b"\0") == b"Bahrain"
    assert table["mWheels.mTemperature.1.2"][0].as_py() == 351.0
    assert pa.types.is_fixed_size_binary(table["mExpansion"].type)


def test_flatten_one_gives_plain_values(game: np.void) -> None:
    """One struct becomes numbers and text cut at the first null (the paths.json of a session)."""
    paths = game["paths"].copy()
    paths["stewardResults"] = _chars(b"C:\\results.xml", 260)
    assert flatten_one(paths)["stewardResults"] == "C:\\results.xml"
    assert all(isinstance(value, str) for value in flatten_one(paths).values())


def test_text_and_safe() -> None:
    """Text ends at the first null; a folder name keeps letters, digits, dash, underscore and dot."""
    assert text(b"Bahrain\0\0junk") == "Bahrain"
    assert safe("Circuit de la Sarthe_race1_20:30/00") == "Circuit_de_la_Sarthe_race1_20_30_00"


def _chars(value: bytes, size: int) -> NDArray[np.bytes_]:
    """Text as the game stores it: one char per element, padded with nulls.
    In:  value  bytes
         size   int                 length of the char array
    Out:        NDArray[np.bytes_]  shape (size,)
    """
    return np.frombuffer(value.ljust(size, b"\0"), dtype="S1")
