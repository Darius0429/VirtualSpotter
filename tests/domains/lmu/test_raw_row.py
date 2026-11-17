import numpy as np
import pytest

from ai_race_engineer.domains.lmu.raw_row import RawRow

COLUMNS: dict[str, np.ndarray] = {
    "mID": np.array([3, 7], dtype=np.int32),
    "mPos.x": np.array([1.5, 2.5]),
    "mOri.z.0": np.array([0.1, 0.2]),
    "mSectorFlag.0": np.array([11, 1], dtype=np.uint8),
    "mWheels.mTemperature.0.0": np.array([300.0, 310.0]),
    "mWheels.mTemperature.0.1": np.array([301.0, 311.0]),
    "mDriverName": np.array([b"Ann\x00junk", b"Bob"], dtype=object),
}


def test_values_by_struct_path() -> None:
    """Fields, arrays of values and arrays of structs resolve to the row's value as a Python value."""
    row = RawRow(COLUMNS, 1)
    assert (row.mID, type(row.mID)) == (7, int)
    assert row.mPos.x == 2.5
    assert row.mOri[0].z == 0.2
    assert row.mSectorFlag[0] == 1
    assert row.mDriverName == b"Bob"
    assert RawRow(COLUMNS, 0).mDriverName == b"Ann"


def test_value_array_ends_with_index_error() -> None:
    """Iterating an array of values stops after its last column."""
    assert list(RawRow(COLUMNS, 0).mWheels[0].mTemperature) == [300.0, 301.0]


def test_unknown_name_raises() -> None:
    """A name that is no column and starts no column is an error, not a deeper view."""
    row = RawRow(COLUMNS, 0)
    with pytest.raises(AttributeError):
        _ = row.mFuel
    with pytest.raises(AttributeError):
        _ = row.mPos.w
