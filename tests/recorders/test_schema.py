from dataclasses import dataclass

import pyarrow as pa

from ai_race_engineer.recorders.schema import schema_of


@dataclass(frozen=True)
class _Row:
    """Row type covering every supported field type.
    speed  float
    laps   int | None
    pit    bool
    name   str | None
    """

    speed: float
    laps: int | None
    pit: bool
    name: str | None


def test_type_mapping_and_optional() -> None:
    """float/int/bool/str map to float64/int64/bool/string, '| None' like the base type, all nullable."""
    schema = schema_of(_Row)
    assert schema.types == [pa.float64(), pa.int64(), pa.bool_(), pa.string()]
    assert all(field.nullable for field in schema)


def test_field_order() -> None:
    """Columns are in field order."""
    assert schema_of(_Row).names == ["speed", "laps", "pit", "name"]
