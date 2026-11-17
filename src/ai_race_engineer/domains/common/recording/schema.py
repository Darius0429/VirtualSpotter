import dataclasses
import types
import typing

import pyarrow as pa

_ARROW_TYPES: dict[type, pa.DataType] = {float: pa.float64(), int: pa.int64(), bool: pa.bool_(), str: pa.string()}


def schema_of(row_type: type) -> pa.Schema:
    """Arrow schema of a row dataclass: one nullable column per field, in field order.
    In:  row_type  type       dataclass with float/int/bool/str fields, optionally '| None'
    Out:           pa.Schema
    """
    hints = typing.get_type_hints(row_type)
    return pa.schema([pa.field(f.name, _arrow_type(hints[f.name])) for f in dataclasses.fields(row_type)])


def _arrow_type(hint: object) -> pa.DataType:
    """Arrow type of one field type hint; 'X | None' maps like X.
    In:  hint  object       float, int, bool, str or a union of one of them with None
    Out:       pa.DataType
    """
    if isinstance(hint, types.UnionType):
        hint = next(arg for arg in typing.get_args(hint) if arg is not types.NoneType)
    return _ARROW_TYPES[hint]
