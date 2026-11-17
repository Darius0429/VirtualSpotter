from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RawRow:
    """Struct-like view of one row of a flattened raw-dump table, so the converters written for the game structs run
    on dump rows. Column key = names joined by dots, then indices joined by dots (mPos.x, mOri.z.0, mSectorFlag.0).
    _columns  Mapping[str, NDArray[Any]]  column name -> values of the whole table
    _row      int                         row number
    _names    tuple[str, ...]             field names followed so far
    _indices  tuple[int, ...]             array indices followed so far
    """

    def __init__(
        self,
        columns: Mapping[str, NDArray[Any]],
        row: int,
        names: tuple[str, ...] = (),
        indices: tuple[int, ...] = (),
    ) -> None:
        """Store the table, the row and the path followed so far.
        In:  columns  Mapping[str, NDArray[Any]]  column name -> values of the whole table
             row      int                         row number
             names    tuple[str, ...]             field names followed so far
             indices  tuple[int, ...]             array indices followed so far
        Out:          None
        """
        self._columns: Mapping[str, NDArray[Any]] = columns
        self._row: int = row
        self._names: tuple[str, ...] = names
        self._indices: tuple[int, ...] = indices

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Field of the struct.
        In:  name  str  field name
        Out:       Any  the row's value when the path is a column (text cut at the first null), else a deeper RawRow;
                        AttributeError when no column starts with the path
        """
        value = self._resolve((*self._names, name), self._indices)
        prefix = ".".join((*self._names, name)) + "."
        if isinstance(value, RawRow) and not any(c.startswith(prefix) for c in self._columns):
            raise AttributeError(name)
        return value

    def __getitem__(self, index: int) -> Any:  # noqa: ANN401
        """Element of the array.
        In:  index  int  array index
        Out:        Any  the row's value when the path is a column, else a deeper RawRow; IndexError past the end of
                         an array of values (ends iteration)
        """
        value = self._resolve(self._names, (*self._indices, index))
        if isinstance(value, RawRow) and not self._has_fields():
            raise IndexError(index)
        return value

    def _resolve(self, names: tuple[str, ...], indices: tuple[int, ...]) -> Any:  # noqa: ANN401
        """Value or deeper view at a path.
        In:  names    tuple[str, ...]  field names
             indices  tuple[int, ...]  array indices
        Out:          Any              Python value of the column at this row, else a RawRow at the path
        """
        column = self._columns.get(".".join((*names, *map(str, indices))))
        if column is None:
            return RawRow(self._columns, self._row, names, indices)
        value = column[self._row]
        if isinstance(value, bytes):
            return value.split(b"\0")[0]
        return value.item() if isinstance(value, np.generic) else value

    def _has_fields(self) -> bool:
        """Whether the path followed so far is an array of structs (some column continues it with a field name).
        In:  -
        Out: bool
        """
        prefix = ".".join(self._names) + "."
        return any(c.startswith(prefix) and not c[len(prefix)].isdigit() for c in self._columns)
