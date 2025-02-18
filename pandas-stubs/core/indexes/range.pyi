from collections.abc import (
    Hashable,
    Sequence,
)
from typing import overload

import numpy as np
from pandas.core.indexes.base import Index

from pandas._typing import (
    HashableT,
    MaskType,
    np_ndarray_anyint,
    npt,
)

class RangeIndex(Index[int]):
    def __new__(
        cls,
        start: int | RangeIndex | range = ...,
        stop: int = ...,
        step: int = ...,
        dtype=...,
        copy: bool = ...,
        name: Hashable = ...,
    ): ...
    @classmethod
    def from_range(cls, data, name: Hashable = ..., dtype=...): ...
    def __reduce__(self): ...
    @property
    def start(self) -> int: ...
    @property
    def stop(self) -> int: ...
    @property
    def step(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @property
    def has_duplicates(self) -> bool: ...
    def __contains__(self, key: int | np.integer) -> bool: ...
    def get_indexer(self, target, method=..., limit=..., tolerance=...): ...
    def tolist(self): ...
    def copy(self, name: Hashable = ..., deep: bool = ..., dtype=..., **kwargs): ...
    def min(self, axis=..., skipna: bool = ..., *args, **kwargs): ...
    def max(self, axis=..., skipna: bool = ..., *args, **kwargs): ...
    def argsort(self, *args, **kwargs): ...
    def factorize(
        self, sort: bool = ..., use_na_sentinel: bool = ...
    ) -> tuple[npt.NDArray[np.intp], RangeIndex]: ...
    def equals(self, other): ...
    def join(
        self,
        other,
        *,
        how: str = ...,
        level=...,
        return_indexers: bool = ...,
        sort: bool = ...,
    ): ...
    def __len__(self) -> int: ...
    @property
    def size(self) -> int: ...
    def __floordiv__(self, other): ...
    def all(self) -> bool: ...
    def any(self) -> bool: ...
    def union(
        self, other: list[HashableT] | Index, sort=...
    ) -> Index | Index[int] | RangeIndex: ...
    @overload  # type: ignore[override]
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index | MaskType,
    ) -> Index: ...
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, idx: int
    ) -> int: ...
