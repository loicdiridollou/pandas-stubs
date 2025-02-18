from collections.abc import (
    Hashable,
    Iterable,
)
from typing import Literal

import numpy as np
from pandas.core import accessor
from pandas.core.indexes.base import Index
from pandas.core.indexes.extension import ExtensionIndex
from typing_extensions import Self

from pandas._typing import (
    S1,
    DtypeArg,
)

class CategoricalIndex(ExtensionIndex[S1], accessor.PandasDelegate):
    codes: np.ndarray = ...
    categories: Index = ...
    def __new__(
        cls,
        data: Iterable[S1] = ...,
        categories=...,
        ordered=...,
        dtype=...,
        copy: bool = ...,
        name: Hashable = ...,
    ) -> Self: ...
    def equals(self, other): ...
    @property
    def inferred_type(self) -> str: ...
    @property
    def values(self): ...
    def __contains__(self, key) -> bool: ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    def astype(self, dtype: DtypeArg, copy: bool = ...) -> Index: ...
    def fillna(self, value=..., downcast=...): ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    def unique(self, level=...): ...
    def duplicated(self, keep: Literal["first", "last", False] = ...): ...
    def where(self, cond, other=...): ...
    def reindex(self, target, method=..., level=..., limit=..., tolerance=...): ...
    def get_indexer(self, target, method=..., limit=..., tolerance=...): ...
    def get_indexer_non_unique(self, target): ...
    def delete(self, loc): ...
    def insert(self, loc, item): ...
