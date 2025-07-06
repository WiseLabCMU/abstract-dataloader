"""Data type system framework following the [ADL recommendations](./types.md).

!!! abstract "Programming Model"

    - Types are dataclasses which are registered to the global optree namespace
        using the provided [`dataclass`][.] decorator.
    - Outer data types are [`Timestamped`][.] with a leading batch timestamp
        field.

!!! warning

    This module requires `optree` to be installed.
"""

from dataclasses import field
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from jaxtyping import Float
from optree.dataclasses import dataclass as _optree_dataclass
from typing_extensions import dataclass_transform


@dataclass_transform(field_specifiers=(field,))
def dataclass(cls):  # noqa: D103
    """A dataclass decorator which registers into optree's global namespace."""
    return _optree_dataclass(cls, namespace='')


@runtime_checkable
class ArrayLike(Protocol):
    """Array with shape and dtype, e.g., `torch.Tensor`, `jax.Array`.

    Use this type to specify arbitrary array types.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...


TArray = TypeVar("TArray", bound=ArrayLike)


@runtime_checkable
class Timestamped(Protocol, Generic[TArray]):
    """Types which have a timestamp.

    Attributes:
        timestamps: data timestamps; batch must be the leading axis, but the
            `timestamps` can have an extra `window` axis if each sample
            represents a sequence of frames.
    """

    timestamps: Float[TArray, "batch *window"]
