"""Data type system framework following the [ADL recommendations](./types.md).

!!! abstract "Programming Model"

    - Types are dataclasses which are registered to the global optree namespace
        using the provided [`dataclass`][.] decorator.
    - Outer data types are [`Timestamped`][.] with a leading batch timestamp
        field.
    - Data types can be [`Augmented`][.]. Each specified augmentation should
        be applied by some step in the data processing pipeline.

!!! warning

    This module requires `optree` to be installed.
"""

from collections.abc import Sequence
from dataclasses import field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from jaxtyping import Float
from optree.dataclasses import dataclass as _optree_dataclass
from typing_extensions import dataclass_transform

TArray = TypeVar("TArray")

@dataclass_transform(field_specifiers=(field,))
def dataclass(cls):  # noqa: D103
    """A dataclass decorator which registers into optree's global namespace."""
    return _optree_dataclass(cls, namespace='')


@runtime_checkable
class ArrayLike(Protocol):
    """Array with shape and dtype, e.g., `torch.Tensor`, `jax.Array`."""

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


TData = TypeVar("TData")

@dataclass
class Augmented(Generic[TData]):
    """Data with augmentations applied.

    Attributes:
        data: wrapped data.
        augmentations: remaining augmentations which have not been applied.
    """

    data: TData
    augmentations: dict[str, Any]

    def pop(self, key: str) -> Any:
        """Pop an augmentation by key."""
        return self.augmentations.pop(key, None)

    def filter(self, keys: Sequence[str]) -> "Augmented[TData]":
        """Filter augmentations by keys.

        Args:
            keys: keys to keep, i.e., augmentations which are relevant to this
                pipeline.

        Returns:
            Filtered `Augmented` wrapper.
        """
        filtered = {k: v for k, v in self.augmentations.items() if k in keys}
        return Augmented(data=self.data, augmentations=filtered)

    def unbox(self) -> TData:
        """Unbox the data, returning the original data type.

        !!! warning

            If any un-applied augmentations are remaining, a `ValueError` is
            raised. Augmentations which are not relevant to a particular data
            type should be explicitly removed.
        """
        return self.data
