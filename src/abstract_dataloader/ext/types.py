"""Data type system framework following the [ADL recommendations](./types.md).

!!! abstract "Programming Model"

    - Samples are single-layer dictionaries of sensor names and sensor values.
    - Types are dataclasses which are registered to the global optree namespace
        using the provided [`dataclass`][.] decorator.
    - Outer data types are [`Timestamped`][.] with a leading batch timestamp
        field.
    - Data types can be [`Augmented`][.]. Each specified augmentation should
        be applied by some step in the data processing pipeline.

!!! warning

    This module requires `optree` to be installed.
"""

from dataclasses import field
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

from jaxtyping import Float
from optree.dataclasses import dataclass as _optree_dataclass
from typing_extensions import dataclass_transform

from abstract_dataloader import spec

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
TNew = TypeVar("TNew")

@dataclass
class Augmented(Generic[TData]):
    """Data with augmentations applied.

    Attributes:
        data: wrapped data.
        augmentations: remaining augmentations which have not been applied.
    """

    data: TData
    augmentations: dict[str, Any]

    def pop(self, key: str, default: Any) -> Any:
        """Pop an augmentation by key."""
        return self.augmentations.pop(key, default)

    def filter(self, *keys: str) -> "Augmented[TData]":
        """Filter augmentations by keys.

        Args:
            keys: keys to keep, i.e., augmentations which are relevant to this
                pipeline.

        Returns:
            Filtered `Augmented` wrapper.
        """
        filtered = {
            k: v for k, v in self.augmentations.items() if k in set(keys)}
        return Augmented(data=self.data, augmentations=filtered)

    def unbox(self) -> TData:
        """Unbox the data, returning the original data type.

        !!! warning

            If any un-applied augmentations are remaining, a `ValueError` is
            raised. Augmentations which are not relevant to a particular data
            type should be explicitly removed.
        """
        if len(self.augmentations) > 0:
            raise ValueError(
                f"Augmented[{type(self.data).__name__}] still has "
                f"un-applied augmentations: {list(self.augmentations.keys())}")
        return self.data

    def update(self, data: TNew) -> "Augmented[TNew]":
        """Update the data with a new value, keeping the augmentations."""
        return Augmented(data=data, augmentations=self.augmentations)


TSample = TypeVar("TSample", bound=dict[str, Any])

class Augment(Generic[TSample]):
    """Apply augmentations to multimodal data.

    Args:
        **kwargs: augmentations, where each key is the name of the augmented
            property, and the value is a callable which samples the value of
            that property.
    """

    def __init__(self, **kwargs: Callable[[], Any]) -> None:
        self.augmentations = kwargs

    def __call__(self, data: TSample) -> dict[str, Augmented[Any]]:
        """Apply augmentations to a collection of multimodal data."""
        aug = {k: v() for k, v in self.augmentations.items()}

        return {
            k: Augmented(data=v, augmentations=aug.copy())
            for k, v in data.items()
        }


TAugmented = TypeVar("TAugmented", bound=dict[str, Augmented[Any]])

class Unbox:
    """Alias for `.unbox`."""

    @overload
    def __call__(self, data: Augmented[TSample]) -> TSample: ...

    @overload
    def __call__(self, data: dict[str, Augmented]) -> dict[str, Any]: ...

    def __call__(
        self, data: Augmented[TSample] | dict[str, Augmented[Any]]
    ) -> TSample | dict[str, Any]:
        """Unbox and raise if there are unapplied values."""
        if isinstance(data, dict):
            return {k: v.unbox() for k, v in data.items()}
        else:
            return data.unbox()


TRaw = TypeVar("TRaw")
TTransformed = TypeVar("TTransformed")

class IgnoreAugmentations(
    spec.Transform[Augmented[TRaw], Augmented[TTransformed]]
):
    """Pass through to a transform which does not use augmentations."""

    def __init__(self, transform: spec.Transform[TRaw, TTransformed]) -> None:
        self.transform = transform

    def __call__(self, data: Augmented[TRaw]) -> Augmented[TTransformed]:
        return Augmented(
            data=self.transform(data.data),
            augmentations=data.augmentations)
