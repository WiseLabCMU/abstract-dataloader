"""Data augmentation specifications.

!!! abstract "Programming Model"

    - Data augmentations consist of a set of properties which correspond to
        physical properties of the data, such as `azimuth_flip` or
        `range_scale`.
    - Multiple data augmentation specifications can be combined into a single
        set of `Augmentations`, which generates a dictionary of values.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, overload

import numpy as np

from abstract_dataloader import spec

from .types import dataclass

T = TypeVar("T")
TNew = TypeVar("TNew")


@dataclass
class Augmented(Generic[T]):
    """Data with augmentations applied.

    Attributes:
        data: wrapped data.
        augmentations: remaining augmentations which have not been applied.
    """

    data: T
    augmentations: dict[str, Any]

    def pop(self, key: str, default: Any) -> Any:
        """Pop an augmentation by key."""
        return self.augmentations.pop(key, default)

    def filter(self, *keys: str) -> "Augmented[T]":
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

    def unbox(self) -> T:
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


class Augmentation(ABC, Generic[T]):
    """Data augmentation random generation policy."""

    def __init__(self) -> None:
        self._rng = {}

    @property
    def rng(self) -> np.random.Generator:
        """Random number generator, using the PID as a seed."""
        pid = os.getpid()
        if pid in self._rng:
            return self._rng[pid]
        else:
            rng = np.random.default_rng(seed=pid)
            self._rng[pid] = rng
            return rng

    @abstractmethod
    def __call__(self) -> T:
        """Sample the value of a data augmentation parameter."""
        ...


class Augment(spec.Transform[dict[str, Any], dict[str, Augmented[Any]]]):
    """Collection of data augmentations.

    Args:
        kwargs: augmentations to apply, where each key is the
            name of a physical property (e.g., `azimuth_flip`, `range_scale`)
            and each value is the corresponding augmentation generator.
    """

    def __init__(self, **kwargs: Augmentation) -> None:
        self.augmentations = kwargs

    def sample(self) -> dict[str, Any]:
        """Generate a dictionary of augmentations."""
        return {k: v() for k, v in self.augmentations.items()}

    def __call__(
        self, data: dict[str, Any], train: bool = False
    ) -> dict[str, Augmented[Any]]:
        """Apply data augmentations to a collection of multimodal data."""
        if train:
            aug = self.sample()
        else:
            aug = {}

        return {
            k: Augmented(data=v, augmentations=aug.copy())
            for k, v in data.items()
        }


TSample = TypeVar("TSample", bound=dict[str, Any])

class Unbox:
    """Alias for `.unbox`."""

    @overload
    def __call__(self, data: Augmented[TSample]) -> TSample: ...

    @overload
    def __call__(self, data: dict[str, Augmented]) -> dict[str, Any]: ...

    def __call__(
        self, data: Augmented[TSample] | dict[str, Augmented[Any]],
        train: bool = False
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

    def __call__(
        self, data: Augmented[TRaw], train: bool = False
    ) -> Augmented[TTransformed]:
        return Augmented(
            data=self.transform(data.data, train=train),
            augmentations=data.augmentations)


class Bernoulli(Augmentation[bool]):
    """Enable augmentation with certain probability.

    Type: `bool` (`True` if enabled).

    Args:
        p: probability of enabling.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def __call__(self) -> bool:
        return self.rng.random() < self.p


class TruncatedLogNormal(Augmentation[float]):
    """Truncated log-normal distribution.

    The underlying normal is always centered around zero.

    Type: `float`; returns `1.0` if not enabled.

    Args:
        p: probability of enabling this augmentation (`True`).
        std: standard deviation of the underlying normal distribution.
        clip: clip to this many standard deviations; don't clip if 0.
    """

    def __init__(
        self, p: float = 1.0, std: float = 0.2, clip: float = 2.0
    ) -> None:
        super().__init__()
        self.p = p
        self.std = std
        self.clip = clip

    def __call__(self) -> float:
        if self.rng.random() > self.p:
            return 1.0

        z = self.rng.normal()
        if self.clip > 0:
            z = np.clip(z, -self.clip, self.clip)
        return np.exp(z * self.std)


class Uniform(Augmentation[float]):
    """Uniform distribution.

    Type: `float`; returns `0.0` if not enabled.

    Args:
        p: probability of enabling this augmentation.
        lower: lower bound.
        upper: upper bound.
    """

    def __init__(
        self, p: float = 1.0, lower: float = -np.pi, upper: float = np.pi
    ) -> None:
        super().__init__()
        self.p = p
        self.lower = lower
        self.upper = upper

    def __call__(self) -> float:
        if self.rng.random() > self.p:
            return 0.0

        return self.rng.uniform(self.lower, self.upper)
