"""Abstract specifications for data augmentations.

!!! abstract "Programming Model"

    - Data augmentations consist of a set of properties which correspond to
        common properties of the data, such as `azimuth_flip` or `range_scale`.
    - The user is responsible for defining and keeping track of each
        augmentation property and its meaning.
    - Multiple data augmentation specifications can be combined into a single
        set of `Augmentations`, which generates a dictionary of values.

In addition to the general framework, we include wrappers for a few common
distributions:

- [`Bernoulli`][.]: `Bernoulli(p)`
- [`Normal`][.]: `Normal(0, std) * Bernoulli(p)`
- [`TruncatedLogNormal`][.]:
    `exp(clamp(Normal(0, std), -clip, clip)) * Bernoulli(p))`
- [`Uniform`][.]: `Unif(lower, upper) * Bernoulli(p)`
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np

T = TypeVar("T")

class Augmentation(ABC, Generic[T]):
    """A generic augmentation random generation policy."""

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


class Augmentations:
    """A collection of data augmentations.

    Args:
        kwargs: augmentations to apply, where each key is the
            name of a physical property (e.g., `azimuth_flip`, `range_scale`)
            and each value is the corresponding augmentation generator.
    """

    def __init__(self, **kwargs: Augmentation) -> None:
        self.augmentations = kwargs

    def __call__(self, meta: dict[str, Any] = {}) -> dict[str, Any]:
        """Generate a dictionary of augmentations.

        If a `train=False` flag is passed in `meta`, no augmentations are
        generated.

        Args:
            meta: data processing configuration inputs.

        Returns:
            Augmentation specifications.
        """
        if meta.get("train", True):
            return {k: v() for k, v in self.augmentations.items()}
        else:
            return {}


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


class Normal(Augmentation[float]):
    """Normal distribution.

    Type: `float`; returns `0.0` if not enabled. Always zero-centered.

    Args:
        p: probability of enabling this augmentation (`True`).
        std: standard deviation of the normal distribution.
    """

    def __init__(self, p: float = 1.0, std: float = 1.0) -> None:
        super().__init__()
        self.p = p
        self.std = std

    def __call__(self) -> float:
        if self.p < 1.0 and self.rng.random() > self.p:
            return 0.0

        return self.rng.normal(scale=self.std)


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
        if self.p < 1.0 and self.rng.random() > self.p:
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
        if self.p < 1.0 and self.rng.random() > self.p:
            return 0.0

        return self.rng.uniform(self.lower, self.upper)
