"""Objective base classes and specifications.

!!! abstract "Programming Model"

    - An [`Objective`][.] is a callable which returns a (batched) scalar loss
      and a dictionary of metrics.
    - Objectives can be combined into a higher-order objective which combines
      their losses and aggregates their metrics ([`MultiObjective`][.]);
      specify these objectives using a [`MultiObjectiveSpec`][.].
"""

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
from jaxtyping import Float, UInt8
from typing_extensions import TypeVar


@runtime_checkable
class ArrayLike(Protocol):
    """Array with shape and dtype, e.g., `torch.Tensor`, `jax.Array`."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...


TArray = TypeVar("TArray", bound=ArrayLike)
YTrue = TypeVar("YTrue", infer_variance=True)
YPred = TypeVar("YPred", infer_variance=True)


@runtime_checkable
class Objective(Protocol, Generic[TArray, YTrue, YPred]):
    """Composable training objective.

    !!! note

        Metrics should use `torch.no_grad()` to make sure gradients are not
        computed for non-loss metrics!

    Type Parameters:
        - `YTrue`: ground truth data type.
        - `YPred`: model output data type.
    """

    @abstractmethod
    def __call__(
        self, y_true: YTrue, y_pred: YPred, train: bool = True
    ) -> tuple[Float[TArray, "batch"], dict[str, Float[TArray, "batch"]]]:
        """Training metrics implementation.

        Args:
            y_true: Data channels (i.e. dataloader output).
            y_pred: Model outputs.
            train: Whether in training mode (i.e. skip expensive metrics).

        Returns:
            A tuple containing the loss and a dict of metric values.
        """
        ...

    def visualizations(
        self, y_true: YTrue, y_pred: YPred
    ) -> dict[str, UInt8[np.ndarray, "H W 3"]]:
        """Generate visualizations for each entry in a batch.

        !!! note

            This method should be called only from a "detached" CPU thread so
            as not to affect training throughput; the caller is responsible for
            detaching gradients and sending the data to the CPU. As such,
            implementations are free to use CPU-specific methods.

        Args:
            y_true: Data channels (i.e., dataloader output).
            y_pred: Model outputs.

        Returns:
            A dict, where each key is the name of a visualization, and the
                value is a stack of RGB images in HWC order, detached from
                Torch and sent to a numpy array.
        """
        ...


YTrueAll = TypeVar("YTrueAll", infer_variance=True)
YPredAll = TypeVar("YPredAll", infer_variance=True)


@dataclass
class MultiObjectiveSpec(Generic[YTrue, YPred, YTrueAll, YPredAll]):
    """Specification for a single objective in a multi-objective setup.

    The inputs and outputs for each objective are specified using `y_true` and
    `y_pred`:

    - `None`: The provided `y_true` and `y_pred` are passed directly to the
        objective. This means that if multiple objectives all use `None`, they
        will all receive the same data that comes from the dataloader.
    - `str`: The key indexes into a mapping which has the `y_true`/`y_pred` key,
        or an object which has a matching attribute.
    - `Sequence[str]`: Each key indexes into the layers of a nested mapping or
        object.
    - `Callable`: The callable is applied to the provided `y_true` and `y_pred`.

    !!! warning

        The user is responsible for ensuring that the `y_true` and `y_pred`
        keys or callables index the appropriate types for this objective.

    Type Parameters:
        - `YTrue`: objective ground truth data type.
        - `YHat`: objective model prediction data type.
        - `YTrueAll`: type of all ground truth data (as loaded by the
            dataloader).
        - `YHatAll`: type of all model output data (as produced by the model).

    Attributes:
        objective: The objective to use.
        weight: Weight of the objective in the overall loss.
        y_true: Key or callable to index into the ground truth data.
        y_pred: Key or callable to index into the model output data.
    """

    objective: Objective
    weight: float = 1.0
    y_true: str | Sequence[str] | Callable[[YTrueAll], YTrue] | None = None
    y_pred: str | Sequence[str] | Callable[[YPredAll], YPred] | None = None

    def _index(
        self, data: Any, key: str | Sequence[str] | Callable | None
    ) -> Any:
        """Index into data using the key or callable."""
        def dereference(obj, k):
            if isinstance(obj, Mapping):
                try:
                    return obj[k]
                except KeyError as e:
                    raise KeyError(
                        f"Key {k} not found: {obj}") from e
            else:
                try:
                    return getattr(obj, k)
                except AttributeError as e:
                    raise AttributeError(
                        f"Attribute {k} not found: {obj}") from e

        if isinstance(key, str):
            return dereference(data, key)
        elif isinstance(key, Sequence):
            for k in key:
                data = dereference(data, k)
            return data
        elif callable(key):
            return key(data)
        else:   # key is None
            return data

    def index_y_true(self, y_true: YTrueAll) -> YTrue:
        """Get indexed ground truth data.

        Args:
            y_true: All ground truth data (as loaded by the dataloader).

        Returns:
            Indexed ground truth data.
        """
        return self._index(y_true, self.y_true)

    def index_y_pred(self, y_pred: YPredAll) -> YPred:
        """Get indexed model output data.

        Args:
            y_pred: All model output data (as produced by the model).

        Returns:
            Indexed model output data.
        """
        return self._index(y_pred, self.y_pred)


class MultiObjective(Objective[TArray, YTrue, YPred]):
    """Composite objective that combines multiple objectives.

    ??? example "Hydra Configuration"

        If using [Hydra](https://hydra.cc/docs/intro/) for dependency
        injection, a `MultiObjective` configuration should look like this:
        ```yaml
        objectives:
        name:
            objective:
                _target_: ...
                kwargs: ...
            weight: 1.0
            y_true: "y_true_key"
            y_pred: "y_pred_key"
        ...
        ```

    Type Parameters:
        - `YTrue`: ground truth data type.
        - `YHat`: model output data type.

    Args:
        objectives: multiple objectives, organized by name; see
            [`MultiObjectiveSpec`][^.]. Each objective can also be provided as
            a dict, in which case the key/values are passed to
            `MultiObjectiveSpec`.
    """

    def __init__(self, **objectives: Mapping | MultiObjectiveSpec) -> None:
        if len(objectives) == 0:
            raise ValueError("At least one objective must be provided.")

        self.objectives = {
            k: v if isinstance(v, MultiObjectiveSpec)
            else MultiObjectiveSpec(**v)
            for k, v in objectives.items()}

    def __call__(
        self, y_true: YTrue, y_pred: YPred, train: bool = True
    ) -> tuple[Float[TArray, "batch"], dict[str, Float[TArray, "batch"]]]:
        """Calculate training metrics.

        Args:
            y_true: Data channels (i.e., dataloader output).
            y_pred: Model outputs.
            train: Whether in training mode (e.g., skip expensive metrics).

        Returns:
            A tuple containing the loss and a dict of metric values.
        """
        loss = 0.
        metrics = {}
        for k, v in self.objectives.items():
            k_loss, k_metrics = v.objective(
                v.index_y_true(y_true), v.index_y_pred(y_pred), train=train)
            loss += k_loss * v.weight

            for name, value in k_metrics.items():
                metrics[f"{k}/{name}"] = value

        # We assure that there's at least one objective.
        loss = cast(Float[TArray, ""] | Float[TArray, "batch"], loss)
        return loss, metrics

    def visualizations(
        self, y_true: YTrue, y_pred: YPred
    ) -> dict[str, UInt8[np.ndarray, "H W 3"]]:
        """Generate visualizations for each entry in a batch.

        !!! note

            This method should be called only from a "detached" CPU thread so
            as not to affect training throughput; the caller is responsible for
            detaching gradients and sending the data to the CPU. As such,
            implementations are free to use CPU-specific methods.

        Args:
            y_true: Data channels (i.e. dataloader output).
            y_pred: Model outputs.

        Returns:
            A dict, where each key is the name of a visualization, and the
                value is a stack of RGB images in HWC order, detached from
                Torch and sent to a numpy array.
        """
        images = {}
        for k, v in self.objectives.items():
            k_images = v.objective.visualizations(
                v.index_y_true(y_true), v.index_y_pred(y_pred))
            for name, image in k_images.items():
                images[f"{k}/{name}"] = image
        return images
