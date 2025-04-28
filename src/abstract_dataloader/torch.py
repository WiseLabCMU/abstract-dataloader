"""Pytorch-ADL wrappers.

These implementations provide interoperability with pytorch dataloaders,
modules, etc.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from abstract_dataloader import torch as adl_torch
    ```

    Since pytorch is not declared as a required dependency, you will also need
    to install `torch` (or install the `torch` extra with
    `pip install abstract_dataloader[torch]`).

!!! note

    Recursive tree operations such as reshaping and stacking are performed
    using the `optree` library, or, if that is not present,
    `torch.utils._pytree`, which implements equivalent functionality. If
    `torch.utils._pytree` is removed in a later version, the constructor will
    raise `NotImplementedError`, and this fallback will need to be replaced.

!!! warning

    Custom data container classes such as `@dataclass` are only supported if
    `optree` is installed, and they are
    [registered with optree][dataclass]. However, `dict`, `list`,
    `tuple`, and equivalent types such as `TypedDict` and `NamedTuple` will
    work [out of the box][creating-a-type-system].
"""

from typing import Any, Generic, Literal, Sequence, TypeVar, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from . import generic, spec

TRaw = TypeVar("TRaw")
TTransformed = TypeVar("TTransformed")
TCollated = TypeVar("TCollated")
TProcessed = TypeVar("TProcessed")


def _get_treelib():
    """Get a tree manipulation library."""
    try:
        # Don't give type checking errors since this is optional
        import optree  # type: ignore
        return optree
    except ImportError:
        try:
            return torch.utils._pytree  # type: ignore
        except AttributeError:
            raise NotImplementedError(
                "No tree_map implementation found: `optree` is not "
                "installed, and the pytorch `_pytree` utility is not "
                "present.")


class TransformedDataset(Dataset[TTransformed], Generic[TRaw, TTransformed]):
    """Pytorch-compatible dataset with transformation applied.

    Extends [`torch.utils.data.Dataset`][torch.utils.data.Dataset],
    implementing a torch "map-style" dataset.

    Type Parameters:
        - `TRaw`: raw data type from the dataloader.
        - `TTransformed`: output data type from the provided transform function.

    Args:
        dataset: source dataset.
        transform: transformation to apply to each sample when loading (note
            that `Transform[TRaw, TTransformed]` is equivalent to
            `Callable[[TRaw], TTransformed]`).
    """

    def __init__(
        self, dataset: spec.Dataset[TRaw],
        transform: spec.Transform[TRaw, TTransformed]
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int | np.integer) -> TTransformed:
        """Map-style dataset indexing.

        Args:
            index: dataset index; passthrough to the underlying `Dataset`.

        Returns:
            Transformed sample.
        """
        return self.transform(self.dataset[index])

    def __len__(self) -> int:
        """Dataset length; passthrough to the underlying `Dataset`."""
        return len(self.dataset)

    def __repr__(self) -> str:
        """Friendly name."""
        return f"Transformed({repr(self.dataset)})"


PRaw = TypeVar("PRaw", bound=dict[str, Any])
PTransformed = TypeVar("PTransformed", bound=dict[str, Any])
PCollated = TypeVar("PCollated", bound=dict[str, Any])
PProcessed = TypeVar("PProcessed", bound=dict[str, Any])


class ParallelPipelines(
    torch.nn.Module,
    generic.ParallelPipelines[PRaw, PTransformed, PCollated, PProcessed]
):
    """Transform Compositions, modified for Pytorch compatibility.

    Any [`nn.Module`][torch.] transforms are registered to a separate
    [`nn.ModuleDict`][torch.]; the original `.transforms` attribute is
    maintained with references to the full pipeline.

    See [`generic.ParallelPipelines`][abstract_dataloader.]
    for more details about this implementation. `.forward` and `.__call__`
    should work as expected within pytorch.

    Type Parameters:
        - `PRaw`, `PTransformed`, `PCollated`, `PProcessed`: see
          [`Pipeline`][abstract_dataloader.spec.].

    Args:
        transforms: pipelines to compose. The key indicates the subkey to
            apply each transform to.
    """

    def __init__(self, **transforms: spec.Pipeline) -> None:
        super().__init__()
        self.transforms = transforms
        self._transforms = torch.nn.ModuleDict({
            k: v for k, v in transforms.items()
            if isinstance(v, torch.nn.Module)})

    def forward(self, data: PCollated) -> PProcessed:
        # We have to redefine this for some reason to make torch happy.
        # I think `nn.Module` has a generic `forward` implementation which
        # is clobbering `ComposeTransform`.
        return cast(
            PProcessed,
            {k: v.batch(data[k]) for k, v in self.transforms.items()})

    def batch(self, data: PCollated) -> PProcessed:
        """Alias `batch` to `__call__` to `forward` via `nn.Module`."""
        return self(data)


class StackedSequencePipeline(
    generic.SequencePipeline[TRaw, TTransformed, TCollated, TProcessed]
):
    """Modify a transform to act on sequences.

    Unlike the generic [`generic.SequencePipeline`][abstract_dataloader.]
    implementation, this class places the sequence axis directly inside each
    tensor, so that each data type has axes `(batch, sequence, ...)`. For the
    same input,

    ```
    [
        [Raw[s=0, t=0], Raw[s=0, t=1], ... Raw[s=0, t=n]]
        [Raw[s=1, t=0], Raw[s=1, t=1], ... Raw[s=1, t=n]]
        ...
        [Raw[s=b, t=0], Raw[s=b, t=1], ... Raw[s=b, t=n]
    ]
    ```

    this transform instead yields

    ```python
    Processed[s=0...b][t=0...n].
    ```

    !!! info

        This class requires that all outputs of `.collate()` are pytorch
        tensors. Furthermore, batches must be treated as an additional leading
        axis by both `.collate` and `.forward`.

    !!! warning

        Since the output has an additional axis, it does not necessarily have
        the same type as the underlying transform!

    This is accomplished by appropriately reshaping the data to use the
    batch-vectorized underlying implementation:

    - `.transform`: apply the transform to each sample across the additional
      sequence axis.
    - `.collate`: concatenate all sequences into a single `list[Raw]`, instead
      of a `list[list[Raw]]`. Then, collate the list, and reshape back into
      `batch sequence ...` order.
    - `.transform`: flatten the collated data back to a `(batch sequence) ...`
      single leading batch axis, apply the transform, and reshape back.

    Type Parameters:
        - `PRaw`, `PTransformed`, `PCollated`, `PProcessed`: see
          [`Pipeline`][abstract_dataloader.spec.].

    Args:
        transform: pipeline to transform to accept sequences.
    """

    def __init__(
        self, transform: spec.Pipeline[
            TRaw, TTransformed, TCollated, TProcessed]
    ) -> None:
        super().__init__(transform)
        self.treelib = _get_treelib()

    def collate(self, data: Sequence[Sequence[TTransformed]]) -> Any:
        data_flat = sum((list(x) for x in data), start=[])
        collated_flat = self.transform.collate(data_flat)
        unflattened = self.treelib.tree_map(
            lambda x: x.reshape(len(data), -1, *x.shape[1:]),
            collated_flat)   # type: ignore
        return unflattened

    def batch(self, data: Any) -> Any:
        batch = self.treelib.tree_leaves(data)[0].shape[0]  # type: ignore
        flattened = self.treelib.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), data)
        transformed = self.transform.batch(cast(TCollated, flattened))
        unflattened = self.treelib.tree_map(
            lambda x: x.reshape(batch, -1, *x.shape[1:]),
            transformed)  # type: ignore
        return unflattened


class Collate(spec.Collate[TTransformed, TCollated]):
    """Generic numpy to pytorch collation.

    Converts numpy arrays to pytorch tensors, and either stacks or concatenates
    each value.

    Type Parameters:
        - `TTransformed`: input sample type.
        - `TCollated`: output collated type.

    Args:
        mode: whether to `stack` or `concat` during collation.
    """

    def __init__(self, mode: Literal["stack", "concat"] = "concat") -> None:
        self.mode = mode
        self.treelib = _get_treelib()

    def __call__(self, data: Sequence[TTransformed]) -> TCollated:
        if self.mode == "concat":
            return self.treelib.tree_map(
                lambda *x: torch.concat([torch.from_numpy(s) for s in x]),
                *data)  # type: ignore
        else:
            return self.treelib.tree_map(
                lambda *x: torch.stack([torch.from_numpy(s) for s in x]),
                *data)  # type: ignore
