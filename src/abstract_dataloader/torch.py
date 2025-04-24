"""Pytorch-ADL wrappers.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from abstract_dataloader import torch as adl_torch
    ```

    Since pytorch is not declared as a required dependency, you will also need
    to install `torch` (or install the `torch` extra with
    `pip install abstract_dataloader[torch]`).
"""

from typing import Any, Callable, Generic, Literal, Sequence, TypeVar, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from . import abstract, generic, spec

Raw = TypeVar("Raw")
Transformed = TypeVar("Transformed")
Collated = TypeVar("Collated")
Processed = TypeVar("Processed")


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


class TransformedDataset(Dataset[Transformed], Generic[Raw, Transformed]):
    """Pytorch-compatible dataset with transformation applied.

    Extends [`torch.utils.data.Dataset`][torch.utils.data.Dataset],
    implementing a torch "map-style" dataset.

    Args:
        dataset: source dataset.
        transform: transformation to apply to each sample when loading.

    Type Parameters:
        - `Raw`: raw data type from the dataloader.
        - `Transformed`: output data type from the provided transform function.
    """

    def __init__(
        self, dataset: spec.Dataset[Raw],
        transform: Callable[[Raw], Transformed]
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int | np.integer) -> Transformed:
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


ComposedRaw = TypeVar("ComposedRaw", bound=dict[str, Any])
ComposedTransformed = TypeVar("ComposedTransformed", bound=dict[str, Any])
ComposedCollated = TypeVar("ComposedCollated", bound=dict[str, Any])
ComposedProcessed = TypeVar("ComposedProcessed", bound=dict[str, Any])


class ComposeTransforms(
    torch.nn.Module,
    generic.ComposeTransforms[
        ComposedRaw, ComposedTransformed,
        ComposedCollated, ComposedProcessed]
):
    """Transform Compositions, modified for Pytorch compatibility.

    Any `nn.Module` transforms are registered to a separate `nn.ModuleDict`;
    the original `.transforms` attribute is maintained with references to all
    transforms.

    See [`generic.ComposeTransforms`][abstract_dataloader.generic.ComposeTransforms]
    for more details about this implementation. `.forward` and `.__call__`
    should work as expected within pytorch.

    Args:
        transforms: transforms to compose. The key indicates the subkey to
            apply each transform to.

    Type Parameters:
        - `ComposedRaw`: Input data format.
        - `ComposedTransformed`: Data after the first `.transform` step.
        - `ComposedCollated`: Data after the second `.collate` step.
        - `ComposedProcessed`: Output data format.
    """

    def __init__(self, **transforms: spec.Transforms) -> None:
        super().__init__()
        self.transforms = transforms
        self._transforms = torch.nn.ModuleDict({
            k: v for k, v in transforms.items()
            if isinstance(v, torch.nn.Module)})

    def forward(self, data: ComposedCollated) -> ComposedProcessed:
        # We have to redefine this for some reason to make torch happy.
        # I think `nn.Module` has a generic `forward` implementation which
        # is clobbering `ComposeTransform`.
        return cast(
            ComposedProcessed,
            {k: v.batch(data[k]) for k, v in self.transforms.items()})

    def batch(self, data: ComposedCollated) -> ComposedProcessed:
        """Alias `batch` to `__call__` to `forward` via `nn.Module`."""
        return self(data)


class StackedSequenceTransforms(
    generic.SequenceTransforms[Raw, Transformed, Collated, Processed]
):
    """Modify a transform to act on sequences.

    Unlike the generic [`generic.SequenceTransforms`][abstract_dataloader.]
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

    !!! note

        Reshaping is performed using the `optree` library, or, if that is not
        present, `torch.utils._pytree`, which implements equivalent
        functionality. If `torch.utils._pytree` is removed in a later version,
        the constructor will raise `NotImplementedError`, and this fallback
        will need to be replaced.
    """

    def __init__(
        self, transform: spec.Transforms[Raw, Transformed, Collated, Processed]
    ) -> None:
        super().__init__(transform)
        self.treelib = _get_treelib()

    def collate(self, data: Sequence[Sequence[Transformed]]) -> Any:
        data_flat = sum((list(x) for x in data), start=[])
        collated_flat = self.transforms.collate(data_flat)
        unflattened = self.treelib.tree_map(
            lambda x: x.reshape(len(data), -1, *x.shape[1:]),
            collated_flat)   # type: ignore
        return unflattened

    def batch(self, data: Any) -> Any:
        batch = self.treelib.tree_leaves(data)[0].shape[0]  # type: ignore
        flattened = self.treelib.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), data)
        transformed = self.transforms.batch(cast(Collated, flattened))
        unflattened = self.treelib.tree_map(
            lambda x: x.reshape(batch, -1, *x.shape[1:]),
            transformed)  # type: ignore
        return unflattened


class Transforms(abstract.Transforms[Raw, Raw, Collated, Collated]):
    """Generic numpy/pytorch transforms.

    Converts numpy arrays to pytorch tensors, and either stacks or concatenates
    each value.

    TODO: convert this to a `Collate`.

    !!! note

        Stacking is performed using the `optree` library, or, if that is not
        present, `torch.utils._pytree`, which implements equivalent
        functionality. If `torch.utils._pytree` is removed in a later version,
        the constructor will raise `NotImplementedError`, and this fallback
        will need to be replaced.

    Args:
        mode: whether to `stack` or `concat` during collation.
    """

    def __init__(self, mode: Literal["stack", "concat"] = "concat") -> None:
        self.mode = mode
        self.treelib = _get_treelib()

    def collate(self, data: Sequence[Raw]) -> Collated:
        if self.mode == "concat":
            return self.treelib.tree_map(
                lambda *x: torch.concat([torch.from_numpy(s) for s in x]),
                *data)  # type: ignore
        else:
            return self.treelib.tree_map(
                lambda *x: torch.stack([torch.from_numpy(s) for s in x]),
                *data)  # type: ignore
