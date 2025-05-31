"""Pytorch-ADL wrappers.

These implementations are a superset of the [`generic`][abstract_dataloader.]
components, and provide interoperability with pytorch dataloaders, modules,
etc. For example, any [`Pipeline`][abstract_dataloader.spec.]-related
components which could contain pytorch [`nn.Module`][torch.nn.Module]s are
modified to subclass `nn.Module` in order to properly register them.

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
    `optree` is installed, and they are [registered with optree](
    https://optree.readthedocs.io/en/stable/dataclasses.html). However, `dict`,
    `list`, `tuple`, and equivalent types such as `TypedDict` and `NamedTuple`
    will work [out of the box](types.md).
"""

# Reimports to expose the same interface as `generic`.
from abstract_dataloader.generic import Empty, Metadata, Nearest, Next, Window

from .generic import ComposedPipeline, ParallelPipelines, SequencePipeline
from .torch import Collate, StackedSequencePipeline, TransformedDataset

__all__ = [
    "ComposedPipeline", "ParallelPipelines",
    "Metadata", "SequencePipeline", "Window",
    "Empty", "Nearest", "Next",
    # Pytorch-specific Components
    "StackedSequencePipeline", "TransformedDataset", "Collate"
]
