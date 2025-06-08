"""Pytorch versions of generic components."""

from typing import Any, TypeVar, cast

import torch

from abstract_dataloader import generic, spec

PRaw = TypeVar("PRaw", bound=dict[str, Any])
PTransformed = TypeVar("PTransformed", bound=dict[str, Any])
PCollated = TypeVar("PCollated", bound=dict[str, Any])
PProcessed = TypeVar("PProcessed", bound=dict[str, Any])


class ParallelPipelines(
    torch.nn.Module,
    generic.ParallelPipelines[PRaw, PTransformed, PCollated, PProcessed]
):
    """Transform Compositions, modified for Pytorch compatibility.

    Any [`nn.Module`][?torch.] transforms are registered to a separate
    [`nn.ModuleDict`][?torch.]; the original `.transforms` attribute is
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


TRawInner = TypeVar("TRawInner")
TRaw = TypeVar("TRaw")
TTransformed = TypeVar("TTransformed")
TCollated = TypeVar("TCollated")
TProcessed = TypeVar("TProcessed")
TProcessedInner = TypeVar("TProcessedInner")

class ComposedPipeline(
    torch.nn.Module,
    generic.ComposedPipeline[
        TRaw, TRawInner, TTransformed, TCollated, TProcessedInner, TProcessed]
):
    """Compose pipeline sequentially with pre and post transforms.

    Type Parameters:
        - `TRaw`: initial input type.
        - `TRawInner`: output of the pre-composed transform, and input to the
            provided [`Pipeline`][abstract_dataloader.spec].
        - `TCollated`, `TProcessed`: intermediate values for the provided
            [`Pipeline`][abstract_dataloader.spec].
        - `TProcessedInner`: output of the transforms, and input to the
            post-composed transform.
        - `TProcessed`: output type.

    Args:
        pipeline: pipeline to compose.
        pre: pre-transform to apply on the CPU side; skipped if `None`.
        post: post-transform to apply on the GPU side; skipped if `None`.
    """

    def __init__(
        self, pipeline: spec.Pipeline[
            TRawInner, TTransformed, TCollated, TProcessedInner],
        pre: spec.Transform[TRaw, TRawInner] | None = None,
        post: spec.Transform[TProcessedInner, TProcessed] | None = None
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.pre = pre
        self.post = post

        self.collate = pipeline.collate


class SequencePipeline(
    torch.nn.Module,
    generic.SequencePipeline[TRaw, TTransformed, TCollated, TProcessed]
):
    """Transform which passes an additional sequence axis through.

    The given `Pipeline` is modified to accept `Sequence[...]` for each
    data type in its pipeline, and return a `list[...]` across the additional
    axis, thus "passing through" the axis.

    For example, suppose a sequence dataloader reads

    ```
    [
        [Raw[s=0, t=0], Raw[s=0, t=1], ... Raw[s=0, t=n]]
        [Raw[s=1, t=0], Raw[s=1, t=1], ... Raw[s=1, t=n]]
        ...
        [Raw[s=b, t=0], Raw[s=b, t=1], ... Raw[s=b, t=n]
    ]
    ```

    for sequence length `t = 0...n` and batch sample `s = 0...b`. For sequence
    length `t`, the output of the transforms will be batched with the sequence
    on the outside:

    ```
    [
        Processed[s=0...b] [t=0],
        Processed[s=0...b] [t=1],
        ...
        Processed[s=0...b] [t=n]
    ]
    ```

    Type Parameters:
        - `TRaw`, `TTransformed`, `TCollated`, `TProcessed`: see
          [`Pipeline`][abstract_dataloader.spec.].

    Args:
        pipeline: input pipeline.
    """

    def __init__(
        self, pipeline: spec.Pipeline[
            TRaw, TTransformed, TCollated, TProcessed]
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
