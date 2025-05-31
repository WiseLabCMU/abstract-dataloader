"""Transform compositions."""

from typing import Any, Generic, Sequence, TypeVar, cast

from abstract_dataloader import spec

PRaw = TypeVar("PRaw", bound=dict[str, Any])
PTransformed = TypeVar("PTransformed", bound=dict[str, Any])
PCollated = TypeVar("PCollated", bound=dict[str, Any])
PProcessed = TypeVar("PProcessed", bound=dict[str, Any])


class ParallelPipelines(
    spec.Pipeline[PRaw, PTransformed, PCollated, PProcessed],
):
    """Compose multiple transforms in parallel.

    For example, with transforms `{"radar": radar_tf, "lidar": lidar_tf, ...}`,
    the composed transform performs:

    ```python
    {
        "radar": radar_tf.transform(data["radar"]),
        "lidar": lidar_tf.transform(data["lidar"]),
        ...
    }
    ```

    !!! note

        This implies that the type parameters must be `dict[str, Any]`, so this
        class is parameterized by a separate set of
        `Composed(Raw|Transformed|Collated|Processed)` types with this bound.

    !!! tip

        See [`torch.ParallelPipelines`][abstract_dataloader.] for an
        implementation which is compatible with [`nn.Module`][torch.]-based
        pipelines.

    Type Parameters:
        - `PRaw`, `PTransformed`, `PCollated`, `PProcessed`: see
          [`Pipeline`][abstract_dataloader.spec.].

    Args:
        transforms: transforms to compose. The key indicates the subkey to
            apply each transform to.
    """

    def __init__(self, **transforms: spec.Pipeline) -> None:
        self.transforms = transforms

    def sample(self, data: PRaw) -> PTransformed:
        return cast(
            PTransformed,
            {k: v.sample(data[k]) for k, v in self.transforms.items()})

    def collate(self, data: Sequence[PTransformed]) -> PCollated:
        return cast(PCollated, {
            k: v.collate([x[k] for x in data])
            for k, v in self.transforms.items()
        })

    def batch(self, data: PCollated) -> PProcessed:
        return cast(
            PProcessed,
            {k: v.batch(data[k]) for k, v in self.transforms.items()})


TRawInner = TypeVar("TRawInner")
TRaw = TypeVar("TRaw")
TTransformed = TypeVar("TTransformed")
TCollated = TypeVar("TCollated")
TProcessed = TypeVar("TProcessed")
TProcessedInner = TypeVar("TProcessedInner")


class ComposedPipeline(
    spec.Pipeline[TRaw, TTransformed, TCollated, TProcessed],
    Generic[
        TRaw, TRawInner, TTransformed,
        TCollated, TProcessedInner, TProcessed]
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
        self.pipeline = pipeline
        self.pre = pre
        self.post = post

        self.collate = pipeline.collate

    def sample(self, data: TRaw) -> TTransformed:
        """Transform single samples.

        Args:
            data: A single `TRaw` data sample.

        Returns:
            A single `TTransformed` data sample.
        """
        if self.pre is None:
            transformed = cast(TRawInner, data)
        else:
            transformed = self.pre(data)
        return self.pipeline.sample(transformed)

    def batch(self, data: TCollated) -> TProcessed:
        """Transform data batch.

        Args:
            data: A `TCollated` batch of data, nominally already sent to the
                GPU.

        Returns:
            The `TProcessed` output, ready for the downstream model.
        """
        transformed = self.pipeline.batch(data)
        if self.post is None:
            return cast(TProcessed, transformed)
        else:
            return self.post(transformed)
