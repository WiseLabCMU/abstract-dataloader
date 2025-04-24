"""Transform compositions."""

from typing import Any, Sequence, TypeVar, cast

from abstract_dataloader import spec

ComposedRaw = TypeVar("ComposedRaw", bound=dict[str, Any])
ComposedTransformed = TypeVar("ComposedTransformed", bound=dict[str, Any])
ComposedCollated = TypeVar("ComposedCollated", bound=dict[str, Any])
ComposedProcessed = TypeVar("ComposedProcessed", bound=dict[str, Any])


class ComposeTransforms(
    spec.Transforms[
        ComposedRaw, ComposedTransformed,
        ComposedCollated, ComposedProcessed],
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

    Args:
        transforms: transforms to compose. The key indicates the subkey to
            apply each transform to.

    Type Parameters:
        - `ComposedRaw`, `ComposedTransformed`, `ComposedCollated`,
          `ComposedProcessed`: see [`Transforms`][abstract_dataloader.spec.].
    """

    def __init__(self, **transforms: spec.Transforms) -> None:
        self.transforms = transforms

    def sample(self, data: ComposedRaw) -> ComposedTransformed:
        return cast(
            ComposedTransformed,
            {k: v.sample(data[k]) for k, v in self.transforms.items()})

    def collate(self, data: Sequence[ComposedTransformed]) -> ComposedCollated:
        return cast(ComposedCollated, {
            k: v.collate([x[k] for x in data])
            for k, v in self.transforms.items()
        })

    def batch(self, data: ComposedCollated) -> ComposedProcessed:
        return cast(
            ComposedProcessed,
            {k: v.batch(data[k]) for k, v in self.transforms.items()})
