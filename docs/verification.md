# Verification using Type Checking

!!! abstract "Recommendations"

    - Set up a [type system][creating-a-type-system] with [jaxtyping array annotations](https://docs.kidger.site/jaxtyping/), and provide [type parameters][type-annotations-in-the-adl] accordingly.
    - Use [pyright](https://microsoft.github.io/pyright) for static type checking (just install the vscode Python extension).
    - Use [beartype](https://github.com/beartype/beartype) for runtime checking - don't forget to [use the jaxtyping hooks](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/)!

## Type Annotations in the ADL

The abstract dataloader specification makes heavy use of [generic types](https://typing.python.org/en/latest/reference/generics.html) to allow type checkers to work even when indirected by various transformations. For example, consider the definition of [`Sensor`][abstract_dataloader.spec.Sensor]:

```python
# Definition simplified for readability
class Sensor(Protocol, Generic[TSample, TMetadata]):

    metadata: TMetadata

    def __getitem__(self, index: int | np.integer) -> TSample:
        ...
```

- `TSample` and `TMetadata` declare the types of the metadata and data sample type.
- Implementations which provide a `TSample` and/or `TMetadata` allow type checkers to infer that `.metadata` should refer to a `TMetadata` object, and indexing via `__getitem__` should return a `TSample` type.

!!! note

    In this particular example, `TMetadata` is expected to implement [`Metadata`][abstract_dataloader.spec.Metadata]:
    ```python
    TMetadata = TypeVar("TMetadata", bound=Metadata)
    ```
    where [`Metadata`][abstract_dataloader.spec.Metadata] is a protocol which requires a `.timestamps` attribute.

Now, suppose we are declaring an interface which takes any [`Sensor`][abstract_dataloader.spec.Sensor] which loads a (statically and dynamically checkable) `CameraFrame` type and is associated with `VideoMetadata`. We can define this interface as such:

```python
read_frame_with_cond(
    video: spec.Sensor[CameraFrame, VideoMetadata], cond: dict
) -> CameraFrame:
    # `video` must have `metadata`, since `video` is a `Sensor`
    # `video.metadata` must be a `VideoMetadata` due to the type parameter
    metadata: VideoMetadata = video.metadata
    idx: int = some_specific_way_to_search_the_metadata(metadata, cond)
    # `video[idx]` must be a `CameraFrame` due to the type parameter
    return video[idx]
```

## Static Type Checking

Static type checking via tools such as [pyright](https://microsoft.github.io/pyright), which is included with the VSCode python plugin, provides a first line of defense against violating the ADL specification, either by incorrectly using compliant components, or declaring a ADL-compatible component that isn't actually compliant.

!!! bug "Mypy does not fully support PEP 695"

    The spec currently makes heavy use of backported (via [`typing_extensions`](https://typing-extensions.readthedocs.io/en/latest)) PEP 695 `TypeVar(..., infer_variance=True)` syntax. Unfortunately, Mypy does not [support this yet](https://github.com/python/mypy/issues/17811), so Mypy will always raise type errors.

!!! warning "Static type checking cannot verify array shapes or data types"

    Since array shapes and types are determined at runtime, static type checkers cannot verify [jaxtyping](https://docs.kidger.site/jaxtyping/) array annotations.

## Runtime Type Checking

The protocols declared in the [`spec`][abstract_dataloader.spec] are [`runtime_checkable`][typing.runtime_checkable], and can be `isinstance` or `issubclass` checked at runtime. As a last line of defense, this allows type checking by runtime type checkers such as [beartype](https://github.com/beartype/beartype), which can also deeply check the data types used as inputs and returns by the various MDL components if they have a [well-defined type system][creating-a-type-system].

Notably, since python is fundamentally a dynamically typed language, this enables checking of many types (or aspects of types) which cannot be checked statically. This is especially true for data with runtime-dynamic shapes.

!!! warning "Runtime type checking does not deeply verify protocols"

    Runtime type checkers are based on `isinstance` and `issubclass` calls; however, these builtin functions verify protocols only by checking the presence of declared methods and parameters. In typing jargon, this means that objects can only be checked against the ["type-erased" versions of protocols at runtime](https://peps.python.org/pep-0544/#runtime-checkable-decorator-and-narrowing-types-by-isinstance).

    In practice, this means that it is still possible for false positives: objects which appear to match a protocol when checked (e.g., at initialization), but cause something to explode later (e.g., when an incompatible method is called).
