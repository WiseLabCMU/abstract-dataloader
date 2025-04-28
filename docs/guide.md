# Using the Abstract Dataloader

!!! abstract "Recommendations"

    - Define your components using [`abstract`][abstract_dataloader.abstract] base classes, and specify input components using [`spec`][abstract_dataloader.spec] interfaces.
    - Set up a [type system][creating-a-type-system], and provide your types to the ADL [`spec`][abstract_dataloader.spec] and [`generic`][abstract_dataloader.generic] components as type parameters.
    - [Use a static type checker and a runtime type checker][verification-using-type-checking].

## Using ADL Components

Code using ADL-compliant components (which may itself be another ADL-compliant component) should use the exports provided in the [`spec`][abstract_dataloader.spec] as type annotations. When combined with static type checkers such as
[mypy](https://mypy-lang.org/) or [pyright](https://microsoft.github.io/pyright/) and runtime (dynamic)
type checkers such as [beartype](https://github.com/beartype/beartype) and [typeguard](https://github.com/agronholm/typeguard), this allows code to be verified for some degree of type safety and spec compliance.

``` python
from dataclasses import dataclass
from abstract_dataloader.spec import Sensor

@dataclass
class Foo:
    x: float

def example(adl_sensor: Sensor[Foo, Any], idx: int) -> float:
    x1 = adl_sensor[idx].x
    # pyright: x: float
    x2 = adl_sensor[x1]
    # pyright: argument of type "float" cannot be assigned to parameter
    # "index" of type "int | Integer[Any]"...
    return x2
```

!!! info "Spec Verification via Type Checking"

    The `abstract_dataloader` module does not include any runtime type checking; downstream modules should enable runtime type checking if desired, for example with [install hooks](https://github.com/beartype/beartype) like
    `beartype_this_package()` or [jaxtyping's](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/) `install_import_hook(...)`.

## Implementing ADL Components

There are three ways to implement ADL-compliant components, in order of preference:

1.  **Use the abstract base classes** provided in [`abstract_dataloader.abstract`][abstract_dataloader.abstract].
2.  **Explicitly implement the protocols** described in [`abstract_dataloader.spec`][abstract_dataloader.spec].
3.  **Implement the protocols using the specifications as guidance** with neither [`abstract_dataloader.abstract`][abstract_dataloader.abstract] nor [`abstract_dataloader.spec`][abstract_dataloader.spec] classes as base classes.

### Using the Abstract Base Classes

The [`abstract_dataloader.abstract`][abstract_dataloader.abstract] submodule provides [Abstract Base Class](https://docs.python.org/3/library/abc.html) implementations of applicable specifications, with required methods annotated, and methods which can be written in terms of others ["polyfilled"](https://developer.mozilla.org/en-US/docs/Glossary/Polyfill) by default. This is the fastest way to get started:

=== "ADL-compliant Sensor"

    ```python
    class Lidar(abstract.Sensor[LidarData, LidarMetadata]):

        def __init__(
            self, metadata: LidarMetadata, path: str, name: str = "sensor"
        ) -> None:
            self.path = path
            super().__init__(metadata=metadata, name=name)

        def __getitem__(self, batch: int | np.integer) -> LidarData:
            blob_path = os.path.join(self.path, self.name, f"{batch:06}.npz")
            with open(blob_path) as f:
                return dict(np.load(f))
    ```

=== "Setup & Types"

    ``` python
    from dataclasses import dataclass
    import numpy as np
    from abstract_dataloader import spec, abstract

    @dataclass
    class LidarMetadata(abstract.Metadata):
        timestamps: Float64[np.ndarray, "N"]
        # ... other data fields ...

    @dataclass
    class LidarData:
        # ... definition of data fields ...
    ```

!!! info "Prefer `spec` to `abstract` as type bounds"

    While this example does not take any ADL components as inputs, ADL-compliant implementations which take components as inputs should use [`spec`][abstract_dataloader.spec] types as annotations instead of [`abstract`][abstract_dataloader.abstract] base classes as annotations. This preserves modularity; if an [`abstract`][abstract_dataloader.abstract] base class is used as an input type annotation, this prevents the use of other compatible implementations which do not use the specified base class.

### Explicitly Implementing an ADL Component

Implementations do not necessarily need to use the abstract base classes provided, and can instead use the provided specifications as base classes. This explicitly implements the defined specifications, which allows static type checkers to [provide some degree of verification](https://typing.python.org/en/latest/spec/protocol.html#explicitly-declaring-implementation) that you have implemented the specification.

!!! example

    The provided [`generic.Next`][abstract_dataloader.generic.Next] explicitly implements [`spec.Synchronization`][abstract_dataloader.spec.Synchronization]:

    ``` python
    import numpy as np
    from jaxtyping import Float64, UInt32
    from abstract_dataloader import spec

    class Next(spec.Synchronization):

        def __init__(self, reference: str) -> None:
            self.reference = reference

        def __call__(
            self, timestamps: dict[str, Float64[np.ndarray, "_N"]]
        ) -> dict[str, UInt32[np.ndarray, "M"]]:
            ref_time_all = timestamps[self.reference]
            start_time = max(t[0] for t in timestamps.values())
            end_time = min(t[-1] for t in timestamps.values())

            start_idx = np.searchsorted(ref_time_all, start_time)
            end_idx = np.searchsorted(ref_time_all, end_time)
            ref_time = ref_time_all[start_idx:end_idx]
            return {
                k: np.searchsorted(v, ref_time).astype(np.uint32)
                for k, v in timestamps.items()}
    ```

### Implicitly Implementing an ADL Component

Since the abstract dataloader is based entirely on structural subtyping - "static duck typing", implementing an ADL-compliant component does not actually require the abstract dataloader!

- Simply implementing the protocols described by the ADL, using the same types (or with more general inputs and more specific outputs), automatically allows for ADL compatibility with downstream components which use only the ADL spec and correctly apply ADL spec type-checking.
- Since the types only describe behavior, parts or all of the [`spec`][abstract_dataloader.spec] can be copied into other modules; these copies then become totally interchangeable with the originals!

## Extending Protocols

When extending the abstract dataloader specifications with additional domain-specific features, keep in mind that your implementation must be usable wherever an ADL-compliant dataloader is used. This is known as the [Liskov Substitution Principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle) (and is the error you will get from static type checkers if you break that compatibility). The key implications are as follows:

**Preconditions cannot be strengthened in the subtype**.

- Implementations may not impose stricter requirements on the input types; for example, where `__getitem__` is expected to allow `index: int | np.integer`, implementations may not require more specific types, e.g. `np.int32`.
- This also implies that additional arguments must be optional, and have defaults, so that they are not required to be set when called.

**Postconditions cannot be weakened in the subtype**.

- Return types cannot be more general than in the specifications. In general, this should never be an issue, since the return types provided in the abstract dataloader [specifications][abstract_dataloader.spec] are extremely general.
- However, the generic types provided should be followed; for example, when [`Sensor.__getitem__`][abstract_dataloader.spec.Sensor.__getitem__] returns a certain `Sample`, [`Sensor.stream`][abstract_dataloader.spec.Sensor.stream] should also return that same type.

!!! example

    The [`abstract.Sensor`][abstract_dataloader.abstract.Sensor] provides a `stream` implementation with an additional `batch` argument that modifies the return type, while still remaining fully compliant with the [`Sensor` specification][abstract_dataloader.spec.Sensor].

    ```python
    @overload
    def stream(self, batch: None = None) -> Iterator[TSample]: ...

    @overload
    def stream(self, batch: int) -> Iterator[list[TSample]]: ...

    def stream(
        self, batch: int | None = None
    ) -> Iterator[TSample | list[TSample]]:
        if batch is None:
            for i in range(len(self)):
                yield self[i]
        else:
            for i in range(len(self) // batch):
                yield [self[j] for j in range(i * batch, (i + 1) * batch)]
    ```

    This is accomplished using an `@overload` with a default that falls back to the specification:

    - If `.stream()` is used only as described by the [spec][abstract_dataloader.spec.Sensor.stream], `batch=None` matches the first overload, which returns an `Iterator[TSample]`, just as in the spec.
    - If `.stream()` is passed a `batch=...` argument, the second overload now matches, instead returning an `Iterator[list[TSample]]`.
