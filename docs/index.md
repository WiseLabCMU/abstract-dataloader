# Abstract Dataloader: Dataloader Not Included

## What is the Abstract Dataloader?

The **abstract dataloader** (ADL) is a minimalist [specification][abstract_dataloader.spec] for creating composable and interoperable dataloaders and data transformations, along with [abstract template implementations][abstract_dataloader.abstract] and reusable [generic components][abstract_dataloader.generic], including a [pytorch interface][abstract_dataloader.torch].

```
Metadata─────────────────┐
   │                     ▼
   └────►Sensor   Synchronization
           │             │
           └────►Trace◄──┘
                   │
                   └────►Dataset───►Transform
```

The ADL's specifications and bundled implementations lean heavily on generic type annotations in order to enable type checking using static type checkers such as [mypy](https://mypy-lang.org/) or [pyright](https://microsoft.github.io/pyright/) and runtime (dynamic) type checkers such as [beartype](https://github.com/beartype/beartype) and [typeguard](https://github.com/agronholm/typeguard), even when applying functor-like generic transforms such as [sequence loading][abstract_dataloader.generic.Window] and [transforms][abstract_dataloader.generic.SequencePipeline].

!!! tip "Structural Subtyping"

    Since the abstract dataloader uses python's [structural subtyping](https://typing.python.org/en/latest/spec/protocol.html) - `Protocol` - feature, the `abstract_dataloader` is not a required dependency for using the abstract dataloader! Implementations which follow the specifications are fully interoperable, including with type checkers, even if they do not have any mutual dependencies - including this library.

!!! info "Type Checking is Optional"

    While most of the non-documentation code in this library goes towards facilitating type checking of the abstract dataloader specifications, static and runtime type checking are fully optional, in line with Python's gradual typing paradigm.

    Users also do not need to fully define the abstract dataloader's typed interfaces. For example, specifying a [`Sensor`][abstract_dataloader.spec.Sensor] instead of a `Sensor[TData, TMetadata]` is perfectly valid, as type checkers will simply interpret the sensor as loading `Any` data and accepting `Any` metadata.

## Why Abstract?

Loading, preprocessing, and training models on time-series data is ubiquitous in machine learning for cyber-physical systems. However, unlike mainstream machine learning research, which has largely standardized around "canonical modalities" in computer vision (RGB images) and natural language processing (ordinary unstructured text), each new setting, dataset, and modality comes with a new set of tasks, questions, challenges - and data types which must be loaded and processed.

This poses a substantial software engineering challenge. With many different modalities, processing algorithms which operate on the power set of those different modalities, and downstream tasks which also each depend on some subset of modalities, two undesirable potential outcomes emerge:

1.  Data loading and processing components fragment into an exponential number of incompatible chunks, each of which encapsulates its required loading and processing functionality in a slightly different way. The barrier this presents to rapid prototyping needs no further explanation.
2.  The various software components coalesce into a monolith which nominally supports the power set of all functionality. However, in addition to the compatibility issues that come with bundling heterogeneous requirements such as managing "non-dependencies" (i.e. dependencies which are required by the monolith, but not a particular task), this also presents a hidden challenge in that by support exponentially many possible configurations, such an architecture is also exponentially hard to debug and verify.

However, we do not believe that these outcomes are a foregone conclusion. In particular, we believe that it's possible to write "one true dataloader" which can scale while maintaining intercompability by **not writing a common dataloader at all** -- but rather a common specification for writing dataloaders. We call this the **"abstract dataloader"**.

## Setup

While it is not necessary to install the `abstract_dataloader` in order to take advantage of ADL-compliant components, installing this library provides access to [`Protocol`][typing.Protocol] [types which describe each interface][abstract_dataloader.spec], as well as [generic][abstract_dataloader.generic] components which may be useful for working with ADL-compliant components.

=== "PyPI"

    To get the latest version:
    ```sh
    pip install abstract-dataloader
    ```

=== "Github"

    To get the latest development version:
    ```sh
    pip install git+git@github.com:WiseLabCMU/abstract-dataloader.git
    ```

## Dependencies

As an explicit goal is to minimize dependency constraints, only the following dependencies are required:

- **`python >= 3.10`**: a somewhat recent version of python is required, since the python type annotation specifications are rapidly evolving. 

- **`numpy >= 1.14`**: any remotely recent version of numpy is compatible, with the `1.14` minimum version only being required since this version first defined the `np.integer` type.

- **`jaxtyping >= 0.2.32`**: a fairly recent version of jaxtyping is also required due to the rapid pace of type annotation tooling. In particular, `jaxtyping 0.2.32` added support for `TypeVar` as array types, which is helpful for expressing [array type polymorphisms](https://github.com/patrick-kidger/jaxtyping/releases/tag/v0.2.32).

- **`typing_extensions >= 3.12`**: we [pull forward](https://typing-extensions.readthedocs.io/en/latest/) typing features from Python 3.12. This minimum version may be increased as we use newer typing features.

!!! example "Minimum Python Version"

    We may consider upgrading our minimum python version in the future, since `3.11` and newer versions support useful typing-related features such as the [`Self` type](https://docs.python.org/3/whatsnew/3.11.html).

!!! warning "Pytorch Integration"

    To use the optional pytorch integrations, we also require either **`torch >= 2.2`** (first version to add `torch.utils._pytree.tree_leaves`) or **`torch` and [`optree >= 0.13`](https://github.com/metaopt/optree)** (first "mostly stable" version) in order to have access to a fully-featured tree manipulation module. The included `torch` extra will install the latest pytorch and optree, with constraints `torch >= 2.2` and `optree >= 0.13`.

## Contributing

Please report any bugs, type-related issues/inconsistencies, and feel free to suggest new generic components! Any issues or PRs are welcome.

=== "Run Tests"

    The test suite isn't fully automated yet; run tests with
    ```sh
    uv run --extra testing coverage run -m pytest tests
    uv run --extra testing coverage report
    ```

    !!! info

        We currently have 100% coverage, with a few "non-functional" items excluded:

        - `__repr__` methods
        - Protocols, `...` placeholders
        - `NotImplementedError`s

=== "Build Docs"

    Build docs with
    ```sh
    uv run --extra docs mkdocs serve
    ```

    Push to github pages with
    ```sh
    uv run --extra docs mkdocs build
    ./update_gh_pages.sh
    ```

    !!! info

        The documentation builder fetches external inventories (i.e., `objects.inv`) in order to properly link external references. This requires internet access; if behind a firewall, make sure that these inventories are not blocked!
