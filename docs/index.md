# Abstract Dataloader: Dataloader Not Included

The abstract

## Setup

## Dependencies

As an explicit goal is to minimize dependency constraints, only the following
dependencies are required:

- **`numpy >= 1.14`**: any remotely recent version of numpy is compatible, with
  the `1.14` minimum version only being required since this version first
  defined the `np.integer` type.

- **`python >= 3.10`**: a somewhat recent version of python is required, since
  the python type annotation specifications are rapidly evolving. 

- **`jaxtyping >= 0.2.32`**: a fairly recent version of jaxtyping is also
  required due to the rapid pace of type annotation tooling. In particular,
  `jaxtyping 0.2.32` added support for `TypeVar` as array types, which is
  helpful for expressing
  [array type polymorphisms](https://github.com/patrick-kidger/jaxtyping/releases/tag/v0.2.32).

!!! example "Minimum Python Version"

    We may consider upgrading our minimum python version in the future,
    since `3.11` and newer versions support useful typing-related features such
    as the [`Self` type](https://docs.python.org/3/whatsnew/3.11.html).

!!! warning "Pytorch Integration"

    To use the optional pytorch integrations, we also require either
    **`torch >= 2.2`** or **`torch` and
    [`optree`](https://github.com/metaopt/optree)** in order to have access to
    a fully-featured tree manipulation module. The included `torch` extra
    will install the latest pytorch and optree, with constraints `torch >= 2.2`
    and `optree >= 0.13`.


## Why Abstract?

Loading, preprocessing, and training models on time-series data is a
ubiquitous primitive in applied machine learning for cyber-physical
systems. However, unlike mainstream machine learning research, which has
largely standardized around "canonical problems" in computer vision
(operating on RGB images) and natural language processing (operating on
ordinary unstructured text), each new setting, dataset, and modality
comes with a new set of tasks, questions, challenges -- and data types
which must be loaded and processed.

This poses a substantial software engineering challenge. With many
different modalities, processing algorithms which various operate on the
power set of those different modalities, and downstream tasks which also
each depend on some subset, two undesirable potential outcomes emerge:

1.  Data loading and processing components fragment into an exponential
    number of incompatible chunks, each of which encapsulates its
    required loading and processing functionality in a slightly
    different way. The barrier this presents to rapid prototyping needs
    no further explanation.
2.  The various software components coalesce into a monolith which
    nominally supports the power set of all functionality. However, in
    addition to the compatibility issues that come with bundling
    heterogeneous requirements such as managing "non-dependencies"
    (i.e. dependencies which are required by the monolith, but not a
    particular task), this also presents a hidden challenge in that by
    support exponentially many possible configurations, such an
    architecture is also exponentially hard to debug and verify.

Fortunately, we do not believe that these outcomes are a foregone
conclusion. In particular, we believe that it's possible to write "one
true dataloader" which can scale while maintaining intercompability by
**not writing a dataloader at all** -- but rather a common
specification for writing dataloaders: an **"abstract dataloader"**.

## What is the Abstract Dataloader?

This library describes an "abstract dataloader" specification, and
should be conceptualized as three major logical components:

1.  A [multimodal, asynchronous time-series dataloader programming model](model.md)
    which organizes datasets into a collection of traces, traces into a
    (possibly asynchronous) collection of sensors, and defines sensors as
    synchronous discrete-time time-series measurements.

    ``` 
    ┌─────────────────────────────────────────────────────────────┐
    │Dataset                                                      │
    │┌─────────────────────────┐┌─────────────────────────┐       │
    ││Trace 1                  ││Trace 2                  │       │
    ││┌────────┐┌────────┐     ││┌────────┐┌────────┐     │       │
    │││Sensor 1││Sensor 2│ ... │││Sensor 1││Sensor 2│ ... │  ...  │
    ││└────────┘└────────┘     ││└────────┘└────────┘     │       │
    │└─────────────────────────┘└─────────────────────────┘       │
    └─────────────────────────────────────────────────────────────┘
    ```

2.  A [formal, modular, and composable specification][abstract_dataloader.spec]
    (or at least as formal as we could get it, given the constraints of
    Python's type system) of the interfaces required to implement each building
    block in a dataloader with our programming model.

    ``` 
    Metadata─────────────────┐
       │                     ▼
       └────►Sensor   Synchronization
               │             │
               └────►Trace◄──┘
                       │
                       └────►Dataset───►Transforms
    ```

3.  A ["starter kit" of `abstract`][abstract_dataloader.abstract]
    and [`generic`][abstract_dataloader.generic]
    implementations of applicable parts of the spec, all with minimal
    dependency constraints -- only a version of numpy from 2018 or
    newer and a recent version of jaxtyping.

Each component is implemented to be as generic and extendable as
possible, with the goal of eventually writing a complete specification
which can encapsulate all re-usable and interoperable dataloader
interactions. Notably, by using python's [structural
subtyping](https://typing.python.org/en/latest/spec/protocol.html)
functionality, the `abstract_dataloader` is not a required
dependency for using the abstract dataloader: implementations which
implement the specifications as described are fully interoperable, even
if they do not have any mutual dependencies, including the
`abstract_dataloader`.
