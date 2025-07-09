"""ADL specification extensions and non-generic utilities.

Unlike [`abstract_dataloader.generic`][abstract_dataloader.generic], these
implementations "extend" the ADL spec by imposing a particular conceptual
framework on various functionality.

!!! warning

    This module and its submodules are not automatically imported; you will
    need to explicitly import them:

    ```python
    from abstract_dataloader.ext import sample
    ```

!!! info

    This module is not included in the test suite or CI, and is generally held
    to a lower standard than the core `abstract_dataloader`.

The current extension modules are:

- [`augment`][.]: A protocol for specifying data augmentations.
- [`graph`][.]: A programming model based on composing a DAG of callables
    into a single transform.
- [`lightning`][.]: A lightning datamodule wrapper for ADL datasets and
    pipelines.
- [`objective`][.]: Standardized learning objectives, and a
    programming model for multi-objective learning.
- [`sample`][.]: Dataset sampling utilities, including a low-discrepancy
    subset sampler.
- [`types`][.]: Type-related utilities which are not part of the core ADL spec.
"""
