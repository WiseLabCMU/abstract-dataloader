"""ADL "extensions": generic modules which impose conceptual constraints.

Unlike `abstract_dataloader.generic`, these implementations "extend" the ADL
spec by imposing a particular conceptual framework on various functionality.

!!! warning

    This module and its submodules are not automatically imported; you will
    need to explicitly import them:

    ```python
    from abstract_dataloader.ext import sample
    ```

!!! warning

    This module is not included in the test suite or CI, and is generally held
    to a lower standard than the core `abstract_dataloader`.

The current extension modules are:

- [`objective`][.]: Standardized learning objectives, and a
    programming model for multi-objective learning.
- [`sample`][.]: Dataset sampling utilities, including a low-discrepancy
    subset sampler
"""
