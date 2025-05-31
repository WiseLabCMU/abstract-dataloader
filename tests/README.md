# Tests

The test suite isn't fully automated yet; run tests with
```sh
uv run coverage run -m pytest
```
and get the code coverage report with
```sh
uv run coverage report
```

We currently have 100% coverage on non-pytorch code, but no tests on the pytorch portion yet since I'm not sure how to best test GPU-specific use cases (that generally cause most of the problems).

A few items are excluded:
- `__repr__` methods
- Protocols, `...` placeholders
- `NotImplementedError`s
