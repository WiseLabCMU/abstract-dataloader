# Tests

The test suite isn't fully automated yet; run tests with
```sh
uv run --extra testing coverage run -m pytest tests
```
and get the code coverage report with
```sh
uv run coverage report
```

We currently have 100% coverage, with a few "non-functional" items excluded:
- `__repr__` methods
- Protocols, `...` placeholders
- `NotImplementedError`s
