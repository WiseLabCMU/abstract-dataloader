"""Windowing and sequence transforms."""

import numpy as np
import pytest

from abstract_dataloader import abstract, spec, torch


class _ConcreteMetadata(spec.Metadata):

    def __init__(self, n: int = 10):
        self.timestamps = np.arange(n, dtype=np.float64)

class _ConcreteSensor(abstract.Sensor[int, _ConcreteMetadata]):

    def __getitem__(self, index: int | np.integer) -> int:
        return int(index)


def _apply_pipeline(pipeline, raw):
    transformed = [pipeline.sample(x) for x in raw]
    collated = pipeline.collate(transformed)
    processed = pipeline.batch(collated)
    return processed


@pytest.mark.parametrize("parallel", [None, 1, 2])
def test_window(parallel):
    """Test windowing."""
    meta = _ConcreteMetadata(5)

    partialed = torch.Window.from_partial_sensor(
        lambda path: _ConcreteSensor(meta),
        past=1, future=1, parallel=parallel)
    windowed = partialed("ignored")

    assert windowed[0] == [0, 1, 2]
    assert windowed[1] == [1, 2, 3]
    assert windowed[2] == [2, 3, 4]
    assert np.all(windowed.metadata.timestamps == np.array([1, 2, 3]))


def test_transform():
    """Test sequence transforms."""
    pipeline = abstract.Pipeline(
        sample=lambda data: data * 2,  # type: ignore
        collate=lambda data: data,
        batch=lambda data: [x * 5 for x in data])
    sequenced = torch.SequencePipeline(pipeline)

    raw = [[1, 4], [2, 5], [3, 6]]
    processed = _apply_pipeline(sequenced, raw)
    assert processed == [[10, 20, 30], [40, 50, 60]]


def test_parallel():
    """Test parallel transforms."""
    pipeline_a = abstract.Pipeline(
        sample=lambda data: data * 2,  # type: ignore
        collate=lambda data: data,
        batch=lambda data: [x * 5 for x in data])

    pipeline_b = abstract.Pipeline(
        sample=lambda data: data + 1,  # type: ignore
        collate=lambda data: data,
        batch=lambda data: [x + 2 for x in data])

    composed = torch.ParallelPipelines(a=pipeline_a, b=pipeline_b)
    raw = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    processed = _apply_pipeline(composed, raw)
    assert processed == {"a": [10, 30], "b": [5, 7]}


def test_sequential():
    """Test sequential composition."""
    pipeline = abstract.Pipeline(collate=lambda data: data)
    pre = lambda data: data * 2
    post = lambda data: [x * 5 for x in data]
    raw = [1, 2, 3]

    composed = torch.ComposedPipeline(pipeline=pipeline, pre=pre, post=post)
    processed = _apply_pipeline(composed, raw)
    assert processed == [10, 20, 30]

    composed = torch.ComposedPipeline(pipeline=pipeline, pre=pre)
    processed = _apply_pipeline(composed, raw)
    assert processed == [2, 4, 6]

    composed = torch.ComposedPipeline(pipeline=pipeline, post=post)
    processed = _apply_pipeline(composed, raw)
    assert processed == [5, 10, 15]

    composed = torch.ComposedPipeline(pipeline=pipeline)
    assert _apply_pipeline(composed, raw) == _apply_pipeline(pipeline, raw)
