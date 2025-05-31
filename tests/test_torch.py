"""Pytorch module registration."""

import numpy as np
import torch
from torch import nn

from abstract_dataloader import abstract
from abstract_dataloader import torch as adl_torch


def test_composition():
    """Test that module composition is correctly applied across pipelines."""

    class _BatchModule(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.parameter = nn.Parameter(torch.zeros(()))

    class _PipelineModule(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.collate = adl_torch.Collate(mode='stack')
            self._batch = _BatchModule()

        def forward(self, data):
            raise NotImplementedError

        def sample(self, data):
            raise NotImplementedError

        def batch(self, data):
            raise NotImplementedError

    p1 = _PipelineModule()
    p2 = _PipelineModule()

    assert set(p1.parameters()) == set(p1._batch.parameters())

    sequenced = adl_torch.SequencePipeline(p1)
    assert isinstance(sequenced, nn.Module)
    assert set(sequenced.parameters()) == set(p1.parameters())

    paralleled = adl_torch.ParallelPipelines(p1=p1, p2=p2)
    assert set(paralleled.parameters()) == set(p1.parameters()).union(
        p2.parameters())

    composed = adl_torch.ComposedPipeline(
        pipeline=p1, pre=lambda x: x, post=lambda x: x)  # type: ignore
    assert set(composed.parameters()) == set(p1.parameters())


def test_collate():
    """Test collate function."""
    data = {
        "a": np.random.uniform(size=(3, 5)),
        "b": np.random.uniform(size=(3, 4))
    }

    collate1 = adl_torch.Collate(mode='stack')
    c1 = collate1([{"a": data["a"][0], "b": data["b"][0]},
              {"a": data["a"][1], "b": data["b"][1]},
              {"a": data["a"][2], "b": data["b"][2]}])
    assert isinstance(c1["a"], torch.Tensor)
    assert isinstance(c1["b"], torch.Tensor)
    assert torch.all(c1["a"] == torch.from_numpy(data["a"]))
    assert torch.all(c1["b"] == torch.from_numpy(data["b"]))

    collate2 = adl_torch.Collate(mode='concat')
    c2 = collate2([{"a": data["a"][0][None], "b": data["b"][0][None]},
              {"a": data["a"][1][None], "b": data["b"][1][None]},
              {"a": data["a"][2][None], "b": data["b"][2][None]}])
    assert isinstance(c2["a"], torch.Tensor)
    assert isinstance(c2["b"], torch.Tensor)
    assert torch.all(c2["a"] == torch.from_numpy(data["a"]))
    assert torch.all(c2["b"] == torch.from_numpy(data["b"]))


def test_stacked_sequence():
    """Test StackedSequencePipeline transform."""
    pipeline = abstract.Pipeline(
        sample=lambda data: {
            "a": np.array(data["a"]),  # type: ignore
            "b": np.array(data["b"])   # type: ignore
        },
        collate=lambda data: {
            "a": torch.stack([torch.tensor(d["a"]) for d in data]),
            "b": torch.stack([torch.tensor(d["b"]) for d in data]),
        },
        batch=lambda data: {
            "a": data["a"] * 2,
            "b": data["b"] * 3,
        },
    )
    stacked_pipeline = adl_torch.StackedSequencePipeline(pipeline)

    raw = [
        [{"a": [1, 2], "b": [3, 4]}, {"a": [5, 6], "b": [7, 8]}],
        [{"a": [9, 10], "b": [11, 12]}, {"a": [13, 14], "b": [15, 16]}],
    ]

    transformed = [stacked_pipeline.sample(x) for x in raw]
    collated = stacked_pipeline.collate(transformed)
    processed = stacked_pipeline.batch(collated)

    assert isinstance(processed["a"], torch.Tensor)
    assert isinstance(processed["b"], torch.Tensor)
    assert processed["a"].shape == (2, 2, 2)  # batch, sequence, features
    assert processed["b"].shape == (2, 2, 2)
    assert torch.all(processed["a"] == torch.tensor(
        [[[2, 4], [10, 12]], [[18, 20], [26, 28]]]))
    assert torch.all(processed["b"] == torch.tensor(
        [[[9, 12], [21, 24]], [[33, 36], [45, 48]]]))


def test_transformed_dataset():
    """Test TransformedDataset functionality."""
    class MockDataset:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    def mock_transform(sample):
        return sample * 2

    raw_data = [1, 2, 3]
    dataset = MockDataset(raw_data)
    transformed_dataset = adl_torch.TransformedDataset(
        dataset, mock_transform)  # type: ignore

    assert len(transformed_dataset) == len(raw_data)
    for i in range(len(transformed_dataset)):
        assert transformed_dataset[i] == raw_data[i] * 2
