"""Lightning `DataModule` wrapper for the abstract dataloader.

!!! warning

    This module requires [Pytorch lightning](https://lightning.ai/docs/pytorch/stable/)
    to be installed:
    ```
    pip install lightning
    ```
"""

from functools import cache, cached_property
from typing import Generic, Literal, Mapping, TypeVar

import lightning as L
import numpy as np
from torch.utils.data import DataLoader

import abstract_dataloader.spec as adl
from abstract_dataloader import torch as adl_torch

Sample = TypeVar("Sample")
Raw = TypeVar("Raw", contravariant=True)
Transformed = TypeVar("Transformed")
Collated = TypeVar("Collated")
Processed = TypeVar("Processed", covariant=True)


class ADLDataModule(
    L.LightningDataModule, Generic[Raw, Transformed, Collated, Processed]
):
    """Pytorch dataloader wrapper for ADL-compliant datasets.

    This wrapper imposes the following abstraction:

    - Visualizations on the validation set should always be rendered from a
        fixed, relatively-small number of samples which should be pre-loaded.
    - The same transforms are applied to all splits.

    !!! info

        Train/val/test splits are not all required to be present; if any are
        not present, the corresponding `.{split}_dataloader()` will raise an
        error if called.

    !!! warning

        Only sample-to-sample (`.transform`) and sample-to-batch (`.collate`)
        transforms are applied in the dataloader; the training loop is
        responsible for applying batch-to-batch (`.forward`) transforms.

    Type Parameters:
        - `Raw`: raw data loaded from the dataset.
        - `Transformed`: data following CPU-side transform.
        - `Collated`: data format after collation; should be in pytorch tensors.
        - `Processed`: data after GPU-side transform.

    Args:
        dataset: datasets for each split which load the same data type.
        transforms: data transforms to apply.
        batch_size: dataloader batch size.
        samples: number of validation-set samples for visualizations.
        num_workers: number of worker processes during data loading and
            CPU-side processing.
        prefetch_factor: number of batches to fetch per worker.

    Attributes:
        transforms: data transforms which should be applied to the data; in
            particular, a `.forward()` GPU batch-to-batch stage which is
            expected to be handled by downstream model code.
    """

    def __init__(
        self, dataset: Mapping[
            Literal["train", "val", "test"], adl.Dataset[Raw]],
        transforms: adl.Pipeline[Raw, Transformed, Collated, Processed],
        batch_size: int = 32, samples: int = 0, num_workers: int = 32,
        prefetch_factor: int = 2
    ) -> None:
        super().__init__()

        self.transforms: adl.Pipeline[Raw, Transformed, Collated, Processed]
        self.transforms = transforms
        self._samples = samples

        self._dataset = dataset
        self._dataloader_args = {
            "batch_size": batch_size, "num_workers": num_workers,
            "prefetch_factor": prefetch_factor, "pin_memory": True,
            "collate_fn": transforms.collate
        }

    @cached_property
    def samples(self) -> Collated | None:
        """Validation samples for rendering samples.

        !!! warning

            While this property is cached, accessing this property the first
            time triggers a full load of the dataset validation split metadata!

            The samples are also cached in memory, so setting a large number
            of `samples` may also cause problems.

        Returns:
            Pre-loaded validation samples, nominally for generating
                visualizations. If `samples=0` was specified, or no validation
                split is provided, then no samples are returned.
        """
        if self._samples > 0 and "val" in self._dataset:
            val = self.dataset("val")
            indices = np.linspace(
                0, len(val) - 1, self._samples, dtype=np.int64)
            sampled = [val[i] for i in indices]
            return self.transforms.collate(sampled)
        else:
            return None

    @cache
    def dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> adl_torch.TransformedDataset[Raw, Transformed]:
        """Get dataset for a given split, with sample transformation applied.

        Args:
            split: target split.

        Returns:
            Dataset for that split, using the partially bound constructor
                passed to the `ADLDataModule`; the dataset is cached between
                calls.
        """
        if split not in self._dataset:
            raise KeyError(
                f"No `{split}` split was provided to this DataModule. Only "
                f"the following splits are present: "
                f"{list(self._dataset.keys())}")

        dataset = self._dataset[split]
        return adl_torch.TransformedDataset(
            dataset, transform=self.transforms.sample,
            train=(split == "train"))

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader (`shuffle=True, drop_last=True`).

        !!! note

            The underlying (transformed) dataset is cached (i.e. the same
            dataset object will be used on each call), but the dataloader
            container is not.
        """
        return DataLoader(
            self.dataset("train"), shuffle=True, drop_last=True,
            **self._dataloader_args)

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader (`shuffle=False, drop_last=True`)."""
        return DataLoader(
            self.dataset("val"), shuffle=False, drop_last=True,
            **self._dataloader_args)

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader (`shuffle=False, drop_last=False`)."""
        return DataLoader(
            self.dataset("test"), shuffle=False, drop_last=False,
            **self._dataloader_args)
