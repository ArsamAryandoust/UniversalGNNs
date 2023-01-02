import torch
import numpy as np
from torch.utils.data import Sampler, Dataset
from datasets import CheckedDataset
from typing import Iterator, List, Tuple

class MultiDataset(Dataset):
    def __init__(self, datasets: List[CheckedDataset]):
        super().__init__()
        self.dataset_lengths = [len(d) for d in datasets]
        self.datasets = datasets
        self.num_datasets = len(datasets)

        self.dataset_offsets = []
        sum = 0
        for l in self.dataset_lengths:
            self.dataset_offsets.append(sum)
            sum += l
        self.num_samples = sum
        
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        dataset_id, sample_id = self._get_dataset_sample_id(idx)
        data = self.datasets[dataset_id].data
        x, y = data[0][sample_id], data[1][sample_id]
        return (x, y, dataset_id)

    def _get_dataset_sample_id(self, idx) -> Tuple[int]:
        """
        Returns the correct dataset and index inside the dataset from the sampler index
        """
        for dataset_id, dataset_offset in enumerate(self.dataset_offsets):
            if dataset_offset + self.dataset_lengths[dataset_id] > idx:
                return dataset_id, idx - dataset_offset
        # should never happen
        return None

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        X = torch.zeros((len(batch), batch[0][0].shape[0]))
        Y = torch.zeros((len(batch), batch[0][1].shape[0]))
        dataset = batch[0][2]
        for i, (x, y, d) in enumerate(batch):
            assert d == dataset, "Error: multi-dataset batch contains different datasets!"
            X[i] = x
            Y[i] = y
        return X, Y, self.datasets[d]
        

class MultiDatasetSampler(Sampler):
    def __init__(self, multidataset: MultiDataset, batch_size: int, num_batches_per_epoch, sequential=False):
        """
        The parameter 'sequential' is only used for debugging purposes, the last batch it will generate 
        for a dataset will contain samples from the next dataset, causing an AssertionError.
        """
        super().__init__(multidataset)
        self.multidataset = multidataset
        if not sequential and not (isinstance(batch_size, int) and isinstance(num_batches_per_epoch, int)):
            raise TypeError(f"""Must provide both a integer batch size and num_batches_per_epoch when not using a sequential dataset!
                batch_size={batch_size}, num_batches_per_epoch={num_batches_per_epoch}.""")
        if batch_size <= 0 or num_batches_per_epoch <= 0:
            raise ValueError(f"""Both batch_size and num_batches_per_epoch must be >0!
                batch_size={batch_size}, num_batches_per_epoch={num_batches_per_epoch}.""")
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.sequential = sequential

    def __iter__(self) -> Iterator[int]:

        if self.sequential:
            yield from range(self.multidataset.num_samples)
        else:
            yield from self._generate_batches()

    def _generate_batches(self) -> List[int]:
        generator = np.random.default_rng(seed=42)
        datasets = torch.multinomial(torch.ones((self.multidataset.num_datasets)), self.num_batches_per_epoch, replacement=True).tolist()
        samples_list = []
        for d in datasets:
            # batch_samples = torch.multinomial(torch.ones((self.multidataset.dataset_lengths[d])), self.batch_size, replacement=False)
            batch_samples = generator.integers(0, self.multidataset.dataset_lengths[d], size=self.batch_size, dtype=np.int64)
            batch_samples += self.multidataset.dataset_offsets[d]
            samples_list += batch_samples.tolist()
        return samples_list
