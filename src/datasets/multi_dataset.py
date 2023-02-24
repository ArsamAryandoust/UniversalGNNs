import torch
import numpy as np
import random
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


class MultiDatasetBatchSampler(Sampler):

    def __init__(self,
                 multidataset: MultiDataset,
                 batch_size: int,
                 num_batches_per_epoch: int = None,
                 sequential: bool = False,
                 drop_last: bool = True):
        """
        Batch sampler for the MultiDataset class. For each batch it first samples the dataset uniformly at random,
        then it samples batch_size samples uniformly at random, with repetition.

        If sequential is set to True then the datasets will be sampled in order from the first to the 
        last dataset and from the first to the last sample in each dataset. The last samples that don't 
        Fit a full batch will either be sampled or not depending on the flag drop_last.
        """
        super().__init__(multidataset)
        self.drop_last = drop_last
        self.multidataset = multidataset
        if not isinstance(batch_size, int):
            raise TypeError(
                f"""Must provide an integer batch size to sample: batch_size={batch_size}."""
            )
        if not sequential and (not isinstance(num_batches_per_epoch, int) or num_batches_per_epoch <= 0):
            raise TypeError(
                f"""Must provide an integer num_batches_per_epoch > 0 when not using a sequential dataset!
                batch_size={batch_size}, num_batches_per_epoch={num_batches_per_epoch}."""
            )
        if batch_size <= 0 :
            raise ValueError(
                f"""Both batch_size must be > 0! batch_size={batch_size}."""
            )
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.sequential = sequential
        self.generator = torch.Generator()

    def __len__(self) -> int:
        if self.sequential:
            return self.multidataset.num_samples // self.batch_size + int(not self.drop_last)
        else:
            return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[int]:

        if self.sequential:
            self.sequential_dataset = 0
            self.sequential_sample = 0
            batch = self._generate_sequential_batch()
            while batch is not None:
                yield batch
                batch = self._generate_sequential_batch()

        else:
            for _ in range(self.num_batches_per_epoch):
                yield self._generate_random_batch()

    def _generate_sequential_batch(self):
        while self.sequential_dataset < self.multidataset.num_datasets:
            start = self.sequential_sample
            end = start + self.batch_size
            
            # get the dataset end sample
            dataset_end = 0
            if self.sequential_dataset < self.multidataset.num_datasets - 1:
                dataset_end = self.multidataset.dataset_offsets[self.sequential_dataset + 1]
            else:
                dataset_end = self.multidataset.num_samples
            
            # check if the end is outside of the current dataset
            if end > dataset_end:
                self.sequential_sample = dataset_end
                self.sequential_dataset += 1
                if not self.drop_last:
                    return torch.arange(start=start, end=dataset_end)
            else:
                self.sequential_sample = end
                return torch.arange(start=start, end=end)
        return None

    def _generate_random_batch(self) -> List[int]:
        dataset_idx = random.randint(0, self.multidataset.num_datasets - 1)
        batch_samples = torch.randint(
            low=0,
            high=self.multidataset.dataset_lengths[dataset_idx],
            size=(self.batch_size,),
            generator=self.generator)
        # batch_samples = generator.integers(0, self.multidataset.dataset_lengths[dataset_idx], size=self.batch_size, dtype=np.int64)
        batch_samples += self.multidataset.dataset_offsets[dataset_idx]
        return batch_samples


def _test():
    from loader import load_datasets, load_multidatasets
    sets = {
        "all_datasets": True
    }
    datasets = load_datasets(sets)
    config = {
        "batch_size": 1024,
        "batches_per_epoch": 1000,
        "drop_last": True
    }
    load_multidatasets(config, datasets)
