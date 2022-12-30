import torch
from datasets import CheckedDataset

class MultiSplitDataset(CheckedDataset):
    """
    Allows to load and process multiple splits of the same dataset.
    
    Parameters:
     - dataset_class (class): the dataset from which to load the splits.
     - train (bool): load training split.
     - val (bool): load validation split.
     - test (bool): load testing split.
     - normalize (bool): if True normalizes the training split and then uses the same mean/std to normalize 
     validation and testing splits. Cannot be used when train=False.
     - sanitize (bool): if True sanitizes the dataset (before possible normalization).
    
    Can pass any named argument to the underlying dataset class via kwargs.
    """
    def __init__(self, dataset_class, train=True, val=True, test=True, normalize=True, sanitize=True, **kwargs):
        super().__init__()
        self.dataset_class = dataset_class
        self.train = train
        self.val = val
        self.test = test

        self.datasets : dict[str, CheckedDataset] = {}
        self.samples_range = {}
        self.num_samples = 0
        if train:
            self._add_dataset("training", **kwargs)
        if val:
            self._add_dataset("validation", **kwargs)
        if test:
            self._add_dataset("testing", **kwargs)
    
        X = torch.vstack([d.data[0] for d in self.datasets.values()])
        Y = torch.vstack([d.data[1] for d in self.datasets.values()])
        self.data = X, Y

        if sanitize:
            self._sanitize()

        if normalize:
            if not train:
                raise ValueError("Impossible to normalize a multi-split dataset without train split!")
            full_data = self.data
            train_start, train_end = self.samples_range["training"][0], self.samples_range["training"][1]
            mean, std = self._get_normalization_values(self.data[0][train_start: train_end])
            X = (self.data[0] - mean) / std
            Y = self.data[1]
            self.data = X, Y

        self._sanity_check_data()
        self._set_input_label_dim()
        for split, (start, end) in self.samples_range.items():
            self.datasets[split].data = self.data[0][start:end], self.data[1][start:end]
            self.datasets[split]._set_input_label_dim()
        print("Loaded successfully multi-split dataset!")
    
    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x, y = self.data[0][idx], self.data[1][idx]
        return x, y

    def _add_dataset(self, split: str, **kwargs):
        dataset = self.dataset_class(split=split, sanitize=False, normalize=False, **kwargs)
        self.datasets[split] = dataset
        self.samples_range[split] = (self.num_samples, self.num_samples + len(dataset))
        self.num_samples += len(dataset)

    def get_splits(self) -> tuple[CheckedDataset | None, CheckedDataset | None, CheckedDataset | None]:
        """
        Returns the single-split datasets in (train, val, test) order with None if a split was not loaded.
        """
        train_dataset = None
        val_dataset = None
        test_dataset = None
        if self.train:
            train_dataset = self.datasets["training"]
        if self.val:
            val_dataset = self.datasets["validation"]
        if self.test:
            test_dataset = self.datasets["testing"]
        return train_dataset, val_dataset, test_dataset