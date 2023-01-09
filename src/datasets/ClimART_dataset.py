import torch
import numpy as np
from datasets.checked_dataset import CheckedDataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class ClimARTDataset(CheckedDataset):
    """
    Loads the appropriate split of the ClimART dataset into main memory and converts data to pytorch tensors
    """

    def __init__(self,
                 dataset_path: str | Path = "/EnergyTransitionTasks/ClimART/pristine/",
                 split: str = "training",
                 normalize=False,
                 sanitize=True):
        print("============================================================")
        print(f"Loading ClimART dataset on {split} split:")
        self.edge_level = False
        self.spatial_temporal_indeces = list(range(5))
        self.normalize = normalize
        self.NUM_COLUMNS = 1268
        self.NUM_INPUTS = 970
        self.dataset_path = dataset_path
        self.split = split
        possible_splits = ["training", "validation", "testing"]
        if split not in possible_splits:
            raise ValueError("Split must be one of " +
                             ", ".join(possible_splits) + "!")

        main_data_frame = pd.DataFrame()
        data_path = Path(dataset_path) / split
        files = list(data_path.glob("*.csv"))
        files.sort()
        print("Loading files:")
        for file in tqdm(files):
            frame = pd.read_csv(file)
            if len(frame.columns) != self.NUM_COLUMNS:
                raise RuntimeError(f"""The number of columns in the csv file 
                    is different from the expected: expected {self.NUM_COLUMNS}, got {len(frame.columns)}."""
                                   )
            main_data_frame = pd.concat([main_data_frame, frame], ignore_index=True)

        X_frame = main_data_frame.iloc[:, :self.NUM_INPUTS]
        Y_frame = main_data_frame.iloc[:, self.NUM_INPUTS:self.NUM_COLUMNS]
        X = X_frame.to_numpy()
        Y = Y_frame.to_numpy()

        self.data = (torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        self.input_dim = X.shape[1]
        self.label_dim = Y.shape[1]

        if sanitize:
            self._sanitize()
        if self.normalize:
            self._normalize_data()
        self._set_input_label_dim()
        self._sanity_check_data()

        print(f"Loaded ClimART {split} split! Number of samples: {len(self.data[0])}")
        print("============================================================")

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x, y = self.data[0][idx], self.data[1][idx]
        return x, y


if __name__ == "__main__":
    import time
    t = time.time()
    train_dataset = ClimARTDataset(normalize=True)
    val_dataset = ClimARTDataset(split="validation")
    test_dataset = ClimARTDataset(split="testing")
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("Shape of the data with batch size = 64:")
    for data in train_dataloader:
        x, y = data
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        break
    print(f"Time elapsed: {time.time() - t} seconds.")
