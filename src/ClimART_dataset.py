import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd


class ClimARTDataset(Dataset):
    """
    Loads the appropriate split of the ClimART dataset into main memory and converts data to pytorch tensors
    """

    def __init__(self, dataset_path: str | Path = "/TasksEnergyTransition/ClimART/", split: str = "training"):
        self.NUM_COLUMNS = 1268
        self.NUM_INPUTS = 970
        self.dataset_path = dataset_path
        self.split = split
        possible_splits = ["training", "validation", "testing"]
        if split not in possible_splits:
            raise ValueError("Split must be one of " + ", ".join(possible_splits) + "!")

        data_path = Path(dataset_path) / split
        files = data_path.glob("*.csv")

        # dimension of inputs and labels
        X = np.zeros((0, self.NUM_INPUTS))
        Y = np.zeros((0, self.NUM_COLUMNS - self.NUM_INPUTS))

        for file in files:
            frame = pd.read_csv(file)
            if len(frame.columns) != self.NUM_COLUMNS:
                raise RuntimeError(f"""The number of columns in the csv file 
                    is different from the expected: expected {self.NUM_COLUMNS}, got {len(frame.columns)}.""")
            X_frame = frame.iloc[:, :self.NUM_INPUTS]
            Y_frame = frame.iloc[:, self.NUM_INPUTS:self.NUM_COLUMNS]
            X = np.vstack([X, X_frame.to_numpy()])
            Y = np.vstack([Y, Y_frame.to_numpy()])
        self.data = (torch.from_numpy(X), torch.from_numpy(Y))
        print(f"Loaded ClimART {split} split!")


    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x, y = self.data[0][idx], self.data[1][idx]
        return x, y

if __name__ == "__main__":
    train_dataset = ClimARTDataset()
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("Shape of the data with batch size = 64:")
    for data in train_dataloader:
        x, y = data
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        break