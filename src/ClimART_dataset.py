import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class ClimARTDataset(Dataset):
    """
    Loads the appropriate split of the ClimART dataset into main memory and converts data to pytorch tensors
    """

    def __init__(self, dataset_path: str | Path = "/TasksEnergyTransition/ClimART/", split: str = "training"):
        print("============================================================")
        print(f"Loading ClimART dataset on {split} split:")
        self.NUM_COLUMNS = 1268
        self.NUM_INPUTS = 970
        self.dataset_path = dataset_path
        self.split = split
        possible_splits = ["training", "validation", "testing"]
        if split not in possible_splits:
            raise ValueError("Split must be one of " + ", ".join(possible_splits) + "!")

        main_data_frame = pd.DataFrame()
        data_path = Path(dataset_path) / split
        files = list(data_path.glob("*.csv"))
        files.sort()
        print("Loading files:")
        for file in tqdm(files):
            frame = pd.read_csv(file)
            if len(frame.columns) != self.NUM_COLUMNS:
                raise RuntimeError(f"""The number of columns in the csv file 
                    is different from the expected: expected {self.NUM_COLUMNS}, got {len(frame.columns)}.""")
            main_data_frame = pd.concat([main_data_frame, frame])
        
        X_frame = frame.iloc[:, :self.NUM_INPUTS]
        Y_frame = frame.iloc[:, self.NUM_INPUTS:self.NUM_COLUMNS]
        X = X_frame.to_numpy()
        Y = Y_frame.to_numpy()
        # TODO: some labels are 9 * 10^36, how do we manage them? For now setting them to 0...
        # could also remove entirely the columns
        Y[Y > 1e30] = 0
        self.data = (torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        self.input_dim = X.shape[1]
        self.label_dim = Y.shape[1]
        print(f"Loaded ClimART {split} split!")
        print("============================================================")
        # just to be sure we don't have the same problem in the future ;D
        assert torch.count_nonzero(self.data[0] > 1e30) == 0, "Error: Values > 1e30 in X!"
        assert torch.count_nonzero(self.data[0] < -1e30) == 0, "Error: Values < -1e30 in X!"
        assert torch.count_nonzero(self.data[1] > 1e30) == 0, "Error: Values > 1e30 in y!"
        assert torch.count_nonzero(self.data[1] < -1e30) == 0, "Error: Values < -1e30 in y!"


    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x, y = self.data[0][idx], self.data[1][idx]
        return x, y

if __name__ == "__main__":
    import time
    t = time.time()
    train_dataset = ClimARTDataset()
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