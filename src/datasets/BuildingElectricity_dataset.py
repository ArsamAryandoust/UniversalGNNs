import torch
import numpy as np
from datasets import CheckedDataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class BuildingElectricityDataset(CheckedDataset):
    """
    Loads the appropriate split of the BuildingElectricity dataset into main memory and converts data to pytorch tensors
    """

    def __init__(self,
                 dataset_path: str | Path = "/EnergyTransitionTasks/BuildingElectricity/",
                 split: str = "training",
                 normalize=False,
                 sanitize=True):
        print("============================================================")
        print(f"Loading BuildingElectricity dataset on {split} split:")
        self.edge_level = False
        self.spatial_temporal_indeces = []
        self.normalize = normalize
        self.dataset_path = dataset_path
        self.split = split
        self.NUM_LABELS = 96
        possible_splits = ["training", "validation", "testing"]
        if split not in possible_splits:
            raise ValueError("Split must be one of " + ", ".join(possible_splits) + "!")

        additional_data_path = Path(dataset_path) / "additional/building_images_pixel_histograms_rgb.csv"
        buildings_df = pd.read_csv(additional_data_path)

        main_data_frame = pd.DataFrame()
        data_path = Path(dataset_path) / split
        files = list(data_path.glob("*.csv"))
        files.sort()
        print("Loading files:")
        for file in tqdm(files):
            frame = pd.read_csv(file)
            main_data_frame = pd.concat([main_data_frame, frame], ignore_index=True)

        spatial_data_df = self._get_spatial_data(main_data_frame, buildings_df)
        main_data_frame.drop(labels=["building_id"], axis="columns", inplace=True)
        main_data_frame = pd.concat([spatial_data_df, main_data_frame], axis=1)

        self.NUM_COLUMNS = len(main_data_frame.columns)
        self.NUM_INPUTS = self.NUM_COLUMNS - self.NUM_LABELS
        X_frame = main_data_frame.iloc[:, :self.NUM_INPUTS]
        Y_frame = main_data_frame.iloc[:, self.NUM_INPUTS:self.NUM_COLUMNS]
        X = X_frame.to_numpy()
        Y = Y_frame.to_numpy()

        self.data = (torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        self._set_input_label_dim()

        if sanitize:
            self._sanitize()
        if self.normalize:
            self._normalize_data()
        self._set_input_label_dim()
        self._sanity_check_data()

        print(f"Loaded BuildingElectricity {split} split! Number of samples: {len(self.data[0])}")
        print("============================================================")

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x, y = self.data[0][idx], self.data[1][idx]
        return x, y

    def _get_spatial_data(self, main_data_frame: pd.DataFrame, buildings_df: pd.DataFrame) -> pd.DataFrame:
        buildings = main_data_frame["building_id"]

        spatial_data = np.zeros((len(buildings), buildings_df.shape[0]))
        for i, building_id in enumerate(buildings):
            spatial_data[i] = buildings_df[f"building_{int(building_id)}"].to_numpy()

        return pd.DataFrame(spatial_data)

if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader
    t = time.time()
    train_dataset = BuildingElectricityDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    print("Shape of the data with batch size = 64:")
    for data in train_dataloader:
        x, y = data
        print("X:")
        print(x[0][:20])
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        break
    print(f"Time elapsed: {time.time() - t} seconds.")