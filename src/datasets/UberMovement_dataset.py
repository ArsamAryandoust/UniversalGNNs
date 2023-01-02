import torch
import numpy as np
from datasets.checked_dataset import CheckedDataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class UberMovementDataset(CheckedDataset):
    """
    Loads the appropriate split of the UberMovement dataset into main memory and converts data to pytorch tensors.

    Can load data with either full delimiting points coordinates for regions or using only 2 xyz coordinates to denote the
    centroid of the region and its standard deviation
    """

    def __init__(self,
                 dataset_path: str
                 | Path = "/TasksEnergyTransition/UberMovement/",
                 split: str = "training",
                 use_region_centroids: bool = True,
                 load_data=True,
                 normalize=False,
                 sanitize=True):
        # The columns in the dataset files are:
        ### INPUTS:
        # 1) city_id
        # 2) source_id
        # 3) destination_id
        # 4) year
        # 5) quarter_of_year
        # 6) daytype
        # 7) hour_of_day
        ### LABELS:
        # 8) mean_travel_time
        # 9) standard_deviation_travel_time
        # 10) geometric_mean_travel_time
        # 11) geometric_standard_deviation_travel_time
        print("============================================================")
        print(f"Loading UberMovement dataset on {split} split:")
        self.normalize = normalize
        self.NUM_ORIGINAL_COLUMNS = 11  # original columns
        self.NUM_LABELS = 4
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.save_file = self.dataset_path / f"processed/{split}.pt"
        possible_splits = ["training", "validation", "testing"]
        if split not in possible_splits:
            raise ValueError("Split must be one of " +
                             ", ".join(possible_splits) + "!")

        if load_data and self.save_file.exists():
            print("Detected save file, trying to load it...")
            self.load_data(self.save_file)
        else:
            self._process_data(use_region_centroids)
            print("Saving data for future loads...")
            self.save_data(self.save_file)
        
        if sanitize:
            self._sanitize()
        if self.normalize:
            self._normalize_data()
        self._set_input_label_dim()
        self._sanity_check_data()

        print(f"Loaded UberMovement {self.split} split!")
        print("============================================================")


    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x, y = self.data[0][idx], self.data[1][idx]
        return x, y

    def _process_data(self, use_region_centroids):
        """
        Loads the city-id and zones information, together with the main data csv, then it
        calculates the spatial features and sets the right ones for each sample in the dataset.
        Finally it sets self.data = (X, Y)
        """
        print("Loading city ids and zones...")
        city2id: dict = self._load_city_id_csv()
        self.num_zones = {}
        city_zones_df = {}
        for city, id in city2id.items():
            # read city zones csv and map it to right id
            df = pd.read_csv(self.dataset_path / f"additional/{city}.csv")
            city_zones_df[id] = df
            self.num_zones[id] = (len(city_zones_df[id].columns) - 1) // 3

        # load the main dataframe
        print("Loading main files...")
        main_data_frame = pd.DataFrame()
        data_path = self.dataset_path / self.split
        files = list(data_path.glob("*.csv"))
        files.sort()
        for file in tqdm(files):
            df = pd.read_csv(file)
            main_data_frame = pd.concat([main_data_frame, df])

        print(main_data_frame.head())

        if use_region_centroids:
            # compute centroid and std dev for each zone (for each city)
            # instead of the source and destination ID we hace the xyz coord of the centroid and stddev
            city_zones_centroids_std = {}
            print("Calculating zone centroids...")
            for id in city2id.values():
                city_zones_centroids_std[id] = self._calculate_city_centroids(
                    city_zones_df, id)

            # build a df with the new data
            print("Building spatial data with centroids...")
            spatial_data = self._build_centroid_spatial_data(
                main_data_frame, city_zones_centroids_std)
            print(spatial_data.head())

        else:
            raise NotImplementedError(
                "Using all points for UberMovement regions is not implemented yet!"
            )

        print("Substituting the processed spatial data...")
        # put the processed spatial data inside the main_dataframe
        main_data_frame.drop(labels=["city_id", "source_id", "destination_id"],
                             axis="columns",
                             inplace=True)
        main_data_frame[spatial_data.columns] = spatial_data
        # reorder the columns to have the statial ones at the beginning
        spatial_data_cols_len = len(spatial_data.columns)
        cols = main_data_frame.columns.tolist()
        cols = cols[-spatial_data_cols_len:] + cols[:-spatial_data_cols_len]
        main_data_frame = main_data_frame[cols]

        # build the final X, Y
        self.NUM_COLUMNS = len(main_data_frame.columns)
        self.NUM_INPUTS = self.NUM_COLUMNS - self.NUM_LABELS
        X_frame = main_data_frame.iloc[:, :self.NUM_INPUTS]
        Y_frame = main_data_frame.iloc[:, self.NUM_INPUTS:self.NUM_COLUMNS]
        X = X_frame.to_numpy()
        Y = Y_frame.to_numpy()
        self.data = (torch.from_numpy(X).float(), torch.from_numpy(Y).float())

    def _load_city_id_csv(self) -> dict:
        """
        Loads city -> id mapping from disk and returns a dict where dict[city] = id
        """
        city_id_df = pd.read_csv(
            Path(self.dataset_path) / "additional/0_city_to_id_mapping.csv")
        city_id_columns = city_id_df.columns.tolist()
        city_id_dict = city_id_df.to_dict("list")
        cities_list = city_id_dict[city_id_columns[0]]
        id_list = city_id_dict[city_id_columns[1]]
        city2id = {}
        for c, id in zip(cities_list, id_list):
            city2id[c] = id
        self.num_cities = len(city2id)
        return city2id

    def _calculate_city_centroids(self, city_zones_df: dict, id: int):
        """
        Calculates the mean and std of the points in the city zones for a specific city and returns a np.array of shape 
        [2, num_zones, 3] where the array[0, :, :] contains the centroids and array[1, :, :] contains the stds
        """
        frame = city_zones_df[id]
        # some frames (London) start with x_cord_0, while others with x_cord_1
        offset = 0
        if "x_cord_0" not in frame.columns.tolist():
            offset = 1
        centroids = []
        stds = []
        for z in range(offset, self.num_zones[id] + offset):
            cols = [f"x_cord_{z}", f"y_cord_{z}", f"z_cord_{z}"]
            x_cord = frame[cols[0]].to_numpy()
            x_cord = x_cord[~np.isnan(x_cord)]
            y_cord = frame[cols[1]].to_numpy()
            y_cord = y_cord[~np.isnan(y_cord)]
            z_cord = frame[cols[2]].to_numpy()
            z_cord = z_cord[~np.isnan(z_cord)]
            cord = np.vstack([x_cord, y_cord, z_cord])

            centroids.append(cord.mean(axis=1))
            stds.append(cord.std(axis=1))
        return np.array([centroids, stds])

    def _build_centroid_spatial_data(
            self, data_frame: pd.DataFrame,
            city_zones_centroids_std: np.ndarray) -> pd.DataFrame:
        """
        Builds a spatial DataFrame of size [num_samples, 12] that contains the spatial data given by the zone centroids and stds
        in the same order as data_frame (row i of the output corresponds to row i of data_frame).
        """
        city_ids = data_frame["city_id"].to_numpy()
        source_ids = data_frame["source_id"].to_numpy()
        destination_ids = data_frame["destination_id"].to_numpy()
        source_data = np.zeros((len(city_ids), 6))
        dest_data = np.zeros((len(city_ids), 6))
        for i, (cid, sid, did) in enumerate(
                tqdm(zip(city_ids, source_ids, destination_ids),
                     total=len(city_ids))):
            # TODO: maybe find a more elegant way to do this?
            # temp fix for "London" that starts with zone 0 when all the others start with one
            # print("cid:", cid)
            # print("sid:", sid)
            # print("did:", did)
            # print("centroids shape:", city_zones_centroids_std[cid].shape)
            if cid != 7:
                sid -= 1
                did -= 1
            assert (sid >= 0)
            assert (did >= 0)
            source_data[i] = city_zones_centroids_std[cid][:, sid, :].flatten()
            dest_data[i] = city_zones_centroids_std[cid][:, did, :].flatten()
        data = np.hstack([source_data, dest_data])
        print("spatial data shape:", data.shape)
        return pd.DataFrame(data,
                            columns=[
                                "s_x_mean", "s_y_mean", "s_z_mean", "s_x_std",
                                "s_y_std", "s_z_std", "d_x_mean", "d_y_mean",
                                "d_z_mean", "d_x_std", "d_y_std", "d_z_std"
                            ])

    def save_data(self, save_file: str | Path):
        save_file = Path(save_file)
        save_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.data, save_file)
        print(f"Saved {self.split} data in {str(save_file)}!")

    def load_data(self, save_file: str | Path):
        data = torch.load(save_file)
        self.data = data[0].float(), data[1].float()
        print(f"Loaded {self.split} data from {str(save_file)}!")


if __name__ == "__main__":
    import time
    t = time.time()
    train_dataset = UberMovementDataset()
    train_dataset = UberMovementDataset(split="validation")
    train_dataset = UberMovementDataset(split="testing")
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("Shape of the data with batch size = 64:")
    for data in train_dataloader:
        x, y = data
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        break
    print(f"Time elapsed: {time.time() - t} seconds.")