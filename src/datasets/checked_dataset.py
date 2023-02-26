import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from GraphBuilder import GraphBuilder
import torch.nn as nn

class CheckedDataset(ABC, Dataset):
    def __init__(self):
        super().__init__()
        self.data = None
        # attributes necessary to the UniversalGNN
        self.spatial_temporal_indeces: list[int] = None
        self.edge_level : bool = None
        self.graph_builder : GraphBuilder = None
        self.regressor : nn.Module = None

    @abstractmethod
    def __len__(self, idx):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def _remove_columns_from_tensor(self, tensor: torch.Tensor, mask: torch.Tensor, message: str) -> torch.Tensor:
        """
        removes the columns specified with the mask from the tensor and 
        prints the message if any columns were removed
        """
        tensor = tensor[:, ~mask]
        num_removed = torch.count_nonzero(mask)
        if num_removed != 0:
            print("Removed", int(num_removed), message)
            print("Indices:", mask.nonzero().flatten())
        return tensor


    def _sanitize(self):
        """
        scans the data and removes bad input and label dimensions, where there are:
         - nan
         - inf
         - > 1e18 / < -1e18
         - uninformative features (where std is 0)
        """
        X, Y = self.data

        nan_remove = (torch.isnan(X).sum(dim=0) > 0)
        X = self._remove_columns_from_tensor(X, nan_remove, "nan features!")
        nan_remove = (torch.isnan(Y).sum(dim=0) > 0)
        Y = self._remove_columns_from_tensor(Y, nan_remove, "nan labels!")

        inf_remove = (torch.isinf(X).sum(dim=0) > 0)
        X = self._remove_columns_from_tensor(X, inf_remove, "inf features!")
        inf_remove = (torch.isinf(Y).sum(dim=0) > 0)
        Y = self._remove_columns_from_tensor(Y, inf_remove, "inf labels!")
        


        too_big = torch.bitwise_or(X > 1e18, X < -1e18).sum(dim=0) > 0
        X = self._remove_columns_from_tensor(X, too_big, "features > 1e18 or < -1e18!")
        too_big = torch.bitwise_or(Y > 1e18, Y < -1e18).sum(dim=0) > 0
        Y = self._remove_columns_from_tensor(Y, too_big, "labels > 1e18 or < -1e18!")

        informative = X.std(dim=0) != 0
        X = self._remove_columns_from_tensor(X, ~informative, "uninformative features!")

        self.data = X, Y

    def _get_normalization_values(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate mean and std for normalization and check for nans or std==0.
        """
        X, Y = data
        
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        Y_mean = Y.mean(dim=0)
        Y_std = Y.std(dim=0)

        X_mean_nan = torch.count_nonzero(torch.isnan(X_mean))
        X_std_nan = torch.count_nonzero(torch.isnan(X_std))
        X_std_zero = torch.count_nonzero(X_std == 0)
        Y_mean_nan = torch.count_nonzero(torch.isnan(Y_mean))
        Y_std_nan = torch.count_nonzero(torch.isnan(Y_std))
        Y_std_zero = torch.count_nonzero(Y_std == 0)
        assert X_mean_nan == 0,   "Error: nan values in X mean when trying to normalize!"
        assert X_std_nan == 0,    "Error: nan values in X std when trying to normalize!"
        assert X_std_zero == 0,   "Error: 0 values in X std when trying to normalize!"
        assert Y_mean_nan == 0,   "Error: nan values in Y mean when trying to normalize!"
        assert Y_std_nan == 0,    "Error: nan values in Y std when trying to normalize!"
        assert Y_std_zero == 0,   "Error: 0 values in Y std when trying to normalize!"

        return X_mean, X_std, Y_mean, Y_std

    def _normalize_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize the input data over features and labels across all samples and check that the mean/std are not nan
        """
        X_mean, X_std, Y_mean, Y_std = self._get_normalization_values(self.data)
        X = (X - X_mean) / X_std
        Y = (Y - Y_mean) / Y_std
        self.data = X, Y
        return X_mean, X_std, Y_mean, Y_std

    def _sanity_check_data(self):
        """
        Executes a sanity check on the data to ensure that it has no nans, infinity or other possible problems
        """
        X, Y = self.data
        try:
            # infinity / nan
            assert torch.count_nonzero(torch.isnan(X)) == 0,    "Error: nan values in X!"
            assert torch.count_nonzero(torch.isnan(Y)) == 0,    "Error: nan values in Y!"
            assert torch.count_nonzero(torch.isinf(X)) == 0,    "Error: nan values in X!"
            assert torch.count_nonzero(torch.isinf(Y)) == 0,    "Error: nan values in Y!"

            # checking for values larger/smaller than 1e18 because they can generate 
            # a +/-inf when using a squared error due to limited float max value
            assert torch.count_nonzero(X > 1e18) == 0,      "Error: Values > 1e18 in X!"
            assert torch.count_nonzero(X < -1e18) == 0,     "Error: Values < -1e18 in X!"
            assert torch.count_nonzero(Y > 1e18) == 0,      "Error: Values > 1e18 in y!"
            assert torch.count_nonzero(Y < -1e18) == 0,     "Error: Values < -1e18 in y!"
        except AssertionError as e:
            print(e)
        else:
            print("All sanity checks passed!")

    def _set_input_label_dim(self):
        self.input_dim = self.data[0].shape[1]
        self.label_dim = self.data[1].shape[1]

if __name__ == "__main__":
    from datasets import ClimARTDataset
    climart_train = ClimARTDataset(split="testing", normalize=True)