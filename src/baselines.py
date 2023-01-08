from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import RegressorChain
from datasets import ClimARTDataset, UberMovementDataset, MultiSplitDataset
from models import MLP
import time

SEED = 42


#####################################
#           Random Forests          #
#####################################
def RFRegressor(train_data, test_data):
    print("Fitting a RF regressor:")
    t = time.time()
    RFRegressor = RandomForestRegressor(n_estimators=128, random_state=SEED, n_jobs=256)
    RFRegressor.fit(*train_data)
    print(time.time() - t, "seconds elapsed!")
    score = RFRegressor.score(*test_data)
    print("score:", score)
    return score
    # ============== CLIMART ===================
    # score: 0.41579297007803917 -> full dataset
    # score: 0.8668907242816928  -> "inf" values set to 0


#####################################
#         Gradient Boosting         #
#####################################
def GradBoostRegressor(train_data, test_data):
    print("Fitting a Gradient Boosting regressor:")
    t = time.time()
    regressor = RegressorChain(GradientBoostingRegressor(random_state=SEED), verbose=True)
    regressor.fit(*train_data)
    print(time.time() - t, "seconds elapsed!")

    score = regressor.score(*test_data)
    print("score:", score)
    return score
    # ============== CLIMART ===================
    # score: 0.4399313160686012 -> full dataset


#####################################
#               MLP                 #
#####################################
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import r2_score
from pytorch_lightning.loggers import WandbLogger

device = torch.device("cuda")


class MLP_PL(MLP, pl.LightningModule):

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__(input_size, hidden_sizes, output_size)
        print(f"MLP with: layers of size: {input_size} -> {hidden_sizes} -> {output_size}.")
        assert len(hidden_sizes) > 0, "MLP must have at least 1 hidden layer!"
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden = nn.Sequential()
        for i in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.hidden.append(nn.ReLU())

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.hidden(x)
        x = self.output_layer(x)
        return x

    def common_step(self, batch, split: str):
        x, y = batch
        out = self(x)
        loss = F.mse_loss(out, y)
        r2 = r2_score(out, y)
        self.log(f"{split} loss", loss, on_epoch=True, batch_size=len(x))
        self.log(f"{split} R2", r2, on_epoch=True, batch_size=len(x))
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "training")
    
    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, "validation")
    
    def test_step(self, batch, batch_idx):
        return self.common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


def MLPRegressor(train_dataset, validation_dataset, test_dataset, input_dim, label_dim, batch_size=64, epochs = 30):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=128, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=128, shuffle=False)
    mlp = MLP_PL(input_dim, [512, 256, 256], label_dim)
    logger = WandbLogger(dir=f"./logs/{train_dataset.__class__.__name__}/",
                             project="UniversalGNNs",
                             tags=["BASELINE", "MLP", train_dataset.__class__.__name__])
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=epochs, log_every_n_steps=100, logger=logger)
    trainer.fit(mlp, train_loader, validation_loader)
    return trainer.test(mlp, test_loader)


datasets = [ClimARTDataset]

scores = {}
for dataset_class in datasets:
    batch_size = 2048
    multisplit_dataset = MultiSplitDataset(dataset_class, normalize_full=False)
    train_dataset, val_dataset, test_dataset = multisplit_dataset.get_splits()

    scores[dataset_class.__name__] = {}

    # scores[dataset_class.__name__]["RF"] = RFRegressor(train_dataset.data, test_dataset.data)
    # scores[dataset_class.__name__]["GB"] = GradBoostRegressor(train_dataset.data, test_dataset.data)
    scores[dataset_class.__name__]["MLP"] = MLPRegressor(train_dataset, val_dataset, test_dataset, train_dataset.input_dim,
                                                         train_dataset.label_dim, batch_size, 200)

print(scores)
with open("results_baselines.txt", "w") as f:
    f.write(str(scores))

# CLIMART:
# {'RF': 0.8670452416481914, 'GB': 0.9635100152090146, 'MLP': [{'test_loss': 28312.16015625, 'r2_score': -122365976576.0}]}
# {'ClimARTDataset': {'MLP': [{'test_loss': 21988.9375, 'r2_score': -3057724.0}]}}
# UberMovement:
# {'UberMovementDataset': {'MLP': [{'test_loss': 10136.76171875, 'r2_score': 0.5676587224006653}]}}
# {'UberMovementDataset': {'RF': 0.5863605102531544}}