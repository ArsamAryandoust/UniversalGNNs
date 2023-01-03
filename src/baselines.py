from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import RegressorChain
from datasets import ClimARTDataset, UberMovementDataset, MultiSplitDataset
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
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import r2_score
device = torch.device("cuda")

class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        print(f"MLP with: layers of size: {input_size} -> {hidden_size} -> {hidden_size} -> {output_size}.")
        self.layer_norm = nn.LayerNorm(input_size)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden(x))
        x = self.output_layer(x)
        return x
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x, y = x.float(), y.float()
        out = self(x)
        loss =  F.mse_loss(out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('validation_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.float(), y.float()
        out = self(x)
        loss =  F.mse_loss(out, y)
        r2 = r2_score(y, out)
        self.log('test_loss', loss, on_epoch=True)
        self.log('r2_score', r2, on_epoch=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0].float())
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

def MLPRegressor(train_dataset, validation_dataset, test_dataset, input_dim, label_dim, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=64, shuffle=False)
    mlp = MLP(input_dim, 2048, label_dim)
    # mlp = MLP.load_from_checkpoint("lightning_logs/version_10/checkpoints/epoch=9-step=1410.ckpt", input_size=50, hidden_size=128, output_size=1)
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=30, log_every_n_steps=10)
    trainer.fit(mlp, train_loader, validation_loader)
    # trainer.test(mlp, validation_loader)
    return trainer.test(mlp, test_loader)

datasets = [ClimARTDataset, UberMovementDataset]

scores = {}
for dataset_class in datasets:
    batch_size = 128
    multisplit_dataset = MultiSplitDataset(dataset_class, test=False)
    train_dataset, val_dataset, _ = multisplit_dataset.get_splits()


    scores[dataset_class.__name__] = {}
    if dataset_class == UberMovementDataset:
        batch_size = 4096
        test_dataset = val_dataset
    else:
        test_dataset = dataset_class(split="testing", normalize=True)

    scores[dataset_class.__name__]["RF"] = RFRegressor(train_dataset.data, test_dataset.data)
    # scores[dataset_class.__name__]["GB"] = GradBoostRegressor(train_dataset.data, test_dataset.data)
    scores[dataset_class.__name__]["MLP"] = MLPRegressor(train_dataset, val_dataset, test_dataset, train_dataset.input_dim, train_dataset.label_dim, batch_size)


print(scores)
with open("results_baselines.txt", "w") as f:
    f.write(str(scores))

# CLIMART:
# {'RF': 0.8670452416481914, 'GB': 0.9635100152090146, 'MLP': [{'test_loss': 28312.16015625, 'r2_score': -122365976576.0}]}
# {'ClimARTDataset': {'MLP': [{'test_loss': 21988.9375, 'r2_score': -3057724.0}]}}
# UberMovement:
# {'UberMovementDataset': {'MLP': [{'test_loss': 10136.76171875, 'r2_score': 0.5676587224006653}]}}
# {'UberMovementDataset': {'RF': 0.5863605102531544}}