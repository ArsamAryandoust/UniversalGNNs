from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import RegressorChain
from ClimART_dataset import ClimARTDataset
from UberMovement_dataset import UberMovementDataset
import time

SEED = 42

# load data
climart_train = ClimARTDataset(split="training")
climart_val = ClimARTDataset(split="validation")
climart_test = ClimARTDataset(split="testing")
input_dim = climart_train.data[0].shape[1]
label_dim = climart_train.data[1].shape[1]

train_data = climart_train.data
print("Number of training samples:", len(train_data[0]))
#####################################
#           Random Forests          #
#####################################
def RFRegressor(train_data, test_data):
    print("Fitting a RF regressor:")
    t = time.time()
    RFRegressor = RandomForestRegressor(random_state=SEED)
    RFRegressor.fit(*train_data)
    print(time.time() - t, "seconds elapsed!")
    score = RFRegressor.score(*test_data)
    print("score:", score)
    return score
    # ============== CLIMART ===================
    # score: 0.41579297007803917 -> full dataset
    # score: 0.8668907242816928  -> "inf" values set to 0

#####################################
#               SVM                 #
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
from torchmetrics import R2Score
device = torch.device("cuda")
r2score  = R2Score(num_outputs=label_dim).to(device)

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
        r2 = r2score(out, y)
        self.log('test_loss', loss, on_epoch=True)
        self.log('r2_score', r2, on_epoch=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0].float())
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

def MLPRegressor(train_dataset, validation_dataset, test_dataset, input_dim, label_dim):
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=64, num_workers=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=64, shuffle=False)
    mlp = MLP(input_dim, 2048, label_dim)
    # mlp = MLP.load_from_checkpoint("lightning_logs/version_10/checkpoints/epoch=9-step=1410.ckpt", input_size=50, hidden_size=128, output_size=1)
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=30, log_every_n_steps=10)
    trainer.fit(mlp, train_loader, validation_loader)
    # trainer.test(mlp, validation_loader)
    return trainer.test(mlp, test_loader)

climart_scores = {}
climart_scores["RF"] = RFRegressor(climart_train.data, climart_test.data)
climart_scores["GB"] = GradBoostRegressor(climart_train.data, climart_test.data)
climart_scores["MLP"] = MLPRegressor(climart_train, climart_val, climart_test, climart_train.input_dim, climart_train.label_dim)


print(climart_scores)
with open("results_baselines.txt", "w") as f:
    f.write(str(climart_scores))