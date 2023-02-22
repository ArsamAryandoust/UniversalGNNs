from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import RegressorChain
from models import MLP
from datasets import MultiSplitDataset
import time
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import r2_score
from pytorch_lightning.loggers import WandbLogger


#####################################
#           Random Forests          #
#####################################
def RFRegressor(train_data, test_data, num_estimators, seed):
    """ """

    print("Fitting a RF regressor:")
    t = time.time()
    RFRegressor = RandomForestRegressor(n_estimators=num_estimators, random_state=seed, n_jobs=-1)
    RFRegressor.fit(*train_data)
    print(time.time() - t, "seconds elapsed!")
    score = RFRegressor.score(*test_data)
    print("score:", score)

    return score


#####################################
#         Gradient Boosting         #
#####################################
def GradBoostRegressor(train_data, test_data, seed):
    """ """

    print("Fitting a Gradient Boosting regressor:")
    t = time.time()
    regressor = RegressorChain(GradientBoostingRegressor(random_state=seed), verbose=True)
    regressor.fit(*train_data)
    print(time.time() - t, "seconds elapsed!")

    score = regressor.score(*test_data)
    print("score:", score)

    return score


#####################################
#               MLP                 #
#####################################
def MLPRegressor(config: dict[str], dataset_name: str, multisplit_dataset: MultiSplitDataset, log_run: bool):
    """ """
    use_random_sampler = config["use_random_sampler"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    dropout = config["dropout"]
    train_dataset, validation_dataset, test_dataset = multisplit_dataset.get_splits()
    input_dim = train_dataset.input_dim
    label_dim = train_dataset.label_dim

    if use_random_sampler:
        from loader import load_multidatasets
        train_loader, validation_loader, test_loader = load_multidatasets(config, {"dataset":multisplit_dataset})
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=128, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=128, shuffle=False)
    mlp = MLP(input_dim, [512, 256, 256], label_dim, dropout)
    if log_run:
        logger = WandbLogger(dir=f"./logs/{dataset_name}/",
                            project="UniversalGNNs",
                            tags=["BASELINE", "MLP", dataset_name], config=config)
        
    else:
        logger = False
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=epochs, log_every_n_steps=100, logger=logger, enable_checkpointing=False)
    trainer.fit(mlp, train_loader, validation_loader)

    score = trainer.test(mlp, test_loader)
    wandb.finish()

    return score
