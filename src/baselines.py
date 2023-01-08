from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import RegressorChain
from models import MLP
import time

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
def RFRegressor(
    HYPER, 
    train_data, 
    test_data
):
    
    """ """
    
    print("Fitting a RF regressor:")
    t = time.time()
    RFRegressor = RandomForestRegressor(
        n_estimators=128, 
        random_state=HYPER.SEED, 
        n_jobs=-1
    )
    RFRegressor.fit(*train_data)
    print(time.time() - t, "seconds elapsed!")
    score = RFRegressor.score(*test_data)
    print("score:", score)
    
    return score



#####################################
#         Gradient Boosting         #
#####################################
def GradBoostRegressor(
    HYPER, 
    train_data, 
    test_data
):
    
    """ """
    
    print("Fitting a Gradient Boosting regressor:")
    t = time.time()
    regressor = RegressorChain(
        GradientBoostingRegressor(random_state=HYPER.SEED), 
        verbose=True
    )
    regressor.fit(*train_data)
    print(time.time() - t, "seconds elapsed!")

    score = regressor.score(*test_data)
    print("score:", score)
    
    return score



#####################################
#               MLP                 #
#####################################
def MLPRegressor(
    train_dataset, 
    validation_dataset, 
    test_dataset, 
    input_dim, 
    label_dim, 
    batch_size=64, 
    epochs = 30
):

    """ """
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=128, 
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=batch_size, 
        num_workers=128, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=128, 
        shuffle=False
    )
    mlp = MLP(input_dim, [512, 256, 256], label_dim)
    logger = WandbLogger(
        dir=f"./logs/{train_dataset.__class__.__name__}/",
        project="UniversalGNNs",
        tags=["BASELINE", "MLP", train_dataset.__class__.__name__]
    )
    trainer = pl.Trainer(
        devices=1, 
        accelerator="gpu", 
        max_epochs=epochs, 
        log_every_n_steps=100, 
        logger=logger
    )
    trainer.fit(
        mlp, 
        train_loader, 
        validation_loader
    )
    
    score = trainer.test(mlp, test_loader)
    
    return score
