import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

from GraphBuilder import GraphBuilder
from datasets import CheckedDataset, MultiSplitDataset
from models import UniversalGNN
import baselines


def train_baselines(config, datasets: dict, train_rf: bool, train_gb: bool, train_mlp: bool):
    """
    Trains the specified baseline models on the datasets provided.
    """

    Path(config["results_path"]).mkdir(exist_ok=True, parents=True)

    for multisplit_dataset in datasets:
        train_dataset, val_dataset, test_dataset = multisplit_dataset.get_splits()

        if train_rf:
            score = baselines.RFRegressor(train_dataset.data, test_dataset.data, config["num_estimators"], config["seed"])
            save_baseline_results(config["results_path"], train_dataset.__class__.__name__, 'RF', score)

        if train_gb:
            score = baselines.GradBoostRegressor(train_dataset.data, test_dataset.data, config["seed"])
            save_baseline_results(config["results_path"], train_dataset.__class__.__name__, 'GB', score)

        if train_mlp:
            score = baselines.MLPRegressor(train_dataset, val_dataset, test_dataset, train_dataset.input_dim,
                                           train_dataset.label_dim, config["mlp_batch_size"], config["mlp_epochs"])
            save_baseline_results(config["results_path"], train_dataset.__class__.__name__, 'MLP', score)


def save_baseline_results(results_path, dataset_name, experiment_name, score):
    """ """
    filename = f"baseresult_{dataset_name}_{experiment_name}.txt"
    saving_path = results_path + filename
    with open(saving_path, "w") as f:
        f.write(str(score))


def train_autoencoder(config: dict, autoencoder: nn.Module, train_dataset: CheckedDataset,
                      val_dataset: CheckedDataset) -> nn.Module:
    """
    Trains a new AutoEncoder/VAE on the dataloader provided with the specified 
    latent dimention.

    If a savefile for the same configuration is present and load_data is set to 
    True (default), it loads the saved weights from the savefile.
    """

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=128)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=128)

    logger = WandbLogger(dir=f"./logs/{train_loader.dataset.__class__.__name__}/",
                         project="UniversalGNNs",
                         tags=["ENCODER", train_loader.dataset.__class__.__name__, str(config["latent_dim"])])
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=30, log_every_n_steps=10, logger=logger, max_steps=100_000)
    trainer.fit(autoencoder, train_loader, val_loader)
    wandb.finish()

    return autoencoder


def train_single(config: dict[str], datasets: dict[str, MultiSplitDataset], autoencoders_dict: dict[str, nn.Module],
                 graphbuilders_dict: dict[str, GraphBuilder], regressors_dict: dict[str, nn.Module]):
    """ 
    Trains a UniversalGNN model on each passed dataset independently.
    """

    for dataset_name, dataset in datasets.items():
        train_split, validation_split, test_split = dataset.get_splits()
        train_loader = DataLoader(train_split, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(validation_split, batch_size=config["batch_size"], shuffle=False, num_workers=0)
        test_loader = DataLoader(test_split, batch_size=config["batch_size"], shuffle=False, num_workers=0)
        latent_dim = config["latent_dim"]

        # create GNN
        autoencoder = {dataset_name: autoencoders_dict[dataset_name]}
        graphbuilder = {dataset_name: graphbuilders_dict[dataset_name]}
        regressor = {dataset_name: regressors_dict[dataset_name]}
        model = UniversalGNN(latent_dim, latent_dim, latent_dim, config["n_layers"], autoencoder, graphbuilder, regressor)

        # train the GNN
        logger = WandbLogger(dir="./logs/UniversalGNN/",
                             project="UniversalGNNs",
                             tags=["UNIVERSALGNN", str(latent_dim), dataset_name])
        trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=config["epochs"], log_every_n_steps=50, logger=logger)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
        wandb.config.update(config)
        wandb.finish()


def train_universal(config: dict[str], loaders: tuple[DataLoader, DataLoader, DataLoader], autoencoders_dict: dict[str,
                                                                                                                   nn.Module],
                    graphbuilders_dict: dict[str, GraphBuilder], regressors_dict: dict[str, nn.Module]):
    """ """
    train_loader, val_loader, test_loader = loaders
    latent_dim = config["latent_dim"]

    # create GNN
    model = UniversalGNN(latent_dim, latent_dim, latent_dim, config["n_layers"], autoencoders_dict, graphbuilders_dict,
                         regressors_dict)

    # train the GNN
    datasets_str = [d.__name__ for d in autoencoders_dict.keys()]
    logger = WandbLogger(dir="./logs/UniversalGNN/", project="UniversalGNNs", tags=["UNIVERSALGNN", str(latent_dim)] + datasets_str)
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=config["epochs"], log_every_n_steps=50, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    wandb.config.update(config)
    wandb.finish()