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


def train_baselines(config, datasets: dict[str, MultiSplitDataset], train_rf: bool, train_gb: bool, train_mlp: bool,
                    log_run: bool):
    """
    Trains the specified baseline models on the datasets provided.
    """

    Path(config["results_path"]).mkdir(exist_ok=True, parents=True)

    for dataset_name, multisplit_dataset in datasets.items():
        train_dataset, val_dataset, test_dataset = multisplit_dataset.get_splits()

        if train_rf:
            score = baselines.RFRegressor(train_dataset.data, test_dataset.data, config["num_estimators"], config["seed"])
            save_baseline_results(config["results_path"], train_dataset.__class__.__name__, 'RF', score)

        if train_gb:
            score = baselines.GradBoostRegressor(train_dataset.data, test_dataset.data, config["seed"])
            save_baseline_results(config["results_path"], train_dataset.__class__.__name__, 'GB', score)

        if train_mlp:
            score = baselines.MLPRegressor(config["mlp"], dataset_name, multisplit_dataset, log_run)
            save_baseline_results(config["results_path"], train_dataset.__class__.__name__, 'MLP', score)


def save_baseline_results(results_path, dataset_name, experiment_name, score):
    """ """
    filename = f"baseresult_{dataset_name}_{experiment_name}.txt"
    saving_path = results_path + filename
    with open(saving_path, "w") as f:
        f.write(str(score))


def train_autoencoder(config: dict, autoencoder: nn.Module, train_dataset: CheckedDataset, val_dataset: CheckedDataset,
                      log_run: bool) -> nn.Module:
    """
    Trains a new AutoEncoder/VAE on the datasets provided.
    """

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    logger = WandbLogger(save_dir=f"./logs/{train_loader.dataset.__class__.__name__}/",
                         project="UniversalGNNs",
                         tags=["ENCODER", train_loader.dataset.__class__.__name__,
                               str(config["latent_dim"])],
                         config=config,
                         mode="online" if log_run else "disabled")
    trainer = pl.Trainer(devices=1,
                         accelerator="gpu",
                         max_epochs=config["max_epochs"],
                         log_every_n_steps=10,
                         logger=logger,
                         max_steps=config["max_steps"],
                         enable_checkpointing=False)
    trainer.fit(autoencoder, train_loader, val_loader)
    wandb.finish()

    return autoencoder


def train_single(config_all: dict[str], datasets: dict[str, MultiSplitDataset], autoencoders_dict: dict[str, nn.Module],
                 graphbuilders_dict: dict[str, GraphBuilder], regressors_dict: dict[str, nn.Module], log_run: bool):
    """ 
    Trains a UniversalGNN model on each passed dataset independently.

    If use_random_sampler config flag is set, it uses random samplers for data loading and ignores the max_steps
    argument. 
    """
    config = config_all["train_single"]
    latent_dim = config["latent_dim"]
    backbone = config["backbone"]

    for dataset_name, dataset in datasets.items():

        # create GNN
        autoencoder = {dataset_name: autoencoders_dict[dataset_name]}
        graphbuilder = {dataset_name: graphbuilders_dict[dataset_name]}
        regressor = {dataset_name: regressors_dict[dataset_name]}
        model = UniversalGNN(latent_dim, latent_dim, latent_dim, config["gnn_layers"], autoencoder, graphbuilder, regressor, backbone)

        # train the GNN
        logger = WandbLogger(save_dir="./logs/UniversalGNN_single/",
                             project="UniversalGNNs",
                             tags=["UNIVERSALGNN", str(latent_dim), dataset_name],
                             config=config_all,
                             mode="online" if log_run else "disabled")
        if config["use_random_sampler"]:
            from loader import load_multidatasets
            print(f"Training UniversalGNN single on {dataset_name} using random sampler!")
            train_loader, val_loader, test_loader = load_multidatasets(config, datasets={dataset_name: dataset})
            trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=config["epochs"], log_every_n_steps=50, logger=logger)
        else:
            print(f"Training UniversalGNN single on {dataset_name} using standard data loader!")
            train_split, validation_split, test_split = dataset.get_splits()
            train_loader = DataLoader(train_split, batch_size=config["batch_size"], shuffle=True, num_workers=0)
            val_loader = DataLoader(validation_split, batch_size=config["batch_size"], shuffle=False, num_workers=0)
            test_loader = DataLoader(test_split, batch_size=config["batch_size"], shuffle=False, num_workers=0)
            trainer = pl.Trainer(devices=1,
                                 accelerator="gpu",
                                 max_epochs=config["epochs"],
                                 max_steps=config["max_steps"],
                                 log_every_n_steps=50,
                                 logger=logger,
                                 enable_checkpointing=False)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
        wandb.finish()


def train_mutual(config_all: dict[dict[str]], loaders: tuple[DataLoader, DataLoader,
                                                             DataLoader], autoencoders_dict: dict[str, nn.Module],
                 graphbuilders_dict: dict[str, GraphBuilder], regressors_dict: dict[str, nn.Module], log_run: bool):
    """ """
    train_loader, val_loader, test_loader = loaders
    config = config_all["train_mutual"]
    latent_dim = config["latent_dim"]
    backbone = config["backbone"]


    # create GNN
    model = UniversalGNN(latent_dim, latent_dim, latent_dim, config["gnn_layers"], autoencoders_dict, graphbuilders_dict,
                         regressors_dict, backbone)

    # train the GNN
    datasets_str = [d for d in autoencoders_dict.keys()]
    logger = WandbLogger(save_dir="./logs/UniversalGNN_mutual/",
                         project="UniversalGNNs",
                         tags=["UNIVERSALGNN", str(latent_dim)] + datasets_str,
                         config=config_all,
                         mode="online" if log_run else "disabled")
    trainer = pl.Trainer(devices=1,
                         accelerator="gpu",
                         max_epochs=config["epochs"],
                         log_every_n_steps=50,
                         logger=logger,
                         enable_checkpointing=False)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    wandb.finish()