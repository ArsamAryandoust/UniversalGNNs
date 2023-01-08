from datasets import MultiSplitDataset, MultiDataset, MultiDatasetBatchSampler
from datasets import ClimARTDataset, UberMovementDataset
from models import AutoEncoder, VAE, UniversalGNN, MLP
from GraphBuilder import GraphBuilder

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pathlib import Path

import hyper
import baselines

# TODO: handle multiple configurations from command line arguments


def train_autoencoder(
    encoder_class: type, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    load_data: bool = True, 
    latent_dim: int = 512
) -> nn.Module:
    """
    Trains a new AutoEncoder/VAE on the dataloader provided with the specified latent dimention.

    If a savefile for the same configuration is present and load_data is set to True (default), it loads the 
    saved weights from the savefile.
    """
    autoencoder = encoder_class(train_loader.dataset.input_dim, latent_dim)
    dataset_name = train_loader.dataset.__class__.__name__
    encoder_name = encoder_class.__name__
    savefile_path = Path(f"/UniversalGNNs/checkpoints/encoders/{dataset_name}") / f"{encoder_name}_{latent_dim}.pt"
    if savefile_path.exists() and load_data:
        autoencoder.load_state_dict(torch.load(savefile_path))
        print("Loaded autoencoder checkpoint in: ", savefile_path)
    else:
        print("Training new autoencoder and saving in: ", savefile_path)
        savefile_path.parent.mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(dir=f"./logs/{train_loader.dataset.__class__.__name__}/",
                             project="UniversalGNNs",
                             tags=["ENCODER", train_loader.dataset.__class__.__name__,
                                   str(latent_dim)])
        trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=30, log_every_n_steps=10, logger=logger, max_steps=100_000)
        trainer.fit(autoencoder, train_loader, val_loader)
        wandb.finish()
        torch.save(autoencoder.state_dict(), savefile_path)

    return autoencoder


def load_datasets(
    dataset_classes: list[type],
    batch_size: int,
    num_batches_per_epoch: int = 1000, 
    latent_dim=512
) -> tuple[tuple[MultiDataset, MultiDataset, MultiDataset], dict, dict, dict]:
    """
    Loads the datasets specified from dataset_classes, together with the autoencoder(s) needed to feed the information
    into the GNN and the final regressor. 
    Returns the train, validation and test dataloaders, together with the autoencoders, GraphBuilders and regressors.

    return: ((train_loader, val_loader, test_loader), autoencoders_dict, graphbuilders_dict, regressors_dict)

    All the dicts are of the form dict[dataset_name] = value, where dataset_name == dataset.__name__
    """
    LOAD_ENCODER = True
    GRAPH_CONNECTIVITY = 0.5
    train_datasets = []
    val_datasets = []
    test_datasets = []
    autoencoders_dict = {}
    graphbuilders_dict = {}
    regressors_dict = {}
    for dataset_class in dataset_classes:
        dataset = MultiSplitDataset(dataset_class)
        splits = dataset.get_splits()

        # Load the autoencoder for the dataset and create a graphbuilder to assign to it
        dataloader = DataLoader(splits[0], batch_size=batch_size, shuffle=True, num_workers=128)
        val_dataloader = DataLoader(splits[1], batch_size=batch_size, shuffle=False, num_workers=128)
        autoencoder = train_autoencoder(AutoEncoder, dataloader, val_dataloader, load_data=LOAD_ENCODER, latent_dim=latent_dim)
        # freeze the autoencoder layers
        autoencoder.requires_grad_(False)
        # create a GraphBuilder and assign it to all the splits
        graph_builder = GraphBuilder(distance_function="euclidean",
                                     params_indeces=splits[0].spatial_temporal_indeces,
                                     connectivity=GRAPH_CONNECTIVITY,
                                     encoder=autoencoder,
                                     edge_level_batch=splits[0].edge_level)
        for split in splits:
            split.graph_builder = graph_builder

        # create a regressor and assign it to every split
        regr_input_dims = latent_dim
        # if the dataset is edge level, the out features from the GNN are the concatenation of the features of the 2 nodes.
        if splits[0].edge_level == True:
            regr_input_dims *= 2
        regr_hidden_dims = (regr_input_dims + splits[0].label_dim) // 2
        regressor = MLP(regr_input_dims, [regr_hidden_dims], splits[0].label_dim)
        for split in splits:
            split.regressor = regressor

        train_datasets.append(splits[0])
        val_datasets.append(splits[1])
        test_datasets.append(splits[2])

        autoencoders_dict[dataset_class.__name__] = autoencoder
        graphbuilders_dict[dataset_class.__name__] = graph_builder
        regressors_dict[dataset_class.__name__] = regressor

    train_dataset = MultiDataset(train_datasets)
    val_dataset = MultiDataset(val_datasets)
    test_dataset = MultiDataset(test_datasets)

    train_sampler = MultiDatasetBatchSampler(train_dataset, batch_size, num_batches_per_epoch, drop_last=True)
    val_sampler = MultiDatasetBatchSampler(val_dataset, batch_size, sequential=True, drop_last=True)
    test_sampler = MultiDatasetBatchSampler(test_dataset, batch_size, sequential=True, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=0, collate_fn=test_dataset.collate_fn)

    return (train_loader, val_loader, test_loader), autoencoders_dict, graphbuilders_dict, regressors_dict


def main(HYPER):

    """ """
    
    # load the datasets
    (
        loaders, 
        autoencoders_dict, 
        graphbuilders_dict, 
        regressors_dict 
    )= load_datasets(HYPER.DATASET_CLASS_LIST, 128, 1000, HYPER.LATENT_DIM)
    train_loader, val_loader, test_loader = loaders
    model = UniversalGNN(
        HYPER.LATENT_DIM, 
        HYPER.LATENT_DIM, 
        HYPER.LATENT_DIM, 
        autoencoders_dict, 
        graphbuilders_dict, 
        regressors_dict
    )
    
    # train the GNN
    datasets_str = [d.__name__ for d in HYPER.DATASET_CLASS_LIST]
    logger = WandbLogger(
        dir="./logs/UniversalGNN/",
        project="UniversalGNNs",
        tags=["UNIVERSALGNN", str(HYPER.LATENT_DIM)] + datasets_str
    )
    trainer = pl.Trainer(
        devices=1, 
        accelerator="gpu", 
        max_epochs=HYPER.MAX_EPOCHS, 
        log_every_n_steps=1, 
        logger=logger
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


def run_baseline_experiments(HYPER):
    
    """ """

    device = torch.device("cuda")
    
    scores = {}
    for dataset_class in HYPER.DATASET_CLASS_LIST:
        multisplit_dataset = MultiSplitDataset(dataset_class)
        train_dataset, val_dataset, test_dataset = multisplit_dataset.get_splits()

        scores[dataset_class.__name__] = {}
        if HYPER.RUN_BASELINE_RF:
            scores[dataset_class.__name__]["RF"] = baselines.RFRegressor(
                HYPER,
                train_dataset.data, 
                test_dataset.data
            )
        if HYPER.RUN_BASELINE_GB:
            scores[dataset_class.__name__]["GB"] = baselines.GradBoostRegressor(
                HYPER,
                train_dataset.data, 
                test_dataset.data
            )
        if HYPER.RUN_BASELINE_MLP:
            scores[dataset_class.__name__]["MLP"] = baselines.MLPRegressor(
                train_dataset, 
                val_dataset, 
                test_dataset, 
                train_dataset.input_dim,
                train_dataset.label_dim, 
                HYPER.BATCH_SIZE_BASELINE, 
                HYPER.EPOCHS_BASELINE
            )

    # show results
    print(scores)
    
    # save results
    if HYPER.SAVE_BASELINE_RESULTS:
        filename = 'results_baselines.txt'
        saving_path = HYPER.PATH_TO_RESULTS + filename
        with open(saving_path, "w") as f:
            f.write(str(scores))
    


if __name__ == "__main__":
    
    # create hyper parameter instance
    HYPER = hyper.HyperParameter()
    
    # run main experiments if chosen so
    if HYPER.RUN_MAIN_EXPERIMENTS:
        main(HYPER)
    
    # run baseline experiments if chosen so
    if HYPER.RUN_BASELINE_EXPERIMENTS:
        run_baseline_experiments(HYPER)
        
        
        
