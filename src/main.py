from datasets import MultiSplitDataset, MultiDataset, MultiDatasetBatchSampler, ClimARTDataset
from models import AutoEncoder, VAE, UniversalGNN, MLP
from GraphBuilder import GraphBuilder

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# TODO: handle multiple configurations from command line arguments


def train_autoencoder(encoder_class: type, dataloader: DataLoader, load_data: bool = True, latent_dim: int = 512) -> nn.Module:
    """
    Trains a new AutoEncoder/VAE on the dataloader provided with the specified latent dimention.

    If a savefile for the same configuration is present and load_data is set to True (default), it loads the 
    saved weights from the savefile.
    """
    autoencoder = encoder_class(dataloader.dataset.input_dim, latent_dim)
    dataset_name = dataloader.dataset.__class__.__name__
    encoder_name = encoder_class.__name__
    savefile_path = Path(f"/UniversalGNNs/checkpoints/encoders/{dataset_name}") / f"{encoder_name}_{latent_dim}.pt"
    if savefile_path.exists() and load_data:
        autoencoder.load_state_dict(torch.load(savefile_path))
        print("Loaded autoencoder checkpoint in: ", savefile_path)
    else:
        print("Training new autoencoder and saving in: ", savefile_path)
        savefile_path.parent.mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger("./logs/", name=encoder_class.__name__, version=dataloader.dataset.__class__.__name__)
        trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=30, log_every_n_steps=10, logger=logger)
        trainer.fit(autoencoder, dataloader)
        torch.save(autoencoder.state_dict(), savefile_path)

    return autoencoder


def load_datasets(dataset_classes: list[type],
                  batch_size: int,
                  num_batches_per_epoch: int = 1000) -> tuple[MultiDataset, MultiDataset, MultiDataset]:
    """
    Loads the datasets specified from dataset_classes, together with the autoencoder(s) needed to feed the information
    into the GNN and the final regressor.

    Returns the train, validation and test dataloaders.
    """
    LATENT_DIM = 512
    LOAD_ENCODER = True
    GRAPH_CONNECTIVITY = 0.5
    DEVICE = "cpu"
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for dataset_class in dataset_classes:
        dataset = MultiSplitDataset(dataset_class)
        splits = dataset.get_splits()

        # Load the autoencoder for the dataset and create a graphbuilder to assign to it
        dataloader = DataLoader(splits[0], batch_size=batch_size, shuffle=False, num_workers=128)
        autoencoder = train_autoencoder(AutoEncoder, dataloader, load_data=LOAD_ENCODER, latent_dim=LATENT_DIM)
        # freeze the autoencoder layers
        autoencoder.requires_grad_(False)
        # create a GraphBuilder and assign it to all the splits
        graph_builder = GraphBuilder(distance_function="euclidean",
                                            params_indeces=splits[0].spatial_temporal_indeces,
                                            connectivity=GRAPH_CONNECTIVITY,
                                            encoder=autoencoder,
                                            edge_level_batch=splits[0].edge_level,
                                            device=DEVICE)
        for split in splits:
            split.graph_builder = graph_builder
        
        # create a regressor and assign it to every split
        regr_input_dims = LATENT_DIM
        # if the dataset is edge level, the out features from the GNN are the concatenation of the features of the 2 nodes.
        if splits[0].edge_level == True:
            regr_input_dims *= 2
        regr_hidden_dims = (regr_input_dims + splits[0].label_dim) // 2
        regressor = MLP(regr_input_dims, regr_hidden_dims, splits[0].label_dim)
        for split in splits:
            split.regressor = regressor

        train_datasets.append(splits[0])
        val_datasets.append(splits[1])
        test_datasets.append(splits[2])

    train_dataset = MultiDataset(train_datasets)
    val_dataset = MultiDataset(val_datasets)
    test_dataset = MultiDataset(test_datasets)

    train_sampler = MultiDatasetBatchSampler(train_dataset, batch_size, num_batches_per_epoch, drop_last=True)
    val_sampler = MultiDatasetBatchSampler(val_dataset, batch_size, sequential=True, drop_last=True)
    test_sampler = MultiDatasetBatchSampler(test_dataset, batch_size, sequential=True, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=128, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=128, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=128, collate_fn=test_dataset.collate_fn)

    return train_loader, val_loader, test_loader


def main(args):
    # load the datasets
    dataset_classes = [ClimARTDataset]
    model = UniversalGNN(512, 512, 512)
    train_loader, val_loader, test_loader = load_datasets(dataset_classes, 128, 10)
    # train the GNN
    logger = TensorBoardLogger("./logs/", name="UniversalGNN")
    trainer = pl.Trainer( accelerator="cpu", max_epochs=1, log_every_n_steps=1, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    main(None)