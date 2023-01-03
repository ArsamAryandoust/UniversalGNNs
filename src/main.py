from datasets import MultiSplitDataset, MultiDataset, MultiDatasetBatchSampler, ClimARTDataset
from models import AutoEncoder, VAE, UniversalGNN

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# TODO: handle multiple configurations from command line arguments

def train_autoencoder(encoder_class: type, dataloader: DataLoader, load_data:bool=True, latent_dim:int=512) -> nn.Module:
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
    else:
        savefile_path.parent.mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger("./logs/", name="AE", version="ClimART_train")
        trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=30, log_every_n_steps=10, logger=logger)
        trainer.fit(dataloader)
        torch.save(autoencoder.state_dict(), savefile_path)
    
    return autoencoder


def load_datasets(dataset_classes: list[type], batch_size:int, num_batches_per_epoch:int=1000) -> tuple[MultiDataset, MultiDataset, MultiDataset]:
    """
    Loads the datasets specified from dataset_classes and returns the train, validation and test dataloaders.
    """
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for dataset_class in dataset_classes:
        dataset = MultiSplitDataset(dataset_class)
        splits = dataset.get_splits()
        # TODO: should we load the autoencoders/GraphBuilders here?
        train_datasets.append(splits[0])
        val_datasets.append(splits[1])
        test_datasets.append(splits[2])

    train_dataset = MultiDataset(train_datasets)
    val_dataset = MultiDataset(val_datasets)
    test_dataset = MultiDataset(test_datasets)

    train_sampler = MultiDatasetBatchSampler(train_dataset, batch_size, num_batches_per_epoch)
    val_sampler = MultiDatasetBatchSampler(val_dataset, batch_size, sequential=True)
    test_sampler = MultiDatasetBatchSampler(test_dataset, batch_size, sequential=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, batch_sampler=train_sampler, num_workers=128, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, batch_sampler=val_sampler, num_workers=128, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, batch_sampler=test_sampler, num_workers=128, collate_fn=test_dataset.collate_fn)

    return train_loader, val_loader, test_loader

def main(args):
    # load the datasets
    dataset_classes = [ClimARTDataset]
    train_loader, val_loader, test_loader = load_datasets(dataset_classes, 128)
    model = UniversalGNN(512)
    # TODO: train the GNN
