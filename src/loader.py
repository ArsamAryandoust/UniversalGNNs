from datasets import MultiSplitDataset, MultiDataset, MultiDatasetBatchSampler
from datasets import ClimARTDataset, UberMovementDataset, BuildingElectricityDataset

from GraphBuilder import GraphBuilder
from models import AutoEncoder, VAE, MLP
from train import train_autoencoder

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def load_datasets(args) -> dict[str, MultiSplitDataset]:
    """
    Loads the datasets specified in the args and returns a dict with datasets["dataset_name"] == MultisplitDataset(dataset_class)
    """
    print("Loading datasets...")

    datasets_list = []
    if args.climart:
        datasets_list.append(ClimARTDataset)
    if args.uber:
        datasets_list.append(UberMovementDataset)
    if args.BE:
        datasets_list.append(BuildingElectricityDataset)
    
    datasets = {}
    for dataset_class in datasets_list:
        datasets[dataset_class.__name__] = MultiSplitDataset(dataset_class)
    
    return datasets


def load_multidatasets(config: dict[str], datasets: dict[str, MultiSplitDataset]) -> tuple[MultiDataset, MultiDataset, MultiDataset]:
    """
    Loads the multidatasets associated to the datasets given as input.

    Returns the train, validation and test dataloaders.
    """

    batch_size = config["batch_size"]
    num_batches_per_epoch = config["batches_per_epoch"]
    drop_last = config["drop_last"]

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for dataset in datasets.values():
        splits = dataset.get_splits()

        train_datasets.append(splits[0])
        val_datasets.append(splits[1])
        test_datasets.append(splits[2])

    train_dataset = MultiDataset(train_datasets)
    val_dataset = MultiDataset(val_datasets)
    test_dataset = MultiDataset(test_datasets)

    train_sampler = MultiDatasetBatchSampler(train_dataset, batch_size, num_batches_per_epoch, drop_last=drop_last)
    val_sampler = MultiDatasetBatchSampler(val_dataset, batch_size, sequential=True, drop_last=drop_last)
    test_sampler = MultiDatasetBatchSampler(test_dataset, batch_size, sequential=True, drop_last=drop_last)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=0, collate_fn=test_dataset.collate_fn)

    return train_loader, val_loader, test_loader


def load_regressors(config: dict[str], datasets: dict[str, MultiSplitDataset]) -> dict[str, nn.Module]:
    """
    Creates a regressor for each dataset and returns them in a dict.
    """
    regressors_dict = {}
    for dataset_name in datasets.keys():
        splits = datasets[dataset_name].get_splits()
        regr_input_dims = config["latent_dim"]
        # if the dataset is edge level, the out features from the GNN are the
        # concatenation of the features of the 2 nodes.
        if splits[0].edge_level == True:
            regr_input_dims *= 2
        regr_hidden_dims = (regr_input_dims + splits[0].label_dim) // 2

        regressor = MLP(regr_input_dims, [regr_hidden_dims], splits[0].label_dim)

        for split in splits:
            split.regressor = regressor
        regressors_dict[dataset_name] = regressor
    
    return regressors_dict


def load_graphbuilders(config: dict[str], datasets: dict[str, MultiSplitDataset]) -> dict[str, GraphBuilder]:
    """
    Creates a grapgbuilder object associated with the corresponding datasets and autoencoders.
    """
    graphbuilders_dict = {}
    for dataset_name in datasets.keys():
        splits = datasets[dataset_name].get_splits()
        graph_builder = GraphBuilder(distance_function=config["distance_function"],
                                        params_indeces=splits[0].spatial_temporal_indeces,
                                        connectivity=config["connectivity"],
                                        edge_level_batch=splits[0].edge_level)
        for split in splits:
            split.graph_builder = graph_builder
        graphbuilders_dict[dataset_name] = graph_builder
    
    return graphbuilders_dict


def load_encoders(config: dict[str], datasets: dict[str, MultiSplitDataset], graphbuilders: dict[str, GraphBuilder]) -> dict[str, nn.Module]:
    """
    Loads or trains the autoencoders for all datasets. 
    Returns the encoders dict where encoders_dict["dataset_name"] == encoder
    """
    encoder_classes = {
        "AutoEncoder": AutoEncoder,
        "VAE": VAE
    }
    encoder_class = encoder_classes[config["encoder_class"]]

    autoencoders_dict = {}

    for dataset_name, multidataset in datasets.items():
        train_dataset, validation_dataset, _ = multidataset.get_splits()
        if train_dataset.edge_level:
            autoencoder = encoder_class(train_dataset.encoder_input_dim, config["latent_dim"])
            autoencoder.set_edge_level_graphbuilder(graphbuilders[dataset_name])
        else:
            autoencoder = encoder_class(train_dataset.input_dim, config["latent_dim"])
        encoder_name = encoder_class.__name__
        savefile_path = Path(f"/UniversalGNNs/checkpoints/encoders/{dataset_name}") / f"{encoder_name}_{config['latent_dim']}.pt"
        if savefile_path.exists() and config["load_checkpoint"]:
            autoencoder.load_state_dict(torch.load(savefile_path))
            if train_dataset.edge_level:
                autoencoder.set_edge_level_graphbuilder(graphbuilders[dataset_name])
            print("Loaded autoencoder checkpoint in: ", savefile_path)
        else:
            print("Training new autoencoder and saving in: ", savefile_path)
            savefile_path.parent.mkdir(parents=True, exist_ok=True)
            train_autoencoder(config, autoencoder, train_dataset, validation_dataset)
            torch.save(autoencoder.state_dict(), savefile_path)
        
        autoencoder.requires_grad_(False)
        autoencoders_dict[dataset_name] = autoencoder
        graphbuilders[dataset_name].set_encoder(autoencoder)

    return autoencoders_dict