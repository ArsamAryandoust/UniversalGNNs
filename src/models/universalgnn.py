import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGCNLayer, LayerNorm, Sequential
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from GraphBuilder import GraphBuilder
from torchmetrics.functional import r2_score

class GNN(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.in_conv = GCNConv(in_channels, hidden_channels)
        self.in_norm = LayerNorm(hidden_channels)
        self.out_conv = GCNConv(hidden_channels, out_channels)
        self.out_norm = LayerNorm(out_channels)
        self.act = nn.ReLU()
        self.in_deeplayer = DeepGCNLayer(self.in_conv, self.in_norm, self.act)
        deeplayers_list = []
        for i in range(n_layers - 2):
            conv = GCNConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels)
            deeplayers_list.append((DeepGCNLayer(conv, norm, self.act), 'x, edge_index, edge_weight -> x'))
        self.hidden_deeplayers = Sequential('x, edge_index, edge_weight', deeplayers_list)
        self.out_deeplayer = DeepGCNLayer(self.out_conv, self.out_norm, nn.Identity())

    def forward(self, node_matrix: torch.Tensor, edge_index: torch.Tensor, edge_weights) -> torch.Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.in_deeplayer(node_matrix, edge_index, edge_weights)
        x = self.hidden_deeplayers(x, edge_index, edge_weights)
        x = self.out_deeplayer(x.float(), edge_index, edge_weights)
        return x.float()


class UniversalGNN(pl.LightningModule):

    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, n_layers: int, autoencoders_dict: dict[str, nn.Module],
                 graphbuilders_dict: dict[str, GraphBuilder], regressors_dict: dict[str, nn.Module]):
        super().__init__()
        self.save_hyperparameters()
        self.gnn = GNN(latent_dim, hidden_dim, out_dim, n_layers)
        self.autoencoders = nn.ModuleDict(autoencoders_dict)
        self.graphbuilders = graphbuilders_dict
        self.regressors = nn.ModuleDict(regressors_dict)
        if len(self.autoencoders) == 1:
            for dataset_name in self.autoencoders.keys():
                self.default_dataset_name = dataset_name

    def forward(self, x: torch.Tensor, dataset_name: str):
        batch_size = x.shape[0]
        nodes_matrix, edges_indeces, edges_weights = self.graphbuilders[dataset_name].compute_graph(x, self.device)
        out = self.gnn(nodes_matrix, edges_indeces, edges_weights)
        if self.graphbuilders[dataset_name].edge_level_batch:
            source = out[:batch_size]
            target = out[batch_size:]
            assert len(source) == len(target), f"""
                Error: edge-level batch has different sizes of source and target: {len(source)} vs {len(target)}"""
            out = torch.hstack([source, target])
        return self.regressors[dataset_name](out)

    def common_step(self, batch, split: str):
        if len(batch) == 3:
            x, y, dataset = batch
            dataset_name = type(dataset).__name__
        elif len(batch) == 2:
            x, y = batch
            dataset_name = self.default_dataset_name
        else:
            raise RuntimeError(f"Encountered abnormal batch of length {len(batch)}:\n {batch}")
        out = self(x, dataset_name)
        loss = F.mse_loss(out, y)
        r2 = r2_score(out, y)
        self.log(f"{dataset_name} {split} loss", loss, on_epoch=True, batch_size=len(x))
        self.log(f"{dataset_name} {split} R2", r2, on_epoch=True, batch_size=len(x))
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "training")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, "validation")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer