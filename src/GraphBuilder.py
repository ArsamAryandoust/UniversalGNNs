import torch
import torch_geometric
import numpy as np
import scipy

class GraphBuilder:
    def __init__(self, distance_function, params_indeces, connectivity, encoder, edge_level_batch=False):
        self.distance_function = distance_function
        self.params_indeces = params_indeces
        self.connectivity = connectivity
        self.encoder = encoder
        self.edge_level_batch = edge_level_batch
    
    def compute_nodes_matrix(self, batch):
        return self.encoder.get_latent(batch)

    def compute_edges_matrices(self, batch, device):
        distance_features_indeces = torch.tensor(self.params_indeces, dtype=torch.long, device=device)
        distance_features = torch.index_select(batch, dim=1, index=distance_features_indeces)
        if isinstance(self.distance_function, int):
          distances_matrix = torch.cdist(distance_features, distance_features, self.distance_function)
        else:
          distance_features = distance_features.cpu().detach().numpy()
          distances_matrix = scipy.spatial.distance.cdist(distance_features, distance_features, self.distance_function)
          distances_matrix = torch.from_numpy(distances_matrix).to(device)
        min_distance, max_distance = distances_matrix.min(), distances_matrix.max()
        distances_matrix = (distances_matrix - min_distance) / (max_distance - min_distance)
        scores_matrix = 1 - distances_matrix
        scores_matrix = scores_matrix - torch.eye(scores_matrix.shape[0]).to(device)
        sparsity = 1 - self.connectivity
        quantile = torch.quantile(scores_matrix, q=sparsity)
        scores_matrix = torch.where(scores_matrix > quantile, scores_matrix, 0.)
        edges_indeces, edges_weights = torch_geometric.utils.dense_to_sparse(scores_matrix)
        return edges_indeces, edges_weights

    def score_function(self, x):
        if np.equal(x, 0.) or np.equal(x, np.inf):
            result = 0.
        else:
            result = 1 - x
        return result

    def compute_row_level_batch(self, batch, device):
        distance_features_indeces_1 = torch.tensor(self.params_indeces[0][0], dtype=torch.long, device=device)
        distance_features__indeces_2 = torch.tensor(self.params_indeces[1][0], dtype=torch.long, device=device)
        node_features_indeces_1 = torch.tensor(self.params_indeces[0][1], dtype=torch.long, device=device)
        node_features_indeces_2 = torch.tensor(self.params_indeces[1][1], dtype=torch.long, device=device)
        distance_features_1 = torch.index_select(batch, dim=1, index=distance_features_indeces_1)
        distance_features_2 = torch.index_select(batch, dim=1, index=distance_features__indeces_2)
        node_features_1 = torch.index_select(batch, dim=1, index=node_features_indeces_1)
        node_features_2 = torch.index_select(batch, dim=1, index=node_features_indeces_2)
        row_level_batch_1 = torch.hstack((distance_features_1, node_features_1))
        row_level_batch_2 = torch.hstack((distance_features_2, node_features_2))
        row_level_batch = torch.vstack((row_level_batch_1, row_level_batch_2))
        self.params_indeces = list(range(distance_features_indeces_1.shape[0]))
        return row_level_batch

    def compute_graph(self, batch, device):
        if self.edge_level_batch:
            batch = self.compute_row_level_batch(batch, device)
        nodes_matrix = self.compute_nodes_matrix(batch)
        edges_indeces, edges_weights = self.compute_edges_matrices(batch, device)
        return nodes_matrix, edges_indeces, edges_weights
