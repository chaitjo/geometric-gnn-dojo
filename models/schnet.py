from typing import Optional

import torch
from torch.nn import functional as F
from torch_geometric.nn import SchNet
from torch_geometric.nn import global_add_pool, global_mean_pool


class SchNetModel(SchNet):
    """
    SchNet model from "Schnet - a deep learning architecture for molecules and materials".

    This class extends the SchNet base class for PyG.
    """
    def __init__(
        self, 
        hidden_channels: int = 128, 
        in_dim: int = 1,
        out_dim: int = 1, 
        num_filters: int = 128, 
        num_layers: int = 6,
        num_gaussians: int = 50, 
        cutoff: float = 10, 
        max_num_neighbors: int = 32, 
        pool: str = 'sum'
    ):
        """
        Initializes an instance of the SchNetModel class with the provided parameters.

        Parameters:
        - hidden_channels (int): Number of channels in the hidden layers (default: 128)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - num_filters (int): Number of filters used in convolutional layers (default: 128)
        - num_layers (int): Number of convolutional layers in the model (default: 6)
        - num_gaussians (int): Number of Gaussian functions used for radial filters (default: 50)
        - cutoff (float): Cutoff distance for interactions (default: 10)
        - max_num_neighbors (int): Maximum number of neighboring atoms to consider (default: 32)
        - pool (str): Global pooling method to be used (default: "sum")
        """
        super().__init__(
            hidden_channels, 
            num_filters, 
            num_layers, 
            num_gaussians, 
            cutoff, 
            interaction_graph=None,
            max_num_neighbors=max_num_neighbors, 
            readout=pool, 
            dipole=False, 
            mean=None, 
            std=None, 
            atomref=None
        )

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]
        
        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.Linear(hidden_channels // 2, out_dim)

    def forward(self, batch):
        
        h = self.embedding(batch.atoms)  # (n,) -> (n, d)

        row, col = batch.edge_index
        edge_weight = (batch.pos[row] - batch.pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            # # Message passing layer: (n, d) -> (n, d)
            h = h + interaction(h, batch.edge_index, edge_weight, edge_attr)

        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)  # (batch_size, out_dim)

        return out
