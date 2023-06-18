from typing import Callable

import torch
from torch.nn import functional as F
from torch_scatter import scatter

from models.layers.spherenet_layer import *


class SphereNetModel(torch.nn.Module):
    """
    SphereNet model from "Spherical Message Passing for 3D Molecular Graphs". 
    """
    def __init__(
        self, 
        cutoff: float = 10, 
        num_layers: int = 4,
        hidden_channels: int = 128, 
        in_dim: int = 1, 
        out_dim: int = 1, 
        int_emb_size: int = 64,
        basis_emb_size_dist: int = 8, 
        basis_emb_size_angle: int = 8, 
        basis_emb_size_torsion: int = 8, 
        out_emb_channels: int = 128,
        num_spherical: int = 7, 
        num_radial: int = 6, 
        envelope_exponent: int = 5,
        num_before_skip: int = 1, 
        num_after_skip: int = 2, 
        num_output_layers: int = 2,
        act: Callable = swish, 
        output_init: str = 'GlorotOrthogonal', 
        use_node_features: bool = True
    ):
        """
        Initializes an instance of the SphereNetModel class with the following parameters:

        Parameters:
        - cutoff (int): Cutoff distance for interactions (default: 10)
        - num_layers (int): Number of layers in the model (default: 4)
        - hidden_channels (int): Number of channels in the hidden layers (default: 128)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - int_emb_size (int): Embedding size for interaction features (default: 64)
        - basis_emb_size_dist (int): Embedding size for distance basis functions (default: 8)
        - basis_emb_size_angle (int): Embedding size for angle basis functions (default: 8)
        - basis_emb_size_torsion (int): Embedding size for torsion basis functions (default: 8)
        - out_emb_channels (int): Number of channels in the output embeddings (default: 128)
        - num_spherical (int): Number of spherical harmonics (default: 7)
        - num_radial (int): Number of radial basis functions (default: 6)
        - envelope_exponent (int): Exponent of the envelope function (default: 5)
        - num_before_skip (int): Number of layers before the skip connections (default: 1)
        - num_after_skip (int): Number of layers after the skip connections (default: 2)
        - num_output_layers (int): Number of output layers (default: 2)
        - act (function): Activation function (default: swish)
        - output_init (str): Initialization method for the output layer (default: 'GlorotOrthogonal')
        - use_node_features (bool): Whether to use node features (default: True)
        """
        super().__init__()

        self.cutoff = cutoff

        self.init_e = init(num_radial, hidden_channels, act, use_node_features=use_node_features)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_dim, num_output_layers, act, output_init)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_dim, num_output_layers, act, output_init) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial, num_before_skip, num_after_skip,act) for _ in range(num_layers)])

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()


    def forward(self, batch_data):
        z, pos, batch = batch_data.atoms, batch_data.pos, batch_data.batch
        edge_index = batch_data.edge_index
        num_nodes = z.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)

        emb = self.emb(dist, angle, torsion, idx_kj)

        # Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i)
        # Disable virutal node trick
        # u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)
        
        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i)
            # Disable virutal node trick
            # u = update_u(u, v, batch)
        
        out = scatter(v, batch, dim=0, reduce='add')
        return out
