from typing import Optional

import torch
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
import e3nn

from models.mace_modules.blocks import RadialEmbeddingBlock
from models.layers.tfn_layer import TensorProductConvLayer


class TFNModel(torch.nn.Module):
    """
    Tensor Field Network model from "Tensor Field Networks".
    """
    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        num_layers: int = 5,
        emb_dim: int = 64,
        hidden_irreps: Optional[e3nn.o3.Irreps] = None,
        mlp_dim: int = 256,
        in_dim: int = 1,
        out_dim: int = 1,
        aggr: str = "sum",
        pool: str = "sum",
        gate: bool = True,
        batch_norm: bool = False,
        residual: bool = True,
        equivariant_pred: bool = False
    ):
        """
        Parameters:
        - r_max (float): Maximum distance for Bessel basis functions (default: 10.0)
        - num_bessel (int): Number of Bessel basis functions (default: 8)
        - num_polynomial_cutoff (int): Number of polynomial cutoff basis functions (default: 5)
        - max_ell (int): Maximum degree of spherical harmonics basis functions (default: 2)
        - num_layers (int): Number of layers in the model (default: 5)
        - emb_dim (int): Scalar feature embedding dimension (default: 64)
        - hidden_irreps (Optional[e3nn.o3.Irreps]): Hidden irreps (default: None)
        - mlp_dim (int): Dimension of MLP for computing tensor product weights (default: 256)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - aggr (str): Aggregation method to be used (default: "sum")
        - pool (str): Global pooling method to be used (default: "sum")
        - gate (bool): Whether to use gated equivariant non-linearity (default: True)
        - batch_norm (bool): Whether to use batch normalization (default: False)
        - residual (bool): Whether to use residual connections (default: True)
        - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)

        Note:
        - If `hidden_irreps` is None, the irreps for the intermediate features are computed 
          using `emb_dim` and `max_ell`.
        - The `equivariant_pred` parameter determines whether it is an equivariant prediction task.
          If set to True, equivariant prediction will be performed.
        - At present, only one of `gate` and `batch_norm` can be True.
        """
        super().__init__()
        
        self.r_max = r_max
        self.max_ell = max_ell
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim
        self.residual = residual
        self.batch_norm = batch_norm
        self.gate = gate
        self.hidden_irreps = hidden_irreps
        self.equivariant_pred = equivariant_pred
        
        # Edge embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        # Set hidden irreps if none are provided
        if hidden_irreps is None:
            hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
            # Note: This defaults to O(3) equivariant layers
            # It is possible to use SO(3) equivariance by passing the appropriate irreps

        self.convs = torch.nn.ModuleList()
        # First conv layer: scalar only -> tensor
        self.convs.append(
            TensorProductConvLayer(
                in_irreps=e3nn.o3.Irreps(f'{emb_dim}x0e'),
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                mlp_dim=mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
            )
        )
        # Intermediate conv layers: tensor -> tensor
        for _ in range(num_layers - 1):
            conv = TensorProductConvLayer(
                in_irreps=hidden_irreps,
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                mlp_dim=mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
            )
            self.convs.append(conv)

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.equivariant_pred:
            # Linear predictor for equivariant tasks using geometric features
            self.pred = torch.nn.Linear(hidden_irreps.dim, out_dim)
        else:
            # MLP predictor for invariant tasks using only scalar features
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, out_dim)
            )
    
    def forward(self, batch):
        # Node embedding
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d)

        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]

        edge_sh = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        for conv in self.convs:
            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_sh, edge_feats)

            # Update node features
            h = h_update + F.pad(h, (0, h_update.shape[-1] - h.shape[-1])) if self.residual else h_update

        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        
        if not self.equivariant_pred:
            # Select only scalars for invariant prediction
            out = out[:,:self.emb_dim]
        
        return self.pred(out)  # (batch_size, out_dim)
