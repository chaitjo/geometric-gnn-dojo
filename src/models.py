from typing import Callable, Optional, Union
import torch
from torch.nn import functional as F
import torch_geometric
from torch_geometric.nn import SchNet, DimeNetPlusPlus, global_add_pool, global_mean_pool
import torch_scatter
from torch_scatter import scatter
from e3nn import o3

from src.modules.blocks import (
    EquivariantProductBasisBlock,
    RadialEmbeddingBlock,
)
from src.modules.irreps_tools import reshape_irreps

from src.egnn_layers import MPNNLayer, EGNNLayer
from src.tfn_layers import TensorProductConvLayer
import src.gvp_layers as gvp


class MACEModel(torch.nn.Module):
    def __init__(
        self,
        r_max=10.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        correlation=3,
        num_layers=5,
        emb_dim=64,
        in_dim=1,
        out_dim=1,
        aggr="sum",
        pool="sum",
        residual=True,
        scalar_pred=True
    ):
        super().__init__()
        self.r_max = r_max
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.residual = residual
        self.scalar_pred = scalar_pred
        # Embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        self.convs = torch.nn.ModuleList()
        self.prods = torch.nn.ModuleList()
        self.reshapes = torch.nn.ModuleList()
        hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
        irrep_seq = [
            o3.Irreps(f'{emb_dim}x0e'),
            # o3.Irreps(f'{emb_dim}x0e + {emb_dim}x1o + {emb_dim}x2e'),
            # o3.Irreps(f'{emb_dim//2}x0e + {emb_dim//2}x0o + {emb_dim//2}x1e + {emb_dim//2}x1o + {emb_dim//2}x2e + {emb_dim//2}x2o'),
            hidden_irreps
        ]
        for i in range(num_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            conv = TensorProductConvLayer(
                in_irreps=in_irreps,
                out_irreps=out_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                hidden_dim=emb_dim,
                gate=False,
                aggr=aggr,
            )
            self.convs.append(conv)
            self.reshapes.append(reshape_irreps(out_irreps))
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=out_irreps,
                target_irreps=out_irreps,
                correlation=correlation,
                element_dependent=False,
                num_elements=in_dim,
                use_sc=residual
            )
            self.prods.append(prod)

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.scalar_pred:
            # Predictor MLP
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, out_dim)
            )
        else:
            self.pred = torch.nn.Linear(hidden_irreps.dim, out_dim)
    
    def forward(self, batch):
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d)

        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        for conv, reshape, prod in zip(self.convs, self.reshapes, self.prods):
            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_attrs, edge_feats)
            # Update node features
            sc = F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))
            h = prod(reshape(h_update), sc, None)

        if self.scalar_pred:
            # Select only scalars for prediction
            h = h[:,:self.emb_dim]
        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_dim)


class TFNModel(torch.nn.Module):
    def __init__(
        self,
        r_max=10.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        num_layers=5,
        emb_dim=64,
        in_dim=1,
        out_dim=1,
        aggr="sum",
        pool="sum",
        residual=True,
        scalar_pred=True
    ):
        super().__init__()
        self.r_max = r_max
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.residual = residual
        self.scalar_pred = scalar_pred
        # Embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        self.convs = torch.nn.ModuleList()
        hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
        irrep_seq = [
            o3.Irreps(f'{emb_dim}x0e'),
            # o3.Irreps(f'{emb_dim}x0e + {emb_dim}x1o + {emb_dim}x2e'),
            # o3.Irreps(f'{emb_dim//2}x0e + {emb_dim//2}x0o + {emb_dim//2}x1e + {emb_dim//2}x1o + {emb_dim//2}x2e + {emb_dim//2}x2o'),
            hidden_irreps
        ]
        for i in range(num_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            conv = TensorProductConvLayer(
                in_irreps=in_irreps,
                out_irreps=out_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                hidden_dim=emb_dim,
                gate=True,
                aggr=aggr,
            )
            self.convs.append(conv)

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.scalar_pred:
            # Predictor MLP
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, out_dim)
            )
        else:
            self.pred = torch.nn.Linear(hidden_irreps.dim, out_dim)
    
    def forward(self, batch):
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d)

        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        for conv in self.convs:
            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_attrs, edge_feats)

            # Update node features
            h = h_update + F.pad(h, (0, h_update.shape[-1] - h.shape[-1])) if self.residual else h_update

        if self.scalar_pred:
            # Select only scalars for prediction
            h = h[:,:self.emb_dim]
        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_dim)


class GVPGNNModel(torch.nn.Module):
    def __init__(
        self,
        r_max=10.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        num_layers=5,
        emb_dim=64,
        in_dim=1,
        out_dim=1,
        aggr="sum",
        pool="sum",
        residual=True
    ):
        super().__init__()
        _DEFAULT_V_DIM = (emb_dim, emb_dim)
        _DEFAULT_E_DIM = (emb_dim, 1)
        activations = (F.relu, None)

        self.r_max = r_max
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        # Embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)
        self.W_e = torch.nn.Sequential(
            gvp.LayerNorm((self.radial_embedding.out_dim, 1)),
            gvp.GVP((self.radial_embedding.out_dim, 1), _DEFAULT_E_DIM, 
                activations=(None, None), vector_gate=True)
        )
        self.W_v = torch.nn.Sequential(
            gvp.LayerNorm((emb_dim, 0)),
            gvp.GVP((emb_dim, 0), _DEFAULT_V_DIM,
                activations=(None, None), vector_gate=True)
        )
        
        # Stack of GNN layers
        self.layers = torch.nn.ModuleList(
                gvp.GVPConvLayer(_DEFAULT_V_DIM, _DEFAULT_E_DIM, 
                             activations=activations, vector_gate=True,
                             residual=residual) 
            for _ in range(num_layers))
        
        self.W_out = torch.nn.Sequential(
            gvp.LayerNorm(_DEFAULT_V_DIM),
            gvp.GVP(_DEFAULT_V_DIM, (emb_dim, 0), 
                activations=activations, vector_gate=True)
        )
        
        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, out_dim)
        )
    
    def forward(self, batch):

        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        
        h_V = self.emb_in(batch.atoms)  # (n,) -> (n, d)
        h_E = (self.radial_embedding(lengths), torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2))

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
    
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)
        
        out = self.pool(out, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_dim)


class EGNNModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=5,
        emb_dim=128,
        in_dim=1,
        out_dim=1,
        activation="relu",
        norm="layer",
        aggr="sum",
        pool="sum",
        residual=True
    ):
        """E(n) Equivariant GNN model 
        
        Args:
            num_layers: (int) - number of message passing layers
            emb_dim: (int) - hidden dimension
            in_dim: (int) - initial node feature dimension
            out_dim: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EGNNLayer(emb_dim, activation, norm, aggr))

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, out_dim)
        )
        self.residual = residual

    def forward(self, batch):
        
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d)
        pos = batch.pos  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update 

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update

        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_dim)


class MPNNModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=5,
        emb_dim=128,
        in_dim=1,
        out_dim=1,
        activation="relu",
        norm="layer",
        aggr="sum",
        pool="sum",
        residual=True
    ):
        """Vanilla Message Passing GNN model
        
        Args:
            num_layers: (int) - number of message passing layers
            emb_dim: (int) - hidden dimension
            in_dim: (int) - initial node feature dimension
            out_dim: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, activation, norm, aggr))

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, out_dim)
        )
        self.residual = residual

    def forward(self, batch):
        
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d)
        
        for conv in self.convs:
            # Message passing layer and residual connection
            h = h + conv(h, batch.edge_index) if self.residual else conv(h, batch.edge_index)

        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_dim)


class SchNetModel(SchNet):
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
        readout: str = 'add', 
        dipole: bool = False,
        mean: Optional[float] = None, 
        std: Optional[float] = None, 
        atomref: Optional[torch.Tensor] = None,
    ):
        super().__init__(hidden_channels, num_filters, num_layers, num_gaussians, cutoff, max_num_neighbors, readout, dipole, mean, std, atomref)

        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.Linear(hidden_channels // 2, out_dim)

    def forward(self, batch):
        h = self.embedding(batch.atoms)

        row, col = batch.edge_index
        edge_weight = (batch.pos[row] - batch.pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, batch.edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        out = scatter(h, batch.batch, dim=0, reduce=self.readout)
        return out


class DimeNetPPModel(DimeNetPlusPlus):
    def __init__(
        self, 
        hidden_channels: int = 128, 
        in_dim: int = 1,
        out_dim: int = 1, 
        num_layers: int = 4, 
        int_emb_size: int = 64, 
        basis_emb_size: int = 8, 
        out_emb_channels: int = 256, 
        num_spherical: int = 7, 
        num_radial: int = 6, 
        cutoff: float = 10, 
        max_num_neighbors: int = 32, 
        envelope_exponent: int = 5, 
        num_before_skip: int = 1, 
        num_after_skip: int = 2, 
        num_output_layers: int = 3, 
        act: Union[str, Callable] = 'swish'
    ):
        super().__init__(hidden_channels, out_dim, num_layers, int_emb_size, basis_emb_size, out_emb_channels, num_spherical, num_radial, cutoff, max_num_neighbors, envelope_exponent, num_before_skip, num_after_skip, num_output_layers, act)

    def forward(self, batch):
        
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            batch.edge_index, num_nodes=batch.atoms.size(0))

        # Calculate distances.
        dist = (batch.pos[i] - batch.pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = batch.pos[idx_i]
        pos_ji, pos_ki = batch.pos[idx_j] - pos_i, batch.pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(batch.atoms, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=batch.pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        return P.sum(dim=0) if batch is None else scatter(P, batch.batch, dim=0)
