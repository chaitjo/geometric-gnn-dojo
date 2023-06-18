import torch
from torch_scatter import scatter
import e3nn

from models.mace_modules.irreps_tools import irreps2gate


class TensorProductConvLayer(torch.nn.Module):
    """Tensor Field Network GNN Layer in e3nn

    Implements a Tensor Field Network equivariant GNN layer for higher-order tensors, using e3nn.
    Implementation adapted from: https://github.com/gcorso/DiffDock/

    Paper: Tensor Field Networks, Thomas, Smidt et al.
    """
    def __init__(
        self,
        in_irreps,
        out_irreps,
        sh_irreps,
        edge_feats_dim,
        mlp_dim,
        aggr="add",
        batch_norm=False,
        gate=False,
    ):
        """
        Args:
            in_irreps: (e3nn.o3.Irreps) Input irreps dimensions
            out_irreps: (e3nn.o3.Irreps) Output irreps dimensions
            sh_irreps: (e3nn.o3.Irreps) Spherical harmonic irreps dimensions
            edge_feats_dim: (int) Edge feature dimensions
            mlp_dim: (int) Hidden dimension of MLP for computing tensor product weights
            aggr: (str) Message passing aggregator
            batch_norm: (bool) Whether to apply equivariant batch norm
            gate: (bool) Whether to apply gated non-linearity
        """
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.edge_feats_dim = edge_feats_dim
        self.aggr = aggr

        if gate:
            # Optionally apply gated non-linearity
            irreps_scalars, irreps_gates, irreps_gated = irreps2gate(
                e3nn.o3.Irreps(out_irreps)
            )
            act_scalars = [torch.nn.functional.silu for _, ir in irreps_scalars]
            act_gates = [torch.sigmoid for _, ir in irreps_gates]
            if irreps_gated.num_irreps == 0:
                self.gate = e3nn.nn.Activation(out_irreps, acts=[torch.nn.functional.silu])
            else:
                self.gate = e3nn.nn.Gate(
                    irreps_scalars,
                    act_scalars,  # scalar
                    irreps_gates,
                    act_gates,  # gates (scalars)
                    irreps_gated,  # gated tensors
                )
                # Output irreps for the tensor product must be updated
                self.out_irreps = out_irreps = self.gate.irreps_in
        else:
            self.gate = None

        # Tensor product over edges to construct messages
        self.tp = e3nn.o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        # MLP used to compute weights of tensor product
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(edge_feats_dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dim, self.tp.weight_numel),
        )

        # Optional equivariant batch norm
        self.batch_norm = e3nn.nn.BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_sh, edge_feat):
        src, dst = edge_index
        # Compute messages
        tp = self.tp(node_attr[dst], edge_sh, self.fc(edge_feat))
        # Aggregate messages
        out = scatter(tp, src, dim=0, reduce=self.aggr)
        # Optionally apply gated non-linearity and/or batch norm
        if self.gate:
            out = self.gate(out)
        if self.batch_norm:
            out = self.batch_norm(out)
        return out
