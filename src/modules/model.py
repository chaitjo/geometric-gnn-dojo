from typing import Callable, Optional, Type
import torch
from torch_scatter import scatter
from e3nn import o3

from src.modules.blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
)
from src.modules import (
    interaction_classes,
    gate_dict
)


class OriginalMACEModel(torch.nn.Module):
    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        interaction_cls: Type[InteractionBlock] = interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first: Type[InteractionBlock] = interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions: int = 2,
        num_elements: int = 1,
        hidden_irreps: o3.Irreps = o3.Irreps("64x0e + 64x1o + 64x2e"),
        MLP_irreps: o3.Irreps = o3.Irreps("64x0e"),
        irreps_out: o3.Irreps = o3.Irreps("1x0e"),
        avg_num_neighbors: int = 1,
        correlation: int = 3,
        gate: Optional[Callable] = gate_dict["silu"],
        num_layers=2,
        in_dim=1,
        out_dim=1,
    ):
        super().__init__()
        self.r_max = r_max
        self.num_elements = num_elements
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readout
        self.atomic_energies_fn = LinearReadoutBlock(node_feats_irreps, irreps_out)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            element_dependent=True,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps, irreps_out))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                element_dependent=True,
                num_elements=num_elements,
                use_sc=True
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate, irreps_out)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps, irreps_out))

    def forward(self, batch):
        # MACE expects one-hot-ified input
        batch.atoms.unsqueeze_(-1)
        shape = batch.atoms.shape[:-1] + (self.num_elements,)
        node_attrs = torch.zeros(shape, device=batch.atoms.device).view(shape)
        node_attrs.scatter_(dim=-1, index=batch.atoms, value=1)

        # Node embeddings
        node_feats = self.node_embedding(node_attrs)
        node_e0 = self.atomic_energies_fn(node_feats)
        e0 = scatter(node_e0, batch.batch, dim=0, reduce="sum")  # [n_graphs, irreps_out]
        
        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=batch.edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, irreps_out]
            energy = scatter(node_energies, batch.batch, dim=0, reduce="sum")  # [n_graphs, irreps_out]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, irreps_out]

        return total_energy
