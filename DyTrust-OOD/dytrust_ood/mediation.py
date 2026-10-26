"""Per-snapshot Hodge structural mediation: front-door inspired, not identified."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .hodge import (GCNConv, HodgeLaguerreConv, HodgeOperators,
                     build_hodge, canonical_edge_index)


class StructuralMediator(nn.Module):
    """Equations (12)-(14), receiving already masked causal node features.

    edge_features[e] and hodge.edge_index[:,e] must refer to the SAME canonical
    edge. No time projection or future snapshot is used here. Scalar Laguerre
    coefficients implement Equation (12) by default. The default orientation
    lift and even readout are explicit engineering amendments, not Eqs. (12)-(13)
    verbatim. structural_mode='literal' retains the old readout for comparison.
    """
    def __init__(self, args):
        super().__init__()
        self.dimension = int(args.n_factors) * int(args.delta_d)
        self.edge_feature_dim = int(getattr(args, "edge_feature_dim", 3))
        self.max_triangles = getattr(args, "max_triangles", None)
        self.start_mlp_edge = nn.Linear(self.edge_feature_dim, self.dimension)
        self.spatial_edge = HodgeLaguerreConv(
            self.dimension, self.dimension, int(getattr(args, "hodge_order", 3)),
            normalize=bool(getattr(args, "hodge_normalize", False)),
            component=getattr(args, "hodge_component", "full"),
            coefficients=getattr(args, "hodge_coefficients", "scalar"),
        )
        self.structural_mode = getattr(args, "structural_mode", "equivariant")
        if self.structural_mode not in ("equivariant", "literal"):
            raise ValueError("Unknown structural_mode")
        if self.structural_mode == "equivariant":
            self.orientation_gate = nn.Linear(self.dimension, self.dimension, bias=False)
            # A channel bias after filtering would break odd orientation equivariance.
            if self.spatial_edge.bias is not None:
                self.spatial_edge.register_parameter("bias", None)
            self.edge_causal = nn.Linear(2 * self.dimension, 1)
        else:
            self.orientation_gate = None
            self.edge_causal = nn.Linear(self.dimension, 1)
        self.spatial_node = GCNConv(self.dimension, self.dimension)

    def forward(self, causal_nodes: Tensor, edge_features: Tensor,
                edge_index: Tensor, hodge: HodgeOperators | None = None,
                return_weights: bool = False):
        if causal_nodes.ndim != 2 or causal_nodes.shape[-1] != self.dimension:
            raise ValueError("Pass one snapshot of causal node features [N,K*d].")
        if hodge is None:
            hodge = build_hodge(edge_index, causal_nodes.shape[0], self.max_triangles,
                                dtype=causal_nodes.dtype)
        if not torch.equal(edge_index, hodge.edge_index):
            raise ValueError("Structural edges must be canonical and match the Hodge edge order. "
                             "Call canonical_edge_index before constructing edge features.")
        if edge_features.shape != (edge_index.shape[1], self.edge_feature_dim):
            raise ValueError("edge_features must have one aligned row per canonical edge.")
        encoded = F.relu(self.start_mlp_edge(edge_features.to(dtype=causal_nodes.dtype)))
        if self.orientation_gate is None:
            filtered = self.spatial_edge(encoded, hodge)
            weights = self.edge_causal(filtered).squeeze(-1)
        else:
            source, destination = edge_index
            gate = torch.tanh(self.orientation_gate(causal_nodes[destination] - causal_nodes[source]))
            filtered = self.spatial_edge(encoded * gate, hodge)
            # The sign changes with reference direction, the aggregation weight does not.
            readout = torch.cat((encoded, torch.log1p(filtered.square())), dim=-1)
            weights = F.softplus(self.edge_causal(readout).squeeze(-1))
        mediator = self.spatial_node(causal_nodes, edge_index, weights)
        return (mediator, weights) if return_weights else mediator


__all__ = ["StructuralMediator", "HodgeOperators", "build_hodge", "canonical_edge_index"]
