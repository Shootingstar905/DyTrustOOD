"""Sparse simplicial boundary operators and structural mediation layers."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def canonical_edge_index(edge_index: Tensor, num_nodes: int | None = None) -> Tensor:
    """Unique, sorted, loop-free undirected edges with orientation u < v.

    This orientation is ONLY for the simplicial complex. Supervised trust pairs
    keep their direction. Construct edge features AFTER this function and in
    exactly its returned ordering.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E].")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("Node IDs must be integers.")
    edges = edge_index.to(dtype=torch.long)
    if edges.numel() == 0:
        return edges.new_empty((2, 0))
    if int(edges.min()) < 0:
        raise ValueError("Node IDs must be nonnegative.")
    inferred_nodes = int(edges.max()) + 1
    if num_nodes is not None and inferred_nodes > num_nodes:
        raise ValueError("An edge endpoint exceeds num_nodes.")
    base = inferred_nodes if num_nodes is None else num_nodes
    pairs = torch.sort(edges, dim=0).values
    pairs = pairs[:, pairs[0] != pairs[1]]
    keys = torch.unique(pairs[0] * base + pairs[1], sorted=True)
    return torch.stack((keys // base, keys % base), dim=0)


@dataclass
class HodgeOperators:
    """B1[N,E], B2[E,F]; apply L1 without constructing any E-by-E matrix."""
    edge_index: Tensor
    b1: Tensor
    b2: Tensor
    triangles: Tensor
    spectral_bound: float

    @property
    def num_triangles(self) -> int:
        return self.b2.shape[1]

    def to(self, device=None, dtype=None) -> "HodgeOperators":
        dtype = self.b1.dtype if dtype is None else dtype
        return HodgeOperators(self.edge_index.to(device=device),
                              self.b1.to(device=device, dtype=dtype),
                              self.b2.to(device=device, dtype=dtype),
                              self.triangles.to(device=device), self.spectral_bound)

    def apply(self, x: Tensor, component: str = "full", normalize: bool = False) -> Tensor:
        if component not in {"full", "lower", "upper"}:
            raise ValueError("component must be full, lower or upper.")
        if x.ndim != 2 or x.shape[0] != self.edge_index.shape[1]:
            raise ValueError("Edge signals must be [number_of_canonical_edges,D].")
        output = x * 0.0
        if x.shape[0] == 0:
            return output
        if component in {"full", "lower"}:
            b1 = self.b1.to(device=x.device, dtype=x.dtype)
            output = output + torch.sparse.mm(b1.transpose(0, 1), torch.sparse.mm(b1, x))
        if component in {"full", "upper"} and self.num_triangles:
            b2 = self.b2.to(device=x.device, dtype=x.dtype)
            output = output + torch.sparse.mm(b2, torch.sparse.mm(b2.transpose(0, 1), x))
        return output / self.spectral_bound if normalize else output


def build_hodge(edge_index: Tensor, num_nodes: int, max_triangles: int | None = None,
                dtype: torch.dtype = torch.float32) -> HodgeOperators:
    """Build the complete 2-dimensional clique complex of an observed graph.

    A triangle [a,b,c], a<b<c, has boundary [b,c]-[a,c]+[a,b]; hence B1 B2=0.
    The optional resource guard raises when exceeded: it never silently drops
    triangles. The returned canonical edge order is authoritative.
    """
    if num_nodes < 0 or (max_triangles is not None and max_triangles < 0):
        raise ValueError("Node counts and triangle limits must be nonnegative.")
    canonical = canonical_edge_index(edge_index, num_nodes)
    device = canonical.device
    pairs = canonical.detach().cpu().t().tolist()
    num_edges = len(pairs)
    lookup = {tuple(pair): index for index, pair in enumerate(pairs)}
    adjacency = [set() for _ in range(num_nodes)]
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    triangles, row_ids, column_ids, values = [], [], [], []
    triangle_degree = [0] * num_edges
    for a in range(num_nodes):
        for b in sorted(v for v in adjacency[a] if v > a):
            for c in sorted(v for v in adjacency[a].intersection(adjacency[b]) if v > b):
                triangle_id = len(triangles)
                if max_triangles is not None and triangle_id >= max_triangles:
                    raise ValueError(f"The observed graph has more than {max_triangles} triangles; "
                                     "increase max_triangles or use an explicitly documented approximation.")
                triangles.append((a, b, c))
                boundary = ((lookup[(b, c)], 1.0), (lookup[(a, c)], -1.0),
                            (lookup[(a, b)], 1.0))
                for edge_id, sign in boundary:
                    row_ids.append(edge_id)
                    column_ids.append(triangle_id)
                    values.append(sign)
                    triangle_degree[edge_id] += 1
    ids = torch.arange(num_edges, device=device)
    b1_indices = torch.stack((canonical.reshape(-1), torch.cat((ids, ids))))
    b1_values = torch.cat((-torch.ones(num_edges, device=device, dtype=dtype),
                           torch.ones(num_edges, device=device, dtype=dtype)))
    b1 = torch.sparse_coo_tensor(b1_indices, b1_values, (num_nodes, num_edges),
                                device=device, dtype=dtype).coalesce()
    b2_indices = torch.tensor([row_ids, column_ids], device=device, dtype=torch.long)
    b2_values = torch.tensor(values, device=device, dtype=dtype)
    b2 = torch.sparse_coo_tensor(b2_indices, b2_values, (num_edges, len(triangles)),
                                device=device, dtype=dtype).coalesce()
    triangles_tensor = torch.tensor(triangles, device=device, dtype=torch.long).reshape(-1, 3).t().contiguous()
    max_node_degree = max((len(neighbors) for neighbors in adjacency), default=0)
    max_triangle_degree = max(triangle_degree, default=0)
    # ||B1^T B1|| <= 2*dmax; ||B2 B2^T|| <= 3*triangle_degree_max.
    bound = float(max(1, 2 * max_node_degree + 3 * max_triangle_degree))
    return HodgeOperators(canonical, b1, b2, triangles_tensor, bound)


class HodgeLaguerreConv(nn.Module):
    """Laguerre filtering with scalar paper coefficients or matrix extensions.

    ``coefficients='scalar'`` implements Equation (12) literally:
    sum_u theta_u Q_u(L) H, with one scalar theta per order, unchanged channel
    width and NO additional bias or channel projection. It starts as the
    identity filter (theta_0=1, all other theta=0).

    ``coefficients='matrix'`` adds an independent channel map per order and an
    optional bias; this is an explicit extension, not literal Equation (12).
    The standalone class retains this default for compatibility. The actual
    StructuralMediator pipeline defaults to ``scalar``. ``bias`` is matrix-only.
    """
    def __init__(self, in_channels: int, out_channels: int, K: int,
                 bias: bool = True, normalize: bool = False, component: str = "full",
                 coefficients: str = "matrix"):
        super().__init__()
        if K < 1:
            raise ValueError("At least one Laguerre coefficient is required.")
        if coefficients not in {"scalar", "matrix"}:
            raise ValueError("coefficients must be 'scalar' or 'matrix'.")
        if coefficients == "scalar" and in_channels != out_channels:
            raise ValueError("Scalar Laguerre coefficients require equal input and output channel widths.")
        self.coefficients, self.order = coefficients, K
        if coefficients == "scalar":
            theta = torch.zeros(K)
            theta[0] = 1.0
            self.theta = nn.Parameter(theta)
            self.lins = nn.ModuleList()
            self.register_parameter("bias", None)
        else:
            self.register_parameter("theta", None)
            self.lins = nn.ModuleList(nn.Linear(in_channels, out_channels, bias=False) for _ in range(K))
            self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.normalize, self.component = normalize, component

    def _coefficient(self, signal: Tensor, order: int) -> Tensor:
        return self.theta[order] * signal if self.coefficients == "scalar" else self.lins[order](signal)

    def forward(self, x: Tensor, hodge: HodgeOperators) -> Tensor:
        previous = x
        output = self._coefficient(previous, 0)
        if self.order > 1:
            current = x - hodge.apply(x, self.component, self.normalize)
            output = output + self._coefficient(current, 1)
            for order in range(1, self.order - 1):
                following = ((2 * order + 1) * current
                             - hodge.apply(current, self.component, self.normalize)
                             - order * previous) / (order + 1)
                output = output + self._coefficient(following, order + 1)
                previous, current = current, following
        return output if self.bias is None else output + self.bias


class GCNConv(nn.Module):
    """Scalar-weight aggregation on the undirected structural support.

    Equation (14) leaves normalization unspecified. Self loops and an
    absolute-weight degree give finite symmetric normalization even with the
    unconstrained (possibly negative) scalar weights of Equation (13).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        weights = edge_weight.reshape(-1)
        if edge_index.shape != (2, weights.numel()):
            raise ValueError("One scalar structural weight is required per canonical edge.")
        projected = self.lin(x)
        # Concatenation, rather than reshape, preserves edge/weight alignment.
        source = torch.cat((edge_index[0], edge_index[1]))
        target = torch.cat((edge_index[1], edge_index[0]))
        directed_weights = torch.cat((weights, weights))
        degree = x.new_ones(x.shape[0]).index_add(0, target, directed_weights.abs())
        norm = directed_weights / (degree[source] * degree[target]).sqrt()
        messages = projected[source] * norm.unsqueeze(-1)
        result = projected / degree.unsqueeze(-1)
        result = result.index_add(0, target, messages)
        return F.relu(result + self.bias)
