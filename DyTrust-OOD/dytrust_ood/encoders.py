"""Environment representations, temporal stability and the conditional VAE.

Equation numbers refer to the supplied 12-page DyTrust-OOD manuscript. The
factor partition is a temporal-stability heuristic, not a causal identifier.
Only PyTorch is required; graph messages use ``index_add``.
"""
from __future__ import annotations

import math
from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RelTemporalEncoding(nn.Module):
    """Sinusoidal time encoding without a fixed maximum sequence length."""
    def __init__(self, n_hid: int):
        super().__init__()
        if n_hid < 1:
            raise ValueError("The time-encoding dimension must be positive.")
        self.n_hid = n_hid
        frequency = torch.exp(torch.arange(0, n_hid, 2) * (-math.log(10000.0) / n_hid))
        self.register_buffer("frequency", frequency)
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x: Tensor, t: int | Tensor) -> Tensor:
        position = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        angles = position.unsqueeze(-1) * self.frequency.to(dtype=x.dtype)
        encoding = x.new_zeros((*position.shape, self.n_hid))
        encoding[..., 0::2] = angles.sin()
        encoding[..., 1::2] = angles[..., : self.n_hid // 2].cos()
        return x + self.lin(encoding / math.sqrt(self.n_hid))


class LinkPredictor(nn.Module):
    """Ordered-pair MLP; the manuscript leaves the final decoder unspecified.

    Ordered source/target features distinguish i->j from j->i. ``logits=True``
    is used with BCEWithLogitsLoss; no sigmoid is applied during training.
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int = 1, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if num_layers < 1 or out_channels != 1:
            raise ValueError("Use at least one layer and one binary output.")
        sizes = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.dropout = dropout

    def reset_parameters(self) -> None:
        for layer in self.lins:
            layer.reset_parameters()

    def forward(self, z: Tensor, edge_index: Tensor, logits: bool = False) -> Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        source, target = z[edge_index[0]], z[edge_index[1]]
        x = torch.cat((source, target, source * target, (source - target).abs()), dim=-1)
        for layer in self.lins[:-1]:
            x = F.dropout(F.relu(layer(x)), self.dropout, training=self.training)
        scores = self.lins[-1](x).squeeze(-1)
        return scores if logits else scores.sigmoid()


class EAConv(nn.Module):
    """Equations (2)-(3): environment softmax for each directed graph edge.

    source->target delivers a message to target. Self representations are added
    explicitly, so input self loops are ignored.
    """
    def __init__(self, dim: int, n_factors: int):
        super().__init__()
        if n_factors < 1 or dim % n_factors:
            raise ValueError("The hidden dimension must be divisible by K.")
        self.dim, self.n_factors = dim, n_factors
        self.delta_d = dim // n_factors

    def environment_weights(self, x: Tensor, edge_index: Tensor) -> Tensor:
        factors = x.reshape(x.shape[0], self.n_factors, self.delta_d)
        similarities = (factors[edge_index[0]] * factors[edge_index[1]]).sum(-1)
        return similarities.softmax(dim=-1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        if edge_index.shape[1] == 0:
            return x
        factors = x.reshape(x.shape[0], self.n_factors, self.delta_d)
        weights = self.environment_weights(x, edge_index)
        messages = factors[edge_index[0]] * weights.unsqueeze(-1)
        aggregated = torch.zeros_like(factors).index_add(0, edge_index[1], messages)
        return (factors + aggregated).reshape_as(x)


class EnvironmentEncoder(nn.Module):
    """Environment projections and graph aggregation, Equations (1)-(4)."""
    def __init__(self, args):
        super().__init__()
        self.n_factors = int(args.n_factors)
        self.delta_d = int(args.delta_d)
        self.in_dim = int(args.nfeat)
        self.hid_dim = self.n_factors * self.delta_d
        self.n_layers = int(getattr(args, "n_layers", 1))
        self.use_RTE = bool(getattr(args, "use_RTE", True))
        self.dropout = float(getattr(args, "dropout", 0.0))
        if min(self.n_factors, self.delta_d, self.in_dim, self.n_layers) < 1:
            raise ValueError("Feature, environment and layer counts must be positive.")
        self.rte = RelTemporalEncoding(self.in_dim) if self.use_RTE else None
        # Rows of this matrix are independent W_k blocks, as in Equation (1).
        self.linear = nn.Linear(self.in_dim, self.hid_dim)
        self.layers = nn.ModuleList(EAConv(self.hid_dim, self.n_factors)
                                    for _ in range(self.n_layers))
        self.edge_decoder = LinkPredictor(
            4 * self.hid_dim, max(int(getattr(args, "nhid", 64)), 1),
            # The same decoder is evaluated for every replacement in L2.
            # Independent dropout would conflate replacement variance with
            # prediction noise. Encoder dropout, if used, is shared per pass.
            num_layers=2, dropout=0.0,
        )

    def forward(self, edge_index_list: Sequence[Tensor], x_list: Sequence[Tensor]) -> Tensor:
        if x_list is None:
            raise ValueError("Provide historical node features explicitly; no hidden feature fallback is used.")
        if len(edge_index_list) != len(x_list) or not len(x_list):
            raise ValueError("One feature matrix and edge list are required per historical snapshot.")
        num_nodes = x_list[0].shape[0]
        outputs = []
        for time, (edge_index, features) in enumerate(zip(edge_index_list, x_list)):
            if features.shape != (num_nodes, self.in_dim):
                raise ValueError("Every snapshot must use the same [N, nfeat] feature shape.")
            encoded = self.rte(features, time) if self.rte is not None else features
            hidden = self.linear(encoded).sigmoid()
            for layer_index, layer in enumerate(self.layers):
                hidden = layer(hidden, edge_index)
                if layer_index + 1 < self.n_layers:
                    hidden = F.dropout(hidden, self.dropout, training=self.training)
            outputs.append(hidden)
        return torch.stack(outputs, dim=0)

    def decoder_logits(self, z: Tensor, edge_index: Tensor) -> Tensor:
        return self.edge_decoder(z, edge_index, logits=True)


class GRUEncoder(nn.Module):
    """One-way, shared-parameter temporal encoder after Equation (4)."""
    def __init__(self, args):
        super().__init__()
        dimension = int(args.n_factors) * int(args.delta_d)
        # GRU dropout only applies between layers; this GRU has one layer.
        self.gru = nn.GRU(dimension, dimension, num_layers=1, batch_first=False)

    def forward(self, x: Tensor, active: Tensor | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[0] < 1:
            raise ValueError("The GRU expects a nonempty [T, N, D] history.")
        if active is None:
            output, _ = self.gru(x)
            return output
        if active.shape != x.shape[:2] or active.dtype != torch.bool:
            raise ValueError("active must be boolean [T,N]")
        state = x.new_zeros((1, x.shape[1], x.shape[2]))
        outputs = []
        for t in range(x.shape[0]):
            available = active[t].reshape(1, -1, 1)
            state = state * available
            _, state = self.gru(x[t:t+1] * available, state)
            state = state * available
            outputs.append(state[0])
        return torch.stack(outputs)


def temporal_factor_variation(history: Tensor, n_factors: int,
                              active: Tensor | None = None) -> Tensor:
    """Equation (5): population temporal variance, summed over coordinates."""
    if history.ndim == 3:
        if n_factors < 1 or history.shape[-1] % n_factors:
            raise ValueError("History width must be divisible by K.")
        factors = history.reshape(*history.shape[:2], n_factors, -1)
    elif history.ndim == 4 and history.shape[2] == n_factors:
        factors = history
    else:
        raise ValueError("History must be [T,N,K*d] or [T,N,K,d].")
    if factors.shape[0] < 1:
        raise ValueError("At least one historical snapshot is required.")
    if active is None:
        return ((factors - factors.mean(0, keepdim=True)) ** 2).sum(-1).mean(0)
    if active.shape != factors.shape[:2] or active.dtype != torch.bool:
        raise ValueError("active must be boolean [T,N]")
    weight = active.to(factors.dtype).unsqueeze(-1).unsqueeze(-1)
    count = weight.sum(0).clamp_min(1)
    mean = (factors * weight).sum(0) / count
    return (((factors - mean) ** 2) * weight).sum(0).sum(-1) / count.squeeze(-1)


def identify_factors(history: Tensor, n_factors: int, active: Tensor | None = None,
                     strategy: str = "mean") -> Tensor:
    """Equation (6): True for relatively stable ('causal') blocks [N,K].

    With one snapshot, or K=1, every block is stable. No forced split is added.
    The mask is fixed from the original history before any replacements.
    """
    variation = temporal_factor_variation(history, n_factors, active)
    if strategy == "mean":
        mask = variation <= variation.mean(-1, keepdim=True)
    elif strategy == "median":
        mask = variation <= variation.median(-1, keepdim=True).values
    elif strategy == "random_matched":
        counts = (variation <= variation.mean(-1, keepdim=True)).sum(-1, keepdim=True)
        ranks = torch.rand_like(variation).argsort(-1).argsort(-1)
        mask = ranks < counts
    else:
        raise ValueError("factor_selection must be mean, median, or random_matched")
    return mask if active is None else mask & active.any(0).unsqueeze(-1)


class ConditionalVAE(nn.Module):
    """Gaussian CVAE with environment AND time conditions, Equations (7)-(9).

    Posterior, conditional prior and decoder are learned neural networks.
    ``logvar`` always denotes log variance, never log standard deviation.
    """
    def __init__(self, args):
        super().__init__()
        self.data_dim = int(args.delta_d)
        self.n_factors = int(args.n_factors)
        self.num_times = int(args.length)
        self.latent_dim = int(args.d_for_cvae)
        self.condition_dim = self.n_factors + self.num_times
        if min(self.data_dim, self.n_factors, self.num_times, self.latent_dim) < 1:
            raise ValueError("All CVAE dimensions must be positive.")
        width = max(32, 2 * self.data_dim, 2 * self.latent_dim)
        self.encoder = nn.Sequential(nn.Linear(self.data_dim + self.condition_dim, width), nn.SiLU())
        self.posterior_mu = nn.Linear(width, self.latent_dim)
        self.posterior_logvar = nn.Linear(width, self.latent_dim)
        self.prior_network = nn.Sequential(nn.Linear(self.condition_dim, width), nn.SiLU())
        self.prior_mu = nn.Linear(width, self.latent_dim)
        self.prior_logvar = nn.Linear(width, self.latent_dim)
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim + self.condition_dim, width), nn.SiLU())
        self.decoder_mu = nn.Linear(width, self.data_dim)
        self.decoder_logvar = nn.Linear(width, self.data_dim)

    def make_condition(self, factor_ids: Tensor, time_ids: Tensor) -> Tensor:
        device = self.posterior_mu.weight.device
        factor_ids, time_ids = torch.broadcast_tensors(
            torch.as_tensor(factor_ids, device=device, dtype=torch.long),
            torch.as_tensor(time_ids, device=device, dtype=torch.long),
        )
        if torch.any((factor_ids < 0) | (factor_ids >= self.n_factors)):
            raise ValueError("Environment IDs must lie in [0,K).")
        if torch.any((time_ids < 0) | (time_ids >= self.num_times)):
            raise ValueError("Snapshot IDs must lie in [0,length).")
        condition = torch.cat((F.one_hot(factor_ids, self.n_factors),
                               F.one_hot(time_ids, self.num_times)), dim=-1)
        return condition.to(dtype=self.posterior_mu.weight.dtype)

    def _check_condition(self, condition: Tensor) -> None:
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(f"CVAE conditions must have shape [B,{self.condition_dim}].")

    def encode(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        self._check_condition(condition)
        hidden = self.encoder(torch.cat((x, condition), dim=-1))
        return self.posterior_mu(hidden), self.posterior_logvar(hidden).clamp(-12.0, 8.0)

    def prior(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        self._check_condition(condition)
        hidden = self.prior_network(condition)
        return self.prior_mu(hidden), self.prior_logvar(hidden).clamp(-12.0, 8.0)

    @staticmethod
    def reparametrize(mu: Tensor, logvar: Tensor, generator=None) -> Tensor:
        noise = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        return mu + noise * (0.5 * logvar).exp()

    def decode_distribution(self, z: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        self._check_condition(condition)
        hidden = self.decoder(torch.cat((z, condition), dim=-1))
        return self.decoder_mu(hidden), self.decoder_logvar(hidden).clamp(-8.0, 4.0)

    def decode(self, z: Tensor, condition: Tensor) -> Tensor:
        """Return the Gaussian decoder mean; ``sample`` draws an observation."""
        return self.decode_distribution(z, condition)[0]

    def forward(self, x: Tensor, condition: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(x, condition)
        prior_mu, prior_logvar = self.prior(condition)
        latent = self.reparametrize(mu, logvar)
        recon, recon_logvar = self.decode_distribution(latent, condition)
        return {"recon": recon, "recon_logvar": recon_logvar,
                "mu": mu, "logvar": logvar,
                "prior_mu": prior_mu, "prior_logvar": prior_logvar}

    @staticmethod
    def loss_from_output(x: Tensor, result: dict[str, Tensor]) -> Tensor:
        """Mean negative ELBO: diagonal Gaussian KL plus Gaussian NLL."""
        logvar, prior_logvar = result["logvar"], result["prior_logvar"]
        kl = 0.5 * (prior_logvar - logvar
                    + (logvar.exp() + (result["mu"] - result["prior_mu"]) ** 2)
                    / prior_logvar.exp() - 1.0).sum(-1)
        reconstruction = 0.5 * (math.log(2.0 * math.pi) + result["recon_logvar"]
                                + (x - result["recon"]) ** 2 / result["recon_logvar"].exp()).sum(-1)
        return (kl + reconstruction).mean()

    def loss(self, x: Tensor, condition: Tensor) -> Tensor:
        if x.shape[0] == 0:
            raise ValueError("A CVAE mini-batch must contain at least one observation.")
        return self.loss_from_output(x, self(x, condition))

    def sample(self, condition: Tensor, generator=None) -> Tensor:
        """Draw e~p(e|c), then r~p(r|c,e), from supplied training conditions."""
        mu, logvar = self.prior(condition)
        latent = self.reparametrize(mu, logvar, generator=generator)
        recon_mu, recon_logvar = self.decode_distribution(latent, condition)
        return self.reparametrize(recon_mu, recon_logvar, generator=generator)
