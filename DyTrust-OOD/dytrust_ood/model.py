"""Complete prefix-only predictor and representation replacement procedure."""
from __future__ import annotations

import math
import torch
from torch import nn

from collections.abc import Mapping, Sequence
from torch import Tensor

from .config import ModelConfig
from .encoders import EnvironmentEncoder, GRUEncoder, ConditionalVAE, identify_factors
from .hodge import HodgeOperators
from .losses import edge_bce, intervention_objective
from .mediation import StructuralMediator


def select_users(active_nodes, ratio):
    candidates = active_nodes.nonzero(as_tuple=True)[0]
    count = min(len(candidates), math.ceil(ratio * len(candidates))) if ratio > 0 else 0
    return candidates[torch.randperm(len(candidates), device=candidates.device)[:count]]


def replace_spurious(history, causal_mask, selected_users, observed_pool, generated_pool, mixing_ratio,
                     active=None):
    """Replace independent [d] blocks; stable blocks and other users are exact copies.

    A fresh result is returned. Every call starts from the same ORIGINAL history,
    and neither the supervised pairs nor their labels are changed.
    """
    times, nodes, dimension = history.shape
    factors = causal_mask.shape[1]
    width = dimension // factors
    if causal_mask.shape != (nodes, factors) or dimension % factors:
        raise ValueError("Causal mask does not match history dimensions")
    if not 0 <= mixing_ratio <= 1:
        raise ValueError("mixing_ratio must be in [0,1]")
    replace_mask = torch.zeros_like(causal_mask, dtype=torch.bool)
    replace_mask[selected_users] = ~causal_mask[selected_users]
    replace_mask = replace_mask.unsqueeze(0).expand(times, -1, -1)
    if active is not None:
        if active.shape != (times, nodes) or active.dtype != torch.bool:
            raise ValueError("active must be boolean [T,N]")
        replace_mask = replace_mask & active.unsqueeze(-1)
    replace_mask = replace_mask.reshape(-1)
    count = int(replace_mask.sum())
    if count == 0:
        return history.clone()
    if observed_pool.shape[0] == 0:
        raise ValueError("Cannot replace factors without observed historical representations")
    observed_ids = torch.randint(observed_pool.shape[0], (count,), device=history.device)
    samples = observed_pool[observed_ids]
    if mixing_ratio > 0:
        if generated_pool is None or generated_pool.shape[0] == 0:
            raise ValueError("A nonempty generated pool is required when mixing_ratio > 0")
        generated_ids = torch.randint(generated_pool.shape[0], (count,), device=history.device)
        use_generated = torch.rand(count, device=history.device) < mixing_ratio
        samples = torch.where(use_generated[:, None], generated_pool[generated_ids], samples)
    output = history.reshape(-1, width).clone()
    output[replace_mask] = samples
    return output.reshape_as(history)


class DyTrustOOD(nn.Module):
    """All learnable components belong to one module and one optimizer.

    The paper leaves final fusion unspecified. This implementation combines the
    last GRU representation, the mean of replaced historical representations,
    and the mean of prefix-only structural mediators,
    followed by a learned projection and the ordered-pair decoder. Dropout is
    absent from this final predictor so Eq. (10) measures replacement variation.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig instance")
        self.config = config
        self.encoder = EnvironmentEncoder(config)
        self.gru = GRUEncoder(config)
        self.cvae = None if config.no_cvae else ConditionalVAE(config)
        self.structural = None if config.no_frontdoor else StructuralMediator(config)
        dimension = config.n_factors * config.delta_d
        self.fusion = nn.Sequential(nn.Linear(3 * dimension, dimension), nn.Tanh(),
                                    nn.LayerNorm(dimension) if dimension > 1 else nn.Identity())

    def encode(self, snapshots):
        features = [snapshot["x"] for snapshot in snapshots]
        edges = [snapshot["edge_index"] for snapshot in snapshots]
        active = torch.stack([snapshot["active_nodes"] for snapshot in snapshots])
        encoded = self.encoder(edges, features) * active.unsqueeze(-1)
        return self.gru(encoded, active)

    def causal_mask(self, history, active=None):
        # Low temporal variance is a selection heuristic, not causal identification.
        return identify_factors(history.detach(), self.config.n_factors, active,
                                getattr(self.config, "factor_selection", "mean"))

    def mediator(self, history, causal_mask, structure, active=None):
        if self.structural is None:
            return torch.zeros_like(history)
        full_mask = causal_mask.repeat_interleave(self.config.delta_d, dim=-1)
        outputs = []
        for time, original in enumerate(history):
            cache = structure[time]
            stable = original * full_mask
            attributes = cache["edge_features"]
            if getattr(self.config, "structural_features", "stable") == "stable":
                source, destination = cache["edge_index"]
                a, b = stable[source], stable[destination]
                dot = (a * b).sum(-1)
                a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
                denominator = (a.square().sum(-1) * b.square().sum(-1)).clamp_min(1e-12).sqrt()
                correlation = ((a * b).sum(-1) / denominator).clamp(-1, 1)
                attributes = torch.stack((dot, correlation, attributes[:, 2]), dim=-1)
            output = self.structural(stable, attributes, cache["edge_index"], cache["hodge"])
            outputs.append(output if active is None else output * active[time].unsqueeze(-1))
        return torch.stack(outputs)

    def fuse(self, replaced_history, mediator_history, active=None):
        if active is None:
            history_mean, mediator_mean = replaced_history.mean(0), mediator_history.mean(0)
        else:
            weight = active.to(replaced_history.dtype).unsqueeze(-1)
            count = weight.sum(0).clamp_min(1)
            history_mean = (replaced_history * weight).sum(0) / count
            mediator_mean = (mediator_history * weight).sum(0) / count
        fused = self.fusion(torch.cat((replaced_history[-1], history_mean, mediator_mean), dim=-1))
        return fused if active is None else fused * active[-1].unsqueeze(-1)

    @property
    def decoder(self):
        return self.encoder.edge_decoder

    def observed_pool(self, history, snapshots):
        """Eq. (7), excluding nodes not yet observed at each historical time.

        The optional cap is uniform sampling of observations, not selection of
        low-variance factors. Time/factor conditions follow the SAME flat order.
        """
        active = torch.stack([snapshot["active_nodes"] for snapshot in snapshots])
        coordinates = active.nonzero(as_tuple=False)
        width, factors = self.config.delta_d, self.config.n_factors
        observed = history.reshape(*history.shape[:2], factors, width)[active].reshape(-1, width)
        factors_ids = torch.arange(factors, device=history.device).repeat(coordinates.shape[0])
        time_ids = coordinates[:, 0].repeat_interleave(factors)
        if self.config.pool_size and observed.shape[0] > self.config.pool_size:
            indices = torch.randperm(observed.shape[0], device=history.device)[:self.config.pool_size]
            observed, factors_ids, time_ids = observed[indices], factors_ids[indices], time_ids[indices]
        conditions = None if self.cvae is None else self.cvae.make_condition(factors_ids, time_ids)
        return observed, conditions

    def environment_pools(self, history, snapshots, compute_cvae_loss=True, training_condition_limit=None):
        observed, conditions = self.observed_pool(history, snapshots)
        detached = getattr(self.config, "environment_gradients", "detached") == "detached"
        if detached:
            observed = observed.detach()
        cvae_loss = history.sum() * 0.0
        generated = None
        if self.cvae is not None and observed.shape[0]:
            # For optional MC inference, generated samples must use conditions
            # encountered during training, never untrained validation/test times.
            if compute_cvae_loss:
                size = min(self.config.cvae_batch_size, observed.shape[0])
                sample_ids = torch.randperm(observed.shape[0], device=history.device)[:size]
                cvae_loss = self.cvae.loss(observed[sample_ids], conditions[sample_ids])
            if self.config.gen_ratio > 0 and not self.config.no_intervention:
                limit = len(snapshots) if training_condition_limit is None else min(training_condition_limit, len(snapshots))
                # Draw conditions from all eligible observations, independently of
                # the observed-pool cap. Otherwise a tiny pool could accidentally
                # remove all training-time conditions during MC evaluation.
                eligible = torch.stack([snapshot["active_nodes"] for snapshot in snapshots[:limit]]).nonzero(as_tuple=False)
                if eligible.shape[0]:
                    ids = torch.randint(eligible.shape[0], (self.config.generated_pool_size,), device=history.device)
                    factors = torch.randint(self.config.n_factors, (self.config.generated_pool_size,), device=history.device)
                    generated_conditions = self.cvae.make_condition(factors, eligible[ids, 0])
                    generated = self.cvae.sample(generated_conditions)
                    if detached:
                        generated = generated.detach()
        return observed, generated, cvae_loss

    def replace(self, history, causal_mask, selected, observed, generated, active=None):
        if self.config.no_intervention or observed.shape[0] == 0:
            return history.clone()
        mixing = self.config.gen_ratio if generated is not None else 0.0
        return replace_spurious(history, causal_mask, selected, observed, generated, mixing, active)



    def _validate_inputs(self, snapshots, edge_index, structure=None, labels=None):
        """Validate the tensor API, without loading, sampling or modifying data."""
        if not isinstance(snapshots, Sequence) or len(snapshots) == 0:
            raise ValueError("snapshots must be a nonempty sequence of historical tensors")
        if len(snapshots) > self.config.length:
            raise ValueError("History exceeds ModelConfig.length (time-condition capacity)")
        weight = next(self.parameters())
        num_nodes = None

        def check_edges(edges, name):
            if not isinstance(edges, Tensor) or edges.dtype != torch.long:
                raise TypeError(f"{name} must be a torch.long tensor")
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError(f"{name} must have shape [2,E]")
            if edges.device != weight.device:
                raise ValueError(f"{name} and model must be on the same device")
            if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
                raise ValueError(f"{name} contains an out-of-range node index")

        for time, snapshot in enumerate(snapshots):
            if not isinstance(snapshot, Mapping):
                raise TypeError("Each snapshot must be a mapping of tensors")
            if not {"x", "edge_index", "active_nodes"} <= snapshot.keys():
                raise ValueError("Each snapshot requires x, edge_index and active_nodes")
            x, active = snapshot["x"], snapshot["active_nodes"]
            if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[1] != self.config.nfeat:
                raise ValueError("Snapshot x must have shape [N,config.nfeat]")
            num_nodes = x.shape[0] if num_nodes is None else num_nodes
            if num_nodes < 1 or x.shape[0] != num_nodes:
                raise ValueError("Snapshots must share a nonempty node axis")
            if x.device != weight.device or x.dtype != weight.dtype:
                raise ValueError("Snapshot x must match the model device and floating dtype")
            if not bool(torch.isfinite(x).all()):
                raise ValueError("Snapshot features must be finite")
            if (not isinstance(active, Tensor) or active.dtype != torch.bool
                    or active.shape != (num_nodes,) or active.device != weight.device):
                raise ValueError("active_nodes must be boolean [N] on the model device")
            check_edges(snapshot["edge_index"], f"snapshots[{time}].edge_index")
            if not bool(active[snapshot["edge_index"]].all()):
                raise ValueError("Historical edges cannot involve inactive nodes")
        check_edges(edge_index, "target edge_index")

        if self.structural is not None:
            if not isinstance(structure, Sequence) or len(structure) != len(snapshots):
                raise ValueError("Provide one structural tensor mapping per historical snapshot")
            for time, item in enumerate(structure):
                if not isinstance(item, Mapping) or not {"edge_index", "edge_features", "hodge"} <= item.keys():
                    raise ValueError("Each structure requires edge_index, edge_features and hodge")
                check_edges(item["edge_index"], f"structure[{time}].edge_index")
                edges, features, operators = item["edge_index"], item["edge_features"], item["hodge"]
                if (not isinstance(features, Tensor) or features.shape != (edges.shape[1], 3)
                        or features.device != weight.device or features.dtype != weight.dtype):
                    raise ValueError("Structural edge_features must be [E,3] on the model device/dtype")
                if not bool(torch.isfinite(features).all()):
                    raise ValueError("Structural edge features must be finite")
                if not isinstance(operators, HodgeOperators):
                    raise TypeError("hodge must be a HodgeOperators instance")
                if (operators.b1.shape != (num_nodes, edges.shape[1])
                        or operators.b2.shape[0] != edges.shape[1]
                        or not torch.equal(edges, operators.edge_index)):
                    raise ValueError("Hodge operators and structural edge order do not match")
                if not bool(snapshots[time]["active_nodes"][edges].all()):
                    raise ValueError("Structural edges cannot involve inactive nodes")
        if labels is not None:
            if (not isinstance(labels, Tensor) or labels.shape != (edge_index.shape[1],)
                    or labels.dtype != weight.dtype or labels.device != weight.device):
                raise ValueError("labels must be floating [E] on the model device/dtype")
            if labels.numel() == 0:
                raise ValueError("Training requires at least one supervised pair")
            if not bool(torch.isfinite(labels).all()) or bool(((labels < 0) | (labels > 1)).any()):
                raise ValueError("BCE labels must be finite values in [0,1]")

    def forward(self, snapshots: Sequence[Mapping[str, Tensor]], edge_index: Tensor,
                structure: Sequence[Mapping] | None = None, *, logits: bool = True) -> Tensor:
        """Predict supplied user pairs using only the supplied historical prefix.

        No random replacement is performed. Call ``eval()`` for inference.
        With the default mean selector this is deterministic; ``random_matched``
        remains a stochastic ablation. Returns logits [E], or probabilities when
        ``logits=False``. Labels are neither required nor consumed.
        """
        self._validate_inputs(snapshots, edge_index, structure)
        history = self.encode(snapshots)
        active = torch.stack([snapshot["active_nodes"] for snapshot in snapshots])
        mask = self.causal_mask(history, active)
        mediator = self.mediator(history, mask, structure, active)
        z = self.fuse(history, mediator, active)
        chunks = [self.decoder(z, edge_index[:, start:start + self.config.edge_batch_size], logits=logits)
                  for start in range(0, edge_index.shape[1], self.config.edge_batch_size)]
        return torch.cat(chunks) if chunks else z.new_empty((0,))

    def compute_loss(self, snapshots: Sequence[Mapping[str, Tensor]], edge_index: Tensor,
                     labels: Tensor, structure: Sequence[Mapping] | None = None):
        """One target's objective: alpha*CVAE + beta*Var(loss_j) + mean(loss_j).

        History, supervised pairs, labels, selected users, structural mediators
        and environment pools are fixed across J replacements. Variance uses
        denominator J. This method does not sample labels, split datasets, step
        an optimizer, select checkpoints or evaluate test data.
        """
        self._validate_inputs(snapshots, edge_index, structure, labels)
        history = self.encode(snapshots)
        active = torch.stack([snapshot["active_nodes"] for snapshot in snapshots])
        mask = self.causal_mask(history, active)
        mediator = self.mediator(history, mask, structure, active)
        observed, generated, cvae_loss = self.environment_pools(history, snapshots)
        selected = select_users(snapshots[-1]["active_nodes"], self.config.interv_size_ratio)
        losses = []
        count = 1 if self.config.no_intervention else self.config.n_intervene
        for _ in range(count):
            replaced = self.replace(history, mask, selected, observed, generated, active)
            z = self.fuse(replaced, mediator, active)
            losses.append(edge_bce(z, edge_index, labels, self.decoder, self.config.edge_batch_size))
        return intervention_objective(cvae_loss, losses, self.config.alpha,
                                      0.0 if self.config.no_intervention else self.config.beta)
