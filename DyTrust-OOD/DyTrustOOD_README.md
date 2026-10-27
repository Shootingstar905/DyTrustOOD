# DyTrust-OOD

Core model implementation for dynamic trust prediction. The implementation includes environment-specific block encoding, a unidirectional GRU, temporal stability-based block selection, a conditional variational autoencoder (CVAE), representation replacement, Hodge–Laguerre structural mediation, and the joint training objective.

This repository provides a model library only. It does not include datasets, data preprocessing, negative sampling, synthetic OOD feature generation, a complete experiment runner, tests, or continuous integration files. Data and historical structural inputs must be supplied by the caller.

## Installation

Python 3.10 or later is required. PyTorch is the only runtime dependency. Install a PyTorch build appropriate for your CPU or CUDA environment, then run the following command from the repository root:

```bash
python -m pip install -e .
```

The project and repository name is `DyTrust-OOD`; the Python import name is `dytrust_ood`. Hyphens cannot be used in standard Python package import statements.

```python
from dytrust_ood import DyTrustOOD, ModelConfig, build_hodge

config = ModelConfig(nfeat=8, length=11)
model = DyTrustOOD(config)
```

Set `nfeat` to the width of your input features. `length` specifies the capacity of the CVAE time condition; it does not automatically partition data into training, validation, and test sets. The hidden representation width is `n_factors * delta_d`, which defaults to `8 * 16 = 128`.

## Repository Structure

```text
DyTrust-OOD/
├── dytrust_ood/
│   ├── __init__.py
│   ├── config.py       # Model configuration only
│   ├── encoders.py     # Environment graph encoding, GRU, stability selection, CVAE
│   ├── hodge.py        # Sparse boundary operators, Laguerre filtering, graph aggregation
│   ├── mediation.py    # Structural mediation module
│   ├── model.py        # Full model, replacement, prediction, and loss interfaces
│   └── losses.py       # BCE and the mean and variance of intervention losses
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Input Specification

### Historical Snapshots: `snapshots`

Supply a list of `T` snapshots, where `1 <= T <= config.length`. All snapshots must use the same node indexing and node-axis size `N`. Each snapshot contains:

| Key | Tensor | Description |
|---|---|---|
| `x` | Floating-point `[N, F]` | Node features available by that historical time, with `F=config.nfeat` |
| `edge_index` | `torch.long [2, E_t]` | Directed historical trust edges; messages pass from source to target |
| `active_nodes` | `torch.bool [N]` | Whether each node is already observable at that time |

All historical tensors and prediction pairs must be on the same device as the model. Floating-point inputs must match the model's floating-point dtype. The model does not move or filter caller-provided data. Historical edges must not connect nodes that have not yet appeared. When predicting a future time step, do not include future snapshots or features derived from future labels in the historical inputs.

### Structural Inputs: `structures`

The structural branch is enabled by default. Supply one structural dictionary for each historical snapshot:

| Key | Tensor / Object | Description |
|---|---|---|
| `edge_index` | `torch.long [2, M_t]` | Canonical edges in exactly the same order as `hodge.edge_index` |
| `edge_features` | Floating-point `[M_t, 3]` | Three columns: inner product, Pearson correlation, and DTW similarity of historical degree sequences, in that order |
| `hodge` | `HodgeOperators` | Sparse boundary operators for the corresponding snapshot |

With `structural_features="stable"`, the first two columns are recomputed from the selected stable representations during the forward pass. The third column must still be a valid DTW similarity computed by the caller using historical information only. This library does not compute or cache DTW features, and it does not silently substitute zeros or random features for them.

`build_hodge` constructs the mathematical operators used by the model. It accepts an already prepared historical edge tensor and constructs `B1` and `B2` on the undirected simple skeleton without reading any data files. This function is retained to support full Hodge filtering:

```python
hodge = build_hodge(
    history_edge_index,
    num_nodes=num_nodes,
    max_triangles=config.max_triangles,
    dtype=next(model.parameters()).dtype,
)
# Prepare or reorder edge_features to match hodge.edge_index exactly.
structure = {
    "edge_index": hodge.edge_index,
    "edge_features": edge_features,
    "hodge": hodge,
}
```

The operators act on an undirected structural skeleton, while supervised trust relationships remain directed. The edge order returned by `build_hodge` is authoritative: do not pass features for the original directed edges without aligning them. The caller is responsible for ensuring that all structural tensors come from the corresponding historical snapshot. Their temporal provenance cannot be verified from tensor values alone.

Set `ModelConfig(no_frontdoor=True)` to disable the structural branch and use `structures=None`. This is a different model variant, not an equivalent way to supply inputs to the full model.

### Supervised User Pairs

`pairs` is a `torch.long` tensor of shape `[2, B]`, preserving source–target direction. `labels` is a floating-point tensor of shape `[B]`, containing 0 or 1 for the binary task. The model does not construct positive or negative samples, define cold-start candidates, split data, or choose an evaluation protocol.

## Prediction and Training

The following code assumes that the caller has already prepared `snapshots`, `structures`, `pairs`, and `labels`. It is not a built-in dataset example.

```python
import torch

model.eval()
with torch.no_grad():
    probabilities = model(snapshots, pairs, structures, logits=False)
    # The default logits=True returns pre-sigmoid scores, also with shape [B].

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-7)
model.train()
optimizer.zero_grad(set_to_none=True)
loss, terms = model.compute_loss(snapshots, pairs, labels, structures)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
optimizer.step()
```

`compute_loss` computes the joint objective for one prediction target:

$$
\mathcal L=\alpha\mathcal L_{\mathrm{CVAE}}
+\beta\frac1J\sum_{j=1}^{J}(\ell_j-\bar\ell)^2+\bar\ell,
\qquad \bar\ell=\frac1J\sum_{j=1}^{J}\ell_j.
$$

`terms` contains the three unweighted loss components: `cvae`, `variance`, and `prediction`. Replacement samples share the original history, supervised pairs, labels, selected users, and environment pools. The variance denominator is `J`; the variance is zero when `J=1`. The default forward prediction does not perform random replacement. With `factor_selection="random_matched"`, block selection itself remains stochastic.

The example above performs one parameter update. Gradient accumulation across multiple targets, early stopping, checkpoint selection, and evaluation are the caller's responsibility. It is not a script that automatically reproduces the original experimental training protocol. All learnable components are registered within `model`, so the optimizer does not need separate parameter lists for additional modules.

## Core Defaults

| Configuration | Default |
|---|---|
| Number of environments / block width / total width | `n_factors=8`, `delta_d=16`, total width 128 |
| Graph / temporal encoder | One environment convolution layer and a one-layer unidirectional GRU |
| Dropout | `0.1` applies only between graph encoder layers; the effective value is 0 for the default single-layer encoder |
| CVAE | Latent dimension 16; hidden width `max(32, 2*delta_d, 2*d_for_cvae)` |
| Observed / generated pools | Caps of 4,096 / 512; CVAE batch size capped at 1,024 |
| Intervention | `J=5`, user fraction 0.6, generated-pool sampling probability 0.5 |
| Loss weights | `alpha=0.01`, `beta=0.01` |
| Hodge filter | Full operator, 3 Laguerre terms, scalar coefficients, no spectral rescaling by default |
| Structural and gradient settings | `equivariant` / `stable` / `detached` |
| Edge chunking | `edge_batch_size=65536`; controls decoder computation chunks only |

The configuration retains only parameters needed for model computation. See `config.py` for the complete list. Defaults match the version before repository simplification, including the unscaled Hodge operator. You may explicitly set `hodge_normalize=True`, but this choice must be documented in the experimental configuration. Exceeding `max_triangles` raises an error rather than silently discarding triangles.

## Saving and Loading

```python
from dataclasses import asdict

torch.save({"config": asdict(config), "state_dict": model.state_dict()}, "model.pt")
state = torch.load("model.pt", map_location="cpu", weights_only=True)
restored = DyTrustOOD(ModelConfig(**state["config"]))
restored.load_state_dict(state["state_dict"], strict=True)
restored.eval()
```

The previous experiment runner and its automatic checkpoint restoration entry point have been removed. This example saves and restores only the model configuration and parameter state dictionary.

## Methodological Scope and Release Status

Temporal stability is a representation-selection heuristic, not a guarantee of causal identification. The structural module is a causal-inspired predictive architecture, not an implementation of classical front-door identification. Orientation-consistent readout, stable-representation-based edge attributes, and gradient isolation retain the implementation choices documented in the previous version. This repository simplification introduces no new experimental results and does not establish reproduction of the paper's result tables.

Author identities and a software license have not been supplied in this package. Before public release, the rights holders should confirm code provenance, required attribution, and licensing terms.
