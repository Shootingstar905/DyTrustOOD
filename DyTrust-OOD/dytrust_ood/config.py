"""Model-only settings; no dataset paths, preprocessing or experiment runner."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Defaults retained from the preceding research implementation.

    ``nfeat`` is the caller's input width. Hidden width is ``n_factors*delta_d``.
    ``length`` is the capacity of CVAE time conditions, NOT a dataset split.
    ``edge_batch_size`` chunks decoder computation without stepping an optimizer.
    """
    nfeat: int = 8
    n_factors: int = 8
    delta_d: int = 16
    nhid: int = 64
    n_layers: int = 1
    dropout: float = 0.1  # Between graph layers only; inactive with one layer.
    use_RTE: bool = True
    length: int = 11

    d_for_cvae: int = 16
    cvae_batch_size: int = 1024
    pool_size: int = 4096  # 0 keeps all observed representations.
    generated_pool_size: int = 512
    n_intervene: int = 5
    interv_size_ratio: float = 0.6
    gen_ratio: float = 0.5
    alpha: float = 0.01
    beta: float = 0.01
    edge_batch_size: int = 65536

    hodge_order: int = 3
    hodge_coefficients: Literal["scalar", "matrix"] = "scalar"
    hodge_normalize: bool = False
    hodge_component: Literal["full", "lower", "upper"] = "full"
    max_triangles: int = 1000000
    structural_mode: Literal["equivariant", "literal"] = "equivariant"
    structural_features: Literal["stable", "raw"] = "stable"
    environment_gradients: Literal["detached", "joint"] = "detached"
    factor_selection: Literal["mean", "median", "random_matched"] = "mean"
    no_cvae: bool = False
    no_intervention: bool = False
    no_frontdoor: bool = False

    @property
    def hidden_dim(self) -> int:
        return self.n_factors * self.delta_d

    @property
    def edge_feature_dim(self) -> int:
        return 3

    def __post_init__(self) -> None:
        positive = ("nfeat", "n_factors", "delta_d", "nhid", "n_layers", "length",
                    "d_for_cvae", "cvae_batch_size", "generated_pool_size", "n_intervene",
                    "edge_batch_size", "hodge_order", "max_triangles")
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.pool_size) is not int or self.pool_size < 0:
            raise ValueError("pool_size must be a nonnegative integer")
        for name in ("use_RTE", "hodge_normalize", "no_cvae", "no_intervention", "no_frontdoor"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("dropout", "interv_size_ratio", "gen_ratio", "alpha", "beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 0 <= self.interv_size_ratio <= 1 or not 0 <= self.gen_ratio <= 1:
            raise ValueError("interv_size_ratio and gen_ratio must be in [0,1]")
        if min(self.alpha, self.beta) < 0:
            raise ValueError("alpha and beta must be nonnegative")
        choices = {
            "hodge_coefficients": ("scalar", "matrix"),
            "hodge_component": ("full", "lower", "upper"),
            "structural_mode": ("equivariant", "literal"),
            "structural_features": ("stable", "raw"),
            "environment_gradients": ("detached", "joint"),
            "factor_selection": ("mean", "median", "random_matched"),
        }
        for name, options in choices.items():
            if getattr(self, name) not in options:
                raise ValueError(f"{name} must be one of {options}")
