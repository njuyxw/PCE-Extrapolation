"""Model registries.

Two registries, both populated by import-time decorators inside the encoder /
predictor subpackages. The registries are the single point of extension:

    @register_encoder("my_gnn")
    class MyGNN(BaseEncoder): ...

    @register_predictor("my_pred")
    class MyPredictor(BasePredictor): ...

A new yaml ``encoder.kind: my_gnn`` then routes through ``build_encoder``.
"""
from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

ENCODER_REGISTRY: dict[str, type[nn.Module]] = {}
PREDICTOR_REGISTRY: dict[str, type[nn.Module]] = {}


def register_encoder(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    def deco(cls: type[nn.Module]) -> type[nn.Module]:
        if name in ENCODER_REGISTRY:
            raise ValueError(f"Encoder `{name}` already registered.")
        ENCODER_REGISTRY[name] = cls
        return cls
    return deco


def register_predictor(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    def deco(cls: type[nn.Module]) -> type[nn.Module]:
        if name in PREDICTOR_REGISTRY:
            raise ValueError(f"Predictor `{name}` already registered.")
        PREDICTOR_REGISTRY[name] = cls
        return cls
    return deco


def build_encoder(kind: str, **kwargs: Any) -> nn.Module:
    if kind not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder `{kind}`. Available: {sorted(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[kind](**kwargs)


def build_predictor(kind: str, **kwargs: Any) -> nn.Module:
    if kind not in PREDICTOR_REGISTRY:
        raise KeyError(f"Unknown predictor `{kind}`. Available: {sorted(PREDICTOR_REGISTRY)}")
    return PREDICTOR_REGISTRY[kind](**kwargs)


# Importing the subpackages triggers @register_* side effects.
from . import encoders as _enc  # noqa: E402, F401
from . import predictors as _pred  # noqa: E402, F401

__all__ = [
    "ENCODER_REGISTRY",
    "PREDICTOR_REGISTRY",
    "build_encoder",
    "build_predictor",
    "register_encoder",
    "register_predictor",
]
