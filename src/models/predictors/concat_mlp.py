"""ConcatMLP: minimal donor/acceptor PCE baseline.

Pools donor and acceptor graph embeddings (no cross-attention), concatenates,
and regresses PCE through an MLP. Useful as:
  1. A sanity-check that any new encoder works end-to-end without P3 plumbing.
  2. A weak baseline in the ablation table.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import build_encoder, register_predictor


@register_predictor("concat_mlp")
class ConcatMLP(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        edge_dim: int = 15,
        hidden_channels: int = 512,
        out_channels: int = 512,
        homolumo_targets: int = 2,
        pce_targets: int = 1,
        dropout_rate: float = 0.25,
        num_heads: int = 8,
        encoder_kind: str = "moe2",
    ) -> None:
        super().__init__()
        encoder_kwargs = dict(
            in_channels=in_channels, edge_dim=edge_dim,
            hidden_channels=hidden_channels, out_channels=out_channels,
            regression_targets=homolumo_targets,
            heads=num_heads, dropout_rate=dropout_rate,
        )
        self.donor_encoder = build_encoder(encoder_kind, **encoder_kwargs)
        self.acceptor_encoder = build_encoder(encoder_kind, **encoder_kwargs)
        self.head = nn.Sequential(
            nn.Linear(out_channels * 2, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, pce_targets),
        )
        self.pce_targets = pce_targets

    def load_pretrained_encoders(self, state_dict: dict[str, torch.Tensor]) -> None:
        for branch in (self.donor_encoder, self.acceptor_encoder):
            target = branch.state_dict()
            compat = {k: v for k, v in state_dict.items()
                      if k in target and v.shape == target[k].shape}
            branch.load_state_dict(compat, strict=False)

    def freeze_encoders(self, freeze: bool = True) -> None:
        for p in self.donor_encoder.parameters():
            p.requires_grad = not freeze
        for p in self.acceptor_encoder.parameters():
            p.requires_grad = not freeze

    def forward(self, donor, acceptor) -> torch.Tensor:
        bs = donor.num_graphs
        d = self.donor_encoder(donor.x, donor.edge_index, donor.edge_attr,
                               donor.batch, "embed", batch_size=bs)
        a = self.acceptor_encoder(acceptor.x, acceptor.edge_index, acceptor.edge_attr,
                                  acceptor.batch, "embed", batch_size=bs)
        return self.head(torch.cat([d, a], dim=-1))
