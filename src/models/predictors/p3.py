"""P3: Photovoltaic Performance Predictor (donor-acceptor cross-attention).

Combines:
  1. Donor + Acceptor pooled graph embeddings (from MOE2 encoders).
  2. Donor → Acceptor and Acceptor → Donor cross-attention attention pools.
  3. Predicted (HOMO, LUMO) for both donor and acceptor (from MOE2 heads).

Concatenates (graph × 2 + attn × 2 + projected HL × 2) and regresses PCE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import to_dense_batch

from .. import build_encoder, register_predictor


def _dense_to_sparse(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``to_dense_batch``: keep only valid rows + reconstruct ``batch``."""
    B, N, _ = x.size()
    flat = x[mask]
    batch = torch.arange(B, device=x.device).unsqueeze(1).expand(B, N)[mask]
    return flat, batch


class _CrossGraphAttention(nn.Module):
    """Multihead cross-attention block (Pre-LN style with residual + FFN)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None, src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kpm = (~tgt_mask) if tgt_mask is not None else None
        m = src_mask.unsqueeze(-1) if src_mask is not None else None
        attn_out, _ = self.cross_attn(src, tgt, tgt, key_padding_mask=kpm)
        attn_out = self.dropout(attn_out)
        if m is not None:
            attn_out = attn_out * m
            src = src * m
        src = self.norm1(src + attn_out)
        ffn_out = self.dropout(self.ffn(src))
        if m is not None:
            ffn_out = ffn_out * m
        src = self.norm2(src + ffn_out)
        if m is not None:
            src = src * m
        return src


@register_predictor("p3")
class P3(nn.Module):
    """Donor-acceptor PCE predictor.

    Two MOE2 encoders share architecture but not weights (loadable from the
    same pretrained checkpoint). ``encoder_kind`` lets you swap the GNN
    backbone without touching this file.
    """

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

        self.cross_d_to_a = _CrossGraphAttention(out_channels, num_heads, dropout_rate)
        self.cross_a_to_d = _CrossGraphAttention(out_channels, num_heads, dropout_rate)

        self.attn_pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(out_channels, 128), nn.ReLU(), nn.Linear(128, 1),
        ))
        self.norm_graph = nn.LayerNorm(out_channels)
        self.norm_attn = nn.LayerNorm(out_channels)
        self.homolumo_proj = nn.Sequential(
            nn.Linear(homolumo_targets, 64), nn.ReLU(), nn.LayerNorm(64),
        )

        fused_dim = out_channels * 4 + 64 * 2
        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, pce_targets),
        )
        self.pce_targets = pce_targets

    def load_pretrained_encoders(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a single MOE2 state dict into BOTH donor and acceptor branches.

        Shape-mismatched keys are silently skipped so that a checkpoint trained
        with a different ``out_channels`` is partially usable for ablations.
        """
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
        batch_size = donor.num_graphs

        d_node = self.donor_encoder.encode_nodes(donor.x, donor.edge_index, donor.edge_attr)
        a_node = self.acceptor_encoder.encode_nodes(acceptor.x, acceptor.edge_index, acceptor.edge_attr)

        d_graph = self.donor_encoder.pool(d_node, donor.batch, dim_size=batch_size)
        a_graph = self.acceptor_encoder.pool(a_node, acceptor.batch, dim_size=batch_size)

        d_homolumo = self.donor_encoder.regression_head(d_graph)
        a_homolumo = self.acceptor_encoder.regression_head(a_graph)

        d_dense, d_mask = to_dense_batch(d_node, donor.batch, batch_size=batch_size)
        a_dense, a_mask = to_dense_batch(a_node, acceptor.batch, batch_size=batch_size)
        d_xattn = self.cross_d_to_a(d_dense, a_dense, a_mask, d_mask)
        a_xattn = self.cross_a_to_d(a_dense, d_dense, d_mask, a_mask)
        d_attn_flat, d_attn_batch = _dense_to_sparse(d_xattn, d_mask)
        a_attn_flat, a_attn_batch = _dense_to_sparse(a_xattn, a_mask)
        d_attn = self.attn_pool(d_attn_flat, d_attn_batch, dim_size=batch_size)
        a_attn = self.attn_pool(a_attn_flat, a_attn_batch, dim_size=batch_size)

        d_graph = self.norm_graph(d_graph)
        a_graph = self.norm_graph(a_graph)
        d_attn = self.norm_attn(d_attn)
        a_attn = self.norm_attn(a_attn)
        d_hl = self.homolumo_proj(d_homolumo)
        a_hl = self.homolumo_proj(a_homolumo)

        z = torch.cat([d_graph, a_graph, d_attn, a_attn, d_hl, a_hl], dim=-1)
        return self.regression_head(z)
