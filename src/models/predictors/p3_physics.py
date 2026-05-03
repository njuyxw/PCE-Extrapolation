"""P3Physics — physics-anchored multi-task PCE predictor.

Differences from the baseline ``P3``:

  - Multi-output head: predicts ``[Voc, Jsc, FF]`` separately, then composes
    ``PCE = Voc * Jsc * FF + δ_PCE`` where ``δ_PCE`` is a small learned
    residual (so the model can correct the empirical Scharber product).
  - **Voc is anchored to predicted HOMO_D / LUMO_A** via the Scharber rule:
    ``Voc = clamp(LUMO_A − HOMO_D − 0.3, 0, 2.5) + 0.2 · δ_Voc``.
    The donor's HOMO and acceptor's LUMO are already estimated by the MOE2
    encoders (regression heads from stage 2 pretraining), so this anchor
    works out-of-the-box: a candidate predicted to have a deep HOMO_D and
    a high LUMO_A naturally produces a *high* Voc above the training range.
  - Forward returns a ``dict`` (not a tensor), so the trainer can route
    each component into the composite loss.

The graph + cross-attention featurizer is identical to the baseline ``P3``,
so any pretrained MOE2 encoder loads via ``load_pretrained_encoders``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import to_dense_batch

from .. import build_encoder, register_predictor
from .p3 import _CrossGraphAttention, _dense_to_sparse


@register_predictor("p3_physics")
class P3Physics(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        edge_dim: int = 15,
        hidden_channels: int = 512,
        out_channels: int = 512,
        homolumo_targets: int = 2,
        pce_targets: int = 1,                # ignored — output is a dict
        dropout_rate: float = 0.25,
        num_heads: int = 8,
        encoder_kind: str = "moe2",
        # Physics knobs
        voc_offset: float = 0.30,            # Scharber: Voc = (|HOMO_D| − |LUMO_A|) − 0.3
        voc_clamp_min: float = 0.0,
        voc_clamp_max: float = 2.5,
        voc_residual_scale: float = 0.20,
        ff_min: float = 0.30,
        ff_max: float = 0.85,
        jsc_min: float = 1.0,
        jsc_max: float = 35.0,
        pce_residual_scale: float = 0.5,
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

        # Shared trunk over fused features → multi-output heads.
        fused_dim = out_channels * 4 + 64 * 2
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.head_voc_residual = nn.Linear(256, 1)
        self.head_jsc_raw = nn.Linear(256, 1)
        self.head_ff_raw = nn.Linear(256, 1)
        self.head_pce_residual = nn.Linear(256, 1)

        # Hyper-parameters as buffers so .to(device) carries them but they don't drift.
        self.register_buffer("voc_offset", torch.tensor(voc_offset))
        self.register_buffer("voc_clamp_min", torch.tensor(voc_clamp_min))
        self.register_buffer("voc_clamp_max", torch.tensor(voc_clamp_max))
        self.voc_residual_scale = voc_residual_scale
        self.ff_min = ff_min
        self.ff_max = ff_max
        self.jsc_min = jsc_min
        self.jsc_max = jsc_max
        self.pce_residual_scale = pce_residual_scale

    # -------------------------------- ckpt helpers -----------------------

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

    # -------------------------------- forward ----------------------------

    def forward(self, donor, acceptor) -> dict[str, torch.Tensor]:
        bs = donor.num_graphs

        d_node = self.donor_encoder.encode_nodes(donor.x, donor.edge_index, donor.edge_attr)
        a_node = self.acceptor_encoder.encode_nodes(acceptor.x, acceptor.edge_index, acceptor.edge_attr)
        d_graph = self.donor_encoder.pool(d_node, donor.batch, dim_size=bs)
        a_graph = self.acceptor_encoder.pool(a_node, acceptor.batch, dim_size=bs)

        # HOMO/LUMO predicted by MOE2 heads (units: eV).
        d_hl = self.donor_encoder.regression_head(d_graph)        # [B, 2] (HOMO_D, LUMO_D)
        a_hl = self.acceptor_encoder.regression_head(a_graph)      # [B, 2] (HOMO_A, LUMO_A)

        # Cross-attention donor↔acceptor (same as baseline P3).
        d_dense, d_mask = to_dense_batch(d_node, donor.batch, batch_size=bs)
        a_dense, a_mask = to_dense_batch(a_node, acceptor.batch, batch_size=bs)
        d_xattn = self.cross_d_to_a(d_dense, a_dense, a_mask, d_mask)
        a_xattn = self.cross_a_to_d(a_dense, d_dense, d_mask, a_mask)
        d_attn_flat, d_attn_batch = _dense_to_sparse(d_xattn, d_mask)
        a_attn_flat, a_attn_batch = _dense_to_sparse(a_xattn, a_mask)
        d_attn = self.attn_pool(d_attn_flat, d_attn_batch, dim_size=bs)
        a_attn = self.attn_pool(a_attn_flat, a_attn_batch, dim_size=bs)

        d_graph = self.norm_graph(d_graph); a_graph = self.norm_graph(a_graph)
        d_attn = self.norm_attn(d_attn);    a_attn = self.norm_attn(a_attn)
        d_hl_proj = self.homolumo_proj(d_hl); a_hl_proj = self.homolumo_proj(a_hl)
        z = torch.cat([d_graph, a_graph, d_attn, a_attn, d_hl_proj, a_hl_proj], dim=-1)
        h = self.trunk(z)                                         # [B, 256]

        # ---- Voc: physics anchor + small residual ----------------------
        # Scharber: Voc = (|HOMO_D| - |LUMO_A|) - 0.3 = LUMO_A - HOMO_D - 0.3 (HOMO_D, LUMO_A < 0)
        homo_d = d_hl[:, 0]
        lumo_a = a_hl[:, 1]
        voc_anchor = torch.clamp(lumo_a - homo_d - self.voc_offset,
                                 min=float(self.voc_clamp_min),
                                 max=float(self.voc_clamp_max))
        voc_residual = self.head_voc_residual(h).squeeze(-1)
        voc = voc_anchor + self.voc_residual_scale * voc_residual

        # ---- Jsc: bounded sigmoid in plausible range -------------------
        jsc = self.jsc_min + (self.jsc_max - self.jsc_min) * torch.sigmoid(
            self.head_jsc_raw(h).squeeze(-1))

        # ---- FF: bounded sigmoid ---------------------------------------
        ff = self.ff_min + (self.ff_max - self.ff_min) * torch.sigmoid(
            self.head_ff_raw(h).squeeze(-1))

        # ---- PCE: physics product + small residual ---------------------
        pce_product = voc * jsc * ff                                                      # raw PCE-like
        pce_residual = self.head_pce_residual(h).squeeze(-1)
        pce = pce_product + self.pce_residual_scale * pce_residual

        return {
            "pce": pce, "voc": voc, "jsc": jsc, "ff": ff,
            "homo_d": homo_d, "lumo_a": lumo_a,
            "voc_anchor": voc_anchor,
        }
