"""MOE2: Molecular Orbital Energy Estimator.

3-stage GATv2 hierarchy with edge-conditioned attention, attentional graph
aggregation, and dual heads (MLM atom-type / HOMO-LUMO regression).

Forward dispatch is governed by ``task_type``:

  - ``"mlm"``       → returns per-atom logits over the atom vocabulary.
  - ``"homo_lumo"`` → returns graph-level [HOMO, LUMO] regression.
  - ``"embed"``     → returns graph-level pooled embedding.
  - ``"embed_node"``→ returns per-atom embeddings (used by the P3 cross-attention).
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation

from .. import register_encoder

TaskType = Literal["mlm", "homo_lumo", "embed", "embed_node"]


@register_encoder("moe2")
class MOE2(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        edge_dim: int = 15,
        hidden_channels: int = 512,
        out_channels: int = 512,
        mlm_output_dim: int = 9,
        regression_targets: int = 2,
        heads: int = 8,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads,
                               edge_dim=edge_dim, dropout=dropout_rate)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads,
                               edge_dim=edge_dim, dropout=dropout_rate)
        self.conv3 = GATv2Conv(hidden_channels * heads, out_channels, heads=1,
                               edge_dim=edge_dim, dropout=dropout_rate)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(out_channels, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        ))
        self.regression_head = nn.Sequential(
            nn.Linear(out_channels, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, regression_targets),
        )
        self.mlm_head = nn.Sequential(
            nn.Linear(out_channels, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, mlm_output_dim),
        )

        self.out_channels = out_channels
        self.regression_targets = regression_targets

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x, edge_index, edge_attr))
        h = self.relu(self.conv2(h, edge_index, edge_attr))
        h = self.conv3(h, edge_index, edge_attr)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        task_type: TaskType = "homo_lumo",
        batch_size: int | None = None,
    ) -> torch.Tensor:
        node_emb = self.encode_nodes(x, edge_index, edge_attr)
        if task_type == "mlm":
            return self.mlm_head(node_emb)
        if task_type == "embed_node":
            return node_emb
        graph_emb = self.pool(node_emb, batch, dim_size=batch_size)
        if task_type == "embed":
            return graph_emb
        if task_type == "homo_lumo":
            return self.regression_head(graph_emb)
        raise ValueError(f"Unknown task_type: {task_type}")
