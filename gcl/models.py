"""Two-layer MLP and GCN encoders, with phase-1 projection heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class MLPEncoder(nn.Module):
    """Feature-only encoder: linear, BatchNorm, PReLU, dropout, linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList((nn.Linear(in_dim, hidden_dim), nn.Linear(hidden_dim, out_dim)))
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edges: torch.Tensor | None = None) -> torch.Tensor:
        del edges
        return self.layers[1](self.dropout(self.activation(self.bn(self.layers[0](x)))))


class GCNEncoder(nn.Module):
    """The two-layer GCN used by the included GRACE benchmark implementation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList((GCNConv(in_dim, hidden_dim), GCNConv(hidden_dim, out_dim)))

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        hidden = F.dropout(F.relu(self.convs[0](x, edges)), p=self.dropout, training=self.training)
        return self.convs[1](hidden, edges)


class ProjectionHead(nn.Module):
    """GRACE projection head; final downstream embeddings bypass this head."""

    def __init__(self, in_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.act = nn.PReLU()
        self.fc2 = nn.Linear(in_dim, projection_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc2(self.act(self.fc1(hidden))), dim=-1)


class RepresentationModel(nn.Module):
    """Encoder plus a projector used only for two-view contrastive training."""

    def __init__(self, method: str, in_dim: int, hidden_dim: int, out_dim: int,
                 projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.method = method
        encoder = MLPEncoder if method == "d2gcl" else GCNEncoder
        self.encoder = encoder(in_dim, hidden_dim, out_dim, dropout)
        # D2-GCL retains its existing ReLU projector for phase 1.
        self.projector = (nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(),
                                       nn.Linear(out_dim, projection_dim)) if method == "d2gcl"
                          else ProjectionHead(out_dim, projection_dim))

    def encode(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edges)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.projector(hidden)
        return F.normalize(projected, dim=-1) if self.method == "d2gcl" else projected
