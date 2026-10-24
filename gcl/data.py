"""Dataset loading, with labels kept out of the training interface."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.transforms import NormalizeFeatures

SUPPORTED_DATASETS = ("Cora", "CiteSeer", "PubMed", "Cornell", "Texas")


def load_dataset(name: str, root: Path, normalize_features: bool = False) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return feature/edge training inputs and separate labels/evaluation masks.

    Citation graphs use their public Planetoid masks. WebKB graphs use the first
    predefined Geom-GCN split supplied by PyG. No random split is generated.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Supported datasets: {SUPPORTED_DATASETS}")
    transform = NormalizeFeatures() if normalize_features else None
    folder = str(root / name)
    dataset = (WebKB(folder, name, transform=transform) if name in ("Cornell", "Texas")
               else Planetoid(folder, name, split="public", transform=transform))
    data = dataset[0]
    n = int(data.num_nodes)
    if not torch.isfinite(data.x).all():
        raise ValueError("Features contain NaN or Inf")
    labels = data.y.reshape(-1).long().cpu()
    evaluation: dict[str, Any] = {"y": labels, "dataset": name}
    for split in ("train", "val", "test"):
        mask = getattr(data, split + "_mask")
        mask = mask[:, 0] if mask.ndim == 2 else mask
        mask = mask.bool().cpu()
        if mask.shape != (n,) or not mask.any():
            raise ValueError(f"Invalid {split} mask")
        evaluation[split + "_mask"] = mask
    train, val, test = (evaluation[s + "_mask"] for s in ("train", "val", "test"))
    if bool((train & val).any() or (train & test).any() or (val & test).any()):
        raise ValueError("Evaluation masks overlap")
    graph = {"x": data.x.float().cpu(), "edge_index": data.edge_index.long().cpu()}
    return graph, evaluation
