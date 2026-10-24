"""Post-training, confidence-weighted CAPP on a self-loop-free graph."""
from __future__ import annotations

from argparse import Namespace

import numpy as np
import scipy.sparse.csgraph as csgraph
import torch
from torch.nn import functional as F

from .structure import adjacency_matrix


@torch.no_grad()
def capp(hidden: torch.Tensor, edges: torch.Tensor, positive_weights: torch.Tensor,
         args: Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix hops0..K using retention, weighted positive agreement and oversmoothing.

    Isolated nodes have zero higher-hop features and zero stationary reference;
    hop0 remains available. An empty positive row falls back to self-retention
    for its agreement score. No virtual self edge is inserted.
    """
    n = len(hidden)
    adjacency = adjacency_matrix(edges, n)
    degrees_cpu = np.asarray(adjacency.sum(1)).reshape(-1)
    degree = torch.as_tensor(degrees_cpu, device=hidden.device, dtype=hidden.dtype)
    row, col = adjacency.nonzero()
    indices = torch.as_tensor(np.stack((row, col)), device=hidden.device)
    inverse = torch.zeros_like(degree)
    inverse[degree > 0] = degree[degree > 0].pow(-.5)
    values = inverse[indices[0]] * inverse[indices[1]]
    operator = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    count, components_cpu = csgraph.connected_components(adjacency, directed=False)
    component = torch.as_tensor(components_cpu, device=hidden.device, dtype=torch.long)
    root_degree = degree.sqrt()
    sums = hidden.new_zeros((count, hidden.shape[1]))
    sums.index_add_(0, component, hidden * root_degree[:, None])
    mass = hidden.new_zeros(count)
    mass.index_add_(0, component, degree)
    stationary = root_degree[:, None] * sums[component] / mass[component, None].clamp_min(1e-12)

    if positive_weights.is_sparse:
        confidence = positive_weights.coalesce()
        indices, values = confidence.indices(), confidence.values()
        keep = indices[0] != indices[1]
        indices, values = indices[:, keep], values[keep]
        positive_mass = hidden.new_zeros(n).index_add(0, indices[0], values)
        denominator = torch.where(positive_mass > 0, positive_mass, torch.ones_like(positive_mass))
        confidence = torch.sparse_coo_tensor(indices, values / denominator[indices[0]], (n, n)).coalesce()
        target = torch.sparse.mm(confidence, F.normalize(hidden, dim=-1, eps=1e-8))
    else:
        confidence = positive_weights.clone()
        confidence.diagonal().zero_()
        positive_mass = confidence.sum(1)
        confidence /= torch.where(positive_mass > 0, positive_mass, torch.ones_like(positive_mass))[:, None]
        target = confidence @ F.normalize(hidden, dim=-1, eps=1e-8)
    hops, scores = [hidden], []
    for _ in range(args.K):
        hops.append(torch.sparse.mm(operator, hops[-1]))
    for feature in hops:
        retention = F.cosine_similarity(feature, hidden, dim=-1, eps=1e-8)
        agreement = (F.normalize(feature, dim=-1, eps=1e-8) * target).sum(1)
        agreement = torch.where(positive_mass > 0, agreement, retention)
        oversmoothing = F.cosine_similarity(feature, stationary, dim=-1, eps=1e-8)
        scores.append(args.alpha_ret * retention + args.alpha_pos * agreement - args.alpha_os * oversmoothing)
    hop_weights = (torch.stack(scores, dim=1) / args.tau_capp).softmax(1)
    output = torch.zeros_like(hidden)
    for hop, feature in enumerate(hops):
        output.add_(feature * hop_weights[:, hop, None])
    if not torch.isfinite(output).all():
        raise ValueError("CAPP produced nonfinite embeddings")
    return output, hop_weights
