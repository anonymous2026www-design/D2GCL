"""Graph-only LapPE and candidate pools for the five supported datasets."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import torch
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


def adjacency_matrix(edges: torch.Tensor, n: int) -> sp.csr_matrix:
    """Build a binary undirected graph with no self edges."""
    edges, _ = remove_self_loops(to_undirected(edges.cpu(), num_nodes=n))
    edges = coalesce(edges, num_nodes=n)
    row, col = edges.numpy()
    adjacency = sp.csr_matrix((np.ones(len(row), dtype=np.float64), (row, col)), shape=(n, n))
    adjacency.sort_indices()
    return adjacency


def laplacian_encoding(edges: torch.Tensor, n: int, m: int, dense_limit: int = 4096) -> torch.Tensor:
    """Concatenate the m smallest positive and m largest Laplacian eigenvectors.

    Small graphs use a dense eigendecomposition; PubMed uses sparse extreme
    eigenpairs. Every zero mode is excluded from the low band. No labels enter.
    """
    adjacency = adjacency_matrix(edges, n)
    degrees = np.asarray(adjacency.sum(1)).reshape(-1)
    inverse = np.zeros_like(degrees)
    inverse[degrees > 0] = degrees[degrees > 0] ** -.5
    operator = sp.diags(inverse) @ adjacency @ sp.diags(inverse)
    if n <= dense_limit:
        laplacian = torch.eye(n, dtype=torch.float64) - torch.from_numpy(operator.toarray())
        values, vectors = torch.linalg.eigh(laplacian)
        positive = torch.nonzero(values > 1e-6).flatten()
        if len(positive) < 2 * m:
            raise ValueError("Too few positive eigenvalues for disjoint low/high LapPE bands")
        indices = torch.cat((positive[:m], torch.arange(n - m, n)))
        return vectors[:, indices].float()
    laplacian = sp.eye(n, format="csr") - operator
    _, component = connected_components(adjacency, directed=False)
    zero_modes = int((np.bincount(component, weights=degrees) > 0).sum())
    count = min(n - m - 1, m + zero_modes + 2)
    if n - zero_modes < 2 * m or count < m:
        raise ValueError("Too few positive eigenvalues for disjoint LapPE bands")
    initial = np.random.default_rng(0).standard_normal(n)
    while True:
        low_values, low_vectors = eigsh(laplacian, k=count, which="SA", tol=1e-8, v0=initial)
        order = np.argsort(low_values)
        order = order[low_values[order] > 1e-6]
        if len(order) >= m:
            low = low_vectors[:, order[:m]]
            break
        next_count = min(n - m - 1, max(count + m, 2 * count))
        if next_count == count:
            raise ValueError("Unable to find enough positive low-frequency modes")
        count = next_count
    high_values, high_vectors = eigsh(laplacian, k=m, which="LA", tol=1e-8, v0=initial)
    high = high_vectors[:, np.argsort(high_values)]
    return torch.from_numpy(np.concatenate((low, high), axis=1).astype(np.float32))


def candidate_pool(edges: torch.Tensor, n: int, radius: int) -> torch.Tensor:
    """Return all other nodes reachable within radius hops, without sampling."""
    adjacency = adjacency_matrix(edges, n).astype(bool)
    power = adjacency.copy()
    reach = adjacency.copy()
    for _ in range(1, radius):
        power = power @ adjacency
        reach = reach.maximum(power)
    reach.setdiag(False)
    reach.eliminate_zeros()
    return torch.from_numpy(reach.toarray()).bool()
