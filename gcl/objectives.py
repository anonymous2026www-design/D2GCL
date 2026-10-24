"""Two-view InfoNCE and the weighted inside-log Phase-2 objective."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def drop_features(x: torch.Tensor, probability: float) -> torch.Tensor:
    """Drop feature dimensions, using one mask shared by all nodes in a view."""
    return x if probability == 0 else x * (torch.rand(x.shape[1], device=x.device) > probability).to(x.dtype)


def drop_edges(edges: torch.Tensor, probability: float) -> torch.Tensor:
    """Independently retain the supplied graph edges, as in the GRACE baseline."""
    return edges if probability == 0 else edges[:, torch.rand(edges.shape[1], device=edges.device) > probability]


def two_view_loss(a: torch.Tensor, b: torch.Tensor, temperature: float,
                  chunk_size: int | None = None) -> torch.Tensor:
    """Symmetric InfoNCE with all within-view and cross-view negatives.

    Each anchor has its other-view counterpart as its sole positive. Only
    the anchor's own representation is excluded from its denominator.
    """
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    n = len(a)
    both = torch.cat((a, b))
    if chunk_size is not None:
        total = both.new_zeros(())
        for start in range(0, 2 * n, chunk_size):
            end = min(2 * n, start + chunk_size)

            def block(z: torch.Tensor, begin: int = start, stop: int = end) -> torch.Tensor:
                ids = torch.arange(begin, stop, device=z.device)
                logits = z[begin:stop] @ z.T / temperature
                logits[torch.arange(stop - begin, device=z.device), ids] = -torch.inf
                return F.cross_entropy(logits, (ids + n) % (2 * n), reduction="sum") / (2 * n)

            total = total + (checkpoint(block, both, use_reentrant=False, preserve_rng_state=False)
                             if torch.is_grad_enabled() and both.requires_grad else block(both))
        return total
    logits = both @ both.T / temperature
    logits = logits.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=a.device), -torch.inf)
    targets = (torch.arange(2 * n, device=a.device) + n) % (2 * n)
    return .5 * (F.cross_entropy(logits[:n], targets[:n]) + F.cross_entropy(logits[n:], targets[n:]))


@torch.no_grad()
def mine_positive_weights(embedding: torch.Tensor, proportion: float, temperature: float,
                          pool: torch.Tensor | None, chunk_size: int | None = None) -> torch.Tensor:
    """Select directed top-q nonself candidates and compute row-softmax weights.

    This follows the draft's top-fraction rule; reciprocal selection is not
    required. Empty candidate rows receive zero mass and never self-positives.
    Selection and confidence are recomputed under no_grad at every P2 epoch.
    """
    n = len(embedding)
    normalized = F.normalize(embedding, dim=-1)
    if chunk_size is not None:
        all_rows, all_columns, all_values = [], [], []
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            ids = torch.arange(start, end, device=embedding.device)
            local = torch.arange(end - start, device=embedding.device)
            scores = normalized[start:end] @ normalized.T
            scores[local, ids] = -torch.inf
            if pool is None:
                counts = torch.full((end - start,), n - 1, device=embedding.device, dtype=torch.long)
            else:
                allowed = pool[start:end].clone()
                allowed[local, ids] = False
                scores.masked_fill_(~allowed, -torch.inf)
                counts = allowed.sum(1)
            sizes = (counts.double() * proportion).long().clamp_min(1)
            sizes = torch.where(counts > 0, sizes, torch.zeros_like(sizes))
            largest = int(sizes.max())
            if not largest:
                continue
            values, columns = scores.topk(largest, dim=1)
            keep = torch.arange(largest, device=embedding.device)[None, :] < sizes[:, None]
            values = (values / temperature).masked_fill(~keep, -torch.inf)
            values[sizes == 0] = 0
            confidence = values.softmax(1).masked_fill(~keep, 0)
            all_rows.append(ids[:, None].expand_as(columns)[keep])
            all_columns.append(columns[keep])
            all_values.append(confidence[keep])
        if not all_rows:
            indices = torch.empty((2, 0), device=embedding.device, dtype=torch.long)
            return torch.sparse_coo_tensor(indices, embedding.new_empty(0), (n, n)).coalesce()
        indices = torch.stack((torch.cat(all_rows), torch.cat(all_columns)))
        return torch.sparse_coo_tensor(indices, torch.cat(all_values), (n, n)).coalesce()
    similarity = normalized @ normalized.T
    allowed = ~torch.eye(n, dtype=torch.bool, device=embedding.device)
    if pool is not None:
        allowed &= pool
    counts = allowed.sum(1)
    sizes = (counts.double() * proportion).long().clamp_min(1)
    sizes = torch.where(counts > 0, sizes, torch.zeros_like(sizes))
    largest = int(sizes.max())
    selected = torch.zeros_like(allowed)
    if largest:
        ranked = similarity.masked_fill(~allowed, -torch.inf).topk(largest, dim=1).indices
        keep = torch.arange(largest, device=embedding.device)[None, :] < sizes[:, None]
        selected.scatter_(1, ranked, keep)
    scores = (similarity / temperature).masked_fill(~selected, -torch.inf)
    scores[~selected.any(1)] = 0
    return scores.softmax(1).masked_fill(~selected, 0)


def phase2_loss(hidden: torch.Tensor, weights: torch.Tensor, temperature: float,
                chunk_size: int | None = None) -> torch.Tensor:
    """Compute -log(sum_j Pi_ij exp(sim_ij/tau)/sum_{j!=i} exp(sim_ij/tau)).

    Weights are frozen during an optimizer step. This is a logarithm of a
    weighted positive sum, not the SupCon average of positive log losses.
    Anchors with no local candidates are omitted from this phase's average.
    """
    if weights.is_sparse or chunk_size is not None:
        return sparse_phase2_loss(hidden, weights, temperature, chunk_size or len(hidden))
    weights = weights.detach()
    valid = weights.sum(1) > 0
    if not bool(valid.any()):
        raise ValueError("No anchor has a positive candidate")
    z = F.normalize(hidden, dim=-1)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(len(z), device=z.device, dtype=torch.bool), -torch.inf)
    logits, selected_weights = logits[valid], weights[valid]
    log_denominator = torch.logsumexp(logits, dim=1)
    log_numerator = torch.logsumexp(logits + selected_weights.log(), dim=1)
    return (log_denominator - log_numerator).mean()


def sparse_phase2_loss(hidden: torch.Tensor, weights: torch.Tensor, temperature: float,
                       chunk_size: int) -> torch.Tensor:
    """Exact all-negative P2 loss with sparse positives and checkpointed row blocks."""
    weights = weights.detach().to_sparse().coalesce()
    indices, confidence = weights.indices(), weights.values()
    keep = (confidence > 0) & (indices[0] != indices[1])
    rows, columns, confidence = indices[0, keep], indices[1, keep], confidence[keep]
    n = len(hidden)
    valid = torch.bincount(rows, minlength=n) > 0
    num_valid = int(valid.sum())
    if not num_valid:
        raise ValueError("No anchor has a positive candidate")
    normalized = F.normalize(hidden, dim=-1)
    total = hidden.new_zeros(())
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        if not bool(valid[start:end].any()):
            continue
        first, last = int(torch.searchsorted(rows, start)), int(torch.searchsorted(rows, end))
        local_rows, cols = rows[first:last] - start, columns[first:last]
        values = confidence[first:last]
        eligible = valid[start:end]

        def block(z: torch.Tensor, begin: int = start, stop: int = end,
                  rr: torch.Tensor = local_rows, cc: torch.Tensor = cols,
                  ww: torch.Tensor = values, anchors: torch.Tensor = eligible) -> torch.Tensor:
            logits = z[begin:stop] @ z.T / temperature
            logits[torch.arange(stop - begin, device=z.device), torch.arange(begin, stop, device=z.device)] = -torch.inf
            selected = logits[rr, cc] + ww.log()
            with torch.no_grad():
                maximum = z.new_full((stop - begin,), -torch.inf)
                maximum.scatter_reduce_(0, rr, selected.detach(), reduce="amax", include_self=True)
            numerator = z.new_zeros(stop - begin).index_add(0, rr, torch.exp(selected - maximum[rr]))
            log_numerator = maximum[anchors] + numerator[anchors].log()
            return (torch.logsumexp(logits[anchors], dim=1) - log_numerator).sum() / num_valid

        total = total + (checkpoint(block, normalized, use_reentrant=False, preserve_rng_state=False)
                         if torch.is_grad_enabled() and normalized.requires_grad else block(normalized))
    return total
