"""Linear classification and ten fixed K-means initializations."""
from __future__ import annotations

from argparse import Namespace
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits


def evaluate(embedding: torch.Tensor, reference: dict[str, Any], args: Namespace,
             seed: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Evaluate frozen final embeddings; representation training uses no labels.

    The classifier uses the fixed final probe epoch, matching the existing
    result protocol. Validation accuracy is reported, not used for selection.
    K-means runs exactly ten times, with initialization seeds0 through9.
    """
    torch.set_num_threads(args.threads)
    embedding = embedding.detach().float().cpu()
    labels = reference["y"].long().cpu()
    if embedding.ndim != 2 or embedding.shape[0] != len(labels) or not torch.isfinite(embedding).all():
        raise ValueError("Invalid frozen embeddings")
    torch.manual_seed(seed + args.probe_seed_offset)
    classifier = torch.nn.Linear(embedding.shape[1], int(labels.max()) + 1)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.probe_lr,
                                 weight_decay=args.probe_weight_decay)
    train = reference["train_mask"]
    for _ in range(args.probe_epochs):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(classifier(embedding[train]), labels[train]).backward()
        optimizer.step()
    with torch.no_grad():
        predictions = classifier(embedding).argmax(-1)
    result: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        mask = reference[split + "_mask"]
        result[split + "_accuracy"] = float((predictions[mask] == labels[mask]).float().mean())
    if args.clustering:
        array = embedding.double().numpy()
        array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
        target = labels.numpy()
        classes = len(np.unique(target))
        scores = []
        with threadpool_limits(limits=args.threads):
            for initialization in args.kmeans_seeds:
                clusters = KMeans(n_clusters=classes, init="k-means++", n_init=1,
                                  max_iter=args.kmeans_max_iter, tol=args.kmeans_tol,
                                  algorithm="lloyd", random_state=initialization).fit_predict(array)
                scores.append(float(normalized_mutual_info_score(target, clusters, average_method="arithmetic")))
        result.update(nmi=float(np.mean(scores)), nmi_initialization_std=float(np.std(scores, ddof=0)),
                      nmi_initializations=scores)
    state = {k: v.detach().cpu() for k, v in classifier.state_dict().items()}
    return result, state
