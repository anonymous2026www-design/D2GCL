#!/usr/bin/env python3
"""Train D2-GCL or GRACE with a fixed recipe and frozen-representation evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from arguments import parse_args
from gcl.data import load_dataset
from gcl.evaluation import evaluate
from gcl.models import RepresentationModel
from gcl.objectives import drop_features, drop_edges, mine_positive_weights, phase2_loss, two_view_loss
from gcl.propagation import capp
from gcl.structure import candidate_pool, laplacian_encoding


def write_json(path: Path, value: Any) -> None:
    """Write finite metadata with portable path values."""
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a small result table without an additional dataframe dependency."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    """Set Python, NumPy and Torch RNGs before initializing each model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one(args: Any, graph: dict[str, torch.Tensor], seed: int,
              positional: torch.Tensor | None, pool: torch.Tensor | None,
              device: torch.device) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    """Train only on graph/features; this function accepts no labels or masks."""
    seed_everything(seed)
    x, edges = graph["x"].to(device), graph["edge_index"].to(device)
    model = RepresentationModel(args.method, x.shape[1], args.hidden_dim, args.out_dim,
                                args.projection_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = int(args.epochs * args.rho)
    chunk_size = args.chunk_size if len(x) > args.dense_node_limit else None
    pool = pool.to(device) if pool is not None else None
    structural = None
    if args.method == "d2gcl":
        structural = mine_positive_weights(positional.to(device), args.q, args.tau_str, pool, chunk_size)
    losses = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        if args.method == "grace" or epoch <= warmup:
            a, b = drop_features(x, args.drop_feature), drop_features(x, args.drop_feature)
            ea = drop_edges(edges, args.drop_edge) if args.method == "grace" else edges
            eb = drop_edges(edges, args.drop_edge) if args.method == "grace" else edges
            loss = two_view_loss(model.project(model.encode(a, ea)), model.project(model.encode(b, eb)), args.tau, chunk_size)
            phase = 1
        else:
            hidden = model.encode(x, edges)
            # Latest draft: refresh semantic positives from CURRENT encoder H
            # every epoch. P2 bypasses the projector and excludes self pairs.
            semantic = mine_positive_weights(hidden, args.q, args.tau_sem, pool, chunk_size)
            loss = phase2_loss(hidden, .5 * (semantic + structural), args.tau, chunk_size)
            phase = 2
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        losses.append({"epoch": epoch, "phase": phase, "loss": float(loss.detach())})
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"{args.method} {args.dataset} seed={seed} epoch={epoch}/{args.epochs} phase={phase} loss={float(loss.detach()):.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        hidden = model.encode(x, edges)
        diagnostics: dict[str, Any] = {"H0": hidden.cpu()}
        if args.method == "d2gcl":
            semantic = mine_positive_weights(hidden, args.q, args.tau_sem, pool, chunk_size)
            embedding, weights = capp(hidden, edges, .5 * (semantic + structural), args)
            diagnostics["hop_weights"] = weights.cpu()
        else:
            embedding = hidden
    diagnostics["model_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return embedding.detach().cpu(), diagnostics, losses


def main() -> None:
    """Run one fixed configuration for one or more explicit training seeds."""
    args = parse_args()
    if args.print_config:
        print(json.dumps(vars(args), indent=2, default=str))
        return
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    graph, reference = load_dataset(args.dataset, args.data_root, args.normalize_features)
    folder = args.output_dir / args.method / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "arguments.json", vars(args))
    positional, pool = None, None
    if args.method == "d2gcl":
        print("Preparing graph-only LapPE and candidate pools", flush=True)
        positional = laplacian_encoding(graph["edge_index"], len(graph["x"]), args.m, args.dense_node_limit)
        if args.candidate_pool == "rhop":
            pool = candidate_pool(graph["edge_index"], len(graph["x"]), args.r)
    rows = []
    for seed in args.seeds if args.seeds is not None else [args.seed]:
        directory = folder / f"seed{seed}"
        directory.mkdir(exist_ok=True)
        embedding, diagnostics, losses = train_one(args, graph, seed, positional, pool, device)
        metrics, probe_state = evaluate(embedding, reference, args, seed)
        write_csv(directory / "training_loss.csv", losses)
        write_json(directory / "metrics.json", metrics)
        torch.save({"model_state_dict": diagnostics.pop("model_state_dict"), "probe_state_dict": probe_state,
                    "arguments": vars(args), "training_seed": seed}, directory / "checkpoint.pt")
        if args.save_embeddings:
            torch.save({"embedding": embedding, "dataset": args.dataset, "training_seed": seed,
                        "arguments": vars(args), "stage": "post-CAPP" if args.method == "d2gcl" else "encoder",
                        **diagnostics}, directory / "embedding.pt")
        row = {"dataset": args.dataset, "method": args.method, "training_seed": seed,
               **{k: v for k, v in metrics.items() if not isinstance(v, list)}}
        rows.append(row)
        print(f"Test accuracy: {100 * metrics['test_accuracy']:.2f}%"
              + (f"; NMI: {100 * metrics['nmi']:.2f}%" if args.clustering else ""), flush=True)
    write_csv(folder / "per_seed.csv", rows)
    summary = {"dataset": args.dataset, "method": args.method, "num_training_seeds": len(rows)}
    for metric in ("test_accuracy", "nmi"):
        if metric in rows[0]:
            values = [r[metric] for r in rows]
            summary.update({metric + "_mean": float(np.mean(values)), metric + "_std": float(np.std(values, ddof=0))})
    write_csv(folder / "summary.csv", [summary])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
