"""All command-line arguments and one fixed recipe per method/dataset."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

DATASETS = ("Cora", "CiteSeer", "PubMed", "Cornell", "Texas")
METHODS = ("d2gcl", "grace")

# Fixed example recipes; no search or result-dependent selection is performed.
PRESETS = {
    "d2gcl": {
        "Cora": dict(lr=.001, m=4, q=.50, r=2, K=10, candidate_pool="rhop", seed=132),
        "CiteSeer": dict(lr=.001, m=4, q=.30, r=1, K=5, candidate_pool="rhop", seed=24),
        "Cornell": dict(lr=.005, m=16, q=.03, r=2, K=3, candidate_pool="global", seed=2045),
        "PubMed": dict(lr=0.005, m=8, q=0.3, r=3, K=2, rho=0.5, tau_capp=2.0, tau=1.0, dropout=0.25, drop_feature=0.25, candidate_pool="rhop", seed=173803),
        "Texas": dict(lr=.01, m=8, q=.03, r=2, K=3, candidate_pool="global", seed=2028),
    },
    "grace": {
        "Cora": dict(lr=.001, weight_decay=.00001, tau=.4, epochs=200,
                     drop_feature=.3, drop_edge=.2, normalize_features=True, seed=132),
        "CiteSeer": dict(lr=.001, weight_decay=.00001, tau=.9, epochs=200,
                         drop_feature=.3, drop_edge=.2, normalize_features=True, seed=24),
        "Cornell": dict(lr=.002, weight_decay=.0001, tau=.5, epochs=100,
                        drop_feature=.25, drop_edge=.25, normalize_features=False, seed=2045),
        "PubMed": dict(lr=.002, weight_decay=.00001, tau=.5, epochs=100,
                       drop_feature=.25, drop_edge=.25, normalize_features=False, seed=173803),
        "Texas": dict(lr=.002, weight_decay=.00001, tau=.5, epochs=100,
                      drop_feature=.25, drop_edge=.25, normalize_features=False, seed=2028),
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Resolve the fixed recipe first; explicit CLI values always override it."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--method", choices=METHODS, default="d2gcl")
    selector.add_argument("--dataset", choices=DATASETS, default="Cora")
    selection, _ = selector.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="D2-GCL and GRACE on five graph datasets",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    general = parser.add_argument_group("Run")
    general.add_argument("--method", choices=METHODS, default=selection.method)
    general.add_argument("--dataset", choices=DATASETS, default=selection.dataset)
    general.add_argument("--seed", type=int, default=0, help="One training seed")
    general.add_argument("--seeds", type=int, nargs="+", help="Optional list of training seeds")
    general.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    general.add_argument("--data-root", type=Path, default=Path("data"))
    general.add_argument("--output-dir", type=Path, default=Path("results"))
    general.add_argument("--threads", type=int, default=1)
    general.add_argument("--log-every", type=int, default=10)
    general.add_argument("--chunk-size", type=int, default=1024, help="Rows per exact large-graph loss block")
    general.add_argument("--dense-node-limit", type=int, default=4096,
                         help="Use blocked losses and sparse positives above this node count")
    general.add_argument("--print-config", action="store_true", help="Print the resolved recipe and exit")

    training = parser.add_argument_group("Architecture and optimization")
    training.add_argument("--hidden-dim", type=int, default=256)
    training.add_argument("--out-dim", type=int, default=256)
    training.add_argument("--projection-dim", type=int, default=256)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--lr", type=float, default=.001)
    training.add_argument("--weight-decay", type=float, default=.0001)
    training.add_argument("--dropout", type=float, default=.5)
    training.add_argument("--drop-feature", type=float, default=.25)
    training.add_argument("--drop-edge", type=float, default=.25, help="GRACE graph augmentation")
    training.add_argument("--tau", type=float, default=1.)
    training.add_argument("--normalize-features", action=argparse.BooleanOptionalAction, default=False)

    mining = parser.add_argument_group("D2-GCL: phase allocation and positive mining")
    mining.add_argument("--rho", type=float, default=.5)
    mining.add_argument("--m", type=int, default=4, help="LapPE eigenvectors at EACH end of the spectrum")
    mining.add_argument("--q", type=float, default=.5)
    mining.add_argument("--candidate-pool", choices=("rhop", "global"), default="rhop")
    mining.add_argument("--r", type=int, default=2)
    mining.add_argument("--tau-sem", type=float, default=.5)
    mining.add_argument("--tau-str", type=float, default=.5)

    propagation = parser.add_argument_group("D2-GCL: weighted CAPP")
    propagation.add_argument("--K", type=int, default=10, help="Maximum propagation hop (hop zero is included)")
    propagation.add_argument("--tau-capp", type=float, default=1.)
    propagation.add_argument("--alpha-ret", type=float, default=1.)
    propagation.add_argument("--alpha-pos", type=float, default=1.)
    propagation.add_argument("--alpha-os", type=float, default=1.)

    evaluation = parser.add_argument_group("Frozen-representation evaluation")
    evaluation.add_argument("--probe-epochs", type=int, default=100)
    evaluation.add_argument("--probe-lr", type=float, default=.01)
    evaluation.add_argument("--probe-weight-decay", type=float, default=.0001)
    evaluation.add_argument("--clustering", action=argparse.BooleanOptionalAction, default=True)
    evaluation.add_argument("--kmeans-max-iter", type=int, default=300)
    evaluation.add_argument("--kmeans-tol", type=float, default=.0001)
    evaluation.add_argument("--save-embeddings", action=argparse.BooleanOptionalAction, default=True)

    # Fixed evaluation protocol values also live here, not inside metric code.
    parser.set_defaults(kmeans_seeds=list(range(10)), probe_seed_offset=100000)
    parser.set_defaults(**PRESETS[selection.method][selection.dataset])
    args = parser.parse_args(argv)
    floating = (args.lr, args.probe_lr, args.tau, args.tau_sem, args.tau_str, args.tau_capp,
                args.rho, args.q, args.dropout, args.drop_feature, args.drop_edge,
                args.weight_decay, args.probe_weight_decay, args.alpha_ret, args.alpha_pos, args.alpha_os,
                args.kmeans_tol)
    if not all(math.isfinite(value) for value in floating):
        parser.error("Floating-point arguments must be finite")
    positive = (args.hidden_dim, args.out_dim, args.projection_dim, args.epochs,
                args.probe_epochs, args.m, args.threads, args.log_every, args.kmeans_max_iter,
                args.chunk_size, args.dense_node_limit)
    if min(positive) < 1:
        parser.error("Dimensions, epoch counts, m, thread count and log interval must be positive")
    if not (0 < args.rho < 1 and 0 < args.q <= 1):
        parser.error("rho must lie in (0,1) and q in (0,1]")
    if min(args.lr, args.probe_lr, args.tau, args.tau_sem, args.tau_str, args.tau_capp) <= 0:
        parser.error("Learning rates and temperatures must be positive")
    if min(args.weight_decay, args.probe_weight_decay, args.K) < 0 or args.r < 1:
        parser.error("Weight decay and K must be nonnegative; r must be positive")
    if min(args.alpha_ret, args.alpha_pos, args.alpha_os) < 0:
        parser.error("CAPP coefficients must be nonnegative")
    if args.kmeans_tol < 0:
        parser.error("K-means tolerance must be nonnegative")
    if any(not 0 <= p < 1 for p in (args.dropout, args.drop_feature, args.drop_edge)):
        parser.error("Drop rates must lie in [0,1)")
    if args.method == "d2gcl" and not 1 <= int(args.epochs * args.rho) < args.epochs:
        parser.error("D2-GCL needs at least one epoch in each phase")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("Training seeds must be distinct")
    return args


def parse_evaluation_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Arguments for re-evaluating a saved final representation."""
    parser = argparse.ArgumentParser(description="Evaluate a saved frozen node representation")
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args(argv)
