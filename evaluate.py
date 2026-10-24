#!/usr/bin/env python3
"""Re-evaluate a final embedding using its saved arguments and dataset split."""
from __future__ import annotations

import json

import torch

from arguments import parse_args, parse_evaluation_args
from gcl.data import load_dataset
from gcl.evaluation import evaluate


def main() -> None:
    """Load a locally produced representation and rerun its fixed evaluation."""
    cli = parse_evaluation_args()
    cached = torch.load(cli.embedding, map_location="cpu", weights_only=False)
    args = parse_args(["--method", cached["arguments"]["method"], "--dataset", cached["dataset"]])
    vars(args).update(cached["arguments"])
    _, reference = load_dataset(cached["dataset"], cli.data_root, args.normalize_features)
    metrics, _ = evaluate(cached["embedding"], reference, args, cached["training_seed"])
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
