# Anonymous D2-GCL code

A compact implementation of D2-GCL and the GRACE baseline on **Cora,
CiteSeer, PubMed, Cornell and Texas**. The full training method is retained; additional
baselines, dataset adapters, parameter searches and experiment schedulers
are not included.

## Installation

Use Python 3.10 or newer and install an appropriate PyTorch build for your
CPU or CUDA environment, followed by:

```bash
pip install -r requirements.txt
```

Run the commands below from this directory. PyG downloads each supported
dataset on first use into `data/`. No private files or external source
directories are required. `--device cpu` selects CPU execution.

## Run a fixed configuration

**All parameters and fixed recipes are together in `arguments.py`.**
There is one recipe per method and dataset; it is shared across training
seeds. Command-line values override the recipe. No parameter search or
label-based representation selection is performed by these commands.

```bash
python train.py --method d2gcl --dataset Cora
python train.py --method grace --dataset Cora

python train.py --method d2gcl --dataset CiteSeer
python train.py --method d2gcl --dataset PubMed
python train.py --method grace --dataset PubMed
python train.py --method d2gcl --dataset Cornell
python train.py --method d2gcl --dataset Texas

# Inspect the actual settings without training.
python train.py --method d2gcl --dataset Cora --print-config

# Override one setting explicitly.
python train.py --method d2gcl --dataset Cora --lr 0.001 --q 0.5

# Run the same recipe on three training seeds and aggregate the results.
python train.py --method d2gcl --dataset Cora --seeds 0 1 2
python train.py --method grace --dataset Cora --seeds 0 1 2
```

The fixed D2-GCL examples are:

| Dataset | Learning rate | LapPE m per band | q | Candidate rule | r | K |
|---|---:|---:|---:|---|---:|---:|
| Cora | 0.001 | 4 | 0.50 | within r hops | 2 | 10 |
| CiteSeer | 0.001 | 4 | 0.30 | within r hops | 1 | 5 |
| PubMed | 0.005 | 8 | 0.30 | within r hops | 3 | 2 |
| Cornell | 0.005 | 16 | 0.03 | all other nodes | unused | 3 |
| Texas | 0.01 | 8 | 0.03 | all other nodes | unused | 3 |

These commands use one fixed preset per method and dataset. No parameter
search, seed search or result-based selection code is included.

PubMed uses `lr=0.005`, `m=8`, `q=0.30`, `r=3`, `rho=0.5`,
`tau=1`, `dropout=0.25`, `drop_feature=0.25`, `K=2` and
`tau_capp=2`. All three training seeds share the same preset.
Default seeds are 132 for Cora, 24 for CiteSeer, 173803 for PubMed,
2045 for Cornell and 2028 for Texas. Override them with `--seed` or
`--seeds`. See `VERIFICATION.md` for recipe provenance and measured results.

## D2-GCL implementation

- A two-layer feature-only MLP with BatchNorm, PReLU and dropout is trained
  for 100 epochs by default; `rho=0.5` gives 50 epochs per phase.
- Phase 1 uses two feature-dropped views, a projection head and symmetric
  InfoNCE with within-view and cross-view negatives.
- Phase 2 uses the original features and the current encoder output **H**,
  bypassing the projection head. Semantic positives and their confidence
  are refreshed **every epoch**. LapPE positives are fixed graph-only inputs.
- Each criterion selects the directed top-q fraction of the candidate pool
  (floor, with a minimum of one if the pool is nonempty). Self pairs are
  excluded. No reciprocal-selection condition or degree cap is added.
- Semantic and structural row-softmax confidences are averaged with equal
  weight. Mining and confidence are detached during an optimizer step.
  The loss is the **negative logarithm of a weighted positive sum**, not
  the SupCon average of positive log losses. Its denominator excludes self.
- The low LapPE band uses the smallest **positive** eigenvalues, excluding
  all zero modes; the high band uses the largest eigenvalues. There is no
  label-dependent preprocessing.
- Final positives are refreshed from the final encoder output. Weighted
  CAPP then combines hops 0 through K on the self-loop-free normalized graph,
  using retention, positive agreement and a component-wise stationary
  reference. Propagation never participates in contrastive optimization.
- Empty local candidate rows are excluded from the Phase 2 loss average.
  Their CAPP positive-agreement score falls back to self-retention. Isolated
  nodes have zero higher-hop features, with hop 0 retained as an option.

Unless overridden by a dataset recipe, the semantic and structural confidence
temperatures are 0.5, and the contrastive temperature, CAPP temperature
and three CAPP coefficients are 1. All are visible in `arguments.py` and can be overridden explicitly.

For PubMed, LapPE uses a sparse eigensolver, mined positive weights use
sparse storage, and contrastive losses are computed in checkpointed row
blocks. These are full-batch calculations: every node and every denominator
candidate is retained in each update. `--chunk-size` controls a computation
buffer, not a sampled training batch. Small graphs retain the original dense
calculation by default. Both paths have been compared for losses, gradients,
positive weights and complete training outputs.

## Evaluation and outputs

The dataset's public Planetoid split is used for Cora/CiteSeer/PubMed. Cornell and Texas use
the first predefined Geom-GCN split distributed by PyG. Splits are identical
between methods and never regenerated from a training seed.

Representation training receives only features and edges. Labels are used
only by the separate frozen-representation evaluation:

- Node classification: a linear classifier, 100 epochs, Adam learning rate 0.01,
  weight decay 0.0001. The fixed final classifier epoch is reported;
  validation accuracy is diagnostic, not a checkpoint-selection criterion.
- Clustering: row-L2-normalized final embeddings, K equal to the number of
  classes, ten separate K-means++ runs with initialization seeds 0 through 9,
  `n_init=1` each, arithmetic NMI. All ten runs are retained.
- With multiple training seeds, average each seed's ten NMI scores first,
  then report mean and population standard deviation across training seeds.
  Accuracy and NMI are saved as fractions in [0,1].

For D2-GCL both tasks use the **final post-CAPP representation**. GRACE uses
the final encoder output. The last encoder epoch is used without label-based
checkpoint selection. The package performs no hyperparameter search; the
fixed recipes are loaded from `arguments.py`. Recipe provenance and results
are recorded in `VERIFICATION.md`.

Outputs are under `results/<method>/<dataset>/`:

```text
arguments.json
per_seed.csv
summary.csv
seed<seed>/
  training_loss.csv
  metrics.json
  checkpoint.pt
  embedding.pt
```

`metrics.json` retains all ten clustering scores. Saved embeddings also
include the unpropagated H0 for D2-GCL. To repeat evaluation:

```bash
python evaluate.py --embedding results/d2gcl/Cora/seed132/embedding.pt
```

## Scope

This release provides training, linear evaluation and NMI for the five
listed datasets. Other baselines, datasets, large-graph/mini-batch code,
ablation sweeps, statistical comparisons and paper plotting utilities are
outside this compact package. See `THIRD_PARTY.md` for baseline attribution.
