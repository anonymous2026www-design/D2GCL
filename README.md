 # Anonymous WWW 2027 Submission

This repository contains the PyTorch implementation and experiment scripts for D2-GCL, as described in our WWW 2027 submission.

The repository also includes the GRACE baseline and supports experiments on Cora, CiteSeer, PubMed, Cornell, and Texas.

## Requirements

Python 3.10 or later is required.

### Linux/macOS

Create and activate a clean virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

Create and activate a clean virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install a PyTorch build compatible with your CPU or CUDA environment, and then install the remaining dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To verify that PyTorch can access the GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Quick Start

Run D2-GCL on Cora using the first CUDA GPU:

```bash
python train.py --method d2gcl --dataset Cora --device cuda:0
```

Run the GRACE baseline on Cora:

```bash
python train.py --method grace --dataset Cora --device cuda:0
```

For CPU execution:

```bash
python train.py --method d2gcl --dataset Cora --device cpu
```

Datasets are downloaded automatically on first use and stored under `data/`.

## Supported Datasets

The following datasets are supported:

- Cora
- CiteSeer
- PubMed
- Cornell
- Texas

Cora, CiteSeer, and PubMed use the public Planetoid splits. Cornell and Texas use the first predefined Geom-GCN split distributed by PyTorch Geometric.

## Running Experiments

### D2-GCL

```bash
python train.py --method d2gcl --dataset Cora --device cuda:0
python train.py --method d2gcl --dataset CiteSeer --device cuda:0
python train.py --method d2gcl --dataset PubMed --device cuda:0
python train.py --method d2gcl --dataset Cornell --device cuda:0
python train.py --method d2gcl --dataset Texas --device cuda:0
```

### GRACE baseline

```bash
python train.py --method grace --dataset Cora --device cuda:0
python train.py --method grace --dataset CiteSeer --device cuda:0
python train.py --method grace --dataset PubMed --device cuda:0
python train.py --method grace --dataset Cornell --device cuda:0
python train.py --method grace --dataset Texas --device cuda:0
```

## Configuration

All dataset-specific configurations and fixed experimental recipes are defined in `arguments.py`.

To inspect a resolved configuration without starting training:

```bash
python train.py --method d2gcl --dataset Cora --print-config
```

Command-line arguments can be used to override individual settings:

```bash
python train.py \
  --method d2gcl \
  --dataset Cora \
  --device cuda:0 \
  --lr 0.001 \
  --q 0.5
```

On Windows PowerShell, the same command can be written on one line:

```powershell
python train.py --method d2gcl --dataset Cora --device cuda:0 --lr 0.001 --q 0.5
```

## Multiple Training Seeds

To run the same fixed configuration with multiple training seeds:

```bash
python train.py --method d2gcl --dataset Cora --device cuda:0 --seeds 0 1 2
```

For GRACE:

```bash
python train.py --method grace --dataset Cora --device cuda:0 --seeds 0 1 2
```

The per-seed results and their aggregate mean and population standard deviation are saved automatically.

## Evaluation

The learned representations are evaluated using:

- Node classification with a frozen-representation linear classifier.
- Node clustering with ten fixed K-means initializations.
- Test accuracy for node classification.
- Normalized Mutual Information (NMI) for node clustering.

Representation learning receives only node features and graph edges. Labels and evaluation masks are used only during the separate evaluation stage.

D2-GCL is evaluated using its final post-CAPP representation. GRACE is evaluated using its final encoder representation.

## Output Files

Experiment outputs are stored under:

```text
results/<method>/<dataset>/
```

The output structure is:

```text
results/
  <method>/
    <dataset>/
      arguments.json
      per_seed.csv
      summary.csv
      seed<seed>/
        training_loss.csv
        metrics.json
        checkpoint.pt
        embedding.pt
```

The main files are:

- `arguments.json`: resolved experiment configuration.
- `training_loss.csv`: training loss for every epoch.
- `metrics.json`: classification and clustering results.
- `checkpoint.pt`: trained model and linear probe parameters.
- `embedding.pt`: final node representations.
- `per_seed.csv`: results for individual training seeds.
- `summary.csv`: aggregate results across training seeds.

## Re-evaluating Saved Embeddings

A saved representation can be evaluated again without retraining:

```bash
python evaluate.py --embedding results/d2gcl/Cora/seed132/embedding.pt
```

## Repository Structure

```text
.
├── arguments.py
├── evaluate.py
├── train.py
├── requirements.txt
├── README.md
├── VERIFICATION.md
├── THIRD_PARTY.md
└── gcl/
    ├── __init__.py
    ├── data.py
    ├── evaluation.py
    ├── models.py
    ├── objectives.py
    ├── propagation.py
    └── structure.py
```

The main components are:

- `train.py`: representation training and downstream evaluation.
- `evaluate.py`: re-evaluation of saved representations.
- `arguments.py`: command-line arguments and fixed experiment configurations.
- `gcl/data.py`: dataset loading and evaluation splits.
- `gcl/models.py`: encoder and projection models.
- `gcl/objectives.py`: contrastive objectives and positive mining.
- `gcl/propagation.py`: confidence-aware propagation.
- `gcl/structure.py`: structural encoding and candidate construction.
- `gcl/evaluation.py`: node classification and clustering evaluation.

## Reproducibility

The repository uses one fixed configuration for each method and dataset. No hyperparameter search, seed search, or label-based representation selection is performed during the provided runs.

Default training seeds and complete configurations are defined in `arguments.py`. Additional verification information and measured results are provided in `VERIFICATION.md`.

## Notes

- Run commands from the repository directory.
- Relative data and output paths are resolved from the repository root.
- An internet connection is required when a dataset is downloaded for the first time.
- The `data/`, `results/`, virtual environment, cache, and local configuration files are excluded from Git tracking.
- No private datasets, pretrained models, or external source directories are required.
- Third-party baseline attribution is provided in `THIRD_PARTY.md`.
