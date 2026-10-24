# Anonymous WWW 2027 Submission

This repository contains the PyTorch implementation and scripts for D2-GCL, as described in our WWW 2027 submission.

## Requirements

Create and activate a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Quick Start

Run D2-GCL on Cora:

```bash
python train.py --method d2gcl --dataset Cora --device cuda:0
```

Run the GRACE baseline:

```bash
python train.py --method grace --dataset Cora --device cuda:0
```
