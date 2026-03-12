# Exhaustive Circuit Mapping of a Single-Cell Foundation Model

Code and results for the paper:

> **Exhaustive Circuit Mapping of a Single-Cell Foundation Model Reveals Massive Redundancy, Heavy-Tailed Hub Architecture, and Layer-Dependent Differentiation Control**
>
> Ihor Kendiukhov
>
> Department of Computer Science, University of Tubingen, Germany

## Overview

This repository contains the analysis code and experimental results for three experiments that address systematic limitations in prior mechanistic interpretability work on single-cell foundation models:

1. **Exhaustive Feature Tracing** — Traces all 4,065 active sparse autoencoder (SAE) features at layer 5 of Geneformer V2-316M, yielding 1,393,850 significant downstream edges and revealing heavy-tailed hub architecture with systematic annotation bias.

2. **Higher-Order Combinatorial Ablation** — Extends pairwise ablation to three-way feature triplets (8 triplets, 7 conditions each), demonstrating that redundancy deepens monotonically with interaction order (three-way ratio 0.59 vs. pairwise 0.74) with zero synergy.

3. **Trajectory-Guided Feature Steering** — Causally tests 14 differentiation-associated switch features, establishing that late-layer features (L17) universally push cell states toward maturity while early/mid-layer features push away.

## Repository Structure

```
sae-biological-map/
├── src/
│   ├── sae_model.py                    # TopK sparse autoencoder model (d=1152, 4x expansion, k=32)
│   ├── exhaustive_feature_tracing.py   # Experiment 1: exhaustive L5 circuit tracing
│   ├── higher_order_ablation.py        # Experiment 2: three-way combinatorial ablation
│   └── trajectory_steering.py          # Experiment 3: causal trajectory steering
├── results/
│   ├── exhaustive_tracing/
│   │   └── exhaustive_summary.json     # Summary statistics (1.39M edges, hub distribution)
│   ├── higher_order_ablation/
│   │   ├── summary.json                # Aggregate ablation results
│   │   └── triplet_*.json              # Per-triplet detailed results (8 files)
│   └── trajectory_steering/
│       ├── summary.json                # Aggregate steering results
│       ├── steering_F*_L*.json         # Per-feature steering results (14 files)
│       └── state_signatures.npz        # Early/late pseudotime gene signatures
├── paper/
│   ├── manuscript.tex                  # LaTeX source
│   ├── references.bib                  # Bibliography (63 entries)
│   └── figures/                        # Figures 1-6
├── requirements.txt
├── LICENSE
└── README.md
```

## Prerequisites

### Data

The following external datasets are required to reproduce the experiments:

- **K562 CRISPRi perturbation data** (Replogle et al., 2022): [Figshare](https://plus.figshare.com/articles/dataset/Replogle_2022_K562_gwps/21452470)
- **Tabula Sapiens immune subset** (The Tabula Sapiens Consortium, 2022): [CZ CELLxGENE](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5)
- **Geneformer V2-316M** pretrained model (Theodoris et al., 2023): [HuggingFace](https://huggingface.co/ctheodoris/Geneformer)

### Upstream dependencies

These experiments build on trained SAE models and extracted activations from a companion study ([Kendiukhov, 2025](https://github.com/Biodyn-AI/bio-sae)). You will need:

- Trained SAE checkpoints (`sae_layer{N}.pt`) for each Geneformer layer
- Extracted residual-stream activations (`layer_{N}_activations.npy`)
- Circuit tracing results from prior causal patching (for Experiment 2)
- Trajectory dynamics results from prior pseudotime analysis (for Experiment 3)

## Installation

```bash
conda create -n sae-bio python=3.10
conda activate sae-bio
pip install -r requirements.txt
```

## Usage

Configure data paths via environment variables or edit the path constants at the top of each script:

```bash
export SAE_DATA_ROOT="/path/to/phase1_k562"      # SAE models and activations
export SAE_DATA_PATH="/path/to/replogle_concat.h5ad"  # K562 CRISPRi data
```

### Experiment 1: Exhaustive Feature Tracing

```bash
python src/exhaustive_feature_tracing.py --n-cells 200 --source-layer 5
```

Traces all active features at layer 5 to downstream layers (L6, L11, L17). Outputs per-feature JSON files with resume support. Runtime: ~12 hours on Apple M2 Max.

### Experiment 2: Higher-Order Ablation

```bash
python src/higher_order_ablation.py --n-cells 200 --n-triplets 10
```

Performs single, pairwise, and three-way ablation for 8 biologically motivated feature triplets. Runtime: ~2 hours.

### Experiment 3: Trajectory Steering

```bash
python src/trajectory_steering.py --alphas 2.0,5.0 --n-cells 500
```

Amplifies 14 switch features in early-pseudotime immune cells and measures state shift toward maturity. Runtime: ~1 minute.

## Key Results

| Experiment | Key Finding | Main Metric |
|---|---|---|
| Exhaustive tracing | 27x more edges than selective sampling; 40% of top-20 hubs unannotated | 1,393,850 edges from 4,065 features |
| Higher-order ablation | Redundancy deepens; zero synergy at all orders | Three-way ratio = 0.59 (vs. pairwise 0.74) |
| Trajectory steering | L17 universally pushes toward maturity; L0/L11 push away | L17 fraction positive = 1.00 |

## Compute Environment

All experiments were run on a MacBook Pro with Apple M2 Max (38-core GPU, 96 GB unified memory) using PyTorch 2.1 with MPS backend. Total compute: ~26.3 hours.

## Citation

If you use this code or results, please cite:

```bibtex
@article{kendiukhov2025exhaustive,
  title={Exhaustive Circuit Mapping of a Single-Cell Foundation Model Reveals Massive Redundancy, Heavy-Tailed Hub Architecture, and Layer-Dependent Differentiation Control},
  author={Kendiukhov, Ihor},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
