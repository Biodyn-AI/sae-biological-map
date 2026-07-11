# Exhaustive single-layer circuit mapping of a single-cell foundation model

Code and results for the paper:

> **Exhaustive single-layer circuit mapping of a single-cell foundation model reveals massive redundancy, heavy-tailed hub architecture, and layer-dependent steering directionality**
>
> Ihor Kendiukhov
>
> Department of Computer Science, University of Tübingen, Germany

## Overview

This repository contains the analysis code and experimental results for a mechanistic-interpretability study of Geneformer V2-316M, decomposed with layer-wise sparse autoencoders (SAEs). Three core experiments map and probe the model's feature-level circuit graph; a set of revision analyses (added during peer review) calibrate the edge-calling rule against an empirical null and stress-test every headline claim against explicit statistical or biological controls.

**Core experiments** (`src/`):

1. **Exhaustive feature tracing** — traces all 4,065 active SAE features at layer 5 by causal ablation to downstream layers {6, 11, 17}, yielding 1,393,850 significant edges under the original criterion (≈1.1M after false-discovery control; see below).
2. **Higher-order combinatorial ablation** — extends pairwise ablation to three-way feature triplets, quantifying how redundancy deepens with interaction order.
3. **Trajectory-guided feature steering** — amplifies differentiation-associated "switch" features and measures the induced shift in the model's output.

**Revision analyses** (`src/revision/`) — the analyses that calibrate and bound the core results:

- **Edge-calling calibration** (`p1a`, `p1b`, `p1c`): re-trace retaining per-cell effects; sign-flip permutation null; empirical FDR, Benjamini–Hochberg control, a 6×5 threshold sweep, and a norm-matched specificity control.
- **Expanded steering panel** (`p2`, `p2b`): 80 switch features (ON + OFF), dose–response, and three readouts independent of the pseudotime signature (a literature marker panel and two held-out probes).
- **Extended triplets** (`p3`): same-pathway, cross-pathway and random triplets with bootstrap intervals.
- **Degree-distribution fitting** (`p4`): Clauset–Shalizi–Newman power-law fit with likelihood-ratio comparison against lognormal / exponential / stretched-exponential / truncated power law.
- **Annotation enrichment** (`p5`): formal Fisher / logistic tests and the degree–frequency confound.
- **Hub importance** (`p6`): masked-LM damage from single-feature ablation vs random and frequency-matched controls.
- **Generalization** (`p7`): tracing with a natively trained Tabula Sapiens SAE.
- **Biological grounding** (`p9`): feature degree vs CRISPRi knockdown effect size.

## Key results (calibrated)

| Topic | Finding | Metric |
|---|---|---|
| **Edge calibration** | The uncalibrated `|d|>0.5`, consistency `>0.7` criterion has a measured false-discovery rate of **3.6%**; 79% of edges survive BH control. At `|d|>0.3` a third of the graph would be noise. | FDR 3.6%; ≈1.1M / 1,393,850 edges retained |
| **Degree ≈ firing rate** | A feature's circuit degree is largely a restatement of its activation frequency. | Spearman ρ = 0.84 |
| **Heavy-tailed, not scale-free** | Right-skewed, but formal fitting cannot distinguish a power law from lognormal/exponential. | power-law exponent α = 6.05 (157 tail points) |
| **Annotation vs centrality** | Annotation does not predict centrality; L5 hubs are **not** annotation-depleted (60% vs 53.8% background), though L8/L14 hubs are. | Fisher OR = 1.29, p = 0.66 (L5) |
| **Redundancy, not pathway-specific** | Redundancy deepens with order and equally for triplets sharing no pathway; synergy is rare and cross-pathway. | pairwise 0.70 → three-way 0.53; superadditive 0.087% |
| **Feature-encoded direction** | Layer-17 ON-switch features shift a held-out maturity probe up (83% of cells); OFF-switch features shift it down (88%); direction is set by the feature, not the layer. | Mann–Whitney p = 0.0055 |
| **Biological bound (null)** | Feature degree does not predict CRISPRi knockdown effect size in the same cell line. | ρ = 0.02 (n.s.) |

Earlier framings that the revision superseded — "zero synergy", "scale-free", "hub fragility", "L17 universally drives maturation (3/3, fraction positive = 1.0)", and "40% of top-20 hubs unannotated, a bias" — are corrected in the paper and in `REPRODUCIBILITY.md`.

## Repository structure

```
sae-biological-map/
├── src/
│   ├── sae_model.py                    # TopK SAE (d=1152, 4x expansion, k=32)
│   ├── exhaustive_feature_tracing.py   # Core exp. 1: exhaustive L5 tracing
│   ├── higher_order_ablation.py        # Core exp. 2: three-way ablation
│   ├── trajectory_steering.py          # Core exp. 3: trajectory steering
│   └── revision/                       # Peer-review revision analyses
│       ├── p1a_retrace_with_deltas.py       # retrace keeping per-cell effects
│       ├── p1b_null_fdr_thresholds.py       # permutation null, FDR, threshold sweep
│       ├── p1c_normmatched_control.py       # norm-matched specificity control
│       ├── p2_steering.py / p2b_readout_analysis.py   # expanded steering panel
│       ├── p3_triplets_extended.py          # same/cross/random triplets
│       ├── p4_tail_fitting.py               # power-law vs alternatives
│       ├── p5_annotation_enrichment.py      # hub annotation enrichment tests
│       ├── p6_hub_ablation_lm.py            # masked-LM hub importance
│       ├── p7_ts_native_trace.py            # Tabula Sapiens-native SAE trace
│       ├── p9_crispri_grounding.py          # CRISPRi grounding of degree
│       └── make_figures.py                  # paper figures from result JSONs
├── results/
│   ├── exhaustive_tracing/  higher_order_ablation/  trajectory_steering/   # core summaries
│   └── revision/                       # calibration & revision result summaries (JSON)
├── paper/
│   ├── manuscript.tex  supplementary.tex  references.bib  figures/
├── REPRODUCIBILITY.md                  # script → artifact → paper-object map, seeds, hashes
├── requirements.txt   LICENSE   README.md
```

The large inputs (SAE checkpoints, the Replogle and Tabula Sapiens `.h5ad` files, and the ≈2 GB of per-cell effect arrays that the calibration consumes) are **not** shipped in git owing to size. `results/` holds the small JSON summaries from which every figure and number is a pure function; see `REPRODUCIBILITY.md`.

## Prerequisites

### Data
- **K562 CRISPRi perturbation data** (Replogle et al., 2022): [Figshare](https://plus.figshare.com/articles/dataset/Replogle_2022_K562_gwps/21452470)
- **Tabula Sapiens immune subset** (The Tabula Sapiens Consortium, 2022): [CZ CELLxGENE](https://cellxgene.cziscience.com/)
- **Geneformer V2-316M** (Theodoris et al., 2023): [HuggingFace](https://huggingface.co/ctheodoris/Geneformer)

### Upstream dependencies
The experiments build on trained layer-wise SAE checkpoints and extracted activations from the companion SAE-atlas study. Configure their location via environment variables (`SAE_PROJ_ROOT`, `SAE_BASE_ROOT`, `SAE_DATA_ROOT`, `SAE_DATA_PATH`) or edit the path constants at the top of each script.

## Installation

```bash
conda create -n sae-bio python=3.10
conda activate sae-bio
pip install -r requirements.txt
```

## Usage

```bash
# Core experiments
python src/exhaustive_feature_tracing.py --n-cells 20 --source-layer 5      # exhaustive L5 trace
python src/higher_order_ablation.py --n-cells 200 --n-triplets 10           # three-way ablation
python src/trajectory_steering.py --alphas 2.0,5.0 --n-cells 500            # trajectory steering

# Revision analyses (see REPRODUCIBILITY.md for the full run order and I/O map)
python src/revision/p1a_retrace_with_deltas.py --validate                   # verify retrace ≡ original
python src/revision/p1a_retrace_with_deltas.py --priority-first             # retrace keeping per-cell effects
python src/revision/p1b_null_fdr_thresholds.py --n-perm 500 --perm-subset 600  # permutation FDR + sweep
python src/revision/p4_tail_fitting.py                                      # power-law model comparison
python src/revision/p5_annotation_enrichment.py                            # annotation enrichment tests
python src/revision/p9_crispri_grounding.py --pseudobulk --associate       # CRISPRi grounding
```

## Compute environment

Experiments were run on an Apple M-series laptop GPU (MPS backend). The exhaustive re-trace is the only long job (~9 GPU-hours); the calibration, tail-fitting, annotation, and CRISPRi analyses are CPU-only and run in minutes from the deposited result summaries. Exact versions, checkpoint SHA-256 hashes, and every random seed are in `REPRODUCIBILITY.md`.

## Citation

```bibtex
@article{kendiukhov2025exhaustive,
  title  = {Exhaustive single-layer circuit mapping of a single-cell foundation model reveals massive redundancy, heavy-tailed hub architecture, and layer-dependent steering directionality},
  author = {Kendiukhov, Ihor},
  year   = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
