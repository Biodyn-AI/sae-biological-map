# Reproducibility

> **Repository layout note.** The original analysis lives under `src/` (`exhaustive_feature_tracing.py`, `higher_order_ablation.py`, `trajectory_steering.py`, `sae_model.py`); the revision analyses added during peer review live under `src/revision/`. Result *summaries* are under `results/` and `results/revision/`. Large inputs (SAE checkpoints, the Replogle `.h5ad`, Tabula Sapiens `.h5ad`, and the per-cell effect arrays) are **not** shipped in git owing to size; see the Data section for their public sources. Paths in the scripts are configurable via the `SAE_PROJ_ROOT` / `SAE_BASE_ROOT` / `SAE_DATA_*` environment variables.

Everything reported in *Exhaustive single-layer circuit mapping of a single-cell foundation
model* can be regenerated from this repository. This file gives the exact script for every
number, table and figure, together with the seeds, checkpoints, intermediate artifacts and
software versions needed to reproduce them without re-deriving any upstream result.

## 1. Environment

| Component | Version |
|---|---|
| Python | 3.10.19 |
| PyTorch | 2.3.1 (MPS backend) |
| transformers | 4.57.3 |
| numpy / scipy / scikit-learn | 1.26.4 / 1.15.2 / 1.7.2 |
| h5py / scanpy | 3.15.1 / 1.10.4 |
| Hardware | Apple M2 Pro, macOS 26.3 |

Create the environment with `requirements.txt`. All GPU work runs on the MPS backend; in
`model.eval()` with `torch.no_grad()` the forward pass is bit-deterministic on this device,
which is why the steering precision floor is exactly `SD(Δs) = 0` (S1 Appendix §S7).

## 2. Models and checkpoints

| Artifact | Location | SHA-256 (first 16) |
|---|---|---|
| Geneformer V2-316M | `ctheodoris/Geneformer`, subfolder `Geneformer-V2-316M` (HuggingFace) | pinned by revision in `requirements.txt` |
| K562 SAE, layer 0 | `SAE checkpoints (SAE-atlas release)/layer00_x4_k32/sae_final.pt` | `1331abeee42ff9ed` |
| K562 SAE, layer 5 | `SAE checkpoints (SAE-atlas release)/layer05_x4_k32/sae_final.pt` | `3e8078ee441cd4ab` |
| K562 SAE, layer 11 | `SAE checkpoints (SAE-atlas release)/layer11_x4_k32/sae_final.pt` | `a39b3f3b55b86515` |
| K562 SAE, layer 17 | `SAE checkpoints (SAE-atlas release)/layer17_x4_k32/sae_final.pt` | `b7325896948f7820` |
| Tabula Sapiens SAEs (L0/5/11/17) | `Tabula Sapiens SAE checkpoints (SAE-atlas release)/layer{00,05,11,17}_x4_k32/` | — |

Each SAE directory also contains `activation_mean.npy` (the pre-encoder mean subtracted before
encoding), `feature_catalog.json` (activation frequency, top-50 loading genes) and
`feature_annotations.json` (GO/KEGG/Reactome labels). Hashes for all 18 K562 layers are
produced by `shasum -a 256 SAE checkpoints (SAE-atlas release)/*/sae_final.pt`.

## 3. Data

| Dataset | Source | Used for |
|---|---|---|
| K562 CRISPRi (Replogle et al. 2022) | Figshare DOI 10.6084/m9.figshare.21452470 → `replogle_concat.h5ad` | tracing cells (non-targeting controls), triplet ablation, masked-LM evaluation, CRISPRi grounding |
| Tabula Sapiens immune subset | CZ CELLxGENE → `tabula_sapiens_immune_subset_20000.h5ad` | steering, TS-native trace, probe training |
| GO / KEGG / Reactome | geneontology.org, kegg.jp, reactome.org | feature annotation |
| Geneformer token dictionaries | shipped with Geneformer (`token_dictionary_gc104M.pkl`, `gene_median_dictionary_gc104M.pkl`, `gene_name_id_dict_gc104M.pkl`) | tokenization |

Special tokens: `<pad>`=0, `<mask>`=1, `<cls>`=2, `<eos>`=3.

## 4. Seeds

| Purpose | Seed | Set in |
|---|---|---|
| K562 control-cell sampling (all tracing/ablation) | 42 | `src/exhaustive_feature_tracing.py:load_and_tokenize_cells` |
| Tabula Sapiens 500-cell steering draw | 42 (`np.random.RandomState`) | `src/revision/p2_steering.py:build_cells` |
| Probe-training cell draw (disjoint from the steering cells) | 43 | same |
| Sign-flip permutations, BH family, threshold sweep | 0 | `src/revision/p1b_null_fdr_thresholds.py` |
| Degree-distribution bootstrap / goodness-of-fit | 0 | `src/revision/p4_tail_fitting.py` |
| Frequency-matched background resampling | 0 | `src/revision/p5_annotation_enrichment.py` |
| Triplet selection and bootstrap | 7 | `src/revision/p3_triplets_extended.py` |
| Masked-LM mask positions, arm selection | 3 | `src/revision/p6_hub_ablation_lm.py` |
| TS-native feature subsample | 11 | `src/revision/p7_ts_native_trace.py` |
| CRISPRi control resampling, gene-label permutation | 0 | `src/revision/p9_crispri_grounding.py` |

Because masked-LM masks are drawn once and reused across every ablation arm, each arm is
evaluated on identical inputs.

## 5. Script → artifact → manuscript object

| Script | Output | Appears as |
|---|---|---|
| `src/exhaustive_feature_tracing.py` | `experiments/phase8_exhaustive_tracing/feature_*.json` | uncalibrated L5 trace (Table 1, "raw" column) |
| `src/revision/p1a_retrace_with_deltas.py` | `results/revision/P1_retrace/deltas/feature_*.npz` (per-cell effect vectors, float16), `json/feature_*.json` | input to all calibration |
| `src/revision/p1b_null_fdr_thresholds.py` | `P1_calibration/{summary,edges_fdr,edges_raw,threshold_sweep,per_feature}.json` | Results §"Edge calling…", Fig 1D, Table 1, S1 §S12 |
| `src/revision/p1c_normmatched_control.py` | `P1_normmatched/summary.json` | S1 §S12 (specificity null) |
| `src/revision/p2_steering.py --probes` | `P2_steering/probes.npz` | held-out maturity / pseudotime probes |
| `src/revision/p2_steering.py --steer --analyse` | `P2_steering/features/L*_F*.json`, `summary.json` | Fig 4, Fig 5, Table 4 |
| `src/revision/p2b_readout_analysis.py` | `P2_steering/readout_analysis.json` | Results §"Steering…", S1 §S13 |
| `src/revision/p3_triplets_extended.py` | `P3_triplets/{triplets,summary}.json`, `deltas/*.npz` | Fig 3, Table 3 |
| `src/revision/p4_tail_fitting.py` | `P4_tail/tail_fit_raw.json` | Results §"Heavy-tailed…", S7 Fig |
| `src/revision/p5_annotation_enrichment.py` | `P5_annotation/annotation_enrichment_*.json` | Fig 2C–D, Table 2 |
| `src/revision/p6_hub_ablation_lm.py` | `P6_hub_importance/summary.json` | Results §"Hub features…", S5 Fig |
| `src/revision/p7_ts_native_trace.py` | `P7_ts_native/{json/*.json,summary.json}` | Results §"Generalization…", S6 Fig |
| `src/revision/p9_crispri_grounding.py` | `P9_crispri/{perturbation_shifts,association}.json` | Results §"Biological grounding" |
| `src/revision/make_figures.py` | `paper/plos_one_submission/Fig{1..5}.pdf`, `S{5,6,7}_Fig.pdf` | all main figures |

Run order: `p1a` (≈9 h) → `p1b`, `p1c`; `p2 --probes` → `p2 --steer --analyse` → `p2b`;
`p3`, `p6`, `p7` in any order; `p4`, `p5`, `p9` are CPU-only and depend only on the trace
outputs. `src/revision/run_gpu_queue.sh` runs the GPU jobs sequentially.

## 6. Intermediate artifacts we deposit

Re-running `p1a` costs ≈9 GPU-hours. To make every downstream statistic reproducible without
it, the following are archived with the code release:

* `P1_retrace/deltas/feature_XXXX.npz` — for each of the 4,065 active L5 features, the per-cell
  effect vector (20 cells × 4,608 targets, float16) at each of the three downstream layers.
  Approximately 2.2 GB in total. Every reported edge count, Cohen's *d*, Hedges' *g*,
  permutation p-value, FDR and threshold-sweep entry is a pure function of these arrays.
* `P1_calibration/*.json` — the calibrated edge tables.
* `P2_steering/features/*.json` — per-cell values of all four steering readouts, for every
  feature and every α, so that the tests in `p2b` can be recomputed or replaced.
* `P3_triplets/deltas/*.npz` — per-cell, per-condition effect vectors for all 12 triplets.
* `P9_crispri/perturbation_shifts.json` — per-perturbation transcriptome-shift z-scores, so the
  30 GB expression matrix does not need to be re-streamed.
* `phase7_trajectory_dynamics/sae_activations.npz` and `cell_metadata.json` — the 481-cell SAE
  activation matrices and diffusion pseudotime that define the switch-feature panel.

## 7. Definitions that upstream work supplied and this paper re-derives

Prior work is used as input for exactly three things. Each is restated in the manuscript's
Methods so that no result depends on reading a preprint, and each is re-derived here from the
deposited artifacts rather than taken on trust:

1. **The layer-wise SAEs** (TopK, k=32, 4× overcomplete, trained on 4.05M K562 residual-stream
   token positions). Checkpoints and catalogs above.
2. **The 30 "selective" features** used by the earlier circuit map, selected at each layer by
   *annotation quality score*: the number of significant ontology enrichments across
   GO BP / KEGG / Reactome / STRING / TRRUST, weighted by −log₁₀(p)
   (`(selective-map script; see the SAE-atlas release):select_features`). Reproduced by that function with
   `random_seed=None`.
3. **The switch-feature panel.** The earlier definition (|Cohen's *d*| > 1.0 between cells with
   pseudotime < 0.3 and > 0.7, among features whose Spearman correlation with pseudotime
   survives BH at q < 0.05) is in `src/trajectory_steering.py`. This paper does not
   rely on it: `p2_steering.py:select_features` re-derives the panel from
   `sae_activations.npz` under a single stated rule — activation frequency ≥ 0.05, Cohen's *d*
   between pseudotime terciles ≥ 0.5 (ON) or ≤ −0.5 (OFF), top 12 ON and top 8 OFF per layer —
   and all reported steering statistics use that panel.

## 8. Verifying the re-trace

`p1a_retrace_with_deltas.py --validate` re-traces three features (F0, F11, F898) and compares
against the original run. It must report identical significant-edge counts at every downstream
layer and `max|Δd| < 1e-4` (observed: ≤ 7.7e-5, i.e. float32 rounding). The re-trace changes
only *how* the ablated forward pass is computed — injecting the perturbed residual stream into
layer `source+1` and running the tail, instead of a full forward pass with a hook — and where
the SAE encoder runs (GPU rather than CPU). It is ~2.3× faster and numerically equivalent.
