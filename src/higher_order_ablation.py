#!/usr/bin/env python3
"""
Higher-Order (3-Way) Combinatorial Ablation

Extends combinatorial ablation from pairs to triplets. For each triplet (A, B, C):
  - 3 single ablations: A, B, C
  - 3 pairwise ablations: AB, AC, BC
  - 1 joint ablation: ABC

Tests whether zero-synergy (found in pairwise) holds at higher orders.

Metrics computed:
  - Pairwise additivity: |d_AB|/(|d_A|+|d_B|), etc.
  - Three-way additivity: |d_ABC|/(|d_A|+|d_B|+|d_C|)
  - Marginal contribution: |d_ABC| - max(|d_AB|, |d_AC|, |d_BC|)
  - Higher-order interaction: |d_ABC| - |d_AB| - |d_AC| - |d_BC| + |d_A| + |d_B| + |d_C|

Usage:
    python src/higher_order_ablation.py \
        [--n-cells 200] [--n-triplets 10]
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# ============================================================
# Paths and constants — configure these for your environment
# ============================================================
# Set DATA_ROOT to the directory containing your SAE models and circuit tracing results.
# Expected layout under DATA_ROOT:
#   sae_models/sae_layer{N}.pt        — trained SAE checkpoints
#   circuit_tracing/                   — circuit edge results from causal tracing
# Set DATA_PATH to the Replogle K562 CRISPRi .h5ad file.
# Set PAIR_DIR to the directory containing pairwise combinatorial ablation results.
DATA_ROOT = os.environ.get("SAE_DATA_ROOT", "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_ROOT, "sae_models")
CIRCUIT_DIR = os.path.join(DATA_ROOT, "circuit_tracing")
DATA_PATH = os.environ.get("SAE_DATA_PATH", "data/replogle_concat.h5ad")
PAIR_DIR = os.environ.get("SAE_PAIR_DIR", "experiments/phase7_combinatorial_ablation")
OUT_DIR = os.environ.get("SAE_OUT_DIR", "results/higher_order_ablation")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

EXPANSION = 4
K_VAL = 32
N_CTRL = 2000
MAX_SEQ_LEN = 2048
HIDDEN_DIM = 1152
N_FEATURES = 4608
N_LAYERS = 18
TARGET_LAYER = 17

COHENS_D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ============================================================
# Cell loading (from circuit tracing)
# ============================================================
def load_categorical_column(h5group, col_name):
    import h5py
    col = h5group[col_name]
    if isinstance(col, h5py.Group):
        categories = col['categories'][:]
        codes = col['codes'][:]
        if categories.dtype.kind in ('O', 'S'):
            categories = np.array([x.decode() if isinstance(x, bytes) else x for x in categories])
        return categories[codes]
    else:
        data = col[:]
        if data.dtype.kind in ('O', 'S'):
            return np.array([x.decode() if isinstance(x, bytes) else x for x in data])
        return data


def tokenize_cell(expression_vector, var_indices, token_ids, medians, max_len=2048):
    expr = expression_vector[var_indices]
    nonzero = expr > 0
    if nonzero.sum() == 0:
        return None
    expr_nz = expr[nonzero]
    tokens_nz = token_ids[nonzero]
    medians_nz = medians[nonzero]
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = expr_nz / medians_nz
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0)
    rank_order = np.argsort(-normalized)
    ranked_tokens = tokens_nz[rank_order][:max_len - 2]
    return np.concatenate([[2], ranked_tokens, [3]]).astype(np.int64)


def load_and_tokenize_cells(n_cells):
    import h5py
    print("  Loading tokenizer dictionaries...")
    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    print("  Loading K562 cell data...")
    with h5py.File(DATA_PATH, 'r') as f:
        cell_genes = load_categorical_column(f['obs'], 'gene')
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_cells_total, n_genes_total = f['X'].shape

    control_mask = np.zeros(n_cells_total, dtype=bool)
    for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
        control_mask |= (cell_genes == ctrl_name)
    ctrl_indices = np.where(control_mask)[0]

    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()

    cell_sample = ctrl_indices[:n_cells]
    print(f"    Using {len(cell_sample)} K562 control cells")

    with h5py.File(DATA_PATH, 'r') as f:
        X_sample = np.empty((len(cell_sample), n_genes_total), dtype=np.float32)
        for ci, idx in enumerate(cell_sample):
            X_sample[ci, :] = f['X'][int(idx), :]

    row_sums = X_sample.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_sample = np.log1p(X_sample / row_sums * 1e4)

    mapped_var_indices = []
    mapped_token_ids_list = []
    mapped_medians_list = []
    for i in range(n_genes_total):
        gene_name = var_genes[i]
        ensembl = gene_name_id_dict.get(gene_name)
        if ensembl and ensembl in token_dict:
            mapped_var_indices.append(i)
            mapped_token_ids_list.append(token_dict[ensembl])
            mapped_medians_list.append(gene_median_dict.get(ensembl, 1.0))
    mapped_var_indices = np.array(mapped_var_indices)
    mapped_token_ids = np.array(mapped_token_ids_list)
    mapped_medians = np.array(mapped_medians_list)

    all_tokens = []
    for ci in range(len(cell_sample)):
        tokens = tokenize_cell(X_sample[ci], mapped_var_indices,
                               mapped_token_ids, mapped_medians, MAX_SEQ_LEN)
        if tokens is not None:
            all_tokens.append(tokens)
    print(f"    Tokenized {len(all_tokens)} cells")
    del X_sample
    gc.collect()
    return all_tokens


# ============================================================
# Core classes
# ============================================================
class WelfordAccumulator:
    def __init__(self, n_dims):
        self.n = 0
        self.mean = np.zeros(n_dims, dtype=np.float64)
        self.M2 = np.zeros(n_dims, dtype=np.float64)
        self.pos_count = np.zeros(n_dims, dtype=np.float64)

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.pos_count += (x > 0).astype(np.float64)

    def finalize(self):
        if self.n < 2:
            return np.zeros_like(self.mean), np.zeros_like(self.mean)
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var)
        cohens_d = self.mean / np.maximum(std, 1e-10)
        consistency = np.maximum(self.pos_count, self.n - self.pos_count) / self.n
        return cohens_d, consistency


class SAECache:
    def __init__(self):
        self._saes = {}
        self._means = {}

    def get(self, layer):
        if layer not in self._saes:
            import torch
            sys.path.insert(0, os.path.join(PROJ_DIR, "src"))
            from sae_model import TopKSAE
            run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
            run_dir = os.path.join(SAE_BASE, run_name)
            sae = TopKSAE.load(os.path.join(run_dir, "sae_final.pt"), device='cpu')
            sae.eval()
            act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))
            self._saes[layer] = sae
            self._means[layer] = act_mean
            print(f"    Loaded SAE for layer {layer}")
        return self._saes[layer], self._means[layer]


def make_hook(modified_hidden, device):
    fired = [False]
    def hook_fn(module, input, output):
        if fired[0]:
            return output
        fired[0] = True
        modified = list(output)
        modified[0] = modified_hidden.unsqueeze(0).to(device)
        return tuple(modified)
    return hook_fn


# ============================================================
# Triplet selection from pairwise results
# ============================================================
def select_triplets(n_triplets):
    """Select triplets from the pairwise ablation features.

    Strategy: for each pathway, find features at L0, L5, and L11.
    Form triplets by combining one from each layer.
    """
    # Load pairwise summary to get the features used
    summary_path = os.path.join(PAIR_DIR, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    # Collect unique features per layer per pathway
    features_by_pathway_layer = {}  # pathway -> layer -> [features]
    for pair in summary['per_pair_results']:
        pw = pair['pathway']
        # Skip cross-pathway pairs for forming same-pathway triplets
        if pair['pair_type'] == 'cross_pathway':
            continue

        for side in ['feature_A', 'feature_B']:
            feat = pair[side]
            layer = feat['layer']
            idx = feat['idx']

            if pw not in features_by_pathway_layer:
                features_by_pathway_layer[pw] = {}
            if layer not in features_by_pathway_layer[pw]:
                features_by_pathway_layer[pw][layer] = []

            # Deduplicate
            existing_ids = [f['idx'] for f in features_by_pathway_layer[pw][layer]]
            if idx not in existing_ids:
                features_by_pathway_layer[pw][layer].append(feat)

    # Form triplets: need features at all three layers (0, 5, 11)
    triplets = []
    for pw, layer_feats in features_by_pathway_layer.items():
        layers_available = sorted(layer_feats.keys())
        if len(layers_available) < 3:
            # Need at least 3 layers; try to use what we have
            if 0 in layer_feats and 5 in layer_feats and 11 in layer_feats:
                pass
            else:
                continue

        feats_0 = layer_feats.get(0, [])
        feats_5 = layer_feats.get(5, [])
        feats_11 = layer_feats.get(11, [])

        if not feats_0 or not feats_5 or not feats_11:
            continue

        # Generate all combinations, take top by total edges
        for f0 in feats_0:
            for f5 in feats_5:
                for f11 in feats_11:
                    total_edges = f0.get('n_edges', 0) + f5.get('n_edges', 0) + f11.get('n_edges', 0)
                    triplets.append({
                        'feature_A': {'layer': 0, 'idx': f0['idx'], 'label': f0['label'],
                                      'n_edges': f0.get('n_edges', 0)},
                        'feature_B': {'layer': 5, 'idx': f5['idx'], 'label': f5['label'],
                                      'n_edges': f5.get('n_edges', 0)},
                        'feature_C': {'layer': 11, 'idx': f11['idx'], 'label': f11['label'],
                                      'n_edges': f11.get('n_edges', 0)},
                        'pathway': pw,
                        'triplet_type': 'same_pathway',
                        'total_edges': total_edges,
                    })

    triplets.sort(key=lambda x: -x['total_edges'])

    # Limit per pathway
    same_pathway = []
    pathway_counts = {}
    for t in triplets:
        pw = t['pathway']
        if pw not in pathway_counts:
            pathway_counts[pw] = 0
        if pathway_counts[pw] < 3:
            same_pathway.append(t)
            pathway_counts[pw] += 1
        if len(same_pathway) >= n_triplets - 2:
            break

    # Add 2 cross-pathway triplets using features from different pathways
    cross_triplets = []
    all_feats = {}
    for pw, layer_feats in features_by_pathway_layer.items():
        for layer, feats in layer_feats.items():
            for f in feats:
                key = (layer, f['idx'])
                if key not in all_feats:
                    all_feats[key] = {**f, 'layer': layer, 'pathway': pw}

    # Cross-pathway 1: DDR (L0) × Mitosis (L5) × Mitosis (L11)
    cross_candidates = [
        # DDR-L0 × Mitosis-L5 × Mitosis-L11
        ({'layer': 0, 'pw': 'ddr'}, {'layer': 5, 'pw': 'mitosis'}, {'layer': 11, 'pw': 'mitosis'}),
        # Signaling-L0 × Mitosis-L5 × Vesicle-L11
        ({'layer': 0, 'pw': 'signaling'}, {'layer': 5, 'pw': 'mitosis'}, {'layer': 11, 'pw': 'vesicle'}),
    ]

    for c0, c5, c11 in cross_candidates:
        f0_list = features_by_pathway_layer.get(c0['pw'], {}).get(c0['layer'], [])
        f5_list = features_by_pathway_layer.get(c5['pw'], {}).get(c5['layer'], [])
        f11_list = features_by_pathway_layer.get(c11['pw'], {}).get(c11['layer'], [])

        if f0_list and f5_list and f11_list:
            f0 = f0_list[0]
            f5 = f5_list[0]
            f11 = f11_list[0]
            cross_triplets.append({
                'feature_A': {'layer': 0, 'idx': f0['idx'], 'label': f0['label'],
                              'n_edges': f0.get('n_edges', 0)},
                'feature_B': {'layer': 5, 'idx': f5['idx'], 'label': f5['label'],
                              'n_edges': f5.get('n_edges', 0)},
                'feature_C': {'layer': 11, 'idx': f11['idx'], 'label': f11['label'],
                              'n_edges': f11.get('n_edges', 0)},
                'pathway': f"{c0['pw']}_x_{c5['pw']}_x_{c11['pw']}",
                'triplet_type': 'cross_pathway',
                'total_edges': f0.get('n_edges', 0) + f5.get('n_edges', 0) + f11.get('n_edges', 0),
            })

    result = same_pathway + cross_triplets
    return result[:n_triplets]


# ============================================================
# Single-feature ablation
# ============================================================
def ablate_single(source_layer, feature_idx, all_tokens, model, device,
                  sae_cache, n_cells):
    """Ablate one feature, measure at TARGET_LAYER."""
    import torch

    source_sae, source_mean = sae_cache.get(source_layer)
    source_mean_t = torch.tensor(source_mean, dtype=torch.float32)
    dst_sae, dst_mean = sae_cache.get(TARGET_LAYER)
    dst_mean_t = torch.tensor(dst_mean, dtype=torch.float32)

    acc = WelfordAccumulator(N_FEATURES)
    n_active = 0

    for ci in range(min(n_cells, len(all_tokens))):
        tokens = all_tokens[ci]
        gene_mask = (tokens != 2) & (tokens != 3)
        gene_positions = np.where(gene_mask)[0]
        if len(gene_positions) == 0:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(1, len(tokens), dtype=torch.long).to(device)

        with torch.no_grad():
            outputs_clean = model(input_ids=input_ids, attention_mask=attention_mask)

        clean_hidden = outputs_clean.hidden_states[source_layer + 1][0].cpu()
        gene_hidden = clean_hidden[gene_positions] - source_mean_t

        with torch.no_grad():
            h_sparse, topk = source_sae.encode(gene_hidden)

        active_mask = (topk == feature_idx).any(dim=1)
        if not active_mask.any():
            del outputs_clean, clean_hidden
            continue

        n_active += 1

        h_abl = h_sparse.clone()
        h_abl[:, feature_idx] = 0.0

        with torch.no_grad():
            recon = source_sae.decode(h_sparse) + source_mean_t
            recon_abl = source_sae.decode(h_abl) + source_mean_t

        delta = recon_abl - recon
        modified = clean_hidden.clone()
        modified[gene_positions] = clean_hidden[gene_positions] + delta

        hook = model.bert.encoder.layer[source_layer].register_forward_hook(
            make_hook(modified, device))
        with torch.no_grad():
            outputs_abl = model(input_ids=input_ids, attention_mask=attention_mask)
        hook.remove()

        clean_dl = outputs_clean.hidden_states[TARGET_LAYER + 1][0][gene_positions].cpu()
        abl_dl = outputs_abl.hidden_states[TARGET_LAYER + 1][0][gene_positions].cpu()

        with torch.no_grad():
            clean_sp, _ = dst_sae.encode(clean_dl - dst_mean_t)
            abl_sp, _ = dst_sae.encode(abl_dl - dst_mean_t)

        active_pos = torch.where(active_mask)[0]
        pos_deltas = [(abl_sp[p] - clean_sp[p]).numpy().astype(np.float64) for p in active_pos]
        acc.update(np.mean(pos_deltas, axis=0))

        del outputs_clean, outputs_abl, clean_hidden, modified
        if device.type == 'mps':
            torch.mps.empty_cache()

    return acc, n_active


# ============================================================
# Two-feature ablation (cross-layer)
# ============================================================
def ablate_two(layer_A, fi_A, layer_B, fi_B, all_tokens, model, device,
               sae_cache, n_cells):
    """Ablate two features at different layers, measure at TARGET_LAYER."""
    import torch

    sae_A, mean_A = sae_cache.get(layer_A)
    sae_B, mean_B = sae_cache.get(layer_B)
    mean_A_t = torch.tensor(mean_A, dtype=torch.float32)
    mean_B_t = torch.tensor(mean_B, dtype=torch.float32)
    dst_sae, dst_mean = sae_cache.get(TARGET_LAYER)
    dst_mean_t = torch.tensor(dst_mean, dtype=torch.float32)

    acc = WelfordAccumulator(N_FEATURES)
    n_active = 0

    # Ensure layer_A < layer_B
    if layer_A > layer_B:
        layer_A, layer_B = layer_B, layer_A
        fi_A, fi_B = fi_B, fi_A
        sae_A, sae_B = sae_B, sae_A
        mean_A_t, mean_B_t = mean_B_t, mean_A_t

    for ci in range(min(n_cells, len(all_tokens))):
        tokens = all_tokens[ci]
        gene_mask = (tokens != 2) & (tokens != 3)
        gene_positions = np.where(gene_mask)[0]
        if len(gene_positions) == 0:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(1, len(tokens), dtype=torch.long).to(device)

        with torch.no_grad():
            outputs_clean = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check activity at both layers
        hidden_A = outputs_clean.hidden_states[layer_A + 1][0].cpu()
        h_A_centered = hidden_A[gene_positions] - mean_A_t
        with torch.no_grad():
            h_sp_A, topk_A = sae_A.encode(h_A_centered)
        active_A = (topk_A == fi_A).any(dim=1)

        hidden_B = outputs_clean.hidden_states[layer_B + 1][0].cpu()
        h_B_centered = hidden_B[gene_positions] - mean_B_t
        with torch.no_grad():
            h_sp_B, topk_B = sae_B.encode(h_B_centered)
        active_B = (topk_B == fi_B).any(dim=1)

        joint_active = active_A | active_B
        if not joint_active.any():
            del outputs_clean, hidden_A, hidden_B
            continue

        n_active += 1

        # Ablate A
        h_abl_A = h_sp_A.clone()
        h_abl_A[:, fi_A] = 0.0
        with torch.no_grad():
            recon_A = sae_A.decode(h_sp_A) + mean_A_t
            recon_abl_A = sae_A.decode(h_abl_A) + mean_A_t
        delta_A = recon_abl_A - recon_A
        modified_A = hidden_A.clone()
        modified_A[gene_positions] = hidden_A[gene_positions] + delta_A

        if layer_A == layer_B:
            # Same layer: ablate both in one hook
            h_abl_AB = h_sp_A.clone()
            h_abl_AB[:, fi_A] = 0.0
            h_abl_AB[:, fi_B] = 0.0
            with torch.no_grad():
                recon_abl_AB = sae_A.decode(h_abl_AB) + mean_A_t
            delta_AB = recon_abl_AB - recon_A
            modified_A[gene_positions] = hidden_A[gene_positions] + delta_AB

            hook = model.bert.encoder.layer[layer_A].register_forward_hook(
                make_hook(modified_A, device))
            with torch.no_grad():
                outputs_joint = model(input_ids=input_ids, attention_mask=attention_mask)
            hook.remove()
        else:
            # Two-pass for cross-layer
            hook_A1 = model.bert.encoder.layer[layer_A].register_forward_hook(
                make_hook(modified_A, device))
            with torch.no_grad():
                outputs_p1 = model(input_ids=input_ids, attention_mask=attention_mask)
            hook_A1.remove()

            hidden_B_after = outputs_p1.hidden_states[layer_B + 1][0].cpu()
            h_B_new = hidden_B_after[gene_positions] - mean_B_t
            with torch.no_grad():
                h_sp_B_new, _ = sae_B.encode(h_B_new)
            h_abl_B_new = h_sp_B_new.clone()
            h_abl_B_new[:, fi_B] = 0.0
            with torch.no_grad():
                recon_B_new = sae_B.decode(h_sp_B_new) + mean_B_t
                recon_abl_B_new = sae_B.decode(h_abl_B_new) + mean_B_t
            delta_B_new = recon_abl_B_new - recon_B_new
            modified_B = hidden_B_after.clone()
            modified_B[gene_positions] = hidden_B_after[gene_positions] + delta_B_new

            hook_A2 = model.bert.encoder.layer[layer_A].register_forward_hook(
                make_hook(modified_A, device))
            hook_B2 = model.bert.encoder.layer[layer_B].register_forward_hook(
                make_hook(modified_B, device))
            with torch.no_grad():
                outputs_joint = model(input_ids=input_ids, attention_mask=attention_mask)
            hook_A2.remove()
            hook_B2.remove()
            del outputs_p1, hidden_B_after

        # Downstream measurement
        clean_dl = outputs_clean.hidden_states[TARGET_LAYER + 1][0][gene_positions].cpu()
        joint_dl = outputs_joint.hidden_states[TARGET_LAYER + 1][0][gene_positions].cpu()

        with torch.no_grad():
            clean_sp, _ = dst_sae.encode(clean_dl - dst_mean_t)
            joint_sp, _ = dst_sae.encode(joint_dl - dst_mean_t)

        active_pos = torch.where(joint_active)[0]
        pos_deltas = [(joint_sp[p] - clean_sp[p]).numpy().astype(np.float64) for p in active_pos]
        acc.update(np.mean(pos_deltas, axis=0))

        del outputs_clean, outputs_joint, hidden_A, hidden_B
        if device.type == 'mps':
            torch.mps.empty_cache()

    return acc, n_active


# ============================================================
# Three-feature ablation (cross-layer)
# ============================================================
def ablate_three(layer_A, fi_A, layer_B, fi_B, layer_C, fi_C,
                 all_tokens, model, device, sae_cache, n_cells):
    """Ablate three features at potentially different layers."""
    import torch

    # Sort by layer
    feats = sorted([(layer_A, fi_A), (layer_B, fi_B), (layer_C, fi_C)], key=lambda x: x[0])
    l1, f1 = feats[0]
    l2, f2 = feats[1]
    l3, f3 = feats[2]

    sae1, mean1 = sae_cache.get(l1)
    sae2, mean2 = sae_cache.get(l2)
    sae3, mean3 = sae_cache.get(l3)
    mean1_t = torch.tensor(mean1, dtype=torch.float32)
    mean2_t = torch.tensor(mean2, dtype=torch.float32)
    mean3_t = torch.tensor(mean3, dtype=torch.float32)
    dst_sae, dst_mean = sae_cache.get(TARGET_LAYER)
    dst_mean_t = torch.tensor(dst_mean, dtype=torch.float32)

    acc = WelfordAccumulator(N_FEATURES)
    n_active = 0

    for ci in range(min(n_cells, len(all_tokens))):
        tokens = all_tokens[ci]
        gene_mask = (tokens != 2) & (tokens != 3)
        gene_positions = np.where(gene_mask)[0]
        if len(gene_positions) == 0:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(1, len(tokens), dtype=torch.long).to(device)

        with torch.no_grad():
            outputs_clean = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check activity at all layers
        hidden1 = outputs_clean.hidden_states[l1 + 1][0].cpu()
        h1_c = hidden1[gene_positions] - mean1_t
        with torch.no_grad():
            h_sp1, topk1 = sae1.encode(h1_c)
        active1 = (topk1 == f1).any(dim=1)

        hidden2 = outputs_clean.hidden_states[l2 + 1][0].cpu()
        h2_c = hidden2[gene_positions] - mean2_t
        with torch.no_grad():
            h_sp2, topk2 = sae2.encode(h2_c)
        active2 = (topk2 == f2).any(dim=1)

        hidden3 = outputs_clean.hidden_states[l3 + 1][0].cpu()
        h3_c = hidden3[gene_positions] - mean3_t
        with torch.no_grad():
            h_sp3, topk3 = sae3.encode(h3_c)
        active3 = (topk3 == f3).any(dim=1)

        joint_active = active1 | active2 | active3
        if not joint_active.any():
            del outputs_clean, hidden1, hidden2, hidden3
            continue

        n_active += 1

        # Ablate feature 1 at l1
        h_abl1 = h_sp1.clone()
        h_abl1[:, f1] = 0.0
        with torch.no_grad():
            recon1 = sae1.decode(h_sp1) + mean1_t
            recon_abl1 = sae1.decode(h_abl1) + mean1_t
        delta1 = recon_abl1 - recon1
        modified1 = hidden1.clone()
        modified1[gene_positions] = hidden1[gene_positions] + delta1

        # Three-pass strategy:
        # Pass 1: ablate l1, get hidden at l2
        hook1a = model.bert.encoder.layer[l1].register_forward_hook(
            make_hook(modified1, device))
        with torch.no_grad():
            out_p1 = model(input_ids=input_ids, attention_mask=attention_mask)
        hook1a.remove()

        # Get l2 hidden after l1 ablation
        hidden2_after = out_p1.hidden_states[l2 + 1][0].cpu()
        h2_new = hidden2_after[gene_positions] - mean2_t
        with torch.no_grad():
            h_sp2_new, _ = sae2.encode(h2_new)
        h_abl2_new = h_sp2_new.clone()
        h_abl2_new[:, f2] = 0.0
        with torch.no_grad():
            recon2_new = sae2.decode(h_sp2_new) + mean2_t
            recon_abl2_new = sae2.decode(h_abl2_new) + mean2_t
        delta2_new = recon_abl2_new - recon2_new
        modified2 = hidden2_after.clone()
        modified2[gene_positions] = hidden2_after[gene_positions] + delta2_new

        # Pass 2: ablate l1 + l2, get hidden at l3
        hook1b = model.bert.encoder.layer[l1].register_forward_hook(
            make_hook(modified1, device))
        hook2a = model.bert.encoder.layer[l2].register_forward_hook(
            make_hook(modified2, device))
        with torch.no_grad():
            out_p2 = model(input_ids=input_ids, attention_mask=attention_mask)
        hook1b.remove()
        hook2a.remove()

        # Get l3 hidden after l1+l2 ablation
        hidden3_after = out_p2.hidden_states[l3 + 1][0].cpu()
        h3_new = hidden3_after[gene_positions] - mean3_t
        with torch.no_grad():
            h_sp3_new, _ = sae3.encode(h3_new)
        h_abl3_new = h_sp3_new.clone()
        h_abl3_new[:, f3] = 0.0
        with torch.no_grad():
            recon3_new = sae3.decode(h_sp3_new) + mean3_t
            recon_abl3_new = sae3.decode(h_abl3_new) + mean3_t
        delta3_new = recon_abl3_new - recon3_new
        modified3 = hidden3_after.clone()
        modified3[gene_positions] = hidden3_after[gene_positions] + delta3_new

        # Pass 3: ablate all three
        hook1c = model.bert.encoder.layer[l1].register_forward_hook(
            make_hook(modified1, device))
        hook2b = model.bert.encoder.layer[l2].register_forward_hook(
            make_hook(modified2, device))
        hook3a = model.bert.encoder.layer[l3].register_forward_hook(
            make_hook(modified3, device))
        with torch.no_grad():
            outputs_joint = model(input_ids=input_ids, attention_mask=attention_mask)
        hook1c.remove()
        hook2b.remove()
        hook3a.remove()

        # Downstream measurement
        clean_dl = outputs_clean.hidden_states[TARGET_LAYER + 1][0][gene_positions].cpu()
        joint_dl = outputs_joint.hidden_states[TARGET_LAYER + 1][0][gene_positions].cpu()

        with torch.no_grad():
            clean_sp, _ = dst_sae.encode(clean_dl - dst_mean_t)
            joint_sp, _ = dst_sae.encode(joint_dl - dst_mean_t)

        active_pos = torch.where(joint_active)[0]
        pos_deltas = [(joint_sp[p] - clean_sp[p]).numpy().astype(np.float64) for p in active_pos]
        acc.update(np.mean(pos_deltas, axis=0))

        del outputs_clean, outputs_joint, out_p1, out_p2
        del hidden1, hidden2, hidden3, hidden2_after, hidden3_after
        if device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()

    return acc, n_active


# ============================================================
# Main experiment
# ============================================================
def run_experiment(triplets, all_tokens, model, device, sae_cache, n_cells, out_dir):
    """Run 3-way ablation for all triplets."""

    results = []

    for ti, triplet in enumerate(triplets):
        fA = triplet['feature_A']
        fB = triplet['feature_B']
        fC = triplet['feature_C']

        triplet_id = f"L{fA['layer']}_F{fA['idx']}_x_L{fB['layer']}_F{fB['idx']}_x_L{fC['layer']}_F{fC['idx']}"
        triplet_path = os.path.join(out_dir, f"triplet_{triplet_id}.json")
        if os.path.exists(triplet_path):
            with open(triplet_path) as f:
                results.append(json.load(f))
            print(f"  [{ti+1}/{len(triplets)}] {triplet_id} — loaded from cache")
            continue

        print(f"\n  [{ti+1}/{len(triplets)}] {triplet_id} ({triplet['pathway']}, {triplet['triplet_type']})")
        print(f"    A: L{fA['layer']} F{fA['idx']} ({fA['label'][:40]})")
        print(f"    B: L{fB['layer']} F{fB['idx']} ({fB['label'][:40]})")
        print(f"    C: L{fC['layer']} F{fC['idx']} ({fC['label'][:40]})")

        t0 = time.time()

        # === 3 single ablations ===
        print(f"    [1/7] Ablating A only...")
        acc_A, n_A = ablate_single(fA['layer'], fA['idx'], all_tokens, model, device, sae_cache, n_cells)

        print(f"    [2/7] Ablating B only...")
        acc_B, n_B = ablate_single(fB['layer'], fB['idx'], all_tokens, model, device, sae_cache, n_cells)

        print(f"    [3/7] Ablating C only...")
        acc_C, n_C = ablate_single(fC['layer'], fC['idx'], all_tokens, model, device, sae_cache, n_cells)

        # === 3 pairwise ablations ===
        print(f"    [4/7] Ablating A+B...")
        acc_AB, n_AB = ablate_two(fA['layer'], fA['idx'], fB['layer'], fB['idx'],
                                   all_tokens, model, device, sae_cache, n_cells)

        print(f"    [5/7] Ablating A+C...")
        acc_AC, n_AC = ablate_two(fA['layer'], fA['idx'], fC['layer'], fC['idx'],
                                   all_tokens, model, device, sae_cache, n_cells)

        print(f"    [6/7] Ablating B+C...")
        acc_BC, n_BC = ablate_two(fB['layer'], fB['idx'], fC['layer'], fC['idx'],
                                   all_tokens, model, device, sae_cache, n_cells)

        # === 1 joint ablation ===
        print(f"    [7/7] Ablating A+B+C...")
        acc_ABC, n_ABC = ablate_three(fA['layer'], fA['idx'], fB['layer'], fB['idx'],
                                       fC['layer'], fC['idx'],
                                       all_tokens, model, device, sae_cache, n_cells)

        elapsed = time.time() - t0

        # Finalize all accumulators
        d_A, c_A = acc_A.finalize()
        d_B, c_B = acc_B.finalize()
        d_C, c_C = acc_C.finalize()
        d_AB, c_AB = acc_AB.finalize()
        d_AC, c_AC = acc_AC.finalize()
        d_BC, c_BC = acc_BC.finalize()
        d_ABC, c_ABC = acc_ABC.finalize()

        # Significance masks
        sig_A = (np.abs(d_A) > COHENS_D_THRESHOLD) & (c_A > CONSISTENCY_THRESHOLD)
        sig_B = (np.abs(d_B) > COHENS_D_THRESHOLD) & (c_B > CONSISTENCY_THRESHOLD)
        sig_C = (np.abs(d_C) > COHENS_D_THRESHOLD) & (c_C > CONSISTENCY_THRESHOLD)
        sig_AB = (np.abs(d_AB) > COHENS_D_THRESHOLD) & (c_AB > CONSISTENCY_THRESHOLD)
        sig_AC = (np.abs(d_AC) > COHENS_D_THRESHOLD) & (c_AC > CONSISTENCY_THRESHOLD)
        sig_BC = (np.abs(d_BC) > COHENS_D_THRESHOLD) & (c_BC > CONSISTENCY_THRESHOLD)
        sig_ABC = (np.abs(d_ABC) > COHENS_D_THRESHOLD) & (c_ABC > CONSISTENCY_THRESHOLD)

        sig_any = sig_A | sig_B | sig_C | sig_AB | sig_AC | sig_BC | sig_ABC

        if sig_any.sum() > 0:
            idx = np.where(sig_any)[0]
            abs_dA = np.abs(d_A[idx])
            abs_dB = np.abs(d_B[idx])
            abs_dC = np.abs(d_C[idx])
            abs_dAB = np.abs(d_AB[idx])
            abs_dAC = np.abs(d_AC[idx])
            abs_dBC = np.abs(d_BC[idx])
            abs_dABC = np.abs(d_ABC[idx])

            sum_singles = abs_dA + abs_dB + abs_dC

            # Pairwise ratios
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_AB = np.where((abs_dA + abs_dB) > 0.01,
                                    abs_dAB / (abs_dA + abs_dB), 1.0)
                ratio_AC = np.where((abs_dA + abs_dC) > 0.01,
                                    abs_dAC / (abs_dA + abs_dC), 1.0)
                ratio_BC = np.where((abs_dB + abs_dC) > 0.01,
                                    abs_dBC / (abs_dB + abs_dC), 1.0)
                ratio_ABC = np.where(sum_singles > 0.01,
                                     abs_dABC / sum_singles, 1.0)

            # Higher-order interaction (inclusion-exclusion)
            # I_ABC = d_ABC - d_AB - d_AC - d_BC + d_A + d_B + d_C
            interaction = abs_dABC - abs_dAB - abs_dAC - abs_dBC + abs_dA + abs_dB + abs_dC

            # Marginal contribution of C given AB
            marginal_C = abs_dABC - abs_dAB

            # Classifications for 3-way
            n_sub_3 = int((ratio_ABC < 0.8).sum())
            n_add_3 = int(((ratio_ABC >= 0.8) & (ratio_ABC <= 1.2)).sum())
            n_sup_3 = int((ratio_ABC > 1.2).sum())

            # Pairwise classifications
            n_sub_AB = int((ratio_AB < 0.8).sum())
            n_sub_AC = int((ratio_AC < 0.8).sum())
            n_sub_BC = int((ratio_BC < 0.8).sum())
        else:
            ratio_AB = ratio_AC = ratio_BC = ratio_ABC = np.array([1.0])
            interaction = marginal_C = np.array([0.0])
            n_sub_3 = n_add_3 = n_sup_3 = 0
            n_sub_AB = n_sub_AC = n_sub_BC = 0

        triplet_result = {
            'triplet_id': triplet_id,
            'feature_A': fA,
            'feature_B': fB,
            'feature_C': fC,
            'pathway': triplet['pathway'],
            'triplet_type': triplet['triplet_type'],
            'target_layer': TARGET_LAYER,
            'n_cells': n_cells,
            'n_active': {
                'A': n_A, 'B': n_B, 'C': n_C,
                'AB': n_AB, 'AC': n_AC, 'BC': n_BC,
                'ABC': n_ABC,
            },
            'n_significant': {
                'A': int(sig_A.sum()), 'B': int(sig_B.sum()), 'C': int(sig_C.sum()),
                'AB': int(sig_AB.sum()), 'AC': int(sig_AC.sum()), 'BC': int(sig_BC.sum()),
                'ABC': int(sig_ABC.sum()),
                'any': int(sig_any.sum()),
            },
            'pairwise_ratios': {
                'AB': float(np.median(ratio_AB)),
                'AC': float(np.median(ratio_AC)),
                'BC': float(np.median(ratio_BC)),
            },
            'threeway_ratio': float(np.median(ratio_ABC)),
            'n_subadditive_3way': n_sub_3,
            'n_additive_3way': n_add_3,
            'n_superadditive_3way': n_sup_3,
            'higher_order_interaction_median': float(np.median(interaction)),
            'marginal_C_given_AB_median': float(np.median(marginal_C)),
            'mean_abs_d': {
                'A': float(np.mean(np.abs(d_A))),
                'B': float(np.mean(np.abs(d_B))),
                'C': float(np.mean(np.abs(d_C))),
                'AB': float(np.mean(np.abs(d_AB))),
                'AC': float(np.mean(np.abs(d_AC))),
                'BC': float(np.mean(np.abs(d_BC))),
                'ABC': float(np.mean(np.abs(d_ABC))),
            },
            'elapsed_seconds': elapsed,
        }

        with open(triplet_path, 'w') as f:
            json.dump(triplet_result, f, indent=2, default=_json_default)

        results.append(triplet_result)

        print(f"    => 3-way ratio: {triplet_result['threeway_ratio']:.2f}")
        print(f"    => Pairwise: AB={triplet_result['pairwise_ratios']['AB']:.2f}, "
              f"AC={triplet_result['pairwise_ratios']['AC']:.2f}, "
              f"BC={triplet_result['pairwise_ratios']['BC']:.2f}")
        print(f"    => 3-way: Sub={n_sub_3} Add={n_add_3} Super={n_sup_3}")
        print(f"    => Higher-order interaction: {triplet_result['higher_order_interaction_median']:.3f}")
        print(f"    => Marginal C|AB: {triplet_result['marginal_C_given_AB_median']:.3f}")
        print(f"    => {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return results


def run_analysis(results, out_dir):
    """Aggregate and summarize results."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Higher-Order (3-Way) Ablation Results")
    print("=" * 70)

    same = [r for r in results if r['triplet_type'] == 'same_pathway']
    cross = [r for r in results if r['triplet_type'] == 'cross_pathway']

    for group_name, group in [("Same-Pathway", same), ("Cross-Pathway", cross)]:
        if not group:
            continue

        ratios_3 = [r['threeway_ratio'] for r in group]
        ratios_AB = [r['pairwise_ratios']['AB'] for r in group]
        interactions = [r['higher_order_interaction_median'] for r in group]
        marginals = [r['marginal_C_given_AB_median'] for r in group]

        sub3 = sum(r['n_subadditive_3way'] for r in group)
        add3 = sum(r['n_additive_3way'] for r in group)
        sup3 = sum(r['n_superadditive_3way'] for r in group)
        total = sub3 + add3 + sup3

        print(f"\n  {group_name} triplets ({len(group)}):")
        print(f"    3-way median ratio: {np.median(ratios_3):.3f}")
        print(f"    Pairwise median ratio (AB): {np.median(ratios_AB):.3f}")
        if total > 0:
            print(f"    Subadditive (redundant): {sub3} ({100*sub3/total:.0f}%)")
            print(f"    Additive (independent): {add3} ({100*add3/total:.0f}%)")
            print(f"    Superadditive (synergistic): {sup3} ({100*sup3/total:.0f}%)")
        print(f"    Higher-order interaction: {np.median(interactions):.4f}")
        print(f"    Marginal C|AB: {np.median(marginals):.4f}")

        for r in group:
            print(f"      {r['pathway']}: 3-way={r['threeway_ratio']:.2f}, "
                  f"interaction={r['higher_order_interaction_median']:.3f}")

    # Compare 3-way vs 2-way redundancy
    if same:
        ratios_3 = [r['threeway_ratio'] for r in same]
        ratios_2 = []
        for r in same:
            for k in ['AB', 'AC', 'BC']:
                ratios_2.append(r['pairwise_ratios'][k])

        print(f"\n  KEY COMPARISON:")
        print(f"    2-way median ratio: {np.median(ratios_2):.3f}")
        print(f"    3-way median ratio: {np.median(ratios_3):.3f}")
        print(f"    Deeper redundancy at 3-way: {np.median(ratios_3) < np.median(ratios_2)}")

    summary = {
        'n_triplets': len(results),
        'n_same_pathway': len(same),
        'n_cross_pathway': len(cross),
        'same_pathway_3way_ratio': float(np.median([r['threeway_ratio'] for r in same])) if same else None,
        'cross_pathway_3way_ratio': float(np.median([r['threeway_ratio'] for r in cross])) if cross else None,
        'total_elapsed_min': sum(r['elapsed_seconds'] for r in results) / 60,
        'per_triplet_results': results,
    }

    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\n  Summary saved to {os.path.join(out_dir, 'summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="Higher-Order (3-Way) Feature Ablation")
    parser.add_argument("--n-cells", type=int, default=200)
    parser.add_argument("--n-triplets", type=int, default=10)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Higher-Order (3-Way) Combinatorial Ablation")
    print(f"  Cells: {args.n_cells}")
    print(f"  Max triplets: {args.n_triplets}")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    triplets = select_triplets(args.n_triplets)
    print(f"\n  Selected {len(triplets)} triplets:")
    for i, t in enumerate(triplets):
        fA = t['feature_A']
        fB = t['feature_B']
        fC = t['feature_C']
        print(f"    [{i+1}] L{fA['layer']}_F{fA['idx']} x L{fB['layer']}_F{fB['idx']} "
              f"x L{fC['layer']}_F{fC['idx']} ({t['pathway']}, {t['triplet_type']})")

    if args.analysis_only:
        results = []
        for t in triplets:
            fA, fB, fC = t['feature_A'], t['feature_B'], t['feature_C']
            tid = f"L{fA['layer']}_F{fA['idx']}_x_L{fB['layer']}_F{fB['idx']}_x_L{fC['layer']}_F{fC['idx']}"
            tpath = os.path.join(OUT_DIR, f"triplet_{tid}.json")
            if os.path.exists(tpath):
                with open(tpath) as f:
                    results.append(json.load(f))
        run_analysis(results, OUT_DIR)
        return

    import torch
    from transformers import BertForMaskedLM

    print("\n  Loading Geneformer V2-316M...")
    t0 = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("    Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("    Using CPU")

    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, subfolder=MODEL_SUBFOLDER,
        output_hidden_states=True,
        output_attentions=False,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    print(f"    Model loaded in {time.time()-t0:.1f}s")

    all_tokens = load_and_tokenize_cells(args.n_cells)
    sae_cache = SAECache()

    total_t0 = time.time()
    results = run_experiment(triplets, all_tokens, model, device, sae_cache, args.n_cells, OUT_DIR)

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"Experiment complete. Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hrs)")
    print("=" * 70)

    run_analysis(results, OUT_DIR)


if __name__ == "__main__":
    main()
