#!/usr/bin/env python3
"""
Causal Trajectory Steering

Amplify the 14 switch features (identified in Phase 7 Step 3) in early-pseudotime
cells and measure whether the model's logit output shifts toward the late-pseudotime
(mature cell) gene signature.

Key metric: State Shift Score
  = cos(steered_logits, late_signature) - cos(steered_logits, early_signature)
  - [cos(clean_logits, late_signature) - cos(clean_logits, early_signature)]
  Positive = steering pushes cell toward mature state.

Usage:
    python src/trajectory_steering.py \
        [--alphas 2.0,5.0] [--n-cells 500]
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
# Set DATA_ROOT to the directory containing your SAE models.
# Expected layout under DATA_ROOT:
#   sae_models/sae_layer{N}.pt        — trained SAE checkpoints
# Set IMMUNE_H5AD to the Tabula Sapiens immune subset .h5ad file.
# Set TRAJ_DIR to the directory containing Phase 7 trajectory dynamics results.
DATA_ROOT = os.environ.get("SAE_DATA_ROOT", "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_ROOT, "sae_models")
IMMUNE_H5AD = os.environ.get("SAE_IMMUNE_H5AD", "data/tabula_sapiens_immune_subset_20000.h5ad")
TRAJ_DIR = os.environ.get("SAE_TRAJ_DIR", "experiments/phase7_trajectory_dynamics")
OUT_DIR = os.environ.get("SAE_OUT_DIR", "results/trajectory_steering")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

EXPANSION = 4
K_VAL = 32
MAX_SEQ_LEN = 2048
HIDDEN_DIM = 1152
N_FEATURES = 4608
N_LAYERS = 18

COHENS_D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7

# Hematopoietic tissues
HEMATOPOIETIC_TISSUES = ['Bone_Marrow', 'Blood', 'Spleen']

MYELOID_TYPES = [
    'hematopoietic stem cell', 'hematopoietic precursor cell',
    'common myeloid progenitor',
    'monocyte', 'classical monocyte', 'intermediate monocyte', 'non-classical monocyte',
    'macrophage', 'tissue-resident macrophage',
    'neutrophil', 'granulocyte', 'mast cell', 'basophil',
]
LYMPHOID_TYPES = [
    'hematopoietic stem cell', 'hematopoietic precursor cell',
    'thymocyte', 'cd4-positive, alpha-beta thymocyte', 'cd8-positive, alpha-beta thymocyte',
    'cd4-positive, alpha-beta t cell', 'naive thymus-derived cd4-positive, alpha-beta t cell',
    'cd8-positive, alpha-beta t cell', 'naive thymus-derived cd8-positive, alpha-beta t cell',
    'b cell', 'plasma cell', 'natural killer cell', 'mature nk t cell',
]

# The 14 switch features from Phase 7 Step 3
SWITCH_FEATURES = [
    {'layer': 0, 'idx': 2061, 'd': 1.33, 'label': 'Chromatin Organization'},
    {'layer': 0, 'idx': 25, 'd': 1.13, 'label': 'Mitotic Spindle Microtubule Attachment'},
    {'layer': 0, 'idx': 3567, 'd': 1.07, 'label': 'unannotated'},
    {'layer': 0, 'idx': 1483, 'd': 1.07, 'label': 'L-amino Acid Transport'},
    {'layer': 0, 'idx': 1295, 'd': 1.01, 'label': 'Neurodegeneration Pathways'},
    {'layer': 5, 'idx': 4349, 'd': 0.95, 'label': 'Pre-mRNA Processing'},
    {'layer': 5, 'idx': 2016, 'd': 0.90, 'label': 'unannotated'},
    {'layer': 11, 'idx': 854, 'd': 1.15, 'label': 'RNA Catabolic Process'},
    {'layer': 11, 'idx': 4350, 'd': 1.09, 'label': 'unannotated'},
    {'layer': 11, 'idx': 2174, 'd': 1.01, 'label': 'unannotated'},
    {'layer': 11, 'idx': 3750, 'd': 1.00, 'label': 'unannotated'},
    {'layer': 17, 'idx': 3730, 'd': 1.25, 'label': 'Protein-RNA Complex Assembly'},
    {'layer': 17, 'idx': 1607, 'd': 1.09, 'label': 'unannotated'},
    {'layer': 17, 'idx': 3567, 'd': 1.09, 'label': 'unannotated'},
]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ============================================================
# Cell loading with metadata + pseudotime
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


def load_sparse_row(f_group, row_idx, n_cols):
    indptr = f_group['indptr']
    start = int(indptr[row_idx])
    end = int(indptr[row_idx + 1])
    indices = f_group['indices'][start:end]
    data = f_group['data'][start:end]
    row = np.zeros(n_cols, dtype=np.float32)
    row[indices] = data
    return row


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


def load_cells_and_pseudotime(n_cells):
    """Load TS immune cells with pseudotime, reusing Phase 7 cache if available."""
    metadata_cache = os.path.join(TRAJ_DIR, "cell_metadata.json")

    # Try to load cached pseudotime from Phase 7
    if os.path.exists(metadata_cache):
        print("  Loading cached metadata from Phase 7...")
        with open(metadata_cache) as f:
            meta = json.load(f)
        cached_pseudotime = np.array(meta['pseudotime'])
        cached_cell_types = meta['cell_types']
        cached_tissues = meta['tissues']
        print(f"    Cached: {len(cached_cell_types)} cells with pseudotime")
    else:
        cached_pseudotime = None

    # Load and tokenize cells
    import h5py
    import scanpy as sc

    print("  Loading tokenizer dictionaries...")
    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    print("  Loading Tabula Sapiens immune cells...")
    with h5py.File(IMMUNE_H5AD, 'r') as f:
        var_genes = load_categorical_column(f['var'], '_index')
        n_genes_total = len(var_genes)
        n_cells_total = f['X']['indptr'].shape[0] - 1

        cell_types = load_categorical_column(f['obs'], 'cell_type')
        tissues = load_categorical_column(f['obs'], 'tissue_in_publication')

    # Filter
    tissue_mask = np.zeros(n_cells_total, dtype=bool)
    for t in HEMATOPOIETIC_TISSUES:
        tissue_mask |= (tissues == t)

    all_types = set(ct.lower() for ct in MYELOID_TYPES + LYMPHOID_TYPES)
    type_mask = np.array([ct.lower() in all_types for ct in cell_types])
    combined_mask = tissue_mask & type_mask
    valid_indices = np.where(combined_mask)[0]
    print(f"    {len(valid_indices)} cells in hematopoietic tissues")

    rng = np.random.RandomState(42)
    if len(valid_indices) > n_cells:
        valid_indices = rng.choice(valid_indices, n_cells, replace=False)
        valid_indices.sort()

    # Load expression
    print(f"    Loading expression for {len(valid_indices)} cells...")
    with h5py.File(IMMUNE_H5AD, 'r') as f:
        X_data = np.empty((len(valid_indices), n_genes_total), dtype=np.float32)
        for ci, idx in enumerate(valid_indices):
            X_data[ci, :] = load_sparse_row(f['X'], int(idx), n_genes_total)

    cell_type_arr = cell_types[valid_indices]
    tissue_arr = tissues[valid_indices]

    # Build tokenization arrays
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

    # Normalize
    row_sums = X_data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_norm = np.log1p(X_data / row_sums * 1e4)

    # Tokenize
    all_tokens = []
    all_ct = []
    all_ts = []
    valid_ci = []

    for ci in range(len(valid_indices)):
        tokens = tokenize_cell(X_norm[ci], mapped_var_indices,
                               mapped_token_ids, mapped_medians, MAX_SEQ_LEN)
        if tokens is not None:
            all_tokens.append(tokens)
            all_ct.append(cell_type_arr[ci])
            all_ts.append(tissue_arr[ci])
            valid_ci.append(ci)

    print(f"    Tokenized {len(all_tokens)} cells")

    # Compute pseudotime if not cached (same procedure as Phase 7)
    if cached_pseudotime is not None and len(cached_pseudotime) == len(all_tokens):
        pseudotime = cached_pseudotime
        print(f"    Using cached pseudotime")
    else:
        print("  Computing diffusion pseudotime...")
        import anndata
        X_valid = X_norm[valid_ci]
        adata = anndata.AnnData(X=X_valid)
        adata.obs['cell_type'] = all_ct
        adata.obs['tissue'] = all_ts

        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
        sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
        sc.tl.diffmap(adata, n_comps=10)

        hsc_mask = np.array([ct.lower() == 'hematopoietic stem cell' for ct in all_ct])
        bm_mask = np.array([t == 'Bone_Marrow' for t in all_ts])
        root_candidates = np.where(hsc_mask & bm_mask)[0]
        if len(root_candidates) == 0:
            root_candidates = np.where(hsc_mask)[0]

        if len(root_candidates) > 0:
            adata.uns['iroot'] = int(root_candidates[0])
            sc.tl.dpt(adata)
            pseudotime = adata.obs['dpt_pseudotime'].values
        else:
            sc.tl.umap(adata)
            pseudotime = adata.obsm['X_umap'][:, 0]
            pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())

        del adata
        gc.collect()
        print(f"    Pseudotime range: {np.nanmin(pseudotime):.3f} - {np.nanmax(pseudotime):.3f}")

    del X_data, X_norm
    gc.collect()

    return all_tokens, all_ct, all_ts, pseudotime, token_dict, gene_name_id_dict


# ============================================================
# Core classes
# ============================================================
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
# State signatures and shift metric
# ============================================================
def compute_state_signatures(all_tokens, pseudotime, model, device):
    """Compute mean logit profiles for early and late pseudotime cells."""
    import torch

    # Split into tertiles
    valid_mask = ~np.isnan(pseudotime)
    pt_valid = pseudotime[valid_mask]
    t1 = np.percentile(pt_valid, 33.3)
    t2 = np.percentile(pt_valid, 66.7)

    early_idx = np.where(valid_mask & (pseudotime <= t1))[0]
    late_idx = np.where(valid_mask & (pseudotime >= t2))[0]

    print(f"  Computing state signatures: {len(early_idx)} early, {len(late_idx)} late cells")

    vocab_size = model.config.vocab_size

    def compute_mean_logits(cell_indices, label):
        """Mean logit vector over cell_indices."""
        acc = np.zeros(vocab_size, dtype=np.float64)
        n = 0
        for ci in cell_indices:
            tokens = all_tokens[ci]
            gene_mask = (tokens != 2) & (tokens != 3)
            gene_positions = np.where(gene_mask)[0]
            if len(gene_positions) == 0:
                continue

            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.ones(1, len(tokens), dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits[0][gene_positions].cpu().numpy().astype(np.float64)
            acc += logits.mean(axis=0)
            n += 1

            del outputs
            if device.type == 'mps':
                torch.mps.empty_cache()

        if n > 0:
            acc /= n
        print(f"    {label}: {n} cells processed")
        return acc

    early_sig = compute_mean_logits(early_idx, "Early")
    late_sig = compute_mean_logits(late_idx, "Late")

    return early_sig, late_sig, early_idx, late_idx


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.sqrt(np.sum(a**2))
    norm_b = np.sqrt(np.sum(b**2))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.sum(a * b) / (norm_a * norm_b))


# ============================================================
# Steering loop for one feature
# ============================================================
def steer_feature(feature, alphas, early_cells, all_tokens, model, device,
                  sae_cache, early_sig, late_sig, token_dict, gene_name_id_dict,
                  out_dir):
    """Steer one switch feature in early-pseudotime cells and measure state shift."""
    import torch

    layer = feature['layer']
    fi = feature['idx']
    feat_id = f"F{fi}_L{layer}"

    # Check cache
    result_path = os.path.join(out_dir, f"steering_{feat_id}.json")
    if os.path.exists(result_path):
        print(f"    {feat_id} — loaded from cache")
        with open(result_path) as f:
            return json.load(f)

    source_sae, source_mean = sae_cache.get(layer)
    source_mean_t = torch.tensor(source_mean, dtype=torch.float32)

    vocab_size = model.config.vocab_size

    # Build reverse token dict for gene name lookup
    rev_token_dict = {}
    for gene_name, ensembl in gene_name_id_dict.items():
        if ensembl in token_dict:
            rev_token_dict[token_dict[ensembl]] = gene_name

    result = {
        'feature_idx': fi,
        'layer': layer,
        'label': feature['label'],
        'switch_d': feature['d'],
        'n_early_cells': len(early_cells),
        'alpha_results': {},
    }

    for alpha in alphas:
        print(f"      alpha={alpha}x...")
        state_shifts = []
        logit_deltas_acc = np.zeros(vocab_size, dtype=np.float64)
        n_steered = 0
        n_active = 0

        for ci_idx, ci in enumerate(early_cells):
            tokens = all_tokens[ci]
            gene_mask = (tokens != 2) & (tokens != 3)
            gene_positions = np.where(gene_mask)[0]
            if len(gene_positions) == 0:
                continue

            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.ones(1, len(tokens), dtype=torch.long).to(device)

            with torch.no_grad():
                outputs_clean = model(input_ids=input_ids, attention_mask=attention_mask)

            clean_hidden = outputs_clean.hidden_states[layer + 1][0].cpu()
            gene_hidden = clean_hidden[gene_positions] - source_mean_t

            with torch.no_grad():
                h_sparse, topk = source_sae.encode(gene_hidden)

            active_mask = (topk == fi).any(dim=1)
            if not active_mask.any():
                del outputs_clean, clean_hidden
                continue

            n_active += 1

            # Amplify
            h_steered = h_sparse.clone()
            h_steered[:, fi] = alpha * h_sparse[:, fi]

            with torch.no_grad():
                recon = source_sae.decode(h_sparse) + source_mean_t
                recon_steered = source_sae.decode(h_steered) + source_mean_t

            delta = recon_steered - recon
            modified = clean_hidden.clone()
            modified[gene_positions] = clean_hidden[gene_positions] + delta

            hook = model.bert.encoder.layer[layer].register_forward_hook(
                make_hook(modified, device))
            with torch.no_grad():
                outputs_steered = model(input_ids=input_ids, attention_mask=attention_mask)
            hook.remove()

            # Get logits
            clean_logits = outputs_clean.logits[0][gene_positions].cpu().numpy().astype(np.float64)
            steered_logits = outputs_steered.logits[0][gene_positions].cpu().numpy().astype(np.float64)

            clean_mean_logits = clean_logits.mean(axis=0)
            steered_mean_logits = steered_logits.mean(axis=0)

            # State shift score
            cos_steered_late = cosine_sim(steered_mean_logits, late_sig)
            cos_steered_early = cosine_sim(steered_mean_logits, early_sig)
            cos_clean_late = cosine_sim(clean_mean_logits, late_sig)
            cos_clean_early = cosine_sim(clean_mean_logits, early_sig)

            shift = (cos_steered_late - cos_steered_early) - (cos_clean_late - cos_clean_early)
            state_shifts.append(shift)

            # Accumulate logit deltas
            logit_delta = steered_mean_logits - clean_mean_logits
            logit_deltas_acc += logit_delta
            n_steered += 1

            del outputs_clean, outputs_steered, clean_hidden, modified
            if device.type == 'mps':
                torch.mps.empty_cache()

            if (ci_idx + 1) % 50 == 0:
                print(f"        Cell {ci_idx+1}/{len(early_cells)}, "
                      f"active={n_active}, steered={n_steered}")

        if n_steered > 0:
            mean_logit_delta = logit_deltas_acc / n_steered

            # Top up/down genes
            sorted_up = np.argsort(-mean_logit_delta)
            sorted_down = np.argsort(mean_logit_delta)

            top_up = []
            for tok_id in sorted_up[:20]:
                gene = rev_token_dict.get(int(tok_id), f"token_{tok_id}")
                top_up.append({'gene': gene, 'delta': float(mean_logit_delta[tok_id])})

            top_down = []
            for tok_id in sorted_down[:20]:
                gene = rev_token_dict.get(int(tok_id), f"token_{tok_id}")
                top_down.append({'gene': gene, 'delta': float(mean_logit_delta[tok_id])})

            shifts_arr = np.array(state_shifts)
            alpha_result = {
                'n_cells_steered': n_steered,
                'n_cells_active': n_active,
                'mean_state_shift': float(np.mean(shifts_arr)),
                'median_state_shift': float(np.median(shifts_arr)),
                'std_state_shift': float(np.std(shifts_arr)),
                'frac_positive_shift': float((shifts_arr > 0).mean()),
                'top_upregulated': top_up,
                'top_downregulated': top_down,
                'mean_abs_logit_delta': float(np.mean(np.abs(mean_logit_delta))),
                'max_logit_delta': float(np.max(mean_logit_delta)),
                'min_logit_delta': float(np.min(mean_logit_delta)),
            }
        else:
            alpha_result = {
                'n_cells_steered': 0,
                'n_cells_active': 0,
                'mean_state_shift': 0.0,
                'median_state_shift': 0.0,
                'std_state_shift': 0.0,
                'frac_positive_shift': 0.0,
                'top_upregulated': [],
                'top_downregulated': [],
            }

        result['alpha_results'][str(alpha)] = alpha_result

    # Save
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)

    return result


# ============================================================
# Main
# ============================================================
def run_analysis(results, out_dir):
    """Summarize trajectory steering results."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Causal Trajectory Steering Results")
    print("=" * 70)

    for alpha_key in ['2.0', '5.0']:
        shifts = []
        frac_pos = []
        feature_labels = []

        for r in results:
            if alpha_key in r.get('alpha_results', {}):
                ar = r['alpha_results'][alpha_key]
                if ar['n_cells_steered'] > 0:
                    shifts.append(ar['mean_state_shift'])
                    frac_pos.append(ar['frac_positive_shift'])
                    feature_labels.append(f"L{r['layer']}_F{r['feature_idx']} ({r['label'][:30]})")

        if shifts:
            print(f"\n  Alpha = {alpha_key}x ({len(shifts)} features with data):")
            print(f"    Mean state shift: {np.mean(shifts):.4f}")
            print(f"    Median state shift: {np.median(shifts):.4f}")
            print(f"    Frac positive: {np.mean(frac_pos):.2f}")
            print(f"    Range: [{np.min(shifts):.4f}, {np.max(shifts):.4f}]")

            # Per-feature
            sorted_idx = np.argsort(-np.array(shifts))
            print(f"\n    Per-feature (sorted by shift):")
            for i in sorted_idx:
                print(f"      {feature_labels[i]}: shift={shifts[i]:.4f}, "
                      f"frac_pos={frac_pos[i]:.2f}")

    # Summary
    summary = {
        'n_features': len(results),
        'features': [
            {
                'feature_idx': r['feature_idx'],
                'layer': r['layer'],
                'label': r['label'],
                'switch_d': r['switch_d'],
                'alpha_results': r.get('alpha_results', {}),
            }
            for r in results
        ],
    }

    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\n  Summary saved to {os.path.join(out_dir, 'summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="Causal Trajectory Steering")
    parser.add_argument("--alphas", type=str, default="2.0,5.0",
                        help="Amplification factors (comma-separated)")
    parser.add_argument("--n-cells", type=int, default=500,
                        help="Max cells to load from Tabula Sapiens")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(',')]

    print("=" * 70)
    print("Causal Trajectory Steering")
    print(f"  Alphas: {alphas}")
    print(f"  Max cells: {args.n_cells}")
    print(f"  Switch features: {len(SWITCH_FEATURES)}")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.analysis_only:
        results = []
        for sf in SWITCH_FEATURES:
            feat_id = f"F{sf['idx']}_L{sf['layer']}"
            rpath = os.path.join(OUT_DIR, f"steering_{feat_id}.json")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    results.append(json.load(f))
        run_analysis(results, OUT_DIR)
        return

    import torch
    from transformers import BertForMaskedLM

    # Load cells
    all_tokens, cell_types, tissues, pseudotime, token_dict, gene_name_id_dict = \
        load_cells_and_pseudotime(args.n_cells)

    # Identify early cells (bottom tertile)
    valid_mask = ~np.isnan(pseudotime)
    pt_valid = pseudotime[valid_mask]
    t1 = np.percentile(pt_valid, 33.3)
    early_cells = np.where(valid_mask & (pseudotime <= t1))[0]
    print(f"\n  Early cells (pseudotime <= {t1:.3f}): {len(early_cells)}")

    # Load model
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

    sae_cache = SAECache()

    # Compute state signatures
    print("\n  Computing state signatures...")
    early_sig, late_sig, _, _ = compute_state_signatures(
        all_tokens, pseudotime, model, device)

    # Save signatures
    sig_path = os.path.join(OUT_DIR, "state_signatures.npz")
    np.savez_compressed(sig_path, early=early_sig, late=late_sig)
    print(f"    Signatures saved to {sig_path}")

    # Steer each switch feature
    total_t0 = time.time()
    results = []

    for si, sf in enumerate(SWITCH_FEATURES):
        print(f"\n  [{si+1}/{len(SWITCH_FEATURES)}] L{sf['layer']} F{sf['idx']} "
              f"(d={sf['d']:.2f}, {sf['label']})")

        result = steer_feature(
            sf, alphas, early_cells, all_tokens, model, device,
            sae_cache, early_sig, late_sig, token_dict, gene_name_id_dict,
            OUT_DIR)
        results.append(result)

        for alpha_key in [str(a) for a in alphas]:
            if alpha_key in result.get('alpha_results', {}):
                ar = result['alpha_results'][alpha_key]
                print(f"    alpha={alpha_key}: shift={ar.get('mean_state_shift', 0):.4f}, "
                      f"frac_pos={ar.get('frac_positive_shift', 0):.2f}, "
                      f"n_steered={ar.get('n_cells_steered', 0)}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"Complete. Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hrs)")
    print("=" * 70)

    run_analysis(results, OUT_DIR)


if __name__ == "__main__":
    main()
