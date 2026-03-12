#!/usr/bin/env python3
"""
Exhaustive Feature Tracing at Layer 5

Traces ALL 4,608 features at L5 (most trajectory-sensitive layer) instead of
the usual 30 well-annotated features. Produces the first complete layer-level
circuit map.

Features:
- Per-feature JSON checkpointing (resume-safe)
- Skips dead features (activation_freq < 0.001)
- Progress tracking with rolling ETA
- Periodic progress summaries

Usage:
    python src/exhaustive_feature_tracing.py \
        [--n-cells 200] [--source-layer 5] [--start-idx 0] [--end-idx 4608]
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
# Set DATA_ROOT to the directory containing your SAE models and extracted activations.
# Expected layout under DATA_ROOT:
#   sae_models/sae_layer{N}.pt        — trained SAE checkpoints (one per layer)
#   layer_{N}_activations.npy          — extracted Geneformer residual-stream activations
# Set DATA_PATH to the Replogle K562 CRISPRi .h5ad file.
# Set OUT_DIR to the desired output directory for results.
DATA_ROOT = os.environ.get("SAE_DATA_ROOT", "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_ROOT, "sae_models")
DATA_PATH = os.environ.get("SAE_DATA_PATH", "data/replogle_concat.h5ad")
OUT_DIR = os.environ.get("SAE_OUT_DIR", "results/exhaustive_tracing")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

EXPANSION = 4
K_VAL = 32
N_CTRL = 2000
MAX_SEQ_LEN = 2048
HIDDEN_DIM = 1152
N_FEATURES = 4608
N_LAYERS = 18

COHENS_D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7
TOP_EFFECTS = 50  # Store top N effects per downstream layer


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ============================================================
# Cell loading
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

    def preload(self, layers):
        for l in layers:
            self.get(l)


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
# Load feature catalog to identify dead features
# ============================================================
def load_feature_catalog(source_layer):
    """Load catalog and return set of active feature indices."""
    run_name = f"layer{source_layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)
    catalog_path = os.path.join(run_dir, "feature_catalog.json")

    with open(catalog_path) as f:
        catalog = json.load(f)

    active_features = set()
    feature_info = {}
    for feat in catalog['features']:
        fi = feat['feature_idx']
        freq = feat.get('activation_freq', 0)
        if freq >= 0.001:
            active_features.add(fi)
        feature_info[fi] = {
            'activation_freq': freq,
            'n_top_genes': len(feat.get('top_genes', [])),
        }

    return active_features, feature_info


# ============================================================
# Load feature annotations
# ============================================================
def load_feature_labels(source_layer):
    """Load best annotation label for each feature."""
    run_name = f"layer{source_layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)
    ann_path = os.path.join(run_dir, "feature_annotations.json")

    labels = {}
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            ann_data = json.load(f)
        for fid_str, anns in ann_data.get('feature_annotations', {}).items():
            fi = int(fid_str)
            for a in anns:
                if a['ontology'] in ('GO_BP', 'KEGG', 'Reactome'):
                    labels[fi] = a['term']
                    break
            if fi not in labels and anns:
                labels[fi] = anns[0]['term']

    return labels


# ============================================================
# Pre-cache clean forward passes (run ONCE for all features)
# ============================================================
def precompute_clean_cache(all_tokens, model, device, source_layer,
                           downstream_layers, sae_cache, n_cells):
    """Run all cells once, cache source hidden + SAE encode + downstream SAE activations.

    Returns list of per-cell dicts (or None for skipped cells):
        {
            'tokens': np.array,
            'gene_positions': np.array,
            'source_hidden': Tensor (seq_len, hidden_dim) on CPU,
            'h_sparse': Tensor (n_gene_pos, n_features) on CPU,
            'topk': Tensor (n_gene_pos, k) on CPU,
            'clean_sae': {dl: Tensor (n_gene_pos, n_features) on CPU},
        }
    """
    import torch

    source_sae, source_mean = sae_cache.get(source_layer)
    source_mean_t = torch.tensor(source_mean, dtype=torch.float32)

    dst_mean_tensors = {}
    for dl in downstream_layers:
        _, dm = sae_cache.get(dl)
        dst_mean_tensors[dl] = torch.tensor(dm, dtype=torch.float32)

    cache = []
    nc = min(n_cells, len(all_tokens))

    for ci in range(nc):
        tokens = all_tokens[ci]
        gene_mask = (tokens != 2) & (tokens != 3)
        gene_positions = np.where(gene_mask)[0]
        if len(gene_positions) == 0:
            cache.append(None)
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(1, len(tokens), dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Cache source layer hidden state (full sequence, for hook injection)
        source_hidden = outputs.hidden_states[source_layer + 1][0].cpu()

        # Source SAE encode (gene positions only)
        gene_hidden = source_hidden[gene_positions] - source_mean_t
        with torch.no_grad():
            h_sparse, topk = source_sae.encode(gene_hidden)

        # Downstream clean SAE activations (gene positions only)
        clean_sae = {}
        for dl in downstream_layers:
            dst_sae, _ = sae_cache.get(dl)
            dst_mean_t = dst_mean_tensors[dl]
            clean_dl = outputs.hidden_states[dl + 1][0][gene_positions].cpu()
            with torch.no_grad():
                clean_sp, _ = dst_sae.encode(clean_dl - dst_mean_t)
            clean_sae[dl] = clean_sp

        cache.append({
            'tokens': tokens,
            'gene_positions': gene_positions,
            'source_hidden': source_hidden,
            'h_sparse': h_sparse,
            'topk': topk,
            'clean_sae': clean_sae,
        })

        del outputs
        if device.type == 'mps':
            torch.mps.empty_cache()

        if (ci + 1) % 50 == 0:
            print(f"    Cached {ci+1}/{nc} cells")

    print(f"    Cached all {nc} cells")
    return cache


# ============================================================
# Trace one feature (using pre-cached clean data)
# ============================================================
def trace_feature(feature_idx, source_layer, clean_cache, model, device,
                  sae_cache, downstream_layers):
    """Ablate one feature and measure downstream effects at all layers.

    Uses pre-cached clean forward pass data — only runs the ablated pass.
    """
    import torch

    source_sae, source_mean = sae_cache.get(source_layer)
    source_mean_t = torch.tensor(source_mean, dtype=torch.float32)

    dst_mean_tensors = {}
    for dl in downstream_layers:
        _, dm = sae_cache.get(dl)
        dst_mean_tensors[dl] = torch.tensor(dm, dtype=torch.float32)

    accumulators = {dl: WelfordAccumulator(N_FEATURES) for dl in downstream_layers}
    n_active = 0

    for ci, cell in enumerate(clean_cache):
        if cell is None:
            continue

        tokens = cell['tokens']
        gene_positions = cell['gene_positions']
        source_hidden = cell['source_hidden']
        h_sparse = cell['h_sparse']
        topk = cell['topk']

        # Check if this feature is active in this cell
        active_mask = (topk == feature_idx).any(dim=1)
        if not active_mask.any():
            continue

        n_active += 1
        active_pos = torch.where(active_mask)[0]

        # Ablate: zero out this feature's activation
        h_abl = h_sparse.clone()
        h_abl[:, feature_idx] = 0.0

        with torch.no_grad():
            recon = source_sae.decode(h_sparse) + source_mean_t
            recon_abl = source_sae.decode(h_abl) + source_mean_t

        delta = recon_abl - recon
        modified = source_hidden.clone()
        modified[gene_positions] = source_hidden[gene_positions] + delta

        # Run ablated forward pass (the ONLY forward pass per cell per feature)
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(1, len(tokens), dtype=torch.long).to(device)

        hook = model.bert.encoder.layer[source_layer].register_forward_hook(
            make_hook(modified, device))
        with torch.no_grad():
            outputs_abl = model(input_ids=input_ids, attention_mask=attention_mask)
        hook.remove()

        # Compare ablated SAE activations against cached clean ones
        for dl in downstream_layers:
            dst_sae, _ = sae_cache.get(dl)
            dst_mean_t = dst_mean_tensors[dl]

            abl_dl = outputs_abl.hidden_states[dl + 1][0][gene_positions].cpu()
            with torch.no_grad():
                abl_sp, _ = dst_sae.encode(abl_dl - dst_mean_t)

            clean_sp = cell['clean_sae'][dl]
            pos_deltas = [(abl_sp[p] - clean_sp[p]).numpy().astype(np.float64) for p in active_pos]
            accumulators[dl].update(np.mean(pos_deltas, axis=0))

        del outputs_abl
        if device.type == 'mps':
            torch.mps.empty_cache()

    # Finalize and extract significant edges
    downstream_edges = {}
    total_significant = 0

    for dl in downstream_layers:
        d_vals, c_vals = accumulators[dl].finalize()
        sig_mask = (np.abs(d_vals) > COHENS_D_THRESHOLD) & (c_vals > CONSISTENCY_THRESHOLD)
        n_sig = int(sig_mask.sum())
        total_significant += n_sig

        if n_sig > 0:
            sig_idx = np.where(sig_mask)[0]
            abs_d = np.abs(d_vals[sig_idx])
            top_order = np.argsort(-abs_d)[:TOP_EFFECTS]
            top_idx = sig_idx[top_order]

            edges = []
            for ti in top_idx:
                edges.append({
                    'target': int(ti),
                    'd': float(d_vals[ti]),
                    'consistency': float(c_vals[ti]),
                })
            downstream_edges[str(dl)] = {
                'n_significant': n_sig,
                'top_effects': edges,
                'mean_abs_d': float(np.mean(abs_d)),
            }

    return {
        'n_cells_active': n_active,
        'n_cells_measured': len(clean_cache),
        'total_significant_edges': total_significant,
        'downstream_edges': downstream_edges,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Exhaustive Feature Tracing at Layer N")
    parser.add_argument("--n-cells", type=int, default=50,
                        help="Number of cells (default 50 for speed; original tracing used 200)")
    parser.add_argument("--source-layer", type=int, default=5)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=N_FEATURES)
    parser.add_argument("--downstream-layers", type=str, default="6,11,17",
                        help="Comma-separated downstream layers to measure (default: 6,11,17)")
    parser.add_argument("--all-downstream", action="store_true",
                        help="Measure all downstream layers (overrides --downstream-layers)")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()

    source_layer = args.source_layer
    if args.all_downstream:
        downstream_layers = list(range(source_layer + 1, N_LAYERS))
    else:
        downstream_layers = [int(x) for x in args.downstream_layers.split(',')]

    print("=" * 70)
    print(f"Phase 8, Direction 4: Exhaustive Feature Tracing at Layer {source_layer}")
    print(f"  Cells: {args.n_cells}")
    print(f"  Feature range: [{args.start_idx}, {args.end_idx})")
    print(f"  Downstream layers: {downstream_layers}")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load catalog to identify dead features
    active_features, feature_info = load_feature_catalog(source_layer)
    feature_labels = load_feature_labels(source_layer)
    n_dead = N_FEATURES - len(active_features)
    print(f"\n  Active features: {len(active_features)} / {N_FEATURES} ({n_dead} dead, skipped)")

    # Identify already-completed features
    completed = set()
    for fi in range(args.start_idx, args.end_idx):
        fpath = os.path.join(OUT_DIR, f"feature_{fi:04d}.json")
        if os.path.exists(fpath):
            completed.add(fi)

    to_trace = [fi for fi in range(args.start_idx, args.end_idx)
                if fi in active_features and fi not in completed]
    n_skip_dead = sum(1 for fi in range(args.start_idx, args.end_idx) if fi not in active_features)

    print(f"  Already completed: {len(completed)}")
    print(f"  Skipping (dead): {n_skip_dead}")
    print(f"  To trace: {len(to_trace)}")

    if args.analysis_only:
        # Load all completed results and summarize
        results = []
        for fi in range(args.start_idx, args.end_idx):
            fpath = os.path.join(OUT_DIR, f"feature_{fi:04d}.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    results.append(json.load(f))

        run_summary(results, OUT_DIR, source_layer, feature_labels)
        return

    if not to_trace:
        print("  Nothing to trace. Use --analysis-only to generate summary.")
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

    # Preload all downstream SAEs
    print("\n  Preloading downstream SAEs...")
    sae_cache.preload([source_layer] + downstream_layers)

    # Pre-cache clean forward passes (run ONCE, reuse for all features)
    print("\n  Pre-caching clean forward passes...")
    cache_t0 = time.time()
    clean_cache = precompute_clean_cache(all_tokens, model, device, source_layer,
                                         downstream_layers, sae_cache, args.n_cells)
    print(f"    Clean cache built in {time.time()-cache_t0:.1f}s")
    del all_tokens  # no longer needed
    gc.collect()

    # Rolling ETA tracker
    recent_times = []
    total_t0 = time.time()
    n_completed_this_run = 0

    for ti, fi in enumerate(to_trace):
        feat_path = os.path.join(OUT_DIR, f"feature_{fi:04d}.json")

        label = feature_labels.get(fi, 'unannotated')
        info = feature_info.get(fi, {})

        ft0 = time.time()

        result = trace_feature(fi, source_layer, clean_cache, model, device,
                               sae_cache, downstream_layers)

        elapsed = time.time() - ft0
        n_completed_this_run += 1

        # Build output
        output = {
            'source_layer': source_layer,
            'feature_idx': fi,
            'label': label,
            'activation_freq': info.get('activation_freq', 0),
            **result,
            'elapsed_seconds': elapsed,
        }

        with open(feat_path, 'w') as f:
            json.dump(output, f, indent=2, default=_json_default)

        # Rolling ETA
        recent_times.append(elapsed)
        if len(recent_times) > 50:
            recent_times.pop(0)
        avg_time = np.mean(recent_times)
        remaining = len(to_trace) - (ti + 1)
        eta_min = remaining * avg_time / 60

        edges = result['total_significant_edges']
        active = result['n_cells_active']

        if (ti + 1) % 10 == 0 or ti == 0:
            print(f"  [{ti+1}/{len(to_trace)}] F{fi:04d} ({label[:30]}): "
                  f"{edges} edges, {active}/{args.n_cells} active, "
                  f"{elapsed:.1f}s | ETA {eta_min:.0f} min")

        # Periodic progress summary
        if (n_completed_this_run) % 100 == 0:
            total_elapsed = time.time() - total_t0
            write_progress(OUT_DIR, source_layer, args.start_idx, args.end_idx,
                          n_completed_this_run + len(completed), total_elapsed)

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"Tracing complete. {n_completed_this_run} features in "
          f"{total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hrs)")
    print("=" * 70)

    # Final summary
    results = []
    for fi in range(args.start_idx, args.end_idx):
        fpath = os.path.join(OUT_DIR, f"feature_{fi:04d}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                results.append(json.load(f))

    run_summary(results, OUT_DIR, source_layer, feature_labels)


def write_progress(out_dir, source_layer, start_idx, end_idx, n_completed, elapsed):
    """Write a progress checkpoint."""
    progress = {
        'source_layer': source_layer,
        'range': [start_idx, end_idx],
        'n_completed': n_completed,
        'elapsed_hours': elapsed / 3600,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(out_dir, "progress.json"), 'w') as f:
        json.dump(progress, f, indent=2)


def run_summary(results, out_dir, source_layer, feature_labels):
    """Generate summary statistics from all traced features."""
    print("\n" + "=" * 70)
    print("SUMMARY: Exhaustive Feature Tracing")
    print("=" * 70)

    n_total = len(results)
    edge_counts = [r['total_significant_edges'] for r in results]
    active_counts = [r['n_cells_active'] for r in results]

    print(f"\n  Features traced: {n_total}")
    print(f"  Total edges: {sum(edge_counts):,}")
    print(f"  Mean edges/feature: {np.mean(edge_counts):.1f}")
    print(f"  Median edges/feature: {np.median(edge_counts):.1f}")
    print(f"  Max edges: {np.max(edge_counts)} (F{results[np.argmax(edge_counts)]['feature_idx']})")
    print(f"  Features with 0 edges: {sum(1 for e in edge_counts if e == 0)}")
    print(f"  Features with >100 edges: {sum(1 for e in edge_counts if e > 100)}")

    # Edge count distribution
    bins = [0, 1, 10, 50, 100, 500, 1000, 5000, 10000]
    for i in range(len(bins) - 1):
        count = sum(1 for e in edge_counts if bins[i] <= e < bins[i+1])
        print(f"    [{bins[i]:>5}, {bins[i+1]:>5}): {count}")
    count = sum(1 for e in edge_counts if e >= bins[-1])
    print(f"    [{bins[-1]:>5},    ∞): {count}")

    # Top hub features
    sorted_by_edges = sorted(results, key=lambda r: -r['total_significant_edges'])
    print(f"\n  Top 20 hub features:")
    for i, r in enumerate(sorted_by_edges[:20]):
        fi = r['feature_idx']
        label = r.get('label', feature_labels.get(fi, 'unannotated'))
        print(f"    {i+1:2d}. F{fi:04d} ({label[:40]}): {r['total_significant_edges']} edges, "
              f"active={r['n_cells_active']}")

    # Features with no activity
    inactive = [r for r in results if r['n_cells_active'] == 0]
    print(f"\n  Features with 0 active cells: {len(inactive)}")

    # Per-downstream-layer stats
    print(f"\n  Per-downstream-layer edge counts:")
    all_dl_keys = set()
    for r in results:
        all_dl_keys.update(r['downstream_edges'].keys())
    for dl in sorted(int(k) for k in all_dl_keys):
        dl_key = str(dl)
        counts = [r['downstream_edges'].get(dl_key, {}).get('n_significant', 0) for r in results]
        print(f"    L{dl}: total={sum(counts)}, mean={np.mean(counts):.1f}, "
              f"median={np.median(counts):.0f}")

    # Summary JSON
    summary = {
        'source_layer': source_layer,
        'n_features_traced': n_total,
        'total_edges': int(sum(edge_counts)),
        'mean_edges_per_feature': float(np.mean(edge_counts)),
        'median_edges_per_feature': float(np.median(edge_counts)),
        'max_edges': int(np.max(edge_counts)),
        'n_zero_edge_features': int(sum(1 for e in edge_counts if e == 0)),
        'top_20_hubs': [
            {
                'feature_idx': r['feature_idx'],
                'label': r.get('label', feature_labels.get(r['feature_idx'], 'unannotated')),
                'total_edges': r['total_significant_edges'],
                'n_cells_active': r['n_cells_active'],
            }
            for r in sorted_by_edges[:20]
        ],
    }

    with open(os.path.join(out_dir, "exhaustive_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\n  Summary saved to {os.path.join(out_dir, 'exhaustive_summary.json')}")


if __name__ == "__main__":
    main()
