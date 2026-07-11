#!/usr/bin/env python3
"""P1a — Re-run the exhaustive L5 trace, retaining per-cell delta vectors.

The original trace (src/20_exhaustive_feature_tracing.py) accumulated Cohen's d and
consistency with a Welford accumulator and discarded the per-cell effects.  Calibrating the
edge-calling criterion against an empirical null requires the per-cell deltas, so we re-run
the trace and store them.

Two changes relative to the original, both verified to be numerically equivalent:
  1. Instead of a full forward pass with a hook on encoder.layer[source], we inject the
     modified hidden state directly into encoder.layer[source+1] and run only the tail of the
     network.  Layers 0..source are deterministic given the tokens, so they are cached once.
  2. SAE encoding runs on the GPU rather than the CPU.

Outputs (per feature):
  deltas/feature_XXXX.npz   float16 (n_active_cells, 4608) per downstream layer
  json/feature_XXXX.json    same schema as the original trace (for validation)

Usage:
    python p1a_retrace_with_deltas.py --validate           # check vs original on 3 features
    python p1a_retrace_with_deltas.py --start-idx 0 --end-idx 4608
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np

BASE = os.environ.get("SAE_BASE_ROOT", "..")
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
ORIG_SCRIPT = os.path.join(PROJ_DIR, "src", "20_exhaustive_feature_tracing.py")
ORIG_OUT = os.path.join(PROJ_DIR, "experiments/phase8_exhaustive_tracing")
OUT_DIR = os.path.join(PROJ_DIR, "experiments/revision_plos/P1_retrace")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"
N_FEATURES = 4608
N_LAYERS = 18
SOURCE_LAYER = 5
DOWNSTREAM = [6, 11, 17]
N_CELLS = 20
COHENS_D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7
SEED = 42


def load_orig_module():
    spec = importlib.util.spec_from_file_location("orig_trace", ORIG_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(device):
    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained(
        MODEL_NAME, subfolder=MODEL_SUBFOLDER,
        output_hidden_states=True, output_attentions=False)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_tail(model, hidden, start_layer, capture):
    """Run encoder layers [start_layer, 18) on `hidden`; return captured layer outputs.

    `hidden` is (1, seq, d) and represents the output of layer start_layer-1.
    Captured value for layer l is the output of encoder.layer[l], i.e. hidden_states[l+1].
    """
    import torch
    out = {}
    h = hidden
    with torch.no_grad():
        for l in range(start_layer, N_LAYERS):
            h = model.bert.encoder.layer[l](h)[0]
            if l in capture:
                out[l] = h
    return out


def precompute_clean(all_tokens, model, device, sae_cache):
    """Cache, per cell: source hidden (layer-5 output), source SAE encoding, and clean
    downstream SAE activations."""
    import torch

    source_sae, source_mean = sae_cache.get(SOURCE_LAYER)
    source_sae.to(device)
    source_mean_t = torch.tensor(source_mean, dtype=torch.float32, device=device)

    dst_mean = {}
    for dl in DOWNSTREAM:
        dsae, dm = sae_cache.get(dl)
        dsae.to(device)
        dst_mean[dl] = torch.tensor(dm, dtype=torch.float32, device=device)

    cache = []
    for ci, tokens in enumerate(all_tokens):
        gene_mask = (tokens != 2) & (tokens != 3)
        gene_positions = np.where(gene_mask)[0]
        if len(gene_positions) == 0:
            cache.append(None)
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones(1, len(tokens), dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        gp = torch.tensor(gene_positions, dtype=torch.long, device=device)
        source_hidden = outputs.hidden_states[SOURCE_LAYER + 1][0].detach()

        with torch.no_grad():
            h_sparse, topk = source_sae.encode(source_hidden[gp] - source_mean_t)
            clean_sae = {}
            for dl in DOWNSTREAM:
                dsae, _ = sae_cache.get(dl)
                clean_dl = outputs.hidden_states[dl + 1][0][gp]
                clean_sp, _ = dsae.encode(clean_dl - dst_mean[dl])
                clean_sae[dl] = clean_sp

        cache.append({
            'gene_positions': gp,
            'source_hidden': source_hidden,
            'h_sparse': h_sparse,
            'topk': topk,
            'clean_sae': clean_sae,
        })
        del outputs
        if device.type == 'mps':
            torch.mps.empty_cache()
        if (ci + 1) % 5 == 0:
            print(f"    cached {ci+1}/{len(all_tokens)} cells")

    return cache, source_mean_t, dst_mean


def trace_feature(feature_idx, clean_cache, model, device, sae_cache,
                  source_mean_t, dst_mean):
    """Ablate one feature; return per-cell delta matrices per downstream layer."""
    import torch

    source_sae, _ = sae_cache.get(SOURCE_LAYER)
    per_cell = {dl: [] for dl in DOWNSTREAM}

    for cell in clean_cache:
        if cell is None:
            continue
        h_sparse = cell['h_sparse']
        active_mask = (cell['topk'] == feature_idx).any(dim=1)
        if not active_mask.any():
            continue
        active_pos = torch.where(active_mask)[0]

        h_abl = h_sparse.clone()
        h_abl[:, feature_idx] = 0.0
        with torch.no_grad():
            delta_h = source_sae.decode(h_abl) - source_sae.decode(h_sparse)

        modified = cell['source_hidden'].clone()
        modified[cell['gene_positions']] += delta_h

        outs = run_tail(model, modified.unsqueeze(0), SOURCE_LAYER + 1, set(DOWNSTREAM))

        for dl in DOWNSTREAM:
            dsae, _ = sae_cache.get(dl)
            with torch.no_grad():
                abl_sp, _ = dsae.encode(outs[dl][0][cell['gene_positions']] - dst_mean[dl])
            d = (abl_sp[active_pos] - cell['clean_sae'][dl][active_pos]).mean(dim=0)
            per_cell[dl].append(d.cpu().numpy().astype(np.float64))

        del outs, modified
        if device.type == 'mps':
            torch.mps.empty_cache()

    return {dl: (np.stack(v) if v else np.zeros((0, N_FEATURES))) for dl, v in per_cell.items()}


def stats_from_deltas(mat):
    """Cohen's d and consistency, matching the original Welford accumulator exactly."""
    n = mat.shape[0]
    if n < 2:
        return np.zeros(N_FEATURES), np.zeros(N_FEATURES)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1)
    d = mean / np.maximum(std, 1e-10)
    pos = (mat > 0).sum(axis=0)
    consistency = np.maximum(pos, n - pos) / n
    return d, consistency


def summarize(deltas):
    downstream_edges = {}
    total = 0
    for dl in DOWNSTREAM:
        d, c = stats_from_deltas(deltas[dl])
        sig = (np.abs(d) > COHENS_D_THRESHOLD) & (c > CONSISTENCY_THRESHOLD)
        n_sig = int(sig.sum())
        total += n_sig
        if n_sig:
            idx = np.where(sig)[0]
            order = np.argsort(-np.abs(d[idx]))[:50]
            top = [{'target': int(i), 'd': float(d[i]), 'consistency': float(c[i])}
                   for i in idx[order]]
            downstream_edges[str(dl)] = {
                'n_significant': n_sig, 'top_effects': top,
                'mean_abs_d': float(np.mean(np.abs(d[idx]))),
            }
    return downstream_edges, total


def priority_subset(active_features, n_top=200, n_strat=400, seed=0):
    """The feature set p1b needs: the 200 highest-degree features from the original trace plus
    a degree-stratified random sample of the rest.  Mirrors the selection in p1b."""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.
    degrees = {}
    for p in sorted(os.listdir(ORIG_OUT)):
        if not p.startswith('feature_') or not p.endswith('.json'):
            continue
        with open(os.path.join(ORIG_OUT, p)) as f:
            j = json.load(f)
        if j['feature_idx'] in active_features:
            degrees[j['feature_idx']] = j['total_significant_edges']
    fis = np.array(sorted(degrees))
    if not len(fis):
        return []
    order = np.argsort(-np.array([degrees[f] for f in fis]), kind='stable')
    rng = np.random.default_rng(seed)
    take = list(order[:n_top])
    rest = order[n_top:]
    if len(rest):
        strata = np.array_split(rest, min(8, len(rest)))
        per = max(1, n_strat // len(strata))
        for st in strata:
            take.extend(rng.choice(st, size=min(per, len(st)), replace=False))
    return [int(fis[i]) for i in dict.fromkeys(take)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-idx', type=int, default=0)
    ap.add_argument('--end-idx', type=int, default=N_FEATURES)
    ap.add_argument('--n-cells', type=int, default=N_CELLS)
    ap.add_argument('--priority-first', action='store_true',
                    help='trace the calibration subset before the remaining features')
    ap.add_argument('--validate', action='store_true',
                    help='trace 3 features and compare against the original run')
    args = ap.parse_args()

    import torch
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    os.makedirs(os.path.join(OUT_DIR, 'deltas'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'json'), exist_ok=True)

    orig = load_orig_module()
    np.random.seed(SEED)
    print("Loading cells...")
    all_tokens = orig.load_and_tokenize_cells(args.n_cells)[:args.n_cells]
    active_features, feature_info = orig.load_feature_catalog(SOURCE_LAYER)
    labels = orig.load_feature_labels(SOURCE_LAYER)
    print(f"  {len(all_tokens)} cells, {len(active_features)} active features")

    print("Loading model...")
    model = build_model(device)
    sae_cache = orig.SAECache()

    print("Pre-computing clean cache...")
    t0 = time.time()
    clean_cache, source_mean_t, dst_mean = precompute_clean(all_tokens, model, device, sae_cache)
    print(f"  clean cache in {time.time()-t0:.1f}s")

    if args.validate:
        ok = True
        for fi in [0, 11, 898]:
            deltas = trace_feature(fi, clean_cache, model, device, sae_cache,
                                   source_mean_t, dst_mean)
            edges, total = summarize(deltas)
            with open(os.path.join(ORIG_OUT, f'feature_{fi:04d}.json')) as f:
                ref = json.load(f)
            print(f"\nF{fi}: total edges new={total} orig={ref['total_significant_edges']}")
            for dl in DOWNSTREAM:
                sdl = str(dl)
                if sdl not in ref['downstream_edges']:
                    continue
                new_top = {e['target']: e['d'] for e in edges.get(sdl, {}).get('top_effects', [])}
                ref_top = {e['target']: e['d'] for e in ref['downstream_edges'][sdl]['top_effects']}
                shared = set(new_top) & set(ref_top)
                maxdiff = max((abs(new_top[t] - ref_top[t]) for t in shared), default=0.0)
                print(f"  L{dl}: n_sig new={edges.get(sdl,{}).get('n_significant',0)} "
                      f"orig={ref['downstream_edges'][sdl]['n_significant']}  "
                      f"top-50 overlap={len(shared)}/{len(ref_top)}  max|Δd|={maxdiff:.2e}")
                if maxdiff > 1e-4 or len(shared) < 0.9 * len(ref_top):
                    ok = False
        print("\nVALIDATION", "PASS" if ok else "FAIL")
        return

    todo = [fi for fi in range(args.start_idx, args.end_idx)
            if fi in active_features
            and not os.path.exists(os.path.join(OUT_DIR, 'deltas', f'feature_{fi:04d}.npz'))]

    if args.priority_first:
        # The permutation-FDR analysis needs a specific 600-feature subset (the 200
        # highest-degree features plus a degree-stratified sample).  Trace those first so the
        # calibration does not have to wait for the full 4,065-feature sweep to finish.
        prio = priority_subset(active_features)
        rank = {fi: i for i, fi in enumerate(prio)}
        todo.sort(key=lambda fi: (rank.get(fi, len(prio) + fi)))
        n_prio = sum(1 for fi in todo if fi in rank)
        print(f"  priority ordering: {n_prio} calibration-subset features first")
    print(f"To trace: {len(todo)}")

    t0 = time.time()
    for n, fi in enumerate(todo):
        deltas = trace_feature(fi, clean_cache, model, device, sae_cache,
                               source_mean_t, dst_mean)
        np.savez(os.path.join(OUT_DIR, 'deltas', f'feature_{fi:04d}.npz'),
                 **{f'dl{dl}': deltas[dl].astype(np.float16) for dl in DOWNSTREAM})
        edges, total = summarize(deltas)
        with open(os.path.join(OUT_DIR, 'json', f'feature_{fi:04d}.json'), 'w') as f:
            json.dump({
                'source_layer': SOURCE_LAYER, 'feature_idx': fi,
                'label': labels.get(fi, 'unannotated'),
                'activation_freq': feature_info[fi]['activation_freq'],
                'n_cells_active': int(deltas[DOWNSTREAM[0]].shape[0]),
                'n_cells_measured': len(clean_cache),
                'total_significant_edges': total,
                'downstream_edges': edges,
            }, f)
        if (n + 1) % 25 == 0:
            el = time.time() - t0
            rate = el / (n + 1)
            print(f"  {n+1}/{len(todo)}  {rate:.2f}s/feat  ETA {(len(todo)-n-1)*rate/3600:.2f}h")

    print(f"Done in {(time.time()-t0)/3600:.2f}h")


if __name__ == '__main__':
    main()
