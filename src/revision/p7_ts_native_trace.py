#!/usr/bin/env python3
"""P7 — Is the hub architecture a property of Geneformer, or of the K562-trained SAE?

The published cross-cell-line check applies K562-trained SAE features to Tabula Sapiens immune
cells and finds Spearman rho = 0.11 between hub rankings, with 34% of K562 hubs never firing on
TS cells.  That is uninterpretable: a feature that never fires cannot be ablated, so its degree
is mechanically zero.  It confounds three explanations the reviewers ask us to separate:
  (a) Geneformer's circuit architecture is cell-type-specific;
  (b) the K562-trained SAE's feature basis does not transfer to immune cells;
  (c) hub structure is a general property of the model.

The discriminating experiment is to trace with an SAE trained *natively* on Tabula Sapiens
immune activations (available at L0/L5/L11/L17) and ask whether the degree distribution and the
annotation-centrality relationship reappear, even though the feature identities do not.

Source L5, downstream {11, 17} (the TS-trained SAEs available).  The K562 trace is recomputed
on the same downstream layers for a like-for-like comparison.
"""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# stdout left as-is; run with python -u for unbuffered output

import numpy as np

BASE = os.environ.get("SAE_BASE_ROOT", "..")
PROJ = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
TRACE = os.path.join(PROJ, "src", "20_exhaustive_feature_tracing.py")
STEER = os.path.join(PROJ, "src", "19_causal_trajectory_steering.py")
TS_SAE = os.path.join(PROJ, "experiments/phase3_multitissue/sae_models")
EXH = os.path.join(PROJ, "experiments/phase8_exhaustive_tracing")
OUT = os.path.join(PROJ, "experiments/revision_plos/P7_ts_native")

SOURCE = 5
DOWNSTREAM = [11, 17]
N_FEATURES = 4608
N_CELLS = 20
N_TRACE = 400
D_THRESH, C_THRESH = 0.5, 0.7
SEED = 11


def load_mod(path, name):
    # The helper scripts reopen sys.stdout on fd 1 at import time; chaining two such imports
    # lets the garbage-collected first wrapper close fd 1, after which our own prints raise
    # EBADF.  Strip that line before executing the module source.
    import types
    src = open(path).read()
    src = '\n'.join(l for l in src.split('\n')
                    if 'sys.stdout = os.fdopen' not in l)
    m = types.ModuleType(name)
    m.__file__ = path
    exec(compile(src, path, 'exec'), m.__dict__)
    return m


class TSSAECache:
    def __init__(self, device):
        self._s, self._m = {}, {}
        self.device = device

    def get(self, layer):
        if layer not in self._s:
            import torch
            sys.path.insert(0, os.path.join(PROJ, "src"))
            from sae_model import TopKSAE
            run = os.path.join(TS_SAE, f"layer{layer:02d}_x4_k32")
            sae = TopKSAE.load(os.path.join(run, "sae_final.pt"), device='cpu')
            sae.eval().to(self.device)
            self._s[layer] = sae
            self._m[layer] = np.load(os.path.join(run, "activation_mean.npy"))
            print(f"    loaded TS SAE layer {layer}")
        return self._s[layer], self._m[layer]


def ts_catalog(layer):
    run = os.path.join(TS_SAE, f"layer{layer:02d}_x4_k32")
    with open(os.path.join(run, 'feature_catalog.json')) as f:
        cat = json.load(f)
    active, info = set(), {}
    for feat in cat['features']:
        fi = feat['feature_idx']
        fr = feat.get('activation_freq', 0)
        info[fi] = {'activation_freq': fr}
        if fr >= 0.001:
            active.add(fi)
    labels = {}
    ann = os.path.join(run, 'feature_annotations.json')
    if os.path.exists(ann):
        with open(ann) as f:
            a = json.load(f)
        for k, v in a.get('feature_annotations', {}).items():
            if v:
                labels[int(k)] = v[0]['term']
    return active, info, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-trace', type=int, default=N_TRACE)
    ap.add_argument('--analyse-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'json'), exist_ok=True)

    trace_mod = load_mod(TRACE, 'trace_mod')
    steer_mod = load_mod(STEER, 'steer_mod')
    rng = np.random.default_rng(SEED)

    active, info, labels = ts_catalog(SOURCE)
    print(f"  TS L5 SAE: {len(active)} active features, "
          f"{sum(1 for f in active if f in labels)} annotated")

    fis = np.array(sorted(active))
    sample = np.sort(rng.choice(fis, min(args.n_trace, len(fis)), replace=False))

    if not args.analyse_only:
        import torch
        from transformers import BertForMaskedLM
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # TS immune cells (the same population the TS SAE was trained to describe)
        # request the same 500-cell draw the steering experiment used, so the cached
        # pseudotime matches and no DPT recomputation is triggered; trace on the first N_CELLS
        toks, cts, tissues, pt, _, _ = steer_mod.load_cells_and_pseudotime(500)
        toks = toks[:N_CELLS]
        print(f"  {len(toks)} TS immune cells")

        model = BertForMaskedLM.from_pretrained(trace_mod.MODEL_NAME,
                                                subfolder=trace_mod.MODEL_SUBFOLDER,
                                                output_hidden_states=True)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        sc = TSSAECache(device)
        src_sae, src_mean = sc.get(SOURCE)
        src_mean_t = torch.tensor(src_mean, dtype=torch.float32, device=device)
        dst_mean = {}
        for dl in DOWNSTREAM:
            _, dm = sc.get(dl)
            dst_mean[dl] = torch.tensor(dm, dtype=torch.float32, device=device)

        cache = []
        for t in toks:
            gp_np = np.where((t != 2) & (t != 3))[0]
            ids = torch.tensor(t, dtype=torch.long, device=device).unsqueeze(0)
            am = torch.ones(1, len(t), dtype=torch.long, device=device)
            gp = torch.tensor(gp_np, dtype=torch.long, device=device)
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=am)
                hidden = o.hidden_states[SOURCE + 1][0].clone()
                h_sp, topk = src_sae.encode(hidden[gp] - src_mean_t)
                clean = {}
                for dl in DOWNSTREAM:
                    dsae, _ = sc.get(dl)
                    csp, _ = dsae.encode(o.hidden_states[dl + 1][0][gp] - dst_mean[dl])
                    clean[dl] = csp
            cache.append({'gp': gp, 'hidden': hidden, 'h_sp': h_sp, 'topk': topk, 'clean': clean})
            del o
            if device.type == 'mps':
                torch.mps.empty_cache()
        print("  clean cache built")

        t0 = time.time()
        done = 0
        for fi in sample:
            path = os.path.join(OUT, 'json', f'feature_{int(fi):04d}.json')
            if os.path.exists(path):
                continue
            per = {dl: [] for dl in DOWNSTREAM}
            for cell in cache:
                m = (cell['topk'] == int(fi)).any(dim=1)
                if not m.any():
                    continue
                pos = torch.where(m)[0]
                h_ab = cell['h_sp'].clone()
                h_ab[:, int(fi)] = 0.0
                with torch.no_grad():
                    dh = src_sae.decode(h_ab) - src_sae.decode(cell['h_sp'])
                mod = cell['hidden'].clone()
                mod[cell['gp']] += dh
                h = mod.unsqueeze(0)
                with torch.no_grad():
                    for l in range(SOURCE + 1, 18):
                        h = model.bert.encoder.layer[l](h)[0]
                        if l in DOWNSTREAM:
                            dsae, _ = sc.get(l)
                            asp, _ = dsae.encode(h[0][cell['gp']] - dst_mean[l])
                            per[l].append((asp[pos] - cell['clean'][l][pos])
                                          .mean(dim=0).cpu().numpy().astype(np.float64))
                del h, mod
                if device.type == 'mps':
                    torch.mps.empty_cache()

            edges, total = {}, 0
            for dl in DOWNSTREAM:
                if len(per[dl]) < 2:
                    continue
                mat = np.stack(per[dl])
                d = mat.mean(0) / np.maximum(mat.std(0, ddof=1), 1e-10)
                p_ = (mat > 0).sum(0)
                c = np.maximum(p_, mat.shape[0] - p_) / mat.shape[0]
                n = int(((np.abs(d) > D_THRESH) & (c > C_THRESH)).sum())
                edges[str(dl)] = {'n_significant': n}
                total += n
            with open(path, 'w') as f:
                json.dump({'feature_idx': int(fi), 'label': labels.get(int(fi), 'unannotated'),
                           'activation_freq': info[int(fi)]['activation_freq'],
                           'n_cells_active': len(per[DOWNSTREAM[0]]),
                           'total_significant_edges': total,
                           'downstream_edges': edges}, f)
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                print(f"    {done} traced, {el/60:.1f} min, "
                      f"ETA {(len(sample)-done)*el/done/60:.0f} min")

    # ---------------- analysis ----------------
    import glob
    from scipy import stats
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, 'json', 'feature_*.json'))):
        if '/._' in p:
            continue
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        print("no results")
        return
    edges = np.array([r['total_significant_edges'] for r in rows])
    ann = np.array([r['label'] != 'unannotated' for r in rows])
    freq = np.array([r['activation_freq'] for r in rows])

    # K562 comparison restricted to the same downstream layers
    k_edges, k_ann, k_freq = [], [], []
    for p in sorted(glob.glob(os.path.join(EXH, 'feature_*.json'))):
        if '/._' in p:
            continue
        with open(p) as f:
            j = json.load(f)
        e = sum(j['downstream_edges'].get(str(dl), {}).get('n_significant', 0)
                for dl in DOWNSTREAM)
        k_edges.append(e)
        k_ann.append(j['label'] != 'unannotated')
        k_freq.append(j['activation_freq'])
    k_edges, k_ann, k_freq = np.array(k_edges), np.array(k_ann), np.array(k_freq)

    def top20_unann(e, a):
        o = np.argsort(-e)[:20]
        return float(1 - a[o].mean())

    res = {
        'ts_native': {
            'n_features': len(rows),
            'mean_edges': float(edges.mean()), 'median_edges': float(np.median(edges)),
            'max_edges': int(edges.max()),
            'annotation_rate': float(ann.mean()),
            'top20_unannotated_frac': top20_unann(edges, ann),
            'heavy_tail_frac_gt_5x_median': float((edges > 5 * np.median(edges)).mean()),
            'spearman_freq_vs_edges': float(stats.spearmanr(freq, edges)[0]),
            'spearman_edges_vs_annotated': float(stats.spearmanr(edges, ann)[0]),
            'skewness': float(stats.skew(edges)),
            'gini': float(gini(edges)),
        },
        'k562_same_downstream': {
            'n_features': len(k_edges),
            'mean_edges': float(k_edges.mean()), 'median_edges': float(np.median(k_edges)),
            'max_edges': int(k_edges.max()),
            'annotation_rate': float(k_ann.mean()),
            'top20_unannotated_frac': top20_unann(k_edges, k_ann),
            'heavy_tail_frac_gt_5x_median': float((k_edges > 5 * np.median(k_edges)).mean()),
            'spearman_freq_vs_edges': float(stats.spearmanr(k_freq, k_edges)[0]),
            'spearman_edges_vs_annotated': float(stats.spearmanr(k_edges, k_ann)[0]),
            'skewness': float(stats.skew(k_edges)),
            'gini': float(gini(k_edges)),
        },
    }
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1)

    print("\n=== TS-native L5 SAE vs K562 L5 SAE (downstream 11+17 only) ===")
    keys = list(res['ts_native'])
    for k in keys:
        print(f"  {k:<34} TS={res['ts_native'][k]:<12.4g} K562={res['k562_same_downstream'][k]:.4g}")


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


if __name__ == '__main__':
    main()
