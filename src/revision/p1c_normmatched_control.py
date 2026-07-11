#!/usr/bin/env python3
"""P1c — Specificity null: is the edge count a property of the feature, or of the perturbation?

The sign-flip permutation null (p1b) calibrates the *statistical* criterion: it asks whether an
edge could have arisen from a delta with no consistent direction.  It does not ask whether the
edges are specific to the ablated feature, as opposed to a generic consequence of nudging the
residual stream by a vector of that size.

Here, for each sampled feature, we replace the ablation displacement

    delta_h = decode(h with f zeroed) - decode(h)

by a random Gaussian direction rescaled to the same per-position L2 norm, and run the identical
downstream measurement.  The excess of real over norm-matched edges is the part of the graph
attributable to feature identity.
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
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np

PROJ = os.environ.get("SAE_PROJ_ROOT", ".")
ORIG = os.path.join(PROJ, "src", "20_exhaustive_feature_tracing.py")
EXH = os.path.join(PROJ, "experiments/phase8_exhaustive_tracing")
OUT = os.path.join(PROJ, "experiments/revision_plos/P1_normmatched")

SOURCE = 5
DOWNSTREAM = [6, 11, 17]
N_FEATURES = 4608
N_CELLS = 20
N_SAMPLE = 200
D_THRESH, C_THRESH = 0.5, 0.7
SEED = 5


def load_orig():
    spec = importlib.util.spec_from_file_location("orig_trace", ORIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stats_from(mat):
    n = mat.shape[0]
    if n < 2:
        return np.zeros(N_FEATURES), np.zeros(N_FEATURES)
    d = mat.mean(0) / np.maximum(mat.std(0, ddof=1), 1e-10)
    pos = (mat > 0).sum(0)
    return d, np.maximum(pos, n - pos) / n


def n_edges(mat):
    d, c = stats_from(mat)
    return int(((np.abs(d) > D_THRESH) & (c > C_THRESH)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-sample', type=int, default=N_SAMPLE)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch
    from transformers import BertForMaskedLM
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    orig = load_orig()
    rng = np.random.default_rng(SEED)
    gen = torch.Generator(device='cpu').manual_seed(SEED)

    # stratify by observed degree so the comparison spans hubs and the long tail
    import glob
    obs = {}
    for p in sorted(glob.glob(os.path.join(EXH, 'feature_*.json'))):
        if '/._' in p:
            continue
        with open(p) as f:
            j = json.load(f)
        obs[j['feature_idx']] = j['total_significant_edges']
    fis = np.array(sorted(obs))
    deg = np.array([obs[f] for f in fis])
    order = np.argsort(-deg)
    strata = np.array_split(order, 10)
    per = max(1, args.n_sample // 10)
    take = np.concatenate([rng.choice(s, size=min(per, len(s)), replace=False) for s in strata])
    sample = fis[np.sort(take)]
    print(f"{len(sample)} features sampled across degree deciles")

    all_tokens = orig.load_and_tokenize_cells(N_CELLS)[:N_CELLS]
    model = BertForMaskedLM.from_pretrained(orig.MODEL_NAME, subfolder=orig.MODEL_SUBFOLDER,
                                            output_hidden_states=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    sc = orig.SAECache()
    src_sae, src_mean = sc.get(SOURCE)
    src_sae.to(device)
    src_mean_t = torch.tensor(src_mean, dtype=torch.float32, device=device)
    dmean = {}
    for dl in DOWNSTREAM:
        s, m = sc.get(dl)
        s.to(device)
        dmean[dl] = torch.tensor(m, dtype=torch.float32, device=device)

    cache = []
    for t in all_tokens:
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
                csp, _ = dsae.encode(o.hidden_states[dl + 1][0][gp] - dmean[dl])
                clean[dl] = csp
        cache.append({'gp': gp, 'hidden': hidden, 'h_sp': h_sp, 'topk': topk, 'clean': clean})
        del o
        if device.type == 'mps':
            torch.mps.empty_cache()
    print("clean cache built")

    def measure(delta_fn):
        per = {dl: [] for dl in DOWNSTREAM}
        for cell in cache:
            r = delta_fn(cell)
            if r is None:
                continue
            dh, pos = r
            mod = cell['hidden'].clone()
            mod[cell['gp']] += dh
            h = mod.unsqueeze(0)
            with torch.no_grad():
                for l in range(SOURCE + 1, 18):
                    h = model.bert.encoder.layer[l](h)[0]
                    if l in DOWNSTREAM:
                        dsae, _ = sc.get(l)
                        asp, _ = dsae.encode(h[0][cell['gp']] - dmean[l])
                        per[l].append((asp[pos] - cell['clean'][l][pos])
                                      .mean(dim=0).cpu().numpy().astype(np.float64))
            del h, mod
            if device.type == 'mps':
                torch.mps.empty_cache()
        return {dl: (np.stack(v) if len(v) > 1 else np.zeros((0, N_FEATURES)))
                for dl, v in per.items()}

    rows = []
    t0 = time.time()
    for i, fi in enumerate(sample):
        fi = int(fi)

        def real(cell, fi=fi):
            m = (cell['topk'] == fi).any(dim=1)
            if not m.any():
                return None
            h_ab = cell['h_sp'].clone()
            h_ab[:, fi] = 0.0
            with torch.no_grad():
                dh = src_sae.decode(h_ab) - src_sae.decode(cell['h_sp'])
            return dh, torch.where(m)[0]

        def randdir(cell, fi=fi):
            m = (cell['topk'] == fi).any(dim=1)
            if not m.any():
                return None
            h_ab = cell['h_sp'].clone()
            h_ab[:, fi] = 0.0
            with torch.no_grad():
                dh = src_sae.decode(h_ab) - src_sae.decode(cell['h_sp'])
            norms = dh.norm(dim=1, keepdim=True)
            r = torch.randn(dh.shape, generator=gen).to(dh.device)
            r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-9) * norms
            return r, torch.where(m)[0]

        dr = measure(real)
        dn = measure(randdir)
        e_real = sum(n_edges(dr[dl]) for dl in DOWNSTREAM)
        e_rand = sum(n_edges(dn[dl]) for dl in DOWNSTREAM)
        rows.append({'feature_idx': fi, 'observed_degree': obs[fi],
                     'edges_real': e_real, 'edges_normmatched': e_rand})
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(sample)}  real={e_real} rand={e_rand}  "
                  f"({el/60:.1f} min, ETA {(len(sample)-i-1)*el/(i+1)/60:.0f} min)")

    from scipy import stats
    er = np.array([r['edges_real'] for r in rows], float)
    en = np.array([r['edges_normmatched'] for r in rows], float)
    w = stats.wilcoxon(er, en)
    res = {
        'n_features': len(rows),
        'mean_edges_real': float(er.mean()), 'mean_edges_normmatched': float(en.mean()),
        'median_edges_real': float(np.median(er)), 'median_edges_normmatched': float(np.median(en)),
        'ratio_real_over_normmatched': float(er.mean() / max(en.mean(), 1e-9)),
        'excess_fraction': float((er - en).sum() / max(er.sum(), 1e-9)),
        'wilcoxon_stat': float(w.statistic), 'wilcoxon_p': float(w.pvalue),
        'spearman_degree_vs_normmatched': float(stats.spearmanr(
            [r['observed_degree'] for r in rows], en)[0]),
        'per_feature': rows,
    }
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1)

    print(f"\nreal      mean edges/feature = {er.mean():.1f}")
    print(f"norm-matched random direction = {en.mean():.1f}")
    print(f"ratio = {res['ratio_real_over_normmatched']:.2f}x, "
          f"excess attributable to feature identity = {res['excess_fraction']:.1%}")
    print(f"Wilcoxon signed-rank p = {w.pvalue:.3g}")


if __name__ == '__main__':
    main()
