#!/usr/bin/env python3
"""P9 — Does model-internal centrality predict a biological, out-of-model quantity?

Prediction under test: genes that load on high-degree ("hub") SAE features are genes whose
CRISPRi knockdown produces a large transcriptome shift in K562.

If it holds, the circuit map makes a biological statement.  If it does not, that is the honest
answer to "ablating SAE coefficients does not establish biological regulation" and it bounds
what the map can claim.

Stage 1 (--pseudobulk): stream replogle_concat.h5ad, compute per-perturbation pseudobulk in
log1p-CPM space, and a control-matched shift z-score:

    shift(g) = ||mu_g - mu_ctrl||_2
    z(g)     = (shift(g) - E[shift_null | n_g]) / SD[shift_null | n_g]

where the null is built by resampling non-targeting cells into pseudo-perturbations of the same
size, so that the sampling noise of small perturbations is removed rather than mistaken for
biology.

Stage 2 (--associate): per L5 feature, perturbation strength = mean z over its top-50 loading
genes present in the CRISPRi library.  Test its association with degree, with
frequency-residualized degree, and against a gene-label permutation null, controlling for mean
expression level.
"""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.

import argparse
import json
import os

import numpy as np
from scipy import stats

BASE = os.environ.get("SAE_BASE_ROOT", "..")
PROJ = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
H5 = os.path.join(BASE, "biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad")
CATALOG = os.path.join(PROJ, "experiments/phase1_k562/sae_models/layer05_x4_k32/feature_catalog.json")
EXH = os.path.join(PROJ, "experiments/phase8_exhaustive_tracing")
OUT = os.path.join(PROJ, "experiments/revision_plos/P9_crispri")
SEED = 0
CHUNK = 8192
MIN_CELLS = 20
TOP_GENES = 50


def load_cat(f, col):
    import h5py
    c = f['obs'][col]
    if isinstance(c, h5py.Group):
        cats = c['categories'][:]
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in cats])
        return cats, c['codes'][:]
    raise TypeError(col)


def pseudobulk():
    import h5py
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)

    with h5py.File(H5, 'r') as f:
        cats, codes = load_cat(f, 'gene')
        n_cells, n_genes = f['X'].shape
        var = f['var']['gene_name_index']
        if isinstance(var, h5py.Group):
            vc = var['categories'][:]
            gene_names = np.array([x.decode() if isinstance(x, bytes) else x
                                   for x in vc])[var['codes'][:]]
        else:
            gene_names = np.array([x.decode() if isinstance(x, bytes) else x for x in var[:]])

        print(f"  {n_cells} cells x {n_genes} genes, {len(cats)} perturbation labels")

        n_pert = len(cats)
        sums = np.zeros((n_pert, n_genes), dtype=np.float64)
        counts = np.zeros(n_pert, dtype=np.int64)

        ctrl_codes = [i for i, c in enumerate(cats)
                      if c.lower().replace('_', '-') == 'non-targeting']
        assert ctrl_codes, "no non-targeting label"
        ctrl_code = ctrl_codes[0]
        ctrl_rows = []

        for s in range(0, n_cells, CHUNK):
            e = min(s + CHUNK, n_cells)
            X = f['X'][s:e, :].astype(np.float32)
            rs = X.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            X = np.log1p(X / rs * 1e4)
            cc = codes[s:e]
            for code in np.unique(cc):
                m_ = cc == code
                sums[code] += X[m_].sum(axis=0)
                counts[code] += int(m_.sum())
            m = cc == ctrl_code
            if m.any():
                # subsample control cells to keep the null tractable
                idx = np.where(m)[0]
                if len(idx) > 200:
                    idx = rng.choice(idx, 200, replace=False)
                ctrl_rows.append(X[idx])
            if (s // CHUNK) % 10 == 0:
                print(f"    {e}/{n_cells}")

    ctrl_pool = np.concatenate(ctrl_rows, axis=0)
    print(f"  control pool: {ctrl_pool.shape}")
    mu = sums / np.maximum(counts, 1)[:, None]
    mu_ctrl = mu[ctrl_code]
    mean_expr = mu_ctrl.copy()

    # null distribution of ||mean(n control cells) - mu_ctrl|| as a function of n
    ns = np.unique(np.clip(counts, MIN_CELLS, 2000))
    grid = np.unique(np.round(np.geomspace(MIN_CELLS, max(ns.max(), MIN_CELLS + 1), 25)).astype(int))
    null_mean, null_sd = {}, {}
    for n in grid:
        d = []
        for _ in range(60):
            idx = rng.choice(len(ctrl_pool), size=min(n, len(ctrl_pool)), replace=False)
            d.append(np.linalg.norm(ctrl_pool[idx].mean(axis=0) - mu_ctrl))
        null_mean[int(n)], null_sd[int(n)] = float(np.mean(d)), float(np.std(d) + 1e-9)
    gk = np.array(sorted(null_mean))
    gm = np.array([null_mean[k] for k in gk])
    gs = np.array([null_sd[k] for k in gk])

    out = {}
    for i, name in enumerate(cats):
        if i == ctrl_code or counts[i] < MIN_CELLS:
            continue
        shift = float(np.linalg.norm(mu[i] - mu_ctrl))
        n = int(np.clip(counts[i], gk[0], gk[-1]))
        em = float(np.interp(n, gk, gm))
        es = float(np.interp(n, gk, gs))
        out[name] = {'n_cells': int(counts[i]), 'shift': shift,
                     'z': (shift - em) / es, 'null_mean': em}

    np.save(os.path.join(OUT, 'mean_expression_ctrl.npy'), mean_expr)
    np.save(os.path.join(OUT, 'gene_names.npy'), gene_names)
    with open(os.path.join(OUT, 'perturbation_shifts.json'), 'w') as f:
        json.dump(out, f)
    zs = np.array([v['z'] for v in out.values()])
    print(f"  {len(out)} perturbations; z: median={np.median(zs):.1f} "
          f"IQR=[{np.percentile(zs,25):.1f},{np.percentile(zs,75):.1f}]")


def associate():
    rng = np.random.default_rng(SEED)
    with open(os.path.join(OUT, 'perturbation_shifts.json')) as f:
        shifts = json.load(f)
    gene_names = np.load(os.path.join(OUT, 'gene_names.npy'), allow_pickle=True)
    mean_expr = np.load(os.path.join(OUT, 'mean_expression_ctrl.npy'))
    expr = {g: float(e) for g, e in zip(gene_names, mean_expr)}
    zmap = {g: v['z'] for g, v in shifts.items()}

    with open(CATALOG) as f:
        cat = {c['feature_idx']: c for c in json.load(f)['features']}

    import glob
    feats = []
    for p in sorted(glob.glob(os.path.join(EXH, 'feature_*.json'))):
        if '/._' in p:
            continue
        with open(p) as fh:
            j = json.load(fh)
        fi = j['feature_idx']
        tg = [g['gene_name'] for g in cat[fi]['top_genes'][:TOP_GENES]]
        hits = [g for g in tg if g in zmap]
        if len(hits) < 5:
            continue
        feats.append({
            'idx': fi, 'edges': j['total_significant_edges'],
            'freq': j['activation_freq'],
            'annotated': j['label'] != 'unannotated',
            'n_covered': len(hits),
            'pert_strength': float(np.mean([zmap[g] for g in hits])),
            'mean_expr': float(np.mean([expr.get(g, 0.0) for g in hits])),
        })
    print(f"  {len(feats)} L5 features with >=5 top genes in the CRISPRi library")

    edges = np.array([f['edges'] for f in feats], float)
    strength = np.array([f['pert_strength'] for f in feats])
    freq = np.array([f['freq'] for f in feats])
    mexpr = np.array([f['mean_expr'] for f in feats])

    def rresid(y, *xs):
        X = np.column_stack([np.ones_like(y)] + [stats.rankdata(x) for x in xs])
        y = stats.rankdata(y)
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    rho, p = stats.spearmanr(edges, strength)
    rho_pe, p_pe = stats.spearmanr(mexpr, strength)
    rho_partial, p_partial = stats.spearmanr(rresid(edges, freq, mexpr),
                                             rresid(strength, freq, mexpr))

    # frequency-residualized degree
    lf, le = np.log10(freq), np.log10(edges + 1)
    A = np.column_stack([np.ones_like(lf), lf, lf ** 2])
    rdeg = le - A @ np.linalg.lstsq(A, le, rcond=None)[0]
    rho_r, p_r = stats.spearmanr(rdeg, strength)

    # gene-label permutation null: reshuffle the gene -> z map, recompute rho
    genes = list(zmap)
    zvals = np.array([zmap[g] for g in genes])
    top_lists = {f['idx']: [g for g in [x['gene_name'] for x in cat[f['idx']]['top_genes'][:TOP_GENES]]
                            if g in zmap] for f in feats}
    null_rho = []
    for _ in range(500):
        perm = dict(zip(genes, rng.permutation(zvals)))
        s = np.array([np.mean([perm[g] for g in top_lists[f['idx']]]) for f in feats])
        null_rho.append(stats.spearmanr(edges, s)[0])
    null_rho = np.array(null_rho)
    p_perm = float((np.abs(null_rho) >= abs(rho)).mean())

    # hub vs non-hub (top 1.8% by degree, matching the paper's hub definition)
    k = max(2, int(round(0.018 * len(feats))))
    order = np.argsort(-edges)
    hub, non = strength[order[:k]], strength[order[k:]]
    u, p_u = stats.mannwhitneyu(hub, non, alternative='two-sided')

    res = {
        'n_features': len(feats), 'n_perturbations': len(zmap),
        'spearman_edges_vs_strength': float(rho), 'p': float(p),
        'permutation_p': p_perm,
        'null_rho_mean': float(null_rho.mean()), 'null_rho_sd': float(null_rho.std()),
        'spearman_meanexpr_vs_strength': float(rho_pe), 'p_meanexpr': float(p_pe),
        'partial_spearman_edges_strength_given_freq_expr': float(rho_partial),
        'p_partial': float(p_partial),
        'spearman_residual_degree_vs_strength': float(rho_r), 'p_residual': float(p_r),
        'hub_k': k, 'hub_mean_strength': float(hub.mean()),
        'nonhub_mean_strength': float(non.mean()),
        'mannwhitney_u': float(u), 'mannwhitney_p': float(p_u),
    }
    with open(os.path.join(OUT, 'association.json'), 'w') as f:
        json.dump(res, f, indent=1)

    print(f"\n  rho(degree, CRISPRi shift z)      = {rho:+.3f}  p={p:.3g}  "
          f"(permutation p={p_perm:.3f})")
    print(f"  rho(mean expression, shift z)     = {rho_pe:+.3f}  p={p_pe:.3g}")
    print(f"  partial rho(degree, z | freq,expr)= {rho_partial:+.3f}  p={p_partial:.3g}")
    print(f"  rho(freq-residualized degree, z)  = {rho_r:+.3f}  p={p_r:.3g}")
    print(f"  hub (top {k}) mean z = {hub.mean():.2f} vs non-hub {non.mean():.2f}  "
          f"Mann-Whitney p={p_u:.3g}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pseudobulk', action='store_true')
    ap.add_argument('--associate', action='store_true')
    a = ap.parse_args()
    if a.pseudobulk:
        pseudobulk()
    if a.associate:
        associate()
