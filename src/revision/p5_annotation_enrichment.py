#!/usr/bin/env python3
"""P5 — Formal enrichment/depletion test for annotation status among hub features.

The manuscript states that 40% of top-20 L5 hubs being unannotated is "a bias".  Since the
background unannotated rate is 46.2%, the top-20 hubs are in fact marginally *enriched* for
annotation.  This script replaces the informal comparison with:

  1. Fisher's exact test, top-k hubs vs all active features, k in {20, 100, 1%}.
  2. Fisher's exact against an activation-frequency-matched background (hub-ness may be a
     proxy for how often a feature fires, and frequent features are easier to annotate).
  3. Logistic regression annotated ~ log10(edges) over all features.
  4. Spearman rho(edge count, annotated) and annotation rate per edge-count decile.
  5. The same battery at the L2/L8/L14 subsamples.

Reads either the original trace or the FDR-recalibrated edge table (--edges).
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

PROJ = os.environ.get("SAE_PROJ_ROOT", ".")
EXH = os.path.join(PROJ, "experiments/phase8_exhaustive_tracing")
R2 = os.path.join(PROJ, "experiments/revision_bioinformatics/R2_source_layer")
OUT = os.path.join(PROJ, "experiments/revision_plos/P5_annotation")
SEED = 0


def load_dir(pattern):
    feats = []
    for p in sorted(glob.glob(pattern)):
        if '/._' in p:
            continue
        with open(p) as f:
            j = json.load(f)
        feats.append({
            'idx': j['feature_idx'],
            'edges': j['total_significant_edges'],
            'annotated': j['label'] != 'unannotated',
            'freq': j.get('activation_freq', np.nan),
        })
    return feats


def fisher_topk(feats, k):
    order = sorted(feats, key=lambda f: -f['edges'])
    top = order[:k]
    rest = order[k:]
    a = sum(f['annotated'] for f in top)
    b = k - a
    c = sum(f['annotated'] for f in rest)
    d = len(rest) - c
    odds, p = stats.fisher_exact([[a, b], [c, d]])
    return {
        'k': k, 'top_annotated': a, 'top_unannotated': b,
        'top_annotated_frac': a / k,
        'background_annotated_frac': c / max(len(rest), 1),
        'odds_ratio': float(odds), 'p_value': float(p),
        'direction': 'enriched' if odds > 1 else 'depleted',
    }


def fisher_freq_matched(feats, k, n_boot=2000, rng=None):
    """Compare top-k hubs against a background matched on log activation frequency."""
    rng = rng or np.random.default_rng(SEED)
    feats = [f for f in feats if np.isfinite(f['freq']) and f['freq'] > 0]
    order = sorted(feats, key=lambda f: -f['edges'])
    top, pool = order[:k], order[k:]
    if not top or not pool:
        return None
    lf_pool = np.array([np.log10(f['freq']) for f in pool])
    ann_pool = np.array([f['annotated'] for f in pool])

    obs = sum(f['annotated'] for f in top) / k
    null = []
    for _ in range(n_boot):
        pick = []
        for f in top:
            lf = np.log10(f['freq'])
            # nearest 25 neighbours in log-frequency, sample one
            nn = np.argsort(np.abs(lf_pool - lf))[:25]
            pick.append(ann_pool[rng.choice(nn)])
        null.append(np.mean(pick))
    null = np.array(null)
    p_two = 2 * min((null <= obs).mean(), (null >= obs).mean())
    return {
        'k': k, 'observed_annotated_frac': float(obs),
        'matched_null_mean': float(null.mean()), 'matched_null_sd': float(null.std()),
        'p_value': float(min(p_two, 1.0)),
        'direction': 'enriched' if obs > null.mean() else 'depleted',
    }


def logistic_and_correlation(feats):
    edges = np.array([f['edges'] for f in feats], dtype=float)
    ann = np.array([f['annotated'] for f in feats], dtype=float)
    x = np.log10(edges + 1)

    # Newton-Raphson logistic regression, annotated ~ 1 + log10(edges+1)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-12
        z = eta + (ann - mu) / W
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = 1 / (1 + np.exp(-eta))
    W = mu * (1 - mu) + 1e-12
    cov = np.linalg.inv((X.T * W) @ X)
    se = np.sqrt(np.diag(cov))
    zstat = beta[1] / se[1]
    p_slope = 2 * stats.norm.sf(abs(zstat))

    rho, p_rho = stats.spearmanr(edges, ann)

    deciles = []
    q = np.quantile(edges, np.linspace(0, 1, 11))
    for i in range(10):
        m = (edges >= q[i]) & (edges <= q[i + 1] if i == 9 else edges < q[i + 1])
        if m.sum():
            deciles.append({'decile': i + 1, 'n': int(m.sum()),
                            'mean_edges': float(edges[m].mean()),
                            'annotated_frac': float(ann[m].mean())})

    return {
        'logit_slope': float(beta[1]), 'logit_slope_se': float(se[1]),
        'logit_slope_z': float(zstat), 'logit_slope_p': float(p_slope),
        'odds_ratio_per_10x_edges': float(np.exp(beta[1])),
        'spearman_rho': float(rho), 'spearman_p': float(p_rho),
        'deciles': deciles,
    }


def frequency_confound(feats):
    """Degree is strongly driven by how often a feature fires: a feature that is active on
    more token positions yields a less noisy per-cell mean effect and therefore passes the
    |d| threshold on more targets.  Quantify that, then re-define hubs on the component of
    degree that activation frequency does not explain."""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.
    feats = [f for f in feats if np.isfinite(f['freq']) and f['freq'] > 0]
    edges = np.array([f['edges'] for f in feats], dtype=float)
    freq = np.array([f['freq'] for f in feats], dtype=float)
    ann = np.array([f['annotated'] for f in feats], dtype=float)

    rho_fe, p_fe = stats.spearmanr(freq, edges)

    # partial Spearman: rho(edges, annotated | log freq) via rank residuals
    r_e, r_a, r_f = (stats.rankdata(v) for v in (edges, ann, freq))
    def resid(y, x):
        A = np.column_stack([np.ones_like(x), x])
        return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rho_partial, p_partial = stats.spearmanr(resid(r_e, r_f), resid(r_a, r_f))

    # frequency-residualized degree (quadratic fit in log-frequency)
    lf = np.log10(freq)
    le = np.log10(edges + 1)
    A = np.column_stack([np.ones_like(lf), lf, lf ** 2])
    coef = np.linalg.lstsq(A, le, rcond=None)[0]
    r2 = 1 - np.var(le - A @ coef) / np.var(le)
    residual_degree = le - A @ coef

    order = np.argsort(-residual_degree)[:20]
    top20_resid = [{'feature_idx': feats[i]['idx'], 'edges': int(edges[i]),
                    'activation_freq': float(freq[i]),
                    'residual_degree': float(residual_degree[i]),
                    'annotated': bool(ann[i])} for i in order]
    a = int(sum(t['annotated'] for t in top20_resid))
    rest = np.setdiff1d(np.arange(len(feats)), order)
    c = int(ann[rest].sum())
    odds, p = stats.fisher_exact([[a, 20 - a], [c, len(rest) - c]])

    return {
        'spearman_freq_vs_edges': float(rho_fe), 'spearman_freq_vs_edges_p': float(p_fe),
        'partial_spearman_edges_annotated_given_freq': float(rho_partial),
        'partial_spearman_p': float(p_partial),
        'logfreq_quadratic_r2_on_logedges': float(r2),
        'residual_hub_top20_annotated': a,
        'residual_hub_top20_annotated_frac': a / 20,
        'residual_hub_fisher_odds': float(odds), 'residual_hub_fisher_p': float(p),
        'residual_hub_top20': top20_resid,
    }


def battery(feats, name, rng):
    n = len(feats)
    res = {'layer': name, 'n_features': n,
           'overall_annotated_frac': float(np.mean([f['annotated'] for f in feats])),
           'fisher': [], 'freq_matched': []}
    for k in [20, 100, max(1, round(0.01 * n))]:
        if k <= n // 2:
            res['fisher'].append(fisher_topk(feats, k))
            fm = fisher_freq_matched(feats, k, rng=rng)
            if fm:
                res['freq_matched'].append(fm)
    res.update(logistic_and_correlation(feats))
    res['frequency'] = frequency_confound(feats)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', default=None,
                    help='optional JSON {feature_idx: edges} to override L5 edge counts')
    ap.add_argument('--tag', default='raw')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    l5 = load_dir(os.path.join(EXH, 'feature_*.json'))
    if args.edges:
        with open(args.edges) as f:
            override = {int(k): v for k, v in json.load(f).items()}
        l5 = [dict(f, edges=override[f['idx']]) for f in l5 if f['idx'] in override]

    results = [battery(l5, 'L5', rng)]
    for L in ['l02', 'l08', 'l14']:
        feats = load_dir(os.path.join(R2, L, 'feature_dumps', 'feature_*.json'))
        if feats:
            results.append(battery(feats, L.upper().replace('L0', 'L'), rng))

    out = os.path.join(OUT, f'annotation_enrichment_{args.tag}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)

    for r in results:
        print(f"\n=== {r['layer']}  (n={r['n_features']}, "
              f"annotated={r['overall_annotated_frac']:.1%})")
        for fi in r['fisher']:
            print(f"  top-{fi['k']:>3}: annotated {fi['top_annotated_frac']:.1%} vs bg "
                  f"{fi['background_annotated_frac']:.1%}  OR={fi['odds_ratio']:.2f} "
                  f"p={fi['p_value']:.3g} ({fi['direction']})")
        for fm in r['freq_matched']:
            print(f"  top-{fm['k']:>3} freq-matched: obs {fm['observed_annotated_frac']:.1%} vs "
                  f"null {fm['matched_null_mean']:.1%}±{fm['matched_null_sd']:.1%} "
                  f"p={fm['p_value']:.3g} ({fm['direction']})")
        print(f"  logistic slope={r['logit_slope']:+.3f} (p={r['logit_slope_p']:.3g}), "
              f"OR per 10x edges={r['odds_ratio_per_10x_edges']:.2f}")
        print(f"  spearman rho(edges, annotated)={r['spearman_rho']:+.3f} (p={r['spearman_p']:.3g})")
        fq = r['frequency']
        print(f"  CONFOUND rho(activation_freq, edges)={fq['spearman_freq_vs_edges']:+.3f} "
              f"(p={fq['spearman_freq_vs_edges_p']:.3g}); log-freq explains "
              f"{fq['logfreq_quadratic_r2_on_logedges']:.1%} of log-degree variance")
        print(f"  partial rho(edges, annotated | freq)="
              f"{fq['partial_spearman_edges_annotated_given_freq']:+.3f} "
              f"(p={fq['partial_spearman_p']:.3g})")
        print(f"  frequency-residualized top-20 hubs: annotated "
              f"{fq['residual_hub_top20_annotated_frac']:.1%}  OR={fq['residual_hub_fisher_odds']:.2f} "
              f"p={fq['residual_hub_fisher_p']:.3g}")
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
