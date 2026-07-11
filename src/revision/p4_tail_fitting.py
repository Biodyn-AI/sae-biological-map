#!/usr/bin/env python3
"""P4 — Formal distribution fitting for the feature degree distribution.

The manuscript calls the degree distribution "heavy-tailed" and gestures at scale-free
architectures.  A right-skewed histogram does not license that.  This implements the
Clauset-Shalizi-Newman procedure for discrete data:

  * MLE for a discrete power law above x_min, with x_min chosen by KS minimization;
  * MLE for lognormal, exponential, stretched-exponential (Weibull) and truncated power law,
    all fitted on the same tail (x >= x_min);
  * goodness-of-fit KS p-value for the power law by parametric bootstrap;
  * Vuong likelihood-ratio tests between the power law and each alternative.

No dependency on the `powerlaw` package.
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import optimize, stats

PROJ = os.environ.get("SAE_PROJ_ROOT", ".")
EXH = os.path.join(PROJ, "experiments/phase8_exhaustive_tracing")
OUT = os.path.join(PROJ, "experiments/revision_plos/P4_tail")
SEED = 0
XMAX = 200000  # upper bound for discrete power-law normalization


def _zeta_tail(alpha, xmin, xmax=XMAX):
    k = np.arange(xmin, xmax + 1)
    return np.sum(k.astype(float) ** (-alpha))


def fit_powerlaw_discrete(x, xmin):
    x = x[x >= xmin]
    if len(x) < 10:
        return None
    logx = np.log(x)

    def nll(alpha):
        if alpha <= 1.01 or alpha > 20:
            return 1e12
        return len(x) * np.log(_zeta_tail(alpha, xmin)) + alpha * logx.sum()

    r = optimize.minimize_scalar(nll, bounds=(1.05, 10.0), method='bounded')
    alpha = float(r.x)
    return alpha, -nll(alpha), len(x)


def pl_logpmf(x, alpha, xmin):
    return -alpha * np.log(x) - np.log(_zeta_tail(alpha, xmin))


def pl_cdf(vals, alpha, xmin):
    k = np.arange(xmin, vals.max() + 1).astype(float)
    pmf = k ** (-alpha)
    pmf /= pmf.sum()
    cdf = np.cumsum(pmf)
    return cdf[np.searchsorted(k, vals)]


def ks_stat(x, alpha, xmin):
    x = np.sort(x[x >= xmin])
    n = len(x)
    emp = np.arange(1, n + 1) / n
    theo = pl_cdf(x, alpha, xmin)
    return float(np.max(np.abs(emp - theo)))


def choose_xmin(x, candidates=None):
    x = np.asarray(x)
    uniq = np.unique(x[x > 0])
    if candidates is None:
        candidates = uniq[(uniq >= 1) & (uniq <= np.quantile(uniq, 0.95))]
        if len(candidates) > 60:
            candidates = np.unique(np.quantile(candidates, np.linspace(0, 1, 60)).astype(int))
    best = None
    for xm in candidates:
        fit = fit_powerlaw_discrete(x, int(xm))
        if fit is None:
            continue
        alpha, ll, ntail = fit
        d = ks_stat(x, alpha, int(xm))
        if best is None or d < best['ks']:
            best = {'xmin': int(xm), 'alpha': alpha, 'ks': d, 'n_tail': ntail, 'loglik': ll}
    return best


# ---- alternative distributions, all fitted on the same tail x >= xmin (discrete, normalized)
def _discrete_norm(logpdf, xmin, xmax=XMAX):
    k = np.arange(xmin, xmax + 1).astype(float)
    return np.log(np.sum(np.exp(logpdf(k))))


def fit_alt(x, xmin, kind):
    x = x[x >= xmin].astype(float)
    n = len(x)

    if kind == 'lognormal':
        def nll(p):
            mu, sig = p
            if sig <= 1e-3:
                return 1e12
            lp = lambda k: -np.log(k) - np.log(sig) - 0.5 * ((np.log(k) - mu) / sig) ** 2
            return -(np.sum(lp(x)) - n * _discrete_norm(lp, xmin))
        r = optimize.minimize(nll, [np.log(x).mean(), np.log(x).std() + .1],
                              method='Nelder-Mead')
        return -r.fun, r.x, 2
    if kind == 'exponential':
        def nll(p):
            lam = p[0]
            if lam <= 1e-9:
                return 1e12
            lp = lambda k: -lam * k
            return -(np.sum(lp(x)) - n * _discrete_norm(lp, xmin))
        r = optimize.minimize(nll, [1.0 / x.mean()], method='Nelder-Mead')
        return -r.fun, r.x, 1
    if kind == 'stretched_exp':
        def nll(p):
            lam, beta = p
            if lam <= 1e-9 or beta <= 1e-3 or beta > 10:
                return 1e12
            lp = lambda k: (beta - 1) * np.log(k) - lam * k ** beta
            return -(np.sum(lp(x)) - n * _discrete_norm(lp, xmin))
        r = optimize.minimize(nll, [1.0 / x.mean(), 1.0], method='Nelder-Mead')
        return -r.fun, r.x, 2
    if kind == 'truncated_powerlaw':
        def nll(p):
            alpha, lam = p
            if alpha <= 0.01 or lam <= 1e-9:
                return 1e12
            lp = lambda k: -alpha * np.log(k) - lam * k
            return -(np.sum(lp(x)) - n * _discrete_norm(lp, xmin))
        r = optimize.minimize(nll, [2.0, 1e-3], method='Nelder-Mead')
        return -r.fun, r.x, 2
    raise ValueError(kind)


def pointwise_loglik(x, xmin, kind, params):
    x = x[x >= xmin].astype(float)
    if kind == 'lognormal':
        mu, sig = params
        lp = lambda k: -np.log(k) - np.log(sig) - 0.5 * ((np.log(k) - mu) / sig) ** 2
    elif kind == 'exponential':
        lp = lambda k: -params[0] * k
    elif kind == 'stretched_exp':
        lam, beta = params
        lp = lambda k: (beta - 1) * np.log(k) - lam * k ** beta
    elif kind == 'truncated_powerlaw':
        alpha, lam = params
        lp = lambda k: -alpha * np.log(k) - lam * k
    else:
        raise ValueError(kind)
    return lp(x) - _discrete_norm(lp, xmin)


def vuong(ll1, ll2):
    """Normalized LR test for non-nested models (Clauset et al. eq. C.6)."""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.
    diff = ll1 - ll2
    n = len(diff)
    R = diff.sum()
    sigma = np.sqrt(np.mean((diff - diff.mean()) ** 2))
    if sigma < 1e-12:
        return 0.0, 1.0
    z = R / (np.sqrt(n) * sigma)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def gof_pvalue(x, alpha, xmin, n_sims=200, rng=None):
    rng = rng or np.random.default_rng(SEED)
    x = np.asarray(x)
    d_obs = ks_stat(x, alpha, xmin)
    ntail = int((x >= xmin).sum())
    body = x[x < xmin]
    p_tail = ntail / len(x)

    k = np.arange(xmin, max(int(x.max()) * 3, xmin + 10)).astype(float)
    pmf = k ** (-alpha)
    pmf /= pmf.sum()

    count = 0
    for _ in range(n_sims):
        n_from_tail = rng.binomial(len(x), p_tail)
        synth_tail = rng.choice(k, size=n_from_tail, p=pmf)
        synth_body = rng.choice(body, size=len(x) - n_from_tail, replace=True) if len(body) else np.array([])
        synth = np.concatenate([synth_body, synth_tail])
        fit = choose_xmin(synth, candidates=np.unique(
            np.quantile(np.unique(synth[synth > 0]), np.linspace(0, .95, 25)).astype(int)))
        if fit is None:
            continue
        if fit['ks'] >= d_obs:
            count += 1
    return count / n_sims, d_obs


def analyse(edges, tag):
    rng = np.random.default_rng(SEED)
    x = np.asarray(edges)
    x = x[x > 0]
    print(f"\n=== {tag}: n={len(x)}, mean={x.mean():.1f}, median={np.median(x):.0f}, max={x.max()}")

    best = choose_xmin(x)
    alpha, xmin = best['alpha'], best['xmin']
    print(f"  power law: alpha={alpha:.3f}, x_min={xmin}, n_tail={best['n_tail']}, KS={best['ks']:.4f}")

    p_gof, d_obs = gof_pvalue(x, alpha, xmin, n_sims=200, rng=rng)
    print(f"  goodness of fit (parametric bootstrap, 200 sims): p={p_gof:.3f} "
          f"{'-> power law PLAUSIBLE' if p_gof > 0.1 else '-> power law REJECTED'}")

    ll_pl = pl_logpmf(x[x >= xmin].astype(float), alpha, xmin)
    comparisons = {}
    for kind in ['lognormal', 'exponential', 'stretched_exp', 'truncated_powerlaw']:
        _, params, npar = fit_alt(x, xmin, kind)
        ll_alt = pointwise_loglik(x, xmin, kind, params)
        z, p = vuong(ll_pl, ll_alt)
        favored = 'power law' if z > 0 else kind
        if p > 0.1:
            favored = 'inconclusive'
        comparisons[kind] = {'params': [float(v) for v in np.atleast_1d(params)],
                             'loglik': float(ll_alt.sum()), 'vuong_z': z, 'vuong_p': p,
                             'favored': favored}
        print(f"  vs {kind:<20} LR z={z:+.2f} p={p:.3g}  -> favors {favored}")

    return {
        'tag': tag, 'n': int(len(x)), 'mean': float(x.mean()),
        'median': float(np.median(x)), 'max': int(x.max()),
        'powerlaw': {'alpha': alpha, 'xmin': xmin, 'n_tail': best['n_tail'],
                     'ks': d_obs, 'gof_p': p_gof,
                     'plausible': bool(p_gof > 0.1)},
        'comparisons': comparisons,
        'powerlaw_loglik': float(ll_pl.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', default=None, help='JSON {feature_idx: edges}')
    ap.add_argument('--tag', default='raw')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.edges:
        with open(args.edges) as f:
            edges = np.array(list(json.load(f).values()))
    else:
        edges = []
        for p in sorted(glob.glob(os.path.join(EXH, 'feature_*.json'))):
            if '/._' in p:
                continue
            with open(p) as f:
                edges.append(json.load(f)['total_significant_edges'])
        edges = np.array(edges)

    res = analyse(edges, f'L5 degree distribution ({args.tag})')
    with open(os.path.join(OUT, f'tail_fit_{args.tag}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {os.path.join(OUT, f'tail_fit_{args.tag}.json')}")


if __name__ == '__main__':
    main()
