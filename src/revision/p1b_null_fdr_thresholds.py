#!/usr/bin/env python3
"""P1b — Empirical null, FDR control, and threshold sensitivity for edge calling.

The edge criterion |d| > 0.5 AND consistency > 0.7 is applied to 4,065 x 4,608 x 3 tests with
n = 20 cells and no null.  Under a symmetric-noise null, |d| > 0.5 on n = 20 corresponds to
|t| > 2.24 (p ~ 0.037) and consistency > 0.7 to >= 15/20 same-signed cells (p ~ 0.041); the two
events are positively correlated.  Whether that produces a large false-discovery rate depends
on how many (feature, target) pairs have small-but-nonzero deltas, which is an empirical
question.  This script answers it.

Null: independent sign flips of each cell's delta, conditioned on the observed |delta| values.
It is the exact null "the ablation has no consistently directed effect on this target".  Note
that targets whose delta is identically zero across cells (the source feature has no path to
them) stay zero under sign flips and can never produce a null edge, so the null is properly
conditioned rather than uniform.

Outputs:
  summary.json            headline calibrated numbers
  edges_fdr.json          {feature_idx: n_edges} after BH q<0.05 + |g|>0.5 + consistency>0.7
  edges_raw.json          {feature_idx: n_edges} under the original criterion (sanity check)
  threshold_sweep.json    edge counts / hub-rank stability across (|d|, consistency) grid
  per_feature.json        per-feature observed, expected-null, FDR
"""

import argparse
import glob
import json
import os

import numpy as np

PROJ = os.environ.get("SAE_PROJ_ROOT", ".")
RETRACE = os.path.join(PROJ, "experiments/revision_plos/P1_retrace")
EXH = os.path.join(PROJ, "experiments/phase8_exhaustive_tracing")
OUT = os.path.join(PROJ, "experiments/revision_plos/P1_calibration")
DL = [6, 11, 17]
N_PERM = 200
SEED = 0
D_GRID = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
C_GRID = [0.6, 0.7, 0.8, 0.9, 1.0]


def hedges_g(d, n):
    """One-sample small-sample bias correction."""
    return d * (1.0 - 3.0 / (4.0 * n - 5.0)) if n > 2 else d


def stats_from(mat):
    n = mat.shape[0]
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1)
    d = mean / np.maximum(std, 1e-10)
    pos = (mat > 0).sum(axis=0)
    cons = np.maximum(pos, n - pos) / n
    return d, cons


def perm_stats(mat, signs):
    """Vectorised sign-flip null. signs: (B, n) in {-1,+1}. Returns (B, T) d and consistency."""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.
    n = mat.shape[0]
    means = (signs @ mat) / n
    # sum of squares is invariant to sign flips, so the flipped variance is closed-form
    sumsq = (mat ** 2).sum(axis=0)
    var = (sumsq - n * means ** 2) / (n - 1)
    d = means / np.maximum(np.sqrt(np.maximum(var, 0)), 1e-10)

    # sign(s_i * x_i) > 0  <=>  (x_i > 0) if s_i > 0 else (x_i < 0).
    # Avoids materialising the (B, n, T) product.
    P = (mat > 0).astype(np.float64)     # (n, T)
    N = (mat < 0).astype(np.float64)
    Sp = (signs > 0).astype(np.float64)  # (B, n)
    pos = Sp @ P + (1.0 - Sp) @ N
    cons = np.maximum(pos, n - pos) / n
    return d, cons


def load_feature(path):
    z = np.load(path)
    return {dl: z[f'dl{dl}'].astype(np.float64) for dl in DL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--perm-subset', type=int, default=600,
                    help='features used for the (expensive) permutation FDR estimate')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    paths = [p for p in sorted(glob.glob(os.path.join(RETRACE, 'deltas', 'feature_*.npz')))
             if '/._' not in p]
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths)} traced features")

    # ---------------- pass 1: observed statistics, raw + threshold sweep ----------------
    raw_edges, g_edges = {}, {}
    sweep = {(dt, ct): {} for dt in D_GRID for ct in C_GRID}
    cache = {}
    for p in paths:
        fi = int(os.path.basename(p).split('_')[1].split('.')[0])
        deltas = load_feature(p)
        n = deltas[DL[0]].shape[0]
        if n < 2:
            continue
        cache[fi] = n
        tot_raw = tot_g = 0
        per_dl = {}
        for dl in DL:
            d, c = stats_from(deltas[dl])
            g = hedges_g(d, n)
            per_dl[dl] = (d, c, g)
            tot_raw += int(((np.abs(d) > 0.5) & (c > 0.7)).sum())
            tot_g += int(((np.abs(g) > 0.5) & (c > 0.7)).sum())
        raw_edges[fi] = tot_raw
        g_edges[fi] = tot_g
        for dt in D_GRID:
            for ct in C_GRID:
                s = 0
                for dl in DL:
                    d, c, _ = per_dl[dl]
                    s += int(((np.abs(d) > dt) & (c >= ct if ct == 1.0 else c > ct)).sum())
                sweep[(dt, ct)][fi] = s

    print(f"observed: total raw edges (retraced subset) = {sum(raw_edges.values()):,}  "
          f"(Hedges' g: {sum(g_edges.values()):,})")

    # The retraced set over-samples high-degree features, so the graph-wide extrapolation must
    # use the degree of every active feature, taken from the original trace.  The re-trace
    # reproduces those counts exactly (see p1a --validate), so this is the same quantity.
    all_degrees = {}
    for p_ in sorted(glob.glob(os.path.join(EXH, 'feature_*.json'))):
        if '/._' in p_:
            continue
        with open(p_) as f:
            j = json.load(f)
        all_degrees[j['feature_idx']] = j['total_significant_edges']
    # The per-cell deltas are archived as float16.  Recomputing d from them can move a target
    # across the |d| = 0.5 boundary, so the recomputed degree differs from the original float64
    # degree by a tiny amount.  Quantify it rather than assume it away.
    diffs = np.array([raw_edges[fi] - all_degrees[fi] for fi in raw_edges if fi in all_degrees])
    orig = np.array([all_degrees[fi] for fi in raw_edges if fi in all_degrees])
    rel = np.abs(diffs).sum() / max(orig.sum(), 1)
    print(f"population: {len(all_degrees)} active features, "
          f"{sum(all_degrees.values()):,} raw edges")
    print(f"float16 storage vs float64 original: mean signed diff {diffs.mean():+.2f} edges/feature, "
          f"total |diff| = {rel:.3%} of edges (max {np.abs(diffs).max()})")
    fp16_effect = {'mean_signed_diff': float(diffs.mean()),
                   'total_abs_diff_fraction': float(rel),
                   'max_abs_diff': int(np.abs(diffs).max())}

    # ---------------- pass 2: permutation null on a stratified subset ----------------
    fis = np.array(sorted(all_degrees))
    ecounts = np.array([all_degrees[f] for f in fis])
    order = np.argsort(-ecounts, kind='stable')
    k = min(args.perm_subset, len(fis))
    # all 200 highest-degree features + a stratified random sample of the rest
    take = list(order[:200])
    rest = order[200:]
    if len(rest) and k > 200:
        strata = np.array_split(rest, min(8, len(rest)))
        per = max(1, (k - 200) // len(strata))
        for st in strata:
            take.extend(rng.choice(st, size=min(per, len(st)), replace=False))
    subset = fis[np.array(sorted(set(take)))]
    have = {int(os.path.basename(p).split('_')[1].split('.')[0]) for p in paths}
    missing = [f for f in subset if f not in have]
    if missing:
        print(f"  {len(missing)} subset features not yet retraced; excluded")
        subset = np.array([f for f in subset if f in have])
    print(f"permutation subset: {len(subset)} features x {args.n_perm} permutations")

    per_feature = {}
    null_edges_by_thresh = {(dt, ct): [] for dt in D_GRID for ct in C_GRID}
    all_p, all_key = [], []
    fdr_edges = {}

    n_degenerate = 0
    for j, fi in enumerate(subset):
        deltas = load_feature(os.path.join(RETRACE, 'deltas', f'feature_{fi:04d}.npz'))
        n = deltas[DL[0]].shape[0]
        if n < 2:
            continue
        # B non-identity sign vectors: the identity would reproduce the observed statistic and
        # inflate the null-edge count by exactly obs/B.
        signs = rng.choice([-1.0, 1.0], size=(args.n_perm, n))
        allpos = np.all(signs > 0, axis=1)
        while allpos.any():
            signs[allpos] = rng.choice([-1.0, 1.0], size=(int(allpos.sum()), n))
            allpos = np.all(signs > 0, axis=1)

        obs_tot, null_tot = 0, np.zeros(args.n_perm)
        nulls_by_t = {key: np.zeros(args.n_perm) for key in null_edges_by_thresh}
        for dl in DL:
            mat = deltas[dl]
            d, c = stats_from(mat)
            g = hedges_g(d, n)
            dn, cn = perm_stats(mat, signs)
            obs_tot += int(((np.abs(d) > 0.5) & (c > 0.7)).sum())
            null_tot += ((np.abs(dn) > 0.5) & (cn > 0.7)).sum(axis=1)
            for (dt, ct) in nulls_by_t:
                nulls_by_t[(dt, ct)] += ((np.abs(dn) > dt) &
                                         (cn >= ct if ct == 1.0 else cn > ct)).sum(axis=1)

            # Permutation p on |d|, conservative (1 + #exceed) / (1 + B).  The test family is
            # every target the ablation could possibly have reached, i.e. every target whose
            # delta is not identically zero across cells; degenerate targets are excluded
            # rather than entered with a degenerate p = 1.
            live = np.any(mat != 0, axis=0)
            n_degenerate += int((~live).sum())
            pval = (1.0 + (np.abs(dn) >= np.abs(d)[None, :]).sum(axis=0)) / (1.0 + args.n_perm)
            for t in np.where(live)[0]:
                all_p.append(pval[t])
                all_key.append((int(fi), dl, int(t),
                                bool((abs(g[t]) > 0.5) and (c[t] > 0.7))))

        exp_null = float(null_tot.mean())
        per_feature[int(fi)] = {
            'observed_edges': obs_tot,
            'expected_null_edges': exp_null,
            'null_sd': float(null_tot.std()),
            'fdr': exp_null / obs_tot if obs_tot else np.nan,
        }
        for key in null_edges_by_thresh:
            null_edges_by_thresh[key].append(nulls_by_t[key].mean())
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(subset)}")

    # ---------------- BH across the retained tests ----------------
    all_p = np.array(all_p)
    passes_effect = np.array([k[3] for k in all_key], dtype=bool)
    if len(all_p):
        o = np.argsort(all_p)
        m = len(all_p)
        thresh = np.arange(1, m + 1) / m * 0.05
        passed = all_p[o] <= thresh
        cut = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
        bh_sig = np.zeros(m, dtype=bool)
        bh_sig[o[:cut]] = True
        # calibrated edge = BH-significant AND passes the effect-size / consistency screen
        keep = bh_sig & passes_effect
        for i in np.where(keep)[0]:
            fi = int(all_key[i][0])
            fdr_edges[fi] = fdr_edges.get(fi, 0) + 1
        for fi in subset:
            fdr_edges.setdefault(int(fi), 0)
        bh_frac = cut / m
        n_effect = int(passes_effect.sum())
        effect_surviving = int(keep.sum())
    else:
        bh_frac, cut, m, n_effect, effect_surviving = 0.0, 0, 0, 0, 0

    obs_sub = sum(per_feature[f]['observed_edges'] for f in per_feature)
    null_sub = sum(per_feature[f]['expected_null_edges'] for f in per_feature)
    fdr_global = null_sub / obs_sub if obs_sub else np.nan

    # extrapolate the calibrated graph size: FDR is degree-dependent, so use the
    # subset's observed->surviving ratio within degree strata
    surv_ratio_by_decile = []
    sub_sorted = sorted(per_feature, key=lambda f: raw_edges[f])
    for chunk in np.array_split(np.array(sub_sorted), 10):
        o_ = sum(per_feature[f]['observed_edges'] for f in chunk)
        s_ = sum(fdr_edges.get(int(f), 0) for f in chunk)
        surv_ratio_by_decile.append({'n': len(chunk),
                                     'min_observed': float(min(raw_edges[f] for f in chunk)),
                                     'max_observed': float(max(raw_edges[f] for f in chunk)),
                                     'mean_observed': o_ / len(chunk),
                                     'surviving_ratio': s_ / o_ if o_ else 0.0})

    # The permutation subset over-samples high-degree features, so the global calibrated graph
    # size is estimated by applying each decile's surviving ratio to all features whose observed
    # degree falls in that decile's range.
    bounds = [(dc['min_observed'], dc['max_observed'], dc['surviving_ratio'])
              for dc in surv_ratio_by_decile]
    est_total = 0.0
    for f, e in all_degrees.items():
        ratio = bounds[-1][2]
        for lo, hi, r in bounds:
            if lo <= e <= hi:
                ratio = r
                break
        est_total += e * ratio

    summary = {
        'n_features_traced': len(all_degrees),
        'n_features_retraced': len(raw_edges),
        'total_edges_raw': int(sum(all_degrees.values())),
        'total_edges_raw_retraced_subset': int(sum(raw_edges.values())),
        'total_edges_hedges_g': int(sum(g_edges.values())),
        'mean_edges_raw': float(np.mean(list(all_degrees.values()))),
        'permutation': {
            'n_features': len(per_feature), 'n_perm': args.n_perm,
            'observed_edges_subset': int(obs_sub),
            'expected_null_edges_subset': float(null_sub),
            'empirical_fdr_subset': float(fdr_global),
            'bh_surviving_fraction': float(bh_frac),
            'bh_surviving_tests': int(cut), 'bh_total_tests': int(m),
            'n_degenerate_targets_excluded': int(n_degenerate),
            'effect_size_edges_subset': int(n_effect),
            'effect_size_edges_surviving_bh': int(effect_surviving),
            'effect_size_edges_surviving_frac': (effect_surviving / n_effect) if n_effect else 0.0,
            'surviving_by_degree_decile': surv_ratio_by_decile,
            'estimated_total_edges_fdr_all_features': int(round(est_total)),
            'estimated_mean_edges_fdr_all_features': float(est_total / max(len(all_degrees), 1)),
        },
        'threshold_sweep_null': {f'd>{dt},c>{ct}': float(np.mean(v))
                                 for (dt, ct), v in null_edges_by_thresh.items()},
        'float16_storage_effect': fp16_effect,
    }

    # ---------------- threshold sensitivity: counts + hub-rank stability ----------------
    # Restrict the sweep to the permutation subset so that observed and null edge counts in
    # the FDR column share a denominator.
    sfis = np.array([f for f in sorted(raw_edges) if f in set(int(x) for x in subset)])
    base = np.array([sweep[(0.5, 0.7)][f] for f in sfis])
    from scipy import stats as sps
    sweep_out = []
    for (dt, ct), counts in sweep.items():
        v = np.array([counts[f] for f in sfis])
        rho = sps.spearmanr(base, v)[0]
        top20_base = set(sfis[np.argsort(-base)[:20]])
        top20_v = set(sfis[np.argsort(-v)[:20]])
        null_mean = float(np.mean(null_edges_by_thresh[(dt, ct)])) if null_edges_by_thresh[(dt, ct)] else np.nan
        sweep_out.append({
            'd_threshold': dt, 'consistency_threshold': ct,
            'n_features': int(len(sfis)),
            'total_edges': int(v.sum()), 'mean_edges': float(v.mean()),
            'median_edges': float(np.median(v)),
            'heavy_tail_frac_gt1000': float((v > 1000).mean()),
            'hub_rank_spearman_vs_default': float(rho),
            'top20_overlap_with_default': len(top20_base & top20_v),
            'expected_null_edges_per_feature': null_mean,
            'implied_fdr': null_mean / v.mean() if v.mean() else np.nan,
        })

    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(OUT, 'edges_raw.json'), 'w') as f:
        json.dump({str(k): v for k, v in raw_edges.items()}, f)
    with open(os.path.join(OUT, 'edges_fdr.json'), 'w') as f:
        json.dump({str(k): v for k, v in fdr_edges.items()}, f)
    with open(os.path.join(OUT, 'per_feature.json'), 'w') as f:
        json.dump(per_feature, f)
    with open(os.path.join(OUT, 'threshold_sweep.json'), 'w') as f:
        json.dump(sweep_out, f, indent=1)

    print("\n=== CALIBRATION ===")
    print(f"observed edges (subset)     : {obs_sub:,}")
    print(f"expected null edges (subset): {null_sub:,.0f}")
    print(f"empirical FDR               : {fdr_global:.1%}")
    print(f"BH q<0.05 surviving tests   : {cut:,}/{m:,} live tests = {bh_frac:.1%}")
    print(f"effect-size edges surviving : {effect_surviving:,}/{n_effect:,} = "
          f"{(effect_surviving/n_effect if n_effect else 0):.1%}")
    print(f"degenerate targets excluded : {n_degenerate:,}")
    print(f"estimated calibrated graph  : {est_total:,.0f} edges "
          f"({est_total/max(sum(all_degrees.values()),1):.1%} of raw {sum(all_degrees.values()):,}), "
          f"mean {est_total/max(len(all_degrees),1):.0f} edges/feature")
    print("\nthreshold sweep (per-feature means):")
    for s in sorted(sweep_out, key=lambda s: (s['d_threshold'], s['consistency_threshold'])):
        print(f"  |d|>{s['d_threshold']:<4} c>{s['consistency_threshold']:<4} "
              f"edges/feat={s['mean_edges']:7.1f} null={s['expected_null_edges_per_feature']:7.1f} "
              f"FDR={s['implied_fdr']:6.1%} hub-rank rho={s['hub_rank_spearman_vs_default']:.3f} "
              f"top20 overlap={s['top20_overlap_with_default']}/20")


if __name__ == '__main__':
    main()
