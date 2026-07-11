#!/usr/bin/env python3
"""P2b — Post-hoc analysis of the steering panel.

Three things the first-pass aggregation got wrong or left out:

1. Dose-response was summarised as Spearman(alpha, mean effect).  Most features push *away*
   from maturity, so a perfectly monotone feature whose effect grows more negative with alpha
   scores rho = -1.  The right question is whether the *magnitude* grows with dose and whether
   the *sign* is preserved across doses.

2. The ON/OFF contrast is the decisive test of whether directionality is encoded by the feature
   or is an artifact of the layer, and needs an explicit statistical test.

3. Concordance between the pseudotime-derived metric and the three readouts that never see the
   pseudotime signature has to be reported per layer, not only pooled: a pooled correlation can
   be dominated by between-layer spread.
"""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.

import glob
import json
import os

import numpy as np
from scipy import stats

PROJ = os.environ.get("SAE_PROJ_ROOT", ".")
OUT = os.path.join(PROJ, "experiments/revision_plos/P2_steering")
LAYERS = [0, 5, 11, 17]
READOUTS = ['delta_s_global', 'marker_score', 'delta_p_terminal', 'delta_pseudotime']


MIN_CELLS = 10


def load():
    return [json.load(open(p)) for p in
            sorted(glob.glob(os.path.join(OUT, 'features', '*.json'))) if '/._' not in p]


def main():
    feats_all = load()
    # A feature can only be steered in cells where it fires.  Some features fire in as few as
    # 2 cells, and their per-feature fraction-positive is meaningless.  All headline statistics
    # use features active in >= MIN_CELLS of the 160 early cells; the unfiltered panel is
    # reported alongside so the effect of the filter is visible.
    feats = [f for f in feats_all
             if '5.0' in f['alphas'] and f['alphas']['5.0']['n_cells'] >= MIN_CELLS]
    print(f"{len(feats)}/{len(feats_all)} features active in >= {MIN_CELLS} cells\n")
    res = {'min_cells': MIN_CELLS, 'n_features_kept': len(feats),
           'n_features_total': len(feats_all)}

    # ---------- 1. dose-response on magnitude, and sign stability ----------
    dose = {r: {'monotone_magnitude': 0, 'sign_stable': 0, 'n': 0, 'rho_abs': []}
            for r in READOUTS}
    for f in feats:
        alphas = sorted(float(a) for a in f['alphas'])
        if len(alphas) < 4:
            continue
        for r in READOUTS:
            means = np.array([f['alphas'][str(a)][r]['mean'] for a in alphas])
            rho_abs = stats.spearmanr(alphas, np.abs(means))[0]
            dose[r]['n'] += 1
            dose[r]['rho_abs'].append(float(rho_abs))
            if rho_abs > 0.8:
                dose[r]['monotone_magnitude'] += 1
            if len(set(np.sign(means[means != 0]))) <= 1:
                dose[r]['sign_stable'] += 1
    for r in READOUTS:
        d = dose[r]
        d['median_rho_abs'] = float(np.median(d['rho_abs']))
        del d['rho_abs']
    res['dose_response'] = dose

    # ---------- 2. ON vs OFF contrast, per layer, per readout ----------
    contrast = {}
    for alpha in ['2.0', '5.0']:
        for L in LAYERS:
            on = [f for f in feats if f['layer'] == L and f['kind'] == 'ON' and alpha in f['alphas']]
            off = [f for f in feats if f['layer'] == L and f['kind'] == 'OFF' and alpha in f['alphas']]
            if not on or not off:
                continue
            row = {'n_on': len(on), 'n_off': len(off)}
            for r in READOUTS:
                a = np.array([f['alphas'][alpha][r]['mean'] for f in on])
                b = np.array([f['alphas'][alpha][r]['mean'] for f in off])
                u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                row[r] = {'on_mean': float(a.mean()), 'off_mean': float(b.mean()),
                          'on_frac_pos': float((a > 0).mean()), 'off_frac_pos': float((b > 0).mean()),
                          'mannwhitney_p': float(p),
                          'direction': 'ON>OFF' if a.mean() > b.mean() else 'ON<OFF'}
            contrast[f'alpha{alpha}_L{L}'] = row
    res['on_off_contrast'] = contrast

    # ---------- 3. per-layer concordance with the non-circular readouts ----------
    conc = {}
    for alpha in ['5.0']:
        for L in LAYERS:
            grp = [f for f in feats if f['layer'] == L and alpha in f['alphas']]
            if len(grp) < 6:
                continue
            ds = np.array([f['alphas'][alpha]['delta_s_global']['mean'] for f in grp])
            row = {}
            for r in READOUTS[1:]:
                v = np.array([f['alphas'][alpha][r]['mean'] for f in grp])
                rho, p = stats.spearmanr(ds, v)
                row[r] = {'rho': float(rho), 'p': float(p),
                          'sign_agreement': float(np.mean(np.sign(ds) == np.sign(v)))}
            # agreement among the three non-circular readouts themselves
            for i in range(1, len(READOUTS)):
                for j in range(i + 1, len(READOUTS)):
                    a = np.array([f['alphas'][alpha][READOUTS[i]]['mean'] for f in grp])
                    b = np.array([f['alphas'][alpha][READOUTS[j]]['mean'] for f in grp])
                    rho, p = stats.spearmanr(a, b)
                    row[f'{READOUTS[i]}_vs_{READOUTS[j]}'] = {'rho': float(rho), 'p': float(p)}
            conc[f'L{L}'] = row
    res['per_layer_concordance'] = conc

    # ---------- 4. layer gradient among ON features ----------
    grad = {}
    for alpha in ['5.0']:
        on = [f for f in feats if f['kind'] == 'ON' and alpha in f['alphas']]
        lay = np.array([f['layer'] for f in on])
        for r in READOUTS:
            v = np.array([f['alphas'][alpha][r]['mean'] for f in on])
            fp = np.array([f['alphas'][alpha][r]['frac_positive'] for f in on])
            rho, p = stats.spearmanr(lay, v)
            rho_fp, p_fp = stats.spearmanr(lay, fp)
            # L17 vs all earlier layers
            u, pu = stats.mannwhitneyu(v[lay == 17], v[lay != 17], alternative='greater')
            grad[r] = {'spearman_layer_vs_effect': float(rho), 'p': float(p),
                       'spearman_layer_vs_fracpos': float(rho_fp), 'p_fracpos': float(p_fp),
                       'L17_gt_earlier_p': float(pu),
                       'L17_mean': float(v[lay == 17].mean()),
                       'earlier_mean': float(v[lay != 17].mean())}
    res['layer_gradient_on_features'] = grad

    # ---------- 5. per-feature cell counts (reviewer: report n per layer) ----------
    counts = {}
    for L in LAYERS:
        for kind in ['ON', 'OFF']:
            g = [f for f in feats_all if f['layer'] == L and f['kind'] == kind]
            if g:
                ns = [f['alphas']['5.0']['n_cells'] for f in g if '5.0' in f['alphas']]
                kept = [f for f in feats if f['layer'] == L and f['kind'] == kind]
                counts[f'L{L}_{kind}'] = {'n_features': len(g), 'n_features_kept': len(kept),
                                          'cells_min': int(min(ns)),
                                          'cells_median': float(np.median(ns)),
                                          'cells_max': int(max(ns))}
    res['panel_sizes'] = counts

    with open(os.path.join(OUT, 'readout_analysis.json'), 'w') as f:
        json.dump(res, f, indent=1)

    print("=== dose-response (32 features, alpha in {1.5,2,3,5,8}) ===")
    for r, d in dose.items():
        print(f"  {r:<20} |effect| monotone in alpha: {d['monotone_magnitude']}/{d['n']}   "
              f"sign stable across doses: {d['sign_stable']}/{d['n']}   "
              f"median rho(alpha,|effect|)={d['median_rho_abs']:+.2f}")

    print("\n=== ON vs OFF contrast (alpha=5) ===")
    for L in LAYERS:
        k = f'alpha5.0_L{L}'
        if k not in contrast:
            continue
        c = contrast[k]
        print(f"  L{L} (ON n={c['n_on']}, OFF n={c['n_off']})")
        for r in READOUTS:
            print(f"    {r:<20} ON {c[r]['on_frac_pos']:.2f} vs OFF {c[r]['off_frac_pos']:.2f} "
                  f"frac-pos; means {c[r]['on_mean']:+.2e} vs {c[r]['off_mean']:+.2e}  "
                  f"p={c[r]['mannwhitney_p']:.3g}  {c[r]['direction']}")

    print("\n=== layer gradient among ON features (alpha=5) ===")
    for r, g in grad.items():
        print(f"  {r:<20} rho(layer, effect)={g['spearman_layer_vs_effect']:+.3f} (p={g['p']:.3g})  "
              f"L17 mean={g['L17_mean']:+.2e} vs earlier {g['earlier_mean']:+.2e} "
              f"(one-sided p={g['L17_gt_earlier_p']:.3g})")

    print("\n=== per-layer concordance of delta_s with non-circular readouts (alpha=5) ===")
    for L, row in conc.items():
        s = '  '.join(f"{r.replace('delta_','d_')[:14]}: rho={row[r]['rho']:+.2f}"
                      for r in READOUTS[1:])
        print(f"  {L}: {s}")

    print("\n=== panel sizes ===")
    for k, v in counts.items():
        print(f"  {k:<10} {v['n_features']} features ({v['n_features_kept']} kept), "
              f"cells/feature median {v['cells_median']:.0f} [{v['cells_min']}-{v['cells_max']}]")


if __name__ == '__main__':
    main()
