#!/usr/bin/env python3
"""Regenerate the PLOS ONE figures from the revision experiment outputs.

Fig1  calibrated circuit map: degree distribution, coverage-sensitivity, attenuation, FDR
Fig2  hub architecture: degree vs annotation, top-20 table, enrichment tests, frequency confound
Fig3  higher-order ablation: redundancy by order, marginal contribution, superadditivity by class
Fig4  steering: per-feature directionality with ON/OFF contrast, dose-response, per-layer summary
Fig5  non-circular steering readouts
S5_Fig  hub causal importance (masked-LM damage)
S6_Fig  TS-native SAE trace vs K562
S7_Fig  degree-distribution model comparison
"""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJ = os.environ.get("SAE_PROJ_ROOT", ".")
RP = os.path.join(PROJ, 'experiments/revision_plos')
EXH = os.path.join(PROJ, 'experiments/phase8_exhaustive_tracing')
OUT = os.path.join(PROJ, 'paper/plos_one_submission')

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif', 'axes.spines.top': False, 'axes.spines.right': False,
})
C = {'ann': '#2166ac', 'unann': '#b2182b', 'null': '#999999', 'obs': '#2166ac',
     'L0': '#d73027', 'L5': '#fc8d59', 'L11': '#91bfdb', 'L17': '#4575b4',
     'ON': '#2166ac', 'OFF': '#b2182b',
     'same': '#4575b4', 'cross': '#d73027', 'random': '#999999'}


def jload(p):
    with open(p) as f:
        return json.load(f)


def maybe(p):
    return jload(p) if os.path.exists(p) else None


def load_features(edge_override=None):
    feats = []
    for p in sorted(glob.glob(os.path.join(EXH, 'feature_*.json'))):
        if '/._' in p:
            continue
        j = jload(p)
        e = j['total_significant_edges']
        if edge_override is not None:
            if str(j['feature_idx']) not in edge_override:
                continue
            e = edge_override[str(j['feature_idx'])]
        feats.append({'idx': j['feature_idx'], 'edges': e,
                      'ann': j['label'] != 'unannotated', 'label': j['label'],
                      'freq': j['activation_freq'],
                      'dl': {k: v['n_significant'] for k, v in j['downstream_edges'].items()}})
    return feats


# ---------------------------------------------------------------- Fig 1
def fig1():
    cal = maybe(os.path.join(RP, 'P1_calibration/summary.json'))
    sweep = maybe(os.path.join(RP, 'P1_calibration/threshold_sweep.json'))
    fdr_edges = maybe(os.path.join(RP, 'P1_calibration/edges_fdr.json'))
    feats = load_features()
    edges = np.array([f['edges'] for f in feats])

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))

    ax[0].hist(edges, bins=60, color=C['obs'], alpha=.85)
    ax[0].axvline(np.median(edges), color='k', ls='--', lw=1,
                  label=f'median {np.median(edges):.0f}')
    ax[0].set_xlabel('significant edges per feature')
    ax[0].set_ylabel('features')
    ax[0].set_title(f'A  Degree distribution (n={len(feats)})')
    ax[0].legend(frameon=False)

    # calibrated vs uncalibrated degree, on the 600-feature calibration subset
    if fdr_edges:
        idx = [f['idx'] for f in feats if str(f['idx']) in fdr_edges]
        raw = np.array([e for f, e in zip(feats, edges) if str(f['idx']) in fdr_edges])
        cal_deg = np.array([fdr_edges[str(i)] for i in idx])
        ax[1].scatter(raw, cal_deg, s=5, c=C['obs'], alpha=.5)
        lim = max(raw.max(), cal_deg.max()) * 1.05
        ax[1].plot([0, lim], [0, lim], 'k--', lw=1, label='no loss')
        from scipy import stats as sps
        rho = sps.spearmanr(raw, cal_deg)[0]
        ax[1].set_xlabel('degree, uncalibrated')
        ax[1].set_ylabel('degree, FDR-controlled')
        ax[1].set_title(f'B  Effect of FDR control\n(n={len(raw)} subset, $\\rho$={rho:.2f})')
        ax[1].legend(frameon=False)

    dl = ['6', '11', '17']
    tot = [sum(f['dl'].get(d, 0) for f in feats) for d in dl]
    ax[2].plot([6, 11, 17], tot, 'o-', color=C['obs'])
    for x, y in zip([6, 11, 17], tot):
        ax[2].annotate(f'{y/1000:.0f}K ({100*y/sum(tot):.0f}%)', (x, y),
                       textcoords='offset points', xytext=(0, 8), ha='center', fontsize=7)
    ax[2].set_xlabel('downstream layer'); ax[2].set_ylabel('total edges')
    ax[2].set_title('C  Signal attenuation')
    ax[2].set_ylim(0, max(tot) * 1.25)

    if sweep:
        srows = [r for r in sweep if r['consistency_threshold'] == 0.7]
        srows.sort(key=lambda r: r['d_threshold'])
        d = [r['d_threshold'] for r in srows]
        obs = [r['mean_edges'] for r in srows]
        nul = [max(r['expected_null_edges_per_feature'], 1e-2) for r in srows]
        ax[3].plot(d, obs, 'o-', color=C['obs'], label='observed')
        ax[3].plot(d, nul, 's--', color=C['null'], label='permutation null')
        ax[3].set_yscale('log')
        ax[3].set_xlabel("|Cohen's $d$| threshold")
        ax[3].set_ylabel('edges per feature')
        ax[3].axvline(0.5, color='k', ls=':', lw=1)
        a2 = ax[3].twinx()
        a2.plot(d, [100 * r['implied_fdr'] for r in srows], '^-', color=C['unann'], lw=1)
        a2.set_ylabel('empirical FDR (%)', color=C['unann'])
        a2.tick_params(axis='y', colors=C['unann'])
        a2.spines['right'].set_visible(True)
        ax[3].legend(frameon=False, loc='upper right')
        ax[3].set_title('D  Edge calling vs empirical null')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig1.pdf'))
    plt.close(fig)
    print('Fig1')


# ---------------------------------------------------------------- Fig 2
def fig2():
    ann = maybe(os.path.join(RP, 'P5_annotation/annotation_enrichment_raw.json'))
    feats = load_features()
    edges = np.array([f['edges'] for f in feats])
    isann = np.array([f['ann'] for f in feats])
    freq = np.array([f['freq'] for f in feats])
    order = np.argsort(-edges)

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))

    r = np.arange(1, len(edges) + 1)
    ax[0].scatter(r[isann[order]], edges[order][isann[order]], s=2, c=C['ann'], label='annotated')
    ax[0].scatter(r[~isann[order]], edges[order][~isann[order]], s=2, c=C['unann'], label='unannotated')
    ax[0].set_xscale('log'); ax[0].set_xlabel('rank'); ax[0].set_ylabel('edges')
    ax[0].set_title('A  Features ranked by degree'); ax[0].legend(frameon=False, markerscale=3)

    top = [feats[i] for i in order[:20]]
    y = np.arange(20)[::-1]
    ax[1].barh(y, [t['edges'] for t in top],
               color=[C['ann'] if t['ann'] else C['unann'] for t in top])
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([f"F{t['idx']}" for t in top], fontsize=5.5)
    ax[1].set_xlabel('edges'); ax[1].set_title('B  Top-20 hubs')

    if ann:
        l5 = ann[0]
        labs, ors, los, his, ps = [], [], [], [], []
        for layer in ann:
            f20 = layer['fisher'][0]
            labs.append(layer['layer'])
            ors.append(f20['odds_ratio'])
            ps.append(f20['p_value'])
        x = np.arange(len(labs))
        cols = [C['unann'] if o < 1 else C['ann'] for o in ors]
        ax[2].bar(x, np.log2(np.maximum(ors, 1e-2)), color=cols)
        ax[2].axhline(0, color='k', lw=1)
        for i, p in enumerate(ps):
            ax[2].text(i, np.log2(max(ors[i], 1e-2)) + (.05 if ors[i] > 1 else -.2),
                       f'p={p:.3f}', ha='center', fontsize=6)
        ax[2].set_xticks(x); ax[2].set_xticklabels(labs)
        ax[2].set_ylabel('$\\log_2$ odds ratio\n(annotated among top-20 hubs)')
        ax[2].set_title('C  Annotation enrichment in hubs')

    ax[3].scatter(freq, edges, s=2, c=np.where(isann, C['ann'], C['unann']), alpha=.4)
    ax[3].set_xscale('log')
    ax[3].set_xlabel('activation frequency'); ax[3].set_ylabel('edges')
    if ann:
        rho = ann[0]['frequency']['spearman_freq_vs_edges']
        ax[3].set_title(f'D  Degree tracks firing rate ($\\rho$={rho:.2f})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig2.pdf'))
    plt.close(fig)
    print('Fig2')


# ---------------------------------------------------------------- Fig 3
def fig3():
    s = maybe(os.path.join(RP, 'P3_triplets/summary.json'))
    if not s:
        print('Fig3 skipped (no P3)')
        return
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))

    groups = ['same_pathway', 'cross_pathway', 'random']
    groups = [g for g in groups if g in s['groups']]
    cols = {'same_pathway': C['same'], 'cross_pathway': C['cross'], 'random': C['random']}

    ax[0].axhline(1.0, color='k', ls=':', lw=1)
    for i, g in enumerate(groups):
        rs = [r for r in s['per_triplet'] if r['type'] == g]
        pw = [r['pairwise_ratio_mean'] for r in rs]
        tw = [r['threeway_ratio'] for r in rs]
        ax[0].plot([1, 2], [np.median(pw), np.median(tw)], 'o-', color=cols[g],
                   label=g.replace('_', ' '))
        for a, b in zip(pw, tw):
            ax[0].plot([1, 2], [a, b], '-', color=cols[g], alpha=.2, lw=.8)
    ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(['1 (single)', '2 (pairwise)', '3 (three-way)'])
    ax[0].set_xlim(0.5, 2.5)
    ax[0].set_xlabel('interaction order'); ax[0].set_ylabel('redundancy ratio $R_n$')
    ax[0].legend(frameon=False); ax[0].set_title('A  Redundancy deepens with order')

    for i, g in enumerate(groups):
        rs = [r for r in s['per_triplet'] if r['type'] == g]
        v = [r['marginal_C_given_AB_median'] for r in rs]
        ax[1].bar(i, np.mean(v), yerr=np.std(v), color=cols[g], width=.6, capsize=3)
    ax[1].set_xticks(range(len(groups)))
    ax[1].set_xticklabels([g.replace('_', '\n') for g in groups])
    ax[1].set_ylabel('median $|d_{ABC}| - |d_{AB}|$')
    ax[1].set_title('B  Marginal third-feature effect')

    for i, g in enumerate(groups):
        gg = s['groups'][g]
        rate = 100 * gg['superadditive_rate']
        n, N = gg['n_superadditive'], gg['n_targets']
        lo, hi = wilson(n, N)
        ax[2].bar(i, rate, color=cols[g], width=.6,
                  yerr=[[rate - 100 * lo], [100 * hi - rate]], capsize=3)
        ax[2].text(i, 100 * hi + .05, f'{n}/{N}', ha='center', fontsize=7)
    ax[2].set_xticks(range(len(groups)))
    ax[2].set_xticklabels([g.replace('_', '\n') for g in groups])
    ax[2].set_ylabel('superadditive targets (%)')
    ax[2].set_title('C  Synergy is rare, and cross-pathway')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig3.pdf'))
    plt.close(fig)
    print('Fig3')


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0, c - h), min(1, c + h)


# ---------------------------------------------------------------- Fig 4 / 5
def steering_feats(min_cells=10):
    fs = [jload(p) for p in sorted(glob.glob(os.path.join(RP, 'P2_steering/features/*.json')))
          if '/._' not in p]
    return [f for f in fs if '5.0' in f['alphas'] and f['alphas']['5.0']['n_cells'] >= min_cells]


def fig4():
    feats = steering_feats()
    if not feats:
        print('Fig4 skipped')
        return
    ra = maybe(os.path.join(RP, 'P2_steering/readout_analysis.json'))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))

    layers = [0, 5, 11, 17]
    xpos, k = [], 0
    for L in layers:
        for kind in ['ON', 'OFF']:
            g = sorted([f for f in feats if f['layer'] == L and f['kind'] == kind],
                       key=lambda f: -f['alphas']['5.0']['delta_p_terminal']['frac_positive'])
            for f in g:
                ax[0].bar(k, f['alphas']['5.0']['delta_p_terminal']['frac_positive'],
                          color=C[kind], width=.85)
                k += 1
            xpos.append((k - len(g) / 2, f'L{L}\n{kind}\nn={len(g)}'))
            k += 1
    ax[0].axhline(.5, color='k', ls=':', lw=1)
    ax[0].set_xticks([p for p, _ in xpos]); ax[0].set_xticklabels([l for _, l in xpos], fontsize=6.5)
    ax[0].set_ylabel('fraction of cells with $\\Delta P(\\mathrm{terminal})>0$')
    ax[0].set_title('A  Per-feature directionality ($\\alpha$=5)')
    ax[0].legend(handles=[Line2D([], [], color=C['ON'], lw=6, label='ON-switch'),
                          Line2D([], [], color=C['OFF'], lw=6, label='OFF-switch')],
                 frameon=False, fontsize=7)

    dose = [f for f in feats if len(f['alphas']) >= 4]
    for f in dose:
        a = sorted(float(x) for x in f['alphas'])
        v = [abs(f['alphas'][str(x)]['delta_p_terminal']['mean']) for x in a]
        ax[1].plot(a, v, '-', color=C[f['kind']], alpha=.45, lw=.9)
    ax[1].set_xlabel('steering coefficient $\\alpha$')
    ax[1].set_ylabel('$|\\Delta P(\\mathrm{terminal})|$')
    ax[1].set_yscale('log')
    if ra:
        d = ra['dose_response']['delta_p_terminal']
        ax[1].set_title(f"B  Dose-response ({d['monotone_magnitude']}/{d['n']} monotone)")
    else:
        ax[1].set_title('B  Dose-response')

    w = .35
    for i, kind in enumerate(['ON', 'OFF']):
        m, e = [], []
        for L in layers:
            g = [f['alphas']['5.0']['delta_p_terminal']['frac_positive']
                 for f in feats if f['layer'] == L and f['kind'] == kind]
            m.append(np.mean(g) if g else 0)
            e.append(np.std(g) / max(np.sqrt(len(g)), 1) if g else 0)
        ax[2].bar(np.arange(4) + (i - .5) * w, m, w, yerr=e, capsize=2,
                  color=C[kind], label=f'{kind}-switch')
    ax[2].axhline(.5, color='k', ls=':', lw=1)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels([f'L{L}' for L in layers])
    ax[2].set_ylabel('mean fraction positive')
    ax[2].set_title('C  Layer $\\times$ switch direction')
    ax[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig4.pdf'))
    plt.close(fig)
    print('Fig4')


def fig5():
    feats = steering_feats()
    if not feats:
        print('Fig5 skipped')
        return
    readouts = [('delta_s_global', '$\\Delta s$ (pseudotime signature)'),
                ('marker_score', 'marker-panel score'),
                ('delta_p_terminal', '$\\Delta P$(terminal), held-out probe'),
                ('delta_pseudotime', '$\\Delta$ predicted pseudotime')]
    layers = [0, 5, 11, 17]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.2), sharey=False)
    for j, (r, title) in enumerate(readouts):
        w = .35
        for i, kind in enumerate(['ON', 'OFF']):
            m, e = [], []
            for L in layers:
                g = [f['alphas']['5.0'][r]['frac_positive']
                     for f in feats if f['layer'] == L and f['kind'] == kind]
                m.append(np.mean(g) if g else 0)
                e.append(np.std(g) / max(np.sqrt(len(g)), 1) if g else 0)
            ax[j].bar(np.arange(4) + (i - .5) * w, m, w, yerr=e, capsize=2,
                      color=C[kind], label=f'{kind}')
        ax[j].axhline(.5, color='k', ls=':', lw=1)
        ax[j].set_xticks(range(4)); ax[j].set_xticklabels([f'L{L}' for L in layers])
        ax[j].set_ylim(0, 1)
        ax[j].set_title(title, fontsize=8)
        if j == 0:
            ax[j].set_ylabel('fraction of cells positive')
            ax[j].legend(frameon=False)
    fig.suptitle('Steering directionality under four readouts ($\\alpha$=5); '
                 'only the leftmost uses the pseudotime signature', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig5.pdf'))
    plt.close(fig)
    print('Fig5')


# ---------------------------------------------------------------- supplements
def s5_fig():
    s = maybe(os.path.join(RP, 'P6_hub_importance/summary.json'))
    if not s:
        print('S5 skipped')
        return
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    arms = ['hub', 'freq_matched', 'random']
    data = [[r['delta_ce'] for r in s['arms'][a]] for a in arms]
    bp = ax[0].boxplot(data, labels=['hub', 'frequency-\nmatched', 'random'],
                       patch_artist=True, showfliers=False)
    for p, c in zip(bp['boxes'], [C['unann'], C['L11'], C['null']]):
        p.set_facecolor(c)
    ax[0].axhline(0, color='k', lw=1, ls=':')
    ax[0].set_ylabel('$\\Delta$ masked-LM cross-entropy')
    ax[0].set_title('A  Damage from ablating one feature')

    allr = [r for a in arms for r in s['arms'][a]]
    ax[1].scatter([r['edges'] for r in allr], [r['delta_ce'] for r in allr],
                  c=[r['freq'] for r in allr], s=14, cmap='viridis')
    ax[1].set_xlabel('feature degree'); ax[1].set_ylabel('$\\Delta$ cross-entropy')
    t = s['tests']['partial_spearman_edges_vs_delta_ce_given_freq']
    ax[1].set_title(f"B  degree | frequency: $\\rho$={t['rho']:.2f}, p={t['p']:.2g}")
    plt.colorbar(ax[1].collections[0], ax=ax[1], label='activation freq')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'S5_Fig.pdf'))
    plt.close(fig)
    print('S5_Fig')


def s6_fig():
    s = maybe(os.path.join(RP, 'P7_ts_native/summary.json'))
    if not s:
        print('S6 skipped')
        return
    ts = [jload(p) for p in sorted(glob.glob(os.path.join(RP, 'P7_ts_native/json/*.json')))
          if '/._' not in p]
    e_ts = np.array([r['total_significant_edges'] for r in ts])
    a_ts = np.array([r['label'] != 'unannotated' for r in ts])

    feats = load_features()
    e_k = np.array([sum(f['dl'].get(d, 0) for d in ['11', '17']) for f in feats])

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].hist(e_k, bins=50, density=True, alpha=.6, color=C['L5'], label='K562-trained SAE')
    ax[0].hist(e_ts, bins=50, density=True, alpha=.6, color=C['L17'], label='TS-trained SAE')
    ax[0].set_xlabel('edges per feature (downstream 11+17)'); ax[0].set_ylabel('density')
    ax[0].legend(frameon=False); ax[0].set_title('A  Degree distribution')

    keys = ['skewness', 'gini', 'top20_unannotated_frac', 'spearman_freq_vs_edges']
    lbl = ['skewness', 'Gini', 'top-20 unann.', r'$\rho$(freq, deg)']
    x = np.arange(len(keys)); w = .35
    ax[1].bar(x - w / 2, [s['k562_same_downstream'][k] for k in keys], w,
              color=C['L5'], label='K562 SAE')
    ax[1].bar(x + w / 2, [s['ts_native'][k] for k in keys], w, color=C['L17'], label='TS SAE')
    ax[1].set_xticks(x); ax[1].set_xticklabels(lbl, fontsize=7)
    ax[1].legend(frameon=False); ax[1].set_title('B  Architecture statistics')

    o = np.argsort(-e_ts)[:20]
    ax[2].barh(np.arange(20)[::-1], e_ts[o],
               color=[C['ann'] if a_ts[i] else C['unann'] for i in o])
    ax[2].set_yticks([]); ax[2].set_xlabel('edges')
    ax[2].set_title(f"C  TS-native top-20 hubs\n({int((~a_ts[o]).sum())}/20 unannotated)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'S6_Fig.pdf'))
    plt.close(fig)
    print('S6_Fig')


def s7_fig():
    t = maybe(os.path.join(RP, 'P4_tail/tail_fit_raw.json'))
    if not t:
        print('S7 skipped')
        return
    feats = load_features()
    e = np.sort(np.array([f['edges'] for f in feats]))
    e = e[e > 0]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ccdf = 1 - np.arange(len(e)) / len(e)
    ax[0].loglog(e, ccdf, '.', ms=2, color=C['obs'], label='observed')
    xmin, alpha = t['powerlaw']['xmin'], t['powerlaw']['alpha']
    tail = e[e >= xmin]
    if len(tail):
        c0 = (e >= xmin).mean()
        ax[0].loglog(tail, c0 * (tail / xmin) ** (1 - alpha), '-', color=C['unann'],
                     label=f'power law $\\alpha$={alpha:.2f}\n($x_{{min}}$={xmin}, p={t["powerlaw"]["gof_p"]:.2f})')
    ax[0].set_xlabel('edges'); ax[0].set_ylabel('P(X $\\geq$ x)')
    ax[0].legend(frameon=False, fontsize=7); ax[0].set_title('A  Tail of the degree distribution')

    names = list(t['comparisons'])
    z = [t['comparisons'][n]['vuong_z'] for n in names]
    p = [t['comparisons'][n]['vuong_p'] for n in names]
    ax[1].barh(np.arange(len(names)), z,
               color=[C['null'] if pp > .1 else (C['obs'] if zz > 0 else C['unann'])
                      for zz, pp in zip(z, p)])
    ax[1].axvline(0, color='k', lw=1)
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels([n.replace('_', ' ') for n in names], fontsize=7)
    ax[1].set_xlabel('Vuong $z$  (>0 favours power law)')
    ax[1].set_title('B  Model comparison (grey: inconclusive)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'S7_Fig.pdf'))
    plt.close(fig)
    print('S7_Fig')


if __name__ == '__main__':
    for f in [fig1, fig2, fig3, fig4, fig5, s5_fig, s6_fig, s7_fig]:
        try:
            f()
        except Exception as ex:
            print(f'{f.__name__} FAILED: {ex}')
