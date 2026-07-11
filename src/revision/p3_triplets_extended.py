#!/usr/bin/env python3
"""P3 — Higher-order ablation: more triplets, cross-pathway contrast, random control.

The published result tested 8 triplets (7 same-pathway, 1 cross-pathway) and concluded "zero
synergy".  Two problems:
  * 8 triplets cannot support a universal negative;
  * the single cross-pathway triplet (DDR x mitosis) produced the *most* superadditive targets
    of any triplet (4), which is evidence for the hypothesis being dismissed rather than
    against it.

This script adds same-pathway, cross-pathway and random triplets under one protocol and asks
whether the superadditive rate depends on the pathway relation.

Ablation semantics match src/18_higher_order_ablation.py exactly: features are ablated
sequentially along the forward pass, each one re-encoded on the already-perturbed residual
stream.  The implementation differs only in running the tail of the network from the earliest
ablated layer instead of re-running the full model with hooks (a ~3x saving), and in retaining
the per-cell deltas so that bootstrap CIs can be computed for every quantity.

Classification thresholds follow the original code: ratio < 0.8 subadditive, 0.8-1.2 additive,
> 1.2 superadditive.
"""

import argparse
import gc
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
PROJ = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
ORIG = os.path.join(PROJ, "src", "20_exhaustive_feature_tracing.py")
OUT = os.path.join(PROJ, "experiments/revision_plos/P3_triplets")
PUBLISHED = os.path.join(PROJ, "experiments/phase8_higher_order_ablation")

LAYERS = (0, 5, 11)
TARGET_LAYER = 17
N_FEATURES = 4608
N_CELLS = 100
MIN_FREQ = 0.02
SEED = 7
COND = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
D_THRESH, C_THRESH = 0.5, 0.7

# Ordered: the first matching pathway wins.  DDR precedes metabolism so that "DNA Metabolic
# Process" is not filed under metabolism, and metabolism explicitly excludes DNA/RNA terms.
PATHWAYS = [
    ('ddr', ['dna damage', 'dna repair', 'double-strand', 'dna replication', 'dna metabolic'], []),
    ('mitosis', ['mitotic', 'spindle', 'cell cycle', 'g2/m', 'chromosome segregation',
                 'kinetochore', 'cell division'], []),
    ('chromatin', ['chromatin', 'histone', 'nucleosome'], []),
    ('vesicle', ['vesicle', 'golgi', 'endosom', 'protein transport', 'secretion', 'exocyt'], []),
    ('translation', ['translation', 'ribosom', 'rrna'], []),
    ('rna', ['mrna', 'splic', 'rna processing', 'ncrna'], []),
    ('metabolism', ['metabolic', 'glycoly', 'oxidative phosphorylation', 'lipid', 'cholesterol',
                    'fatty acid', 'glycerophospholipid'], ['dna', 'rna', 'protein']),
]


def load_orig():
    spec = importlib.util.spec_from_file_location("orig_trace", ORIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pathway_of(label):
    ll = label.lower()
    for pw, keys, excl in PATHWAYS:
        if any(k in ll for k in keys) and not any(x in ll for x in excl):
            return pw
    return None


def feature_pool(orig):
    pool = {L: {} for L in LAYERS}
    for L in LAYERS:
        active, info = orig.load_feature_catalog(L)
        labels = orig.load_feature_labels(L)
        for fi in active:
            if info[fi]['activation_freq'] < MIN_FREQ:
                continue
            lab = labels.get(fi, 'unannotated')
            pw = pathway_of(lab) if lab != 'unannotated' else None
            pool[L][fi] = {'label': lab, 'pathway': pw,
                           'freq': info[fi]['activation_freq']}
    for L in LAYERS:
        print(f"  L{L}: {len(pool[L])} features with freq >= {MIN_FREQ}")
    return pool


def select_triplets(pool, n_same=4, n_cross=6, n_random=4):
    rng = np.random.default_rng(SEED)
    triplets = []

    def best(L, pw, exclude):
        cands = [(fi, v) for fi, v in pool[L].items()
                 if v['pathway'] == pw and fi not in exclude]
        cands.sort(key=lambda x: -x[1]['freq'])
        return cands[0] if cands else None

    # same-pathway: one feature per layer from the same pathway
    for pw in ['mitosis', 'vesicle', 'translation', 'chromatin', 'rna', 'metabolism']:
        picks, used = [], set()
        for L in LAYERS:
            b = best(L, pw, used)
            if b is None:
                break
            picks.append((L, b[0], b[1]))
            used.add(b[0])
        if len(picks) == 3:
            triplets.append({'type': 'same_pathway', 'pathway': pw, 'features': picks})
        if len(triplets) >= n_same:
            break

    # cross-pathway: three different pathways, one per layer
    combos = [('ddr', 'mitosis', 'mitosis'), ('mitosis', 'ddr', 'chromatin'),
              ('vesicle', 'metabolism', 'translation'), ('mitosis', 'metabolism', 'vesicle'),
              ('translation', 'ddr', 'rna'), ('chromatin', 'vesicle', 'mitosis'),
              ('rna', 'chromatin', 'metabolism'), ('metabolism', 'translation', 'ddr')]
    for combo in combos:
        if len(set(combo)) < 2:
            continue  # a "cross-pathway" triplet must span at least two pathways
        picks, used = [], set()
        for L, pw in zip(LAYERS, combo):
            b = best(L, pw, used)
            if b is None:
                break
            picks.append((L, b[0], b[1]))
            used.add(b[0])
        if len(picks) == 3 and all(p[2]['pathway'] == pw for p, pw in zip(picks, combo)):
            triplets.append({'type': 'cross_pathway', 'pathway': '_x_'.join(combo),
                             'features': picks})
        if sum(1 for t in triplets if t['type'] == 'cross_pathway') >= n_cross:
            break

    # random: three features that share no annotated pathway, frequency-matched to the above
    target_freqs = [np.median([f[2]['freq'] for t in triplets for f in t['features']
                               if f[0] == L]) for L in LAYERS]
    n_drawn, attempts = 0, 0
    while n_drawn < n_random and attempts < 200:
        attempts += 1
        picks = []
        for L, tf in zip(LAYERS, target_freqs):
            items = list(pool[L].items())
            fr = np.array([v['freq'] for _, v in items])
            near = np.argsort(np.abs(np.log(fr) - np.log(tf)))[:50]
            fi, v = items[rng.choice(near)]
            picks.append((L, fi, v))
        pws = [p[2]['pathway'] for p in picks if p[2]['pathway']]
        if len(set(pws)) < len(pws):
            continue  # two picks share a pathway: redraw
        triplets.append({'type': 'random', 'pathway': 'random', 'features': picks})
        n_drawn += 1

    for t in triplets:
        t['id'] = '_x_'.join(f"L{L}_F{fi}" for L, fi, _ in t['features'])
        print(f"  [{t['type']:<13}] {t['pathway']:<40} " +
              ' | '.join(f"L{L}F{fi} {v['label'][:26]}" for L, fi, v in t['features']))
    return triplets


def build_model(device, orig):
    from transformers import BertForMaskedLM
    m = BertForMaskedLM.from_pretrained(orig.MODEL_NAME, subfolder=orig.MODEL_SUBFOLDER,
                                        output_hidden_states=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def ablate_stream(model, sae_cache, means, clean_hiddens, gp, ablations, device):
    """Run the network from the earliest ablated layer, ablating each feature on the
    already-perturbed stream, and return the target-layer hidden state."""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.
    import torch
    start = min(ablations)
    h = clean_hiddens[start].clone()  # output of layer `start`, still clean

    for L in sorted(ablations):
        if L != start:
            pass  # h is the current output of layer L
        sae, _ = sae_cache.get(L)
        with torch.no_grad():
            h_sp, _ = sae.encode(h[gp] - means[L])
            h_ab = h_sp.clone()
            for fi in ablations[L]:
                h_ab[:, fi] = 0.0
            dh = sae.decode(h_ab) - sae.decode(h_sp)
        h[gp] += dh
        nxt = min((x for x in ablations if x > L), default=TARGET_LAYER)
        with torch.no_grad():
            hh = h.unsqueeze(0)
            for l in range(L + 1, nxt + 1):
                hh = model.bert.encoder.layer[l](hh)[0]
            h = hh[0]
    return h


def active_mask(sae, mean, hidden, gp, feats):
    import torch
    with torch.no_grad():
        _, topk = sae.encode(hidden[gp] - mean)
    m = torch.zeros(len(gp), dtype=torch.bool, device=topk.device)
    for fi in feats:
        m |= (topk == fi).any(dim=1)
    return m


def run_triplet(t, all_tokens, model, device, sae_cache, means, n_cells):
    import torch
    (lA, fA, _), (lB, fB, _), (lC, fC, _) = t['features']
    abl = {'A': {lA: [fA]}, 'B': {lB: [fB]}, 'C': {lC: [fC]},
           'AB': {lA: [fA], lB: [fB]}, 'AC': {lA: [fA], lC: [fC]},
           'BC': {lB: [fB], lC: [fC]},
           'ABC': {lA: [fA], lB: [fB], lC: [fC]}}
    # merge same-layer entries
    for k, a in abl.items():
        merged = {}
        for L, fs in a.items():
            merged.setdefault(L, []).extend(fs)
        abl[k] = merged

    dst_sae, _ = sae_cache.get(TARGET_LAYER)
    per_cell = {c: [] for c in COND}
    n_active = {c: 0 for c in COND}

    for ci in range(min(n_cells, len(all_tokens))):
        tokens = all_tokens[ci]
        gp_np = np.where((tokens != 2) & (tokens != 3))[0]
        if not len(gp_np):
            continue
        ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        am = torch.ones(1, len(tokens), dtype=torch.long, device=device)
        gp = torch.tensor(gp_np, dtype=torch.long, device=device)

        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=am)
        clean_hiddens = {L: o.hidden_states[L + 1][0].clone() for L in set([lA, lB, lC])}
        clean_target = o.hidden_states[TARGET_LAYER + 1][0][gp]
        with torch.no_grad():
            clean_sp, _ = dst_sae.encode(clean_target - means[TARGET_LAYER])
        del o

        masks = {}
        for L, fi in [(lA, fA), (lB, fB), (lC, fC)]:
            sae, _ = sae_cache.get(L)
            masks[(L, fi)] = active_mask(sae, means[L], clean_hiddens[L], gp, [fi])

        for cond in COND:
            keys = [(lA, fA), (lB, fB), (lC, fC)]
            sel = [keys['ABC'.index(ch)] for ch in cond]
            m = masks[sel[0]].clone()
            for s in sel[1:]:
                m |= masks[s]
            if not m.any():
                continue
            n_active[cond] += 1
            h_t = ablate_stream(model, sae_cache, means, clean_hiddens, gp, abl[cond], device)
            with torch.no_grad():
                abl_sp, _ = dst_sae.encode(h_t[gp] - means[TARGET_LAYER])
            pos = torch.where(m)[0]
            per_cell[cond].append(
                (abl_sp[pos] - clean_sp[pos]).mean(dim=0).cpu().numpy().astype(np.float64))

        del clean_hiddens, clean_sp
        if device.type == 'mps':
            torch.mps.empty_cache()

    return {c: (np.stack(v) if v else np.zeros((0, N_FEATURES))) for c, v in per_cell.items()}, n_active


def d_of(mat):
    if mat.shape[0] < 2:
        return np.zeros(N_FEATURES), np.zeros(N_FEATURES)
    d = mat.mean(0) / np.maximum(mat.std(0, ddof=1), 1e-10)
    pos = (mat > 0).sum(0)
    return d, np.maximum(pos, mat.shape[0] - pos) / mat.shape[0]


def analyse_triplet(deltas, rng, n_boot=1000):
    D = {c: d_of(deltas[c]) for c in COND}
    sig = np.zeros(N_FEATURES, bool)
    for c in COND:
        d, cs = D[c]
        sig |= (np.abs(d) > D_THRESH) & (cs > C_THRESH)
    idx = np.where(sig)[0]
    if not len(idx):
        return None

    def ratios(Dd):
        a, b, c_ = (np.abs(Dd[x][0][idx]) for x in ['A', 'B', 'C'])
        ab, ac, bc, abc = (np.abs(Dd[x][0][idx]) for x in ['AB', 'AC', 'BC', 'ABC'])
        s = a + b + c_
        r3 = np.where(s > 0.01, abc / np.maximum(s, 1e-9), 1.0)
        rAB = np.where((a + b) > 0.01, ab / np.maximum(a + b, 1e-9), 1.0)
        rAC = np.where((a + c_) > 0.01, ac / np.maximum(a + c_, 1e-9), 1.0)
        rBC = np.where((b + c_) > 0.01, bc / np.maximum(b + c_, 1e-9), 1.0)
        inter = abc - ab - ac - bc + a + b + c_
        marg = abc - ab
        return r3, (rAB, rAC, rBC), inter, marg

    r3, (rAB, rAC, rBC), inter, marg = ratios(D)
    n_sub = int((r3 < 0.8).sum())
    n_add = int(((r3 >= 0.8) & (r3 <= 1.2)).sum())
    n_sup = int((r3 > 1.2).sum())

    # The redundancy ratio and the interaction term are medians over *targets*, so the target
    # is the resampling unit.  Resampling cells instead perturbs every d estimate (duplicated
    # cells shrink the within-cell SD and inflate |d|), which biases the bootstrap distribution
    # away from the point estimate and yields intervals that do not cover it.
    T = len(idx)
    boot_r3, boot_int = [], []
    for _ in range(n_boot):
        take = rng.integers(0, T, T)
        boot_r3.append(float(np.median(r3[take])))
        boot_int.append(float(np.median(inter[take])))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def wilson(k, n, z=1.96):
        if n == 0:
            return [0.0, 0.0]
        ph = k / n
        d_ = 1 + z ** 2 / n
        c = (ph + z ** 2 / (2 * n)) / d_
        h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / d_
        return [max(0.0, c - h), min(1.0, c + h)]

    return {
        'n_targets': int(len(idx)),
        'threeway_ratio': float(np.median(r3)),
        'threeway_ratio_ci': ci(boot_r3),
        'pairwise_ratios': {'AB': float(np.median(rAB)), 'AC': float(np.median(rAC)),
                            'BC': float(np.median(rBC))},
        'pairwise_ratio_mean': float(np.median(np.concatenate([rAB, rAC, rBC]))),
        'n_subadditive': n_sub, 'n_additive': n_add, 'n_superadditive': n_sup,
        'superadditive_rate': n_sup / len(idx),
        'superadditive_rate_ci': wilson(n_sup, len(idx)),
        'higher_order_interaction_median': float(np.median(inter)),
        'interaction_ci': ci(boot_int),
        'marginal_C_given_AB_median': float(np.median(marg)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-cells', type=int, default=N_CELLS)
    ap.add_argument('--analyse-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'deltas'), exist_ok=True)
    orig = load_orig()

    pool = feature_pool(orig)
    triplets = select_triplets(pool)
    with open(os.path.join(OUT, 'triplets.json'), 'w') as f:
        json.dump([{k: (v if k != 'features' else [[L, int(fi), val] for L, fi, val in v])
                    for k, v in t.items()} for t in triplets], f, indent=1)

    if not args.analyse_only:
        import torch
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        all_tokens = orig.load_and_tokenize_cells(args.n_cells)
        model = build_model(device, orig)
        sae_cache = orig.SAECache()
        for L in list(LAYERS) + [TARGET_LAYER]:
            s, _ = sae_cache.get(L)
            s.to(device)
        means = {L: torch.tensor(sae_cache.get(L)[1], dtype=torch.float32, device=device)
                 for L in list(LAYERS) + [TARGET_LAYER]}

        for i, t in enumerate(triplets):
            path = os.path.join(OUT, 'deltas', f"{t['id']}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            deltas, n_active = run_triplet(t, all_tokens, model, device, sae_cache,
                                           means, args.n_cells)
            np.savez(path, **{c: deltas[c].astype(np.float32) for c in COND})
            print(f"  [{i+1}/{len(triplets)}] {t['id']} ({t['type']}) "
                  f"{(time.time()-t0)/60:.1f} min  n_active={n_active['ABC']}")
            gc.collect()

    # ---------------- analysis ----------------
    from scipy import stats
    rng = np.random.default_rng(SEED)
    results = []
    for t in triplets:
        path = os.path.join(OUT, 'deltas', f"{t['id']}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        deltas = {c: z[c].astype(np.float64) for c in COND}
        a = analyse_triplet(deltas, rng)
        if a:
            results.append({**{k: v for k, v in t.items() if k != 'features'},
                            'features': [[L, int(fi), v['label']] for L, fi, v in t['features']],
                            **a})

    groups = {}
    for r in results:
        groups.setdefault(r['type'], []).append(r)

    summary = {'per_triplet': results, 'groups': {}}
    for g, rs in groups.items():
        sub = sum(r['n_subadditive'] for r in rs)
        add = sum(r['n_additive'] for r in rs)
        sup = sum(r['n_superadditive'] for r in rs)
        tot = sub + add + sup
        summary['groups'][g] = {
            'n_triplets': len(rs), 'n_targets': tot,
            'median_threeway_ratio': float(np.median([r['threeway_ratio'] for r in rs])),
            'median_pairwise_ratio': float(np.median([r['pairwise_ratio_mean'] for r in rs])),
            'n_subadditive': sub, 'n_additive': add, 'n_superadditive': sup,
            'superadditive_rate': sup / tot if tot else 0.0,
        }

    # cross vs same, and cross vs random: is the superadditive rate different?
    tests = {}
    for a, b in [('cross_pathway', 'same_pathway'), ('cross_pathway', 'random'),
                 ('same_pathway', 'random')]:
        if a in summary['groups'] and b in summary['groups']:
            ga, gb = summary['groups'][a], summary['groups'][b]
            table = [[ga['n_superadditive'], ga['n_targets'] - ga['n_superadditive']],
                     [gb['n_superadditive'], gb['n_targets'] - gb['n_superadditive']]]
            odds, p = stats.fisher_exact(table)
            tests[f'{a}_vs_{b}'] = {'odds_ratio': float(odds), 'p_value': float(p),
                                    'table': table}
    summary['superadditive_tests'] = tests

    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    print("\n=== groups ===")
    for g, s in summary['groups'].items():
        print(f"  {g:<14} n={s['n_triplets']:<2} 3-way ratio={s['median_threeway_ratio']:.3f} "
              f"pairwise={s['median_pairwise_ratio']:.3f}  superadditive "
              f"{s['n_superadditive']}/{s['n_targets']} = {s['superadditive_rate']:.3%}")
    print("\n=== superadditive-rate contrasts ===")
    for k, v in tests.items():
        print(f"  {k:<34} OR={v['odds_ratio']:.2f}  p={v['p_value']:.3g}")
    print("\n=== per triplet ===")
    for r in results:
        print(f"  {r['type']:<14} {r['pathway'][:34]:<34} 3way={r['threeway_ratio']:.3f} "
              f"CI[{r['threeway_ratio_ci'][0]:.3f},{r['threeway_ratio_ci'][1]:.3f}] "
              f"super={r['n_superadditive']:>3} ({100*r['superadditive_rate']:.2f}%, "
              f"CI[{100*r['superadditive_rate_ci'][0]:.2f},{100*r['superadditive_rate_ci'][1]:.2f}]%) "
              f"of {r['n_targets']}")


if __name__ == '__main__':
    main()
