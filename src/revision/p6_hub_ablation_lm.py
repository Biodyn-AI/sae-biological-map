#!/usr/bin/env python3
"""P6 — Do hub features actually matter to the model?

The Discussion asserts that connectivity concentrated in ~70 hub features "may be unexpectedly
fragile to targeted perturbation".  Nothing in the paper tests this.  Here we do.

Prediction: ablating a hub feature degrades Geneformer's masked-gene prediction more than
ablating (a) a random active feature or (b) a non-hub feature matched on activation frequency.
The frequency-matched arm is essential, because degree correlates rho = 0.84 with activation
frequency, so an unmatched comparison would only rediscover that frequent features matter.

Metric: on held-out K562 cells with 15% of gene tokens masked, the increase in masked-token
cross-entropy and the drop in top-1 recovery accuracy, relative to the unablated model.
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

BASE = os.environ.get("SAE_BASE_ROOT", "..")
PROJ = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
ORIG = os.path.join(PROJ, "src", "20_exhaustive_feature_tracing.py")
EXH = os.path.join(PROJ, "experiments/phase8_exhaustive_tracing")
OUT = os.path.join(PROJ, "experiments/revision_plos/P6_hub_importance")

SOURCE_LAYER = 5
N_EVAL_CELLS = 50
N_PER_ARM = 50
MASK_FRAC = 0.15
MASK_TOKEN = 1
SEED = 3


def load_orig():
    spec = importlib.util.spec_from_file_location("orig_trace", ORIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_degrees():
    import glob
    feats = {}
    for p in sorted(glob.glob(os.path.join(EXH, 'feature_*.json'))):
        if '/._' in p:
            continue
        with open(p) as f:
            j = json.load(f)
        feats[j['feature_idx']] = {'edges': j['total_significant_edges'],
                                   'freq': j['activation_freq'],
                                   'label': j['label']}
    return feats


def pick_arms(feats, rng):
    fis = np.array(sorted(feats))
    edges = np.array([feats[f]['edges'] for f in fis])
    freq = np.array([feats[f]['freq'] for f in fis])

    order = np.argsort(-edges)
    hubs = fis[order[:N_PER_ARM]]
    hub_set = set(hubs.tolist())

    non_hub = np.array([f for f in fis if f not in hub_set])
    nh_freq = np.array([feats[f]['freq'] for f in non_hub])

    random_arm = rng.choice(non_hub, N_PER_ARM, replace=False)

    # frequency-matched non-hubs: for each hub, the nearest unused non-hub in log-frequency
    matched, used = [], set()
    for h in hubs:
        lf = np.log(feats[h]['freq'])
        cand = np.argsort(np.abs(np.log(nh_freq) - lf))
        for c in cand:
            if non_hub[c] not in used:
                matched.append(non_hub[c])
                used.add(non_hub[c])
                break
    matched = np.array(matched)

    print(f"  hub freq median      {np.median([feats[f]['freq'] for f in hubs]):.4f}, "
          f"edges {np.median([feats[f]['edges'] for f in hubs]):.0f}")
    print(f"  matched freq median  {np.median([feats[f]['freq'] for f in matched]):.4f}, "
          f"edges {np.median([feats[f]['edges'] for f in matched]):.0f}")
    print(f"  random freq median   {np.median([feats[f]['freq'] for f in random_arm]):.4f}, "
          f"edges {np.median([feats[f]['edges'] for f in random_arm]):.0f}")
    return {'hub': hubs, 'freq_matched': matched, 'random': random_arm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-cells', type=int, default=N_EVAL_CELLS)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from transformers import BertForMaskedLM

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    orig = load_orig()
    rng = np.random.default_rng(SEED)

    feats = load_degrees()
    arms = pick_arms(feats, rng)

    # held-out cells: the trace used the first 20 control cells, so evaluate on the next block
    all_tokens = orig.load_and_tokenize_cells(200)
    eval_tokens = all_tokens[100:100 + args.n_cells]
    print(f"  {len(eval_tokens)} held-out evaluation cells")

    model = BertForMaskedLM.from_pretrained(orig.MODEL_NAME, subfolder=orig.MODEL_SUBFOLDER,
                                            output_hidden_states=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    sae_cache = orig.SAECache()
    sae, mean = sae_cache.get(SOURCE_LAYER)
    sae.to(device)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)

    # build the masked inputs once, so every arm sees identical masks
    cells = []
    for tokens in eval_tokens:
        gp = np.where((tokens != 2) & (tokens != 3))[0]
        n_mask = max(1, int(MASK_FRAC * len(gp)))
        mpos = rng.choice(gp, n_mask, replace=False)
        masked = tokens.copy()
        targets = masked[mpos].copy()
        masked[mpos] = MASK_TOKEN
        cells.append({'masked': masked, 'gp': gp, 'mpos': mpos, 'targets': targets})

    def evaluate(feature_idx):
        ce_sum, acc_sum, n_tok, n_used = 0.0, 0, 0, 0
        for c in cells:
            ids = torch.tensor(c['masked'], dtype=torch.long, device=device).unsqueeze(0)
            am = torch.ones(1, len(c['masked']), dtype=torch.long, device=device)
            gp = torch.tensor(c['gp'], dtype=torch.long, device=device)
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=am)
            if feature_idx is None:
                logits = o.logits[0]
            else:
                hidden = o.hidden_states[SOURCE_LAYER + 1][0]
                with torch.no_grad():
                    h_sp, topk = sae.encode(hidden[gp] - mean_t)
                if not (topk == feature_idx).any():
                    del o
                    continue
                h_ab = h_sp.clone()
                h_ab[:, feature_idx] = 0.0
                with torch.no_grad():
                    dh = sae.decode(h_ab) - sae.decode(h_sp)
                mod = hidden.clone()
                mod[gp] += dh
                h = mod.unsqueeze(0)
                with torch.no_grad():
                    for l in range(SOURCE_LAYER + 1, 18):
                        h = model.bert.encoder.layer[l](h)[0]
                    logits = model.cls(h)[0]
            n_used += 1
            mp = torch.tensor(c['mpos'], dtype=torch.long, device=device)
            tg = torch.tensor(c['targets'], dtype=torch.long, device=device)
            lg = logits[mp]
            ce_sum += float(F.cross_entropy(lg, tg, reduction='sum'))
            acc_sum += int((lg.argmax(-1) == tg).sum())
            n_tok += len(tg)
            del o
            if device.type == 'mps':
                torch.mps.empty_cache()
        if n_tok == 0:
            return None
        return {'ce': ce_sum / n_tok, 'acc': acc_sum / n_tok,
                'n_cells_active': n_used, 'n_tokens': n_tok}

    t0 = time.time()
    base = evaluate(None)
    print(f"  baseline: CE={base['ce']:.4f}  top1={base['acc']:.4f}")

    results = {'baseline': base, 'arms': {}}
    for arm, fis in arms.items():
        rows = []
        for i, fi in enumerate(fis):
            r = evaluate(int(fi))
            if r is None:
                continue
            rows.append({'feature_idx': int(fi), 'edges': feats[int(fi)]['edges'],
                         'freq': feats[int(fi)]['freq'],
                         'delta_ce': r['ce'] - base['ce'],
                         'delta_acc': r['acc'] - base['acc'],
                         'n_cells_active': r['n_cells_active']})
            if (i + 1) % 10 == 0:
                print(f"    {arm} {i+1}/{len(fis)}  ({(time.time()-t0)/60:.1f} min)")
        results['arms'][arm] = rows
        dce = [r['delta_ce'] for r in rows]
        print(f"  {arm:<13} mean dCE={np.mean(dce):+.4f}  median={np.median(dce):+.4f}  n={len(rows)}")

    from scipy import stats
    tests = {}
    for a, b in [('hub', 'random'), ('hub', 'freq_matched'), ('freq_matched', 'random')]:
        x = [r['delta_ce'] for r in results['arms'][a]]
        y = [r['delta_ce'] for r in results['arms'][b]]
        u, p = stats.mannwhitneyu(x, y, alternative='greater')
        tests[f'{a}_gt_{b}_delta_ce'] = {'u': float(u), 'p': float(p),
                                         'median_a': float(np.median(x)),
                                         'median_b': float(np.median(y))}
    # is degree predictive of damage once frequency is controlled?
    allr = [r for arm in results['arms'].values() for r in arm]
    e = np.log10([r['edges'] + 1 for r in allr])
    f_ = np.log10([r['freq'] for r in allr])
    y = np.array([r['delta_ce'] for r in allr])
    rho_e = stats.spearmanr(e, y)
    rho_f = stats.spearmanr(f_, y)

    def rresid(v, x):
        X = np.column_stack([np.ones_like(x), stats.rankdata(x)])
        v = stats.rankdata(v)
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    rho_partial = stats.spearmanr(rresid(e, f_), rresid(y, f_))

    tests['spearman_logedges_vs_delta_ce'] = {'rho': float(rho_e[0]), 'p': float(rho_e[1])}
    tests['spearman_logfreq_vs_delta_ce'] = {'rho': float(rho_f[0]), 'p': float(rho_f[1])}
    tests['partial_spearman_edges_vs_delta_ce_given_freq'] = {'rho': float(rho_partial[0]),
                                                              'p': float(rho_partial[1])}
    results['tests'] = tests

    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=1)

    print("\n=== tests ===")
    for k, v in tests.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
