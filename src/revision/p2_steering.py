#!/usr/bin/env python3
"""P2 — Steering: expanded feature panel, ON/OFF contrast, dose-response, non-circular readouts.

Three problems with the published steering result:
  (i)   the L17 claim rests on 3 features;
  (ii)  the switch features were selected because they track late pseudotime, and the state-shift
        metric is built from early/late pseudotime signatures, so the result is close to circular;
  (iii) there is no dose-response and no multiple-testing correction.

This script addresses all three.

Panel.  Switch features are re-derived from the cached 481-cell TS immune SAE activations at
L0/L5/L11/L17 (Cohen's d between early and late pseudotime terciles, activation frequency
>= 0.05).  We take the top ON-switch features (d > 0) and, crucially, the top OFF-switch
features (d < 0) per layer.

The OFF-switch features are the decisive control.  If "L17 features push cells toward maturity"
is a property of the *layer*, then L17 OFF-switch features will also push toward maturity.  If
it is a property of what the feature *encodes*, they will push away.

Readouts.  Besides the published Delta-s (which uses the pseudotime signature and is therefore
retained only for comparability), three readouts that never see the pseudotime signature:
  * marker score: Delta logit on a literature-defined terminal-maturation panel minus a
    progenitor panel;
  * P(terminal): a logistic probe on Geneformer's final-layer embedding, trained on TS
    hematopoietic cells disjoint from the 481 steered cells, with ontology labels;
  * predicted pseudotime: a ridge probe trained on the clean embeddings of the non-steered
    (late 2/3) cells.
Delta-s is additionally recomputed under signatures derived from Bone_Marrow only and from
Blood+Spleen only, to check that the sign transfers across tissue.

Stages:
    --probes    fit the held-out probes (writes probes.npz)
    --steer     run the steering panel (writes per-feature JSON)
    --analyse   aggregate, test, and write summary.json
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
ORIG = os.path.join(PROJ, "src", "19_causal_trajectory_steering.py")
TRAJ = os.path.join(PROJ, "experiments/phase7_trajectory_dynamics")
SIGN = os.path.join(PROJ, "experiments/phase8_trajectory_steering/state_signatures.npz")
OUT = os.path.join(PROJ, "experiments/revision_plos/P2_steering")

LAYERS = [0, 5, 11, 17]
N_ON, N_OFF = 12, 8
SWITCH_D = 0.5
MIN_ACT_FREQ = 0.05
ALPHAS_FULL = [2.0, 5.0]
ALPHAS_DOSE = [1.5, 2.0, 3.0, 5.0, 8.0]
N_DOSE_PER_SIDE = 4
N_TRAIN_PROBE = 1500
SEED = 42

PROGENITOR = ['CD34', 'KIT', 'FLT3', 'MPO', 'GATA2', 'SPINK2', 'PROM1', 'CRHBP', 'AVP',
              'HLF', 'MECOM', 'AZU1', 'ELANE', 'CTSG', 'TCF7', 'LEF1', 'SELL', 'CCR7']
TERMINAL = ['LYZ', 'S100A8', 'S100A9', 'S100A12', 'FCN1', 'CSF3R', 'FCGR3B', 'ITGAM',
            'CD14', 'CST3', 'GNLY', 'NKG7', 'PRF1', 'GZMB', 'MS4A1', 'CD79A', 'JCHAIN', 'MZB1']

IMMATURE_TYPES = {'hematopoietic stem cell', 'hematopoietic precursor cell',
                  'common myeloid progenitor',
                  'naive thymus-derived cd4-positive, alpha-beta t cell',
                  'naive thymus-derived cd8-positive, alpha-beta t cell'}
TERMINAL_TYPES = {'neutrophil', 'macrophage', 'plasma cell', 'mature nk t cell',
                  'granulocyte', 'basophil'}


def load_orig():
    spec = importlib.util.spec_from_file_location("orig_steer", ORIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def select_features():
    """Re-derive ON and OFF switch features per layer from the cached activations."""
    z = np.load(os.path.join(TRAJ, 'sae_activations.npz'))
    meta = json.load(open(os.path.join(TRAJ, 'cell_metadata.json')))
    pt = np.array(meta['pseudotime'])
    lo, hi = np.percentile(pt, [33.3, 66.7])
    early, late = pt <= lo, pt >= hi

    sel = []
    for L in LAYERS:
        A = z[f'layer_{L}']
        freq = (A > 0).mean(0)
        me, ml = A[early].mean(0), A[late].mean(0)
        sp = np.sqrt(((A[early].var(0) * (early.sum() - 1)) +
                      (A[late].var(0) * (late.sum() - 1))) / (early.sum() + late.sum() - 2))
        d = (ml - me) / np.maximum(sp, 1e-9)
        ok = freq >= MIN_ACT_FREQ
        on = np.where(ok & (d >= SWITCH_D))[0]
        off = np.where(ok & (d <= -SWITCH_D))[0]
        on = on[np.argsort(-d[on])][:N_ON]
        off = off[np.argsort(d[off])][:N_OFF]
        for fi in on:
            sel.append({'layer': L, 'idx': int(fi), 'd': float(d[fi]), 'kind': 'ON'})
        for fi in off:
            sel.append({'layer': L, 'idx': int(fi), 'd': float(d[fi]), 'kind': 'OFF'})
        print(f"  L{L}: {len(on)} ON, {len(off)} OFF (of {int(ok.sum())} active features)")
    return sel, pt, early, late


def build_cells(orig, n_total=500):
    """Reproduce the 481-cell steering set and a disjoint probe-training set."""
    import h5py
    with open(os.path.join(orig.TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        import pickle
        token_dict = pickle.load(f)
    with open(os.path.join(orig.TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        import pickle
        median_dict = pickle.load(f)
    with open(os.path.join(orig.TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        import pickle
        name_id = pickle.load(f)

    with h5py.File(orig.IMMUNE_H5AD, 'r') as f:
        var_genes = orig.load_categorical_column(f['var'], '_index')
        n_genes = len(var_genes)
        n_cells = f['X']['indptr'].shape[0] - 1
        cell_types = orig.load_categorical_column(f['obs'], 'cell_type')
        tissues = orig.load_categorical_column(f['obs'], 'tissue_in_publication')

    tmask = np.zeros(n_cells, bool)
    for t in orig.HEMATOPOIETIC_TISSUES:
        tmask |= (tissues == t)
    allowed = set(ct.lower() for ct in orig.MYELOID_TYPES + orig.LYMPHOID_TYPES)
    typemask = np.array([ct.lower() in allowed for ct in cell_types])
    valid = np.where(tmask & typemask)[0]

    rng = np.random.RandomState(SEED)
    steer_idx = np.sort(rng.choice(valid, n_total, replace=False)) if len(valid) > n_total else valid
    pool = np.setdiff1d(valid, steer_idx)
    rng2 = np.random.RandomState(SEED + 1)
    train_idx = np.sort(rng2.choice(pool, min(N_TRAIN_PROBE, len(pool)), replace=False))

    mvi, mti, mmd = [], [], []
    for i in range(n_genes):
        ens = name_id.get(var_genes[i])
        if ens and ens in token_dict:
            mvi.append(i); mti.append(token_dict[ens]); mmd.append(median_dict.get(ens, 1.0))
    mvi, mti, mmd = np.array(mvi), np.array(mti), np.array(mmd)

    def tok(indices):
        """Returns (tokens, cell_types, kept_original_indices). Cells with no expressed gene
        in the vocabulary are dropped, so the kept indices must be carried along rather than
        assuming the first len(toks) entries survived."""
# NOTE: paths below are configurable via environment variables and default to a local
# layout; the large activation/data files are not shipped in this repo (see README /
# REPRODUCIBILITY.md). This script is provided to document the exact analysis method.
        toks, cts, kept = [], [], []
        with h5py.File(orig.IMMUNE_H5AD, 'r') as f:
            for idx in indices:
                row = orig.load_sparse_row(f['X'], int(idx), n_genes)
                s = row.sum() or 1
                row = np.log1p(row / s * 1e4)
                t = orig.tokenize_cell(row, mvi, mti, mmd, orig.MAX_SEQ_LEN)
                if t is not None:
                    toks.append(t); cts.append(cell_types[idx]); kept.append(int(idx))
        return toks, cts, np.array(kept)

    return tok, steer_idx, train_idx, cell_types, tissues, token_dict, name_id


def mean_pool(hidden, gene_positions):
    return hidden[gene_positions].mean(dim=0)


def cmd_probes(orig, device):
    import torch
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score

    tok, steer_idx, train_idx, cell_types, tissues, _, _ = build_cells(orig)
    print(f"  probe training pool: {len(train_idx)} cells disjoint from the {len(steer_idx)} steered cells")

    labels, keep = [], []
    for i, idx in enumerate(train_idx):
        ct = cell_types[idx].lower()
        if ct in IMMATURE_TYPES:
            labels.append(0); keep.append(idx)
        elif ct in TERMINAL_TYPES:
            labels.append(1); keep.append(idx)
    print(f"  labelled: {len(keep)} ({labels.count(0)} immature, {labels.count(1)} terminal)")

    keep = np.array(keep)
    lab = dict(zip(keep, labels))
    toks, _, kept = tok(keep)
    model = build_model(device)
    E = []
    for i, t in enumerate(toks):
        E.append(embed(model, t, device))
        if (i + 1) % 200 == 0:
            print(f"    embedded {i+1}/{len(toks)}")
    E = np.stack(E)
    y = np.array([lab[i] for i in kept])

    clf = LogisticRegression(max_iter=3000, C=0.05)
    auc = cross_val_score(clf, E, y, cv=5, scoring='roc_auc')
    clf.fit(E, y)
    print(f"  maturity probe 5-fold CV AUC = {auc.mean():.3f} +/- {auc.std():.3f}")

    # ridge on pseudotime, trained on the non-steered (late 2/3) cells of the 481
    _, pt, early, _ = select_features()
    toks481, _, _ = tok(steer_idx)
    E481 = np.stack([embed(model, t, device) for t in toks481])
    train_mask = ~early
    rid = Ridge(alpha=10.0)
    r2 = cross_val_score(rid, E481[train_mask], pt[train_mask], cv=5, scoring='r2')
    rid.fit(E481[train_mask], pt[train_mask])
    print(f"  pseudotime probe 5-fold CV R^2 = {r2.mean():.3f} (trained on the {train_mask.sum()} non-steered cells)")

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, 'probes.npz'),
             clf_w=clf.coef_[0], clf_b=clf.intercept_, clf_auc=auc,
             ridge_w=rid.coef_, ridge_b=np.array([rid.intercept_]), ridge_r2=r2)
    print(f"  wrote {os.path.join(OUT, 'probes.npz')}")


def build_model(device):
    from transformers import BertForMaskedLM
    m = BertForMaskedLM.from_pretrained(orig_mod.MODEL_NAME, subfolder=orig_mod.MODEL_SUBFOLDER,
                                        output_hidden_states=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def embed(model, tokens, device):
    import torch
    gp = np.where((tokens != 2) & (tokens != 3))[0]
    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    am = torch.ones(1, len(tokens), dtype=torch.long, device=device)
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=am)
    e = o.hidden_states[-1][0][gp].mean(dim=0).cpu().numpy()
    del o
    if device.type == 'mps':
        torch.mps.empty_cache()
    return e


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0


def cmd_steer(orig, device):
    import torch

    sel, pt, early_mask, late_mask = select_features()
    tok, steer_idx, _, cell_types, tissues, token_dict, name_id = build_cells(orig)
    toks, _, kept = tok(steer_idx)
    ts_arr = tissues[kept]
    assert len(toks) == len(pt), f"{len(toks)} cells vs {len(pt)} pseudotime values"
    print(f"  {len(toks)} cells tokenized")

    early_cells = np.where(early_mask)[0]
    print(f"  {len(early_cells)} early-pseudotime cells to steer")

    probes = np.load(os.path.join(OUT, 'probes.npz'))
    clf_w, clf_b = probes['clf_w'], probes['clf_b'][0]
    rid_w, rid_b = probes['ridge_w'], probes['ridge_b'][0]

    model = build_model(device)
    sae_cache = orig.SAECache()

    # marker token ids
    rev = {}
    for gname, ens in name_id.items():
        if ens in token_dict:
            rev.setdefault(gname, token_dict[ens])
    prog_tok = [rev[g] for g in PROGENITOR if g in rev]
    term_tok = [rev[g] for g in TERMINAL if g in rev]
    print(f"  marker panel: {len(prog_tok)} progenitor, {len(term_tok)} terminal genes mapped")

    # signatures: global, and per-tissue holdouts
    sig = np.load(SIGN)
    sigs = {'global': (sig['early'], sig['late'])}
    for name, mask in [('bone_marrow', ts_arr == 'Bone_Marrow'),
                       ('blood_spleen', (ts_arr == 'Blood') | (ts_arr == 'Spleen'))]:
        e_idx = np.where(early_mask & mask)[0]
        l_idx = np.where(late_mask & mask)[0]
        if len(e_idx) < 5 or len(l_idx) < 5:
            print(f"  skipping {name} signature ({len(e_idx)} early, {len(l_idx)} late)")
            continue
        acc_e = np.zeros(model.config.vocab_size)
        acc_l = np.zeros(model.config.vocab_size)
        for idx, acc in [(e_idx, acc_e), (l_idx, acc_l)]:
            for ci in idx:
                t = toks[ci]
                gp = np.where((t != 2) & (t != 3))[0]
                ids = torch.tensor(t, dtype=torch.long, device=device).unsqueeze(0)
                am = torch.ones(1, len(t), dtype=torch.long, device=device)
                with torch.no_grad():
                    o = model(input_ids=ids, attention_mask=am)
                acc += o.logits[0][gp].mean(dim=0).cpu().numpy().astype(np.float64)
                del o
            acc /= len(idx)
        sigs[name] = (acc_e, acc_l)
        print(f"  {name} signature from {len(e_idx)} early / {len(l_idx)} late cells")

    dose_features = set()
    for L in LAYERS:
        for kind in ['ON', 'OFF']:
            fs = [s for s in sel if s['layer'] == L and s['kind'] == kind]
            for s in fs[:N_DOSE_PER_SIDE]:
                dose_features.add((s['layer'], s['idx']))

    os.makedirs(os.path.join(OUT, 'features'), exist_ok=True)
    t0 = time.time()

    # preload SAEs
    for L in LAYERS:
        s, _ = sae_cache.get(L)
        s.to(device)
    means = {L: torch.tensor(sae_cache.get(L)[1], dtype=torch.float32, device=device)
             for L in LAYERS}

    READOUTS = ['delta_s_global', 'delta_s_bone_marrow', 'delta_s_blood_spleen',
                'marker_score', 'delta_p_terminal', 'delta_pseudotime']
    todo = [f for f in sel
            if not os.path.exists(os.path.join(OUT, 'features', f"L{f['layer']}_F{f['idx']}.json"))]
    if not todo:
        print("  all features already computed")
        return
    print(f"  {len(todo)} features to steer")

    # cells outer, features inner: one clean forward per cell, then a tail-only forward per
    # (feature, alpha).  Recomputing the clean pass per feature would cost ~4 h on its own.
    rec = {(f['layer'], f['idx'], a): {k: [] for k in READOUTS}
           for f in todo
           for a in (ALPHAS_DOSE if (f['layer'], f['idx']) in dose_features else ALPHAS_FULL)}

    by_layer = {}
    for f in todo:
        by_layer.setdefault(f['layer'], []).append(f)

    for n, ci in enumerate(early_cells):
        t = toks[ci]
        gp_np = np.where((t != 2) & (t != 3))[0]
        ids = torch.tensor(t, dtype=torch.long, device=device).unsqueeze(0)
        am = torch.ones(1, len(t), dtype=torch.long, device=device)
        gp = torch.tensor(gp_np, dtype=torch.long, device=device)

        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=am)
        clean_logits = o.logits[0][gp].mean(dim=0).cpu().numpy().astype(np.float64)
        clean_emb = o.hidden_states[-1][0][gp].mean(dim=0).cpu().numpy()
        p_cl = 1 / (1 + np.exp(-(clf_w @ clean_emb + clf_b)))
        clean_cos = {name: cos(clean_logits, sl) - cos(clean_logits, se)
                     for name, (se, sl) in sigs.items()}
        hiddens = {L: o.hidden_states[L + 1][0].clone() for L in by_layer}
        del o
        if device.type == 'mps':
            torch.mps.empty_cache()

        for L, feats_L in by_layer.items():
            sae, _ = sae_cache.get(L)
            hidden = hiddens[L]
            with torch.no_grad():
                h_sparse, topk = sae.encode(hidden[gp] - means[L])
            for feat in feats_L:
                fi = feat['idx']
                if not (topk == fi).any():
                    continue
                alphas = ALPHAS_DOSE if (L, fi) in dose_features else ALPHAS_FULL
                for alpha in alphas:
                    h_st = h_sparse.clone()
                    h_st[:, fi] = alpha * h_sparse[:, fi]
                    with torch.no_grad():
                        dh = sae.decode(h_st) - sae.decode(h_sparse)
                    mod = hidden.clone()
                    mod[gp] += dh
                    h = mod.unsqueeze(0)
                    with torch.no_grad():
                        for l in range(L + 1, 18):
                            h = model.bert.encoder.layer[l](h)[0]
                        st_logits = model.cls(h)[0][gp].mean(dim=0).cpu().numpy().astype(np.float64)
                    st_emb = h[0][gp].mean(dim=0).cpu().numpy()

                    r = rec[(L, fi, alpha)]
                    for name, (se, sl) in sigs.items():
                        r[f'delta_s_{name}'].append(
                            (cos(st_logits, sl) - cos(st_logits, se)) - clean_cos[name])
                    dlog = st_logits - clean_logits
                    r['marker_score'].append(
                        float(np.mean(dlog[term_tok]) - np.mean(dlog[prog_tok])))
                    p_st = 1 / (1 + np.exp(-(clf_w @ st_emb + clf_b)))
                    r['delta_p_terminal'].append(float(p_st - p_cl))
                    r['delta_pseudotime'].append(float(rid_w @ (st_emb - clean_emb)))
                    del h, mod
                if device.type == 'mps':
                    torch.mps.empty_cache()

        del hiddens
        el = time.time() - t0
        print(f"  cell {n+1}/{len(early_cells)}  ({el/60:.1f} min elapsed, "
              f"ETA {(len(early_cells)-n-1)*el/(n+1)/60:.0f} min)")
        gc.collect()

    for feat in todo:
        L, fi = feat['layer'], feat['idx']
        alphas = ALPHAS_DOSE if (L, fi) in dose_features else ALPHAS_FULL
        res = {**feat, 'alphas': {}}
        for alpha in alphas:
            r = rec[(L, fi, alpha)]
            res['alphas'][str(alpha)] = {
                'n_cells': len(r['marker_score']),
                **{k: {'mean': float(np.mean(v)) if v else 0.0,
                       'sd': float(np.std(v)) if v else 0.0,
                       'frac_positive': float(np.mean(np.array(v) > 0)) if v else 0.0,
                       'values': [float(x) for x in v]}
                   for k, v in r.items()},
            }
        with open(os.path.join(OUT, 'features', f"L{L}_F{fi}.json"), 'w') as f:
            json.dump(res, f)
    print(f"  wrote {len(todo)} feature files")


def cmd_analyse():
    import glob
    from scipy import stats

    files = [p for p in sorted(glob.glob(os.path.join(OUT, 'features', '*.json'))) if '/._' not in p]
    feats = [json.load(open(p)) for p in files]
    print(f"{len(feats)} features")

    READOUTS = ['delta_s_global', 'marker_score', 'delta_p_terminal', 'delta_pseudotime',
                'delta_s_bone_marrow', 'delta_s_blood_spleen']
    out = {'per_layer': {}, 'readout_agreement': {}, 'dose_response': {}, 'features': []}

    for alpha in ['2.0', '5.0']:
        for L in LAYERS:
            for kind in ['ON', 'OFF']:
                grp = [f for f in feats if f['layer'] == L and f['kind'] == kind
                       and alpha in f['alphas']]
                if not grp:
                    continue
                entry = {'n_features': len(grp),
                         'n_cells_median': float(np.median([f['alphas'][alpha]['n_cells'] for f in grp]))}
                for r in READOUTS:
                    if r not in grp[0]['alphas'][alpha]:
                        continue
                    fp = [f['alphas'][alpha][r]['frac_positive'] for f in grp]
                    mn = [f['alphas'][alpha][r]['mean'] for f in grp]
                    # per-feature one-sample t on per-cell values, then BH across features
                    pv = []
                    for f in grp:
                        v = np.array(f['alphas'][alpha][r]['values'])
                        pv.append(stats.ttest_1samp(v, 0).pvalue if len(v) > 2 else 1.0)
                    pv = np.array(pv)
                    o = np.argsort(pv)
                    m = len(pv)
                    bh = pv[o] <= np.arange(1, m + 1) / m * 0.05
                    n_sig = int(np.max(np.where(bh)[0]) + 1) if bh.any() else 0
                    n_sig_pos = int(sum(1 for i in o[:n_sig] if mn[i] > 0))
                    entry[r] = {
                        'mean_frac_positive': float(np.mean(fp)),
                        'sd_frac_positive': float(np.std(fp)),
                        'mean_effect': float(np.mean(mn)),
                        'n_features_bh_significant': n_sig,
                        'n_significant_positive': n_sig_pos,
                        'n_significant_negative': n_sig - n_sig_pos,
                        'n_features_frac_pos_eq_1': int(sum(1 for x in fp if x == 1.0)),
                    }
                out['per_layer'][f'alpha{alpha}_L{L}_{kind}'] = entry

    # agreement between the circular metric and the non-circular readouts, across features
    for alpha in ['2.0', '5.0']:
        grp = [f for f in feats if alpha in f['alphas']]
        ds = np.array([f['alphas'][alpha]['delta_s_global']['mean'] for f in grp])
        for r in ['marker_score', 'delta_p_terminal', 'delta_pseudotime',
                  'delta_s_bone_marrow', 'delta_s_blood_spleen']:
            if r not in grp[0]['alphas'][alpha]:
                continue
            v = np.array([f['alphas'][alpha][r]['mean'] for f in grp])
            rho, p = stats.spearmanr(ds, v)
            agree = float(np.mean(np.sign(ds) == np.sign(v)))
            out['readout_agreement'][f'alpha{alpha}_{r}'] = {
                'spearman_rho': float(rho), 'p': float(p), 'sign_agreement': agree}

    # dose-response
    for f in feats:
        av = sorted(float(a) for a in f['alphas'])
        if len(av) < 4:
            continue
        key = f"L{f['layer']}_F{f['idx']}_{f['kind']}"
        d = {}
        for r in ['delta_s_global', 'marker_score', 'delta_p_terminal']:
            y = [f['alphas'][str(a)][r]['mean'] for a in av]
            rho, p = stats.spearmanr(av, y)
            d[r] = {'spearman_rho': float(rho), 'p': float(p),
                    'alphas': av, 'means': [float(x) for x in y]}
        out['dose_response'][key] = d

    for f in feats:
        e = {'layer': f['layer'], 'idx': f['idx'], 'kind': f['kind'], 'switch_d': f['d']}
        for a in f['alphas']:
            e[f'alpha{a}'] = {r: f['alphas'][a][r]['frac_positive']
                              for r in READOUTS if r in f['alphas'][a]}
            e[f'alpha{a}_n'] = f['alphas'][a]['n_cells']
        out['features'].append(e)

    with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)

    print("\n=== fraction positive by layer x switch-direction (alpha=5) ===")
    hdr = f"{'':<12}" + ''.join(f"{r.replace('delta_','d_'):>22}" for r in
                                ['delta_s_global', 'marker_score', 'delta_p_terminal', 'delta_pseudotime'])
    print(hdr)
    for L in LAYERS:
        for kind in ['ON', 'OFF']:
            k = f'alpha5.0_L{L}_{kind}'
            if k not in out['per_layer']:
                continue
            e = out['per_layer'][k]
            row = f"L{L:<2} {kind:<4} n={e['n_features']:<3}"
            for r in ['delta_s_global', 'marker_score', 'delta_p_terminal', 'delta_pseudotime']:
                if r in e:
                    row += f"{e[r]['mean_frac_positive']:>10.2f} ({e[r]['n_significant_positive']:>2}+/" \
                           f"{e[r]['n_significant_negative']:<2}-)"
            print(row)

    print("\n=== agreement of non-circular readouts with delta_s (alpha=5) ===")
    for k, v in out['readout_agreement'].items():
        if k.startswith('alpha5.0'):
            print(f"  {k:<32} rho={v['spearman_rho']:+.3f} p={v['p']:.3g} "
                  f"sign agreement={v['sign_agreement']:.1%}")

    dr = out['dose_response']
    if dr:
        print("\n=== dose-response (Spearman rho of mean effect vs alpha) ===")
        for r in ['delta_s_global', 'marker_score', 'delta_p_terminal']:
            rhos = [v[r]['spearman_rho'] for v in dr.values()]
            print(f"  {r:<20} median rho={np.median(rhos):+.2f}  "
                  f"|rho|>0.8 in {sum(1 for x in rhos if abs(x) > 0.8)}/{len(rhos)} features")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', action='store_true')
    ap.add_argument('--steer', action='store_true')
    ap.add_argument('--analyse', action='store_true')
    a = ap.parse_args()

    orig_mod = load_orig()
    if a.probes or a.steer:
        import torch
        dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        os.makedirs(OUT, exist_ok=True)
        if a.probes:
            cmd_probes(orig_mod, dev)
        if a.steer:
            cmd_steer(orig_mod, dev)
    if a.analyse:
        cmd_analyse()
