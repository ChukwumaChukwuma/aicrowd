#!/usr/bin/env python
"""Offline-trained residual corrector: generate, fit, validate, score, ship.

Modes, in the order the decision is taken.

``--mode anova``
    The premise, measured before anything is built: the population share of
    ``Var(relu(z^32_j))`` explained by the layer-1 Hermite control-variate
    family at each order ``k``.  That share is the ceiling of the mechanism
    and is what tells you ``k <= 2`` is the operating point.

``--mode data``
    Generate TRAINING data.  Fresh LOCAL MLP seeds (``--mlp-seed-base``,
    default 100000), disjoint from every existing suite and from the official
    one.  Each MLP is streamed: weights made, the instrumented sparse pass
    run, two independent Monte-Carlo reference halves accumulated, features
    and labels appended, weights dropped.  Nothing but ``(n_mlps, 256)``
    arrays ever lives on disk.

``--mode refresh``
    Recompute only the features of existing shards after a change to the
    feature block.  Bit-identical to re-running ``--mode data``, ~20x faster.

``--mode fit``
    Ridge, then optionally the small MLP head, on a train/validation split
    **by MLP**.  Selection happens here and only here; the official suite is
    never read by this mode.

``--mode score``
    Score on the official 100-MLP suite (N=1e9 reference, so ``raw_mse`` is
    leaderboard-comparable), reporting ``F/B`` separately from ``C/B`` and the
    adjusted score at 1x/2x/3x this machine's residual.  ``--damp 0`` is the
    exact ablation: the identical code path with the correction switched off.

``--mode ship``
    Run the actual ``submission/estimator.py``, including the fallback probe.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import corrector as C  # noqa: E402
from whestfloor.contract import (  # noqa: E402
    DEPTH,
    FLOP_BUDGET,
    LAMBDA_FLOPS_PER_SECOND,
    MULTIPLIER_FLOOR,
    WIDTH,
    effective_compute,
)
from whestfloor.mc import layer_means, make_mlp  # noqa: E402
from whestfloor.official_seeds import (  # noqa: E402
    derive_estimator_seed,
    make_official_mlp,
)
from whestfloor.suite import Suite  # noqa: E402


def artifacts() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


def data_dir() -> Path:
    p = artifacts() / "corrector"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# mode: anova
# ---------------------------------------------------------------------------
def mode_anova(n_mlps: int, n_samples: int, kmax: int = 6) -> None:
    print(f"# population share of Var(relu(z^32)) explained by the layer-1 "
          f"Hermite family\n# {n_mlps} local MLPs x {n_samples:,} samples\n")
    for k in range(n_mlps):
        W = make_mlp(WIDTH, DEPTH, 900_000 + k)
        n = WIDTH
        sig1 = np.sqrt(np.sum(W[0] * W[0], axis=0)).astype(np.float64)
        rho = (W[0].astype(np.float64).T @ W[0].astype(np.float64))
        rho = rho / np.outer(sig1, sig1)
        Ck = [np.zeros((n, n)) for _ in range(kmax + 1)]
        sy = np.zeros(n)
        sy2 = np.zeros(n)
        rng = np.random.default_rng(11)
        done = 0
        while done < n_samples:
            nb = min(4096, n_samples - done)
            x = rng.standard_normal((nb, n), dtype=np.float32)
            h = x
            for w in W:
                h = np.maximum(h @ w, 0.0)
            y = h.astype(np.float64)
            t = (x.astype(np.float64) @ W[0].astype(np.float64)) / sig1
            hm1, hk = np.ones_like(t), t
            for kk in range(1, kmax + 1):
                Ck[kk] += hk.T @ y
                hm1, hk = hk, t * hk - kk * hm1
            sy += y.sum(0)
            sy2 += (y * y).sum(0)
            done += nb
        vy = sy2 / n_samples - (sy / n_samples) ** 2
        tot = float(np.sum(vy))
        fact, rk, run = 1.0, np.ones((n, n)), 0.0
        line = [f"mlp {900_000+k} v={np.mean(vy):.5f}"]
        for kk in range(1, kmax + 1):
            fact *= kk
            rk = rk * rho
            G = fact * rk
            G = G + 1e-9 * np.trace(G) / n * np.eye(n)
            c = Ck[kk] / n_samples
            run += float(np.sum(c * np.linalg.solve(G, c))) / tot
            line.append(f"k<={kk}: {run*100:5.2f}% ({1/(1-run):5.3f}x)")
        print("  " + "  ".join(line), flush=True)


# ---------------------------------------------------------------------------
# mode: data
# ---------------------------------------------------------------------------
def mode_data(n_mlps: int, seed_base: int, n_gt: int, n_samples: int,
              tau: float | None, shard: int, n_shards: int,
              gt_seed_base: int = 3_000_000) -> None:
    out = data_dir() / f"train_s{shard}of{n_shards}.npz"
    idx = list(range(shard, n_mlps, n_shards))
    prim = {k: [] for k in C.PRIMITIVES}
    scal = {k: [] for k in C.SCALARS}
    ga, gb, seeds = [], [], []
    t0 = time.time()
    for c, i in enumerate(idx):
        ms = seed_base + i
        W = make_mlp(WIDTH, DEPTH, ms)
        _, f = C.sparse_mc_features(W, seed=ms, tau=tau, n_samples=n_samples)
        a, _ = layer_means(W, n_gt, gt_seed_base + 2 * i, want_var=False,
                           all_layers=False)
        b, _ = layer_means(W, n_gt, gt_seed_base + 2 * i + 1, want_var=False,
                           all_layers=False)
        for k in C.PRIMITIVES:
            prim[k].append(f[k])
        for k in C.SCALARS:
            scal[k].append(f[k])
        ga.append(a[-1])
        gb.append(b[-1])
        seeds.append(ms)
        del W
        if (c + 1) % 10 == 0 or c + 1 == len(idx):
            el = time.time() - t0
            print(f"  shard {shard}: {c+1}/{len(idx)}  {el:.0f}s  "
                  f"eta {el/(c+1)*(len(idx)-c-1):.0f}s", flush=True)
    np.savez_compressed(
        out, mlp_seeds=np.asarray(seeds, dtype=np.int64),
        gt_a=np.asarray(ga), gt_b=np.asarray(gb), n_gt=n_gt,
        n_samples=n_samples, tau=(-1.0 if tau is None else tau),
        **{k: np.asarray(v) for k, v in prim.items()},
        **{k: np.asarray(v, dtype=np.float64) for k, v in scal.items()})
    print(f"wrote {out}  ({len(idx)} MLPs)")


def mode_refresh(n_samples: int, tau: float | None) -> None:
    """Recompute the FEATURES of every existing shard with the current code.

    Purely an optimisation.  ``--mode data`` is deterministic in
    ``(seed_base, n_gt, n_samples, tau, shard, n_shards, gt_seed_base)``, so
    re-running it regenerates every shard bit-identically; this mode produces
    the same bytes ~20x faster by keeping the Monte-Carlo reference (which
    depends only on the weights and its own seed) and redoing the cheap part.
    Use it after a change to the feature block; use ``--mode data`` to verify.
    """
    for f in sorted(data_dir().glob("train_s*.npz")):
        with np.load(f) as zf:            # materialise, then close the handle
            z = {k: zf[k] for k in zf.files}
        seeds = z["mlp_seeds"]
        acc = {k: [] for k in C.PRIMITIVES}
        sc = {k: [] for k in C.SCALARS}
        t0 = time.time()
        for c, ms in enumerate(seeds):
            W = make_mlp(WIDTH, DEPTH, int(ms))
            _, pr = C.sparse_mc_features(W, seed=int(ms), tau=tau,
                                         n_samples=n_samples)
            for k in C.PRIMITIVES:
                acc[k].append(pr[k])
            for k in C.SCALARS:
                sc[k].append(pr[k])
            del W
            if (c + 1) % 50 == 0:
                print(f"  {f.name}: {c+1}/{len(seeds)}  "
                      f"{time.time()-t0:.0f}s", flush=True)
        z.update({k: np.asarray(v) for k, v in acc.items()})
        z.update({k: np.asarray(v, dtype=np.float64) for k, v in sc.items()})
        np.savez_compressed(f, **z)
        print(f"refreshed {f.name} ({len(seeds)} MLPs)")


def load_train() -> dict:
    files = sorted(data_dir().glob("train_s*.npz"))
    if not files:
        raise SystemExit(f"no training shards in {data_dir()}")
    parts = [np.load(f) for f in files]
    keys = set(parts[0].files)
    acc: dict = {}
    for k in keys:
        if parts[0][k].ndim == 0:
            acc[k] = parts[0][k]
        else:
            acc[k] = np.concatenate([p[k] for p in parts], axis=0)
    order = np.argsort(acc["mlp_seeds"])
    for k in list(acc):
        if getattr(acc[k], "ndim", 0) >= 1:
            acc[k] = acc[k][order]
    return acc


def design_from(d: dict, sl, cv_kmax: int = 2,
                names: tuple[str, ...] | None = None) -> np.ndarray:
    """Stack the per-MLP design matrices for the MLPs selected by ``sl``.

    ``names`` defaults to the FULL research design, so the group ablations
    below see every column; the shipped subset is then selected from it, which
    guarantees the two cannot disagree about a column's contents.

    ``cv_kmax`` zeroes the Hermite blocks the shipped kernel does not compute,
    so a fit made at ``cv_kmax = 2`` cannot put weight on a feature the
    submission would supply as zero.
    """
    names = names or C.FEATURES_V2
    out = []
    for i in np.arange(len(d["mlp_seeds"]))[sl]:
        f = {k: d[k][i] for k in C.PRIMITIVES}
        for k in C.SCALARS:
            f[k] = float(d[k][i])
        if cv_kmax < 3:
            f["cv3"] = np.zeros_like(f["cv3"])
        if cv_kmax < 2:
            f["cv2"] = np.zeros_like(f["cv2"])
        out.append(C.build_design(f, names))
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# mode: fit
# ---------------------------------------------------------------------------
def _umse(pred, a, b):
    return float(np.mean((pred - a) * (pred - b)))


def mode_fit(val_frac: float, lam_grid, do_mlp: bool, hidden: int,
             epochs: int, out_name: str, cv_kmax: int = 2,
             install: bool = False, seed: int = 0) -> None:
    d = load_train()
    FULL = C.FEATURES_V2
    NF = len(FULL)
    n = len(d["mlp_seeds"])
    n_val = int(round(val_frac * n))
    # Split by MLP, deterministically, so a rerun reproduces the choice.
    # THREE splits: the subset/lambda search below evaluates ~900 candidates
    # on the validation split, so the winner's validation number is
    # optimistically biased; the test split is touched once, at the end, and
    # is the internal check before the official suite is read at all.
    rng = np.random.default_rng(20260807)
    perm = rng.permutation(n)
    xsel, vsel, tsel = perm[:n_val], perm[n_val:2 * n_val], perm[2 * n_val:]
    print(f"{n} MLPs: {len(tsel)} train / {len(vsel)} validation / "
          f"{len(xsel)} test ({len(tsel)*WIDTH:,} training rows, "
          f"{NF} research features "
          f"({C.N_FEATURES} of them shipped), cv_kmax={cv_kmax})")

    # ---- what the raw control variates buy with no head at all --------
    A0, B0, MU0 = d["gt_a"], d["gt_b"], d["mu"]
    b_all = _umse(MU0, A0, B0)
    print(f"\n=== raw control variates, coefficient fixed at 1 (no fitting) ===")
    print(f"  sparse MC (shipped)          {b_all:11.4e}   1.000x")
    for nm, p in (("- cv1 (Hermite k=1)", MU0 - d["cv1"]),
                  ("- cv1 - cv2 (k<=2)", MU0 - d["cv1"] - d["cv2"]),
                  ("- cv1 - cv2 - cv3 (k<=3)",
                   MU0 - d["cv1"] - d["cv2"] - d["cv3"]),
                  ("- cv1mf (mean-field propagation)", MU0 - d["cv1mf"]),
                  *[(f"- cva{m} (adapted k=1, m={m})", MU0 - d[f"cva{m}"])
                    for m in C.CVA_GRID],
                  *[(f"- cva{m} - cv2", MU0 - d[f"cva{m}"] - d["cv2"])
                    for m in C.CVA_GRID]):
        v = _umse(p, A0, B0)
        print(f"  {nm:<28} {v:11.4e}   {b_all/v:6.3f}x")

    # ---- pick m for the adapted block, on the VALIDATION split only ----
    # The official suite is never read here; m is a hyperparameter and is
    # chosen exactly like lambda and the feature subset.
    print("\n=== adapted-block direction count m, on the validation split ===")
    print(f"{'m':>5} {'best lambda':>12} {'val':>12} {'val x':>8}")
    mbest = (C.CVA_GRID[0], -1.0)
    for m in C.CVA_GRID:
        d["cva"] = d[f"cva{m}"]
        Xt_ = design_from(d, tsel, cv_kmax).reshape(-1, NF)
        Xv_ = design_from(d, vsel, cv_kmax)
        yt_ = (0.5 * (d["gt_a"][tsel] + d["gt_b"][tsel])
               - d["mu"][tsel]).ravel()
        sc_ = C.design_scale(Xt_)
        bv = (-1.0, None, None)
        for lm in lam_grid:
            bb = C.ridge_fit(Xt_, yt_, lm, sc_)
            vv = _umse(d["mu"][vsel] + Xv_ @ bb, d["gt_a"][vsel],
                       d["gt_b"][vsel])
            g = _umse(d["mu"][vsel], d["gt_a"][vsel], d["gt_b"][vsel]) / vv
            if g > bv[0]:
                bv = (g, lm, vv)
        print(f"{m:5d} {bv[1]:12.1e} {bv[2]:12.4e} {bv[0]:8.3f}")
        if bv[0] > mbest[1]:
            mbest = (m, bv[0])
    print(f"selected m = {mbest[0]}  ->  {mbest[1]:.3f}x on validation "
          f"(C.CVA_M is {C.CVA_M})")
    d["cva"] = d[f"cva{mbest[0]}"]

    Xt = design_from(d, tsel, cv_kmax)
    Xv = design_from(d, vsel, cv_kmax)
    A, B, MU = d["gt_a"], d["gt_b"], d["mu"]
    yt = (0.5 * (A[tsel] + B[tsel]) - MU[tsel])
    base_t = _umse(MU[tsel], A[tsel], B[tsel])
    base_v = _umse(MU[vsel], A[vsel], B[vsel])
    print(f"baseline sparse-MC unbiased true MSE: train {base_t:.4e}  "
          f"val {base_v:.4e}")

    Xt2 = Xt.reshape(-1, NF)
    yt2 = yt.ravel()
    sc = C.design_scale(Xt2)

    print("\n=== ridge ===")
    print(f"{'lambda':>10} {'train':>12} {'val':>12} {'val x':>8}")
    best = (None, -1.0, None)
    for lam in lam_grid:
        beta = C.ridge_fit(Xt2, yt2, lam, sc)
        pv = MU[vsel] + Xv @ beta
        pt = MU[tsel] + Xt @ beta
        v = _umse(pv, A[vsel], B[vsel])
        t = _umse(pt, A[tsel], B[tsel])
        print(f"{lam:10.1e} {t:12.4e} {v:12.4e} {base_v/v:8.3f}")
        if base_v / v > best[1]:
            best = (lam, base_v / v, beta)
    lam, gain, beta = best
    print(f"\nselected lambda {lam:.1e}  ->  {gain:.3f}x on validation")

    # ---- ablations: which feature groups carry it ----------------------
    groups = {
        "cv1 (Hermite k=1)": ("cv1", "cv1_Phi", "cv1_a"),
        "cv2 (Hermite k=2)": ("cv2", "cv2_Phi", "cv2_a"),
        "cv3 (Hermite k=3)": ("cv3",),
        "cv1mf (mean-field)": ("cv1mf", "cv1mf_Phi", "cv1mf_a"),
        "cva (adapted k=1)": ("cva", "cva_Phi", "cva_a"),
        "RB gap + Edgeworth": ("gap", "gap_Phi", "gap_a", "sk", "ku"),
        "shrink (mu)": ("mu", "mu_Phi"),
        "shape": ("s", "Phi", "phi", "a", "sd_mc", "dpilot"),
        "weights + suite": ("wn", "w4", "vbar", "arms"),
    }
    print("\n=== leave-one-group-out on the validation split ===")
    for name, cols in groups.items():
        keep = np.array([f not in cols for f in FULL])
        bb = C.ridge_fit(Xt2[:, keep], yt2, lam, sc[keep])
        v = _umse(MU[vsel] + Xv[:, :, keep] @ bb, A[vsel], B[vsel])
        print(f"  without {name:<22} {v:11.4e}   {base_v/v:6.3f}x")
    print("\n=== only-one-group on the validation split ===")
    for name, cols in groups.items():
        keep = np.array([(f in cols) or f == "one" for f in FULL])
        bb = C.ridge_fit(Xt2[:, keep], yt2, lam, sc[keep])
        v = _umse(MU[vsel] + Xv[:, :, keep] @ bb, A[vsel], B[vsel])
        print(f"  only    {name:<22} {v:11.4e}   {base_v/v:6.3f}x")

    # ---- exhaustive group subset selection, on validation only ---------
    # 2^5 subsets is cheap and removes the eyeballing.  A dropped group also
    # removes its FLOPs and, more importantly, its passes over the (N, width)
    # sample array -- which is residual wall time, billed at 1e11 FLOP/s.
    names = list(groups)
    print(f"\n=== best feature-group subsets on validation "
          f"(top 10 of {1 << len(names)}) ===")
    # One Gram for the full design; every subset is a submatrix solve, so the
    # whole search costs less than a single extra fit.
    Xs = Xt2 / sc
    Gfull = Xs.T @ Xs
    bfull = Xs.T @ yt2
    nrow = len(Xs)
    res = []
    for mask in range(1 << len(names)):
        cols = {"one"}
        for b, nm in enumerate(names):
            if mask >> b & 1:
                cols |= set(groups[nm])
        keep = np.array([f in cols for f in FULL])
        idx = np.flatnonzero(keep)
        Gk = Gfull[np.ix_(idx, idx)]
        for lm in lam_grid:
            Gr = Gk.copy()
            Gr[np.diag_indices(len(idx))] += lm * nrow
            bb = np.linalg.solve(Gr, bfull[idx]) / sc[idx]
            v = _umse(MU[vsel] + Xv[:, :, keep] @ bb, A[vsel], B[vsel])
            res.append((base_v / v, v, lm, mask, keep, bb))
    res.sort(key=lambda r: -r[0])
    for g, v, lm, mask, _, _ in res[:10]:
        on = "+".join(nm.split()[0] for b, nm in enumerate(names)
                      if mask >> b & 1) or "(intercept only)"
        print(f"  {g:6.3f}x  {v:11.4e}  lam {lm:7.1e}  {on}")
    if res[0][0] > gain * 1.002:
        gain, _, lam, _, keep, bb = res[0][:6]
        beta = np.zeros(NF)
        beta[keep] = bb
        print(f"\nsubset selection improves on the full design: "
              f"{gain:.3f}x at lambda {lam:.1e}; using it")

    # ---- the untouched split, read once -------------------------------
    Xx = design_from(d, xsel, cv_kmax)
    base_x = _umse(MU[xsel], A[xsel], B[xsel])
    vx = _umse(MU[xsel] + Xx @ beta, A[xsel], B[xsel])
    print(f"\n=== TEST split ({len(xsel)} MLPs, never used for selection) ===")
    print(f"  sparse MC baseline           {base_x:11.4e}")
    print(f"  ridge corrector              {vx:11.4e}   {base_x/vx:6.3f}x"
          f"   (validation said {gain:.3f}x)")

    print("\n=== coefficients (original units / scaled) ===")
    for i in np.argsort(-np.abs(beta * sc)):
        print(f"  {FULL[i]:<10} {beta[i]:15.6g}   "
              f"{beta[i]*sc[i]:11.4e}")

    # ---- the SHIPPED design: the same columns, minus C.DROPPED ---------
    # Every dropped group measures at exactly 1.000x above; three of the four
    # cost passes over the (N, width) sample array, which is residual wall
    # time.  The head is RE-FITTED on the reduced design rather than masked,
    # so the ridge shrinkage is the right one for the columns that remain.
    ship_keep = np.array([f in C.FEATURES for f in FULL])
    ship_idx = np.flatnonzero(ship_keep)
    assert [FULL[i] for i in ship_idx] == list(C.FEATURES)
    St2 = Xt2[:, ship_idx]
    ssc = C.design_scale(St2)
    print(f"\n=== SHIPPED design ({C.N_FEATURES} of {NF} "
          f"columns; dropped {', '.join(C.DROPPED)}) ===")
    print(f"{'lambda':>10} {'train':>12} {'val':>12} {'val x':>8}")
    sbest = (None, -1.0, None)
    for lm in lam_grid:
        bb = C.ridge_fit(St2, yt2, lm, ssc)
        v = _umse(MU[vsel] + Xv[:, :, ship_idx] @ bb, A[vsel], B[vsel])
        t = _umse(MU[tsel] + Xt[:, :, ship_idx] @ bb, A[tsel], B[tsel])
        print(f"{lm:10.1e} {t:12.4e} {v:12.4e} {base_v / v:8.3f}")
        if base_v / v > sbest[1]:
            sbest = (lm, base_v / v, bb)
    slam, sgain, sbeta = sbest
    svx = _umse(MU[xsel] + Xx[:, :, ship_idx] @ sbeta, A[xsel], B[xsel])
    print(f"  selected lambda {slam:.1e}: validation {sgain:.3f}x  "
          f"(full design {gain:.3f}x),  TEST {base_x / svx:.3f}x  "
          f"(full design {base_x / vx:.3f}x)")
    print("  coefficients (original units / scaled):")
    for i in np.argsort(-np.abs(sbeta * ssc)):
        print(f"    {C.FEATURES[i]:<10} {sbeta[i]:15.6g}   "
              f"{sbeta[i] * ssc[i]:11.4e}")

    payload = {"beta": sbeta.astype(np.float32),
               "features": np.array(C.FEATURES),
               "beta_full": beta.astype(np.float32),
               "features_full": np.array(FULL),
               "lam": float(slam), "lam_full": float(lam), "cv_kmax": cv_kmax,
               "n_train_mlps": len(tsel),
               "n_val_mlps": len(vsel), "n_test_mlps": len(xsel),
               "val_gain": float(sgain), "test_gain": float(base_x / svx),
               "val_gain_full": float(gain),
               "test_gain_full": float(base_x / vx)}

    if do_mlp:
        print(f"\n=== MLP head ({hidden} hidden units) ===")
        Xs_t = Xt2 / sc
        Xs_v = Xv / sc
        ys = float(np.sqrt(np.mean(yt2 * yt2)))
        best_m = (None, -1.0, None)
        for hs in (hidden,):
            for lr in (3e-2, 1e-2):
                m = C.MLPHead(NF, hs, seed=seed)
                m.fit(Xs_t, yt2 / ys, epochs=epochs, lr=lr, seed=seed)
                pv, _ = m.forward(Xs_v.reshape(-1, NF))
                pred = MU[vsel] + (pv * ys).reshape(Xv.shape[:2])
                v = _umse(pred, A[vsel], B[vsel])
                print(f"  hidden {hs:3d} lr {lr:.0e}: val {v:11.4e}   "
                      f"{base_v/v:6.3f}x   (ridge {gain:.3f}x)")
                if base_v / v > best_m[1]:
                    best_m = (m, base_v / v, hs)
        m, mg, hs = best_m
        px, _ = m.forward((Xx / sc).reshape(-1, NF))
        vmx = _umse(MU[xsel] + (px * ys).reshape(Xx.shape[:2]),
                    A[xsel], B[xsel])
        print(f"  best MLP head {mg:.3f}x on validation, "
              f"{base_x/vmx:.3f}x on TEST  vs ridge {gain:.3f}x / "
              f"{base_x/vx:.3f}x  ->  {(base_x/vmx)/(base_x/vx):.3f}x over "
              f"ridge on TEST (bar 1.5x)")
        payload.update(m.to_dict())
        payload["mlp_yscale"] = np.float64(ys)
        payload["mlp_xscale"] = sc.astype(np.float64)
        payload["mlp_val_gain"] = float(mg)
        payload["mlp_test_gain"] = float(base_x / vmx)

    p = data_dir() / out_name
    np.savez(p, **payload)
    print(f"\nwrote {p}")
    if install:
        # The shipped artifact carries ONLY the coefficient vector, so it is
        # 228 bytes and pickle-free; ``fnp.load`` reads it at 0 FLOPs.
        ship = Path(__file__).resolve().parent.parent / "submission" / \
            "corrector.npz"
        np.savez(ship, beta=sbeta.astype(np.float32))
        print(f"installed {ship}  ({ship.stat().st_size} bytes)")
    (data_dir() / (out_name + ".json")).write_text(json.dumps(
        {"lam": float(slam), "val_gain": float(sgain),
         "test_gain": float(base_x / svx), "cv_kmax": cv_kmax,
         "beta": sbeta.tolist(), "features": list(C.FEATURES),
         "lam_full": float(lam), "val_gain_full": float(gain),
         "test_gain_full": float(base_x / vx),
         "beta_full": beta.tolist(),
         "features_full": list(FULL), "cva_m": int(mbest[0])}, indent=1))


# ---------------------------------------------------------------------------
# mode: score
# ---------------------------------------------------------------------------
def mode_score(suite: Suite, n_mlps: int, grid: str, coef: str) -> None:
    from whestfloor import kernels  # noqa: PLC0415
    from whestfloor.harness import run_billed  # noqa: PLC0415

    beta = np.load(data_dir() / coef)["beta"]
    gt = suite.gt[:, -1, :]
    variants = []
    for spec in grid.split(";"):
        if not spec:
            continue
        kw = dict(kv.split("=") for kv in spec.split(","))
        variants.append((
            spec,
            functools.partial(
                kernels.corrected_sparse_kernel,
                tau=(None if kw.get("tau", "2.5") == "None"
                     else float(kw.get("tau", "2.5"))),
                n_samples=int(kw.get("n", 8500)),
                n_pilot=int(kw.get("P", 150)),
                seed=int(kw.get("seed", 0)),
                damp=float(kw.get("damp", 1.0)),
                kmax=int(kw.get("kmax", 2)),
                beta=beta)))

    print(f"# official suite, {n_mlps} MLPs, N=1e9 reference -> raw_mse is "
          "leaderboard-comparable\n")
    hdr = (f"{'variant':<40} {'raw_mse':>11} {'F/B':>7} {'C/B':>7} "
           f"{'adj@1x':>10} {'adj@2x':>10} {'adj@3x':>10} {'raise':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, fn in variants:
        mses, fls, rss, nfail = [], [], [], 0
        worst = (0.0, -1)
        for i in range(n_mlps):
            Wn = make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])
            sd = derive_estimator_seed(suite.mlp_seeds[i])
            f2 = functools.partial(fn, seed=sd)
            try:
                pred, fl, rs = run_billed(f2, Wn)
            except Exception as e:  # noqa: BLE001
                print(f"  !! RAISED on mlp {i}: {type(e).__name__}: {e}")
                nfail += 1
                pred, fl, rs = np.zeros((DEPTH, WIDTH)), FLOP_BUDGET, 0.0
            mse = float(np.mean((pred[-1] - gt[i]) ** 2))
            if mse > worst[0]:
                worst = (mse, i)
            mses.append(mse)
            fls.append(fl)
            rss.append(rs)
        raw = float(np.mean(mses))
        F, R = float(np.mean(fls)), float(np.mean(rss))
        adj = [raw * max(MULTIPLIER_FLOOR,
                         effective_compute(F, k * R, LAMBDA_FLOPS_PER_SECOND)
                         / FLOP_BUDGET) for k in (1, 2, 3)]
        C1 = effective_compute(F, R, LAMBDA_FLOPS_PER_SECOND)
        print(f"{name:<40} {raw:11.4e} {F/FLOP_BUDGET:7.4f} "
              f"{C1/FLOP_BUDGET:7.4f} " + " ".join(f"{v:10.4e}" for v in adj)
              + f" {nfail:6d}")
        print(f"{'':<40} worst MLP {worst[0]:.4e} (index {worst[1]})")


# ---------------------------------------------------------------------------
# mode: ship
# ---------------------------------------------------------------------------
def mode_ship(suite: Suite, n_mlps: int) -> None:
    import importlib.util  # noqa: PLC0415
    import types  # noqa: PLC0415

    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    if "whestbench" not in sys.modules:
        wb = types.ModuleType("whestbench")

        class BaseEstimator:  # noqa: D401
            pass

        wb.BaseEstimator = BaseEstimator
        sys.modules["whestbench"] = wb
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "shipped", root / "submission" / "estimator.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)

    class _MLP:
        def __init__(self, w, seed):
            self.width = w[0].shape[0]
            self.depth = len(w)
            self.weights = w
            self.seed = seed

    class _Ctx:
        width, depth = WIDTH, DEPTH
        flop_budget = FLOP_BUDGET
        api_version = "1.0"
        scratch_dir = None
        submission_dir = str(root / "submission")
        seed = 0

    t0 = time.time()
    est = sub.Estimator()
    est.setup(_Ctx())
    setup_s = time.time() - t0
    gt = suite.gt[:, -1, :]
    mses, fls, rss, nfail, worst = [], [], [], 0, (0.0, -1)
    for i in range(n_mlps):
        fw = [fnp.asarray(w)
              for w in make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])]
        mlp = _MLP(fw, derive_estimator_seed(suite.mlp_seeds[i]))
        try:
            with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as c:
                out = est.predict(mlp, FLOP_BUDGET)
            pred = np.asarray(out, dtype=np.float64)
            fl, rs = int(c.flops_used), float(c.residual_wall_time_s)
        except Exception as e:  # noqa: BLE001
            print(f"  !! RAISED on mlp {i}: {type(e).__name__}: {e}")
            nfail += 1
            pred, fl, rs = np.zeros((DEPTH, WIDTH)), FLOP_BUDGET, 0.0
        assert pred.shape == (DEPTH, WIDTH) and np.isfinite(pred).all()
        mse = float(np.mean((pred[-1] - gt[i]) ** 2))
        if mse > worst[0]:
            worst = (mse, i)
        mses.append(mse)
        fls.append(fl)
        rss.append(rs)
    raw = float(np.mean(mses))
    F, R = float(np.mean(fls)), float(np.mean(rss))
    print(f"\n=== SHIPPED submission/estimator.py, {n_mlps} official MLPs "
          "(N=1e9 reference) ===")
    print(f"  setup wall time       {setup_s:.3f}s (limit 5s, off budget)")
    print(f"  raw final-layer MSE   {raw:.4e}")
    print(f"  F/B                   {F/FLOP_BUDGET:.4f}  (machine-independent)")
    for k in (1, 2, 3):
        Ck = effective_compute(F, k * R, LAMBDA_FLOPS_PER_SECOND)
        print(f"  adjusted @ {k}x residual {raw*max(MULTIPLIER_FLOOR, Ck/FLOP_BUDGET):.4e}"
              f"   (C/B {Ck/FLOP_BUDGET:.4f})")
    print(f"  raises                {nfail} / {n_mlps}")
    print(f"  worst single MLP      {worst[0]:.4e} (index {worst[1]})")
    print(f"  max C over the suite  "
          f"{max(effective_compute(f, r, LAMBDA_FLOPS_PER_SECOND) for f, r in zip(fls, rss))/FLOP_BUDGET:.4f} of B")

    good = est._sparse
    est._sparse = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe"))
    try:
        with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as c:
            out = np.asarray(est.predict(mlp, FLOP_BUDGET), dtype=np.float64)
    finally:
        est._sparse = good
    fb = float(np.mean((out[-1] - gt[n_mlps - 1]) ** 2))
    print(f"  fallback fires: shape {out.shape}, finite "
          f"{bool(np.isfinite(out).all())}, raw MSE {fb:.4e}, C/B "
          f"{effective_compute(c.flops_used, c.residual_wall_time_s, LAMBDA_FLOPS_PER_SECOND)/FLOP_BUDGET:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("anova", "data", "refresh", "fit", "score",
                             "ship"))
    ap.add_argument("--suite", default=None)
    ap.add_argument("--n-mlps", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=8500)
    ap.add_argument("--n-gt", type=int, default=125_000)
    ap.add_argument("--mlp-seed-base", type=int, default=100_000)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--lams", type=str, default="1e-8,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--mlp-head", action="store_true")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    # head_lean.npz is the 15-column shipped head; head_ridge.npz is the
    # 28-column predecessor, kept so its ablation table stays re-scoreable.
    ap.add_argument("--coef", type=str, default="head_lean.npz")
    ap.add_argument("--cv-kmax", type=int, default=2)
    ap.add_argument("--install", action="store_true",
                    help="write submission/corrector.npz")
    ap.add_argument("--grid", type=str, default="damp=0;damp=1")
    args = ap.parse_args()

    if args.mode == "anova":
        mode_anova(args.n_mlps, 400_000)
        return 0
    if args.mode == "refresh":
        mode_refresh(args.n_samples,
                     None if args.tau < 0 else args.tau)
        return 0
    if args.mode == "data":
        mode_data(args.n_mlps, args.mlp_seed_base, args.n_gt, args.n_samples,
                  args.tau, args.shard, args.n_shards)
        return 0
    if args.mode == "fit":
        mode_fit(args.val_frac, [float(v) for v in args.lams.split(",")],
                 args.mlp_head, args.hidden, args.epochs, args.coef,
                 args.cv_kmax, args.install)
        return 0
    p = (Path(args.suite) if args.suite
         else artifacts() / "suites" / "official_mini.npz")
    suite = Suite.load(p)
    if args.mode == "score":
        mode_score(suite, args.n_mlps, args.grid, args.coef)
    else:
        mode_ship(suite, args.n_mlps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
