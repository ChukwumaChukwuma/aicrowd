#!/usr/bin/env python
"""Scale the offline-trained corrector: generate, study, fit, curve, cost.

The shipped head is 15 floats fitted on 640 MLPs generated at ``(tau, N, P) =
(2.5, 8500, 150)`` and deployed at ``(2.5, 25000, 225)``.  This script rebuilds
that pipeline at the deployed operating point, with a superset feature block,
seed replication, and heads whose parameter count can be swept.

Modes, in the order the decision is taken.

``--mode refcal``
    Calibrate the REFERENCE, before any training data exists.  Ground truth is
    the binding compute constraint of the whole programme, so its cost/accuracy
    curve is measured first: rms error of :func:`bigcorr.reference` at several
    ``n_gt``, with and without the exactly-mean-zero k=1 control variate,
    against a 1e6-sample reference on the same networks.

``--mode data``
    Generate training data at the SHIPPED operating point.  Fresh local MLP
    seeds (``--mlp-seed-base``, default 400000), disjoint from the 100000-block
    used by ``scripts/28``, from every suite, and from the official seeds.  Each
    MLP gets ONE reference (two independent halves) and ``--n-seeds``
    independent estimator seeds, because the reference costs about as much as
    four scored passes and depends only on the weights.  Blocks are written
    every ``--block`` MLPs so a fit can run against partial output and
    generation can be stopped at any time.

``--mode fit``
    Everything selective, on a train/validation split BY MLP SEED.  A third
    split is held back and read once.  Reports, in order: the shipped 15-float
    head evaluated as-is at the deployed operating point; the same design
    refitted here; the rich design; leave-one-group-out and only-one-group; and
    the random-feature head.

``--mode curve``
    The learning curve: held-out gain against the number of training MLPs and
    against the parameter count, which is the deliverable if this bottoms out.

``--mode cost``
    Bill the new feature channels in a real ``flopscope.BudgetContext`` and
    time ``fnp.load`` of a full-size weight file against the 5 s setup window.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import bigcorr as BC  # noqa: E402
from whestfloor import corrector as C  # noqa: E402
from whestfloor.contract import DEPTH, WIDTH  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402

#: Operating point of the shipped estimator.  Everything is generated here.
SHIP_TAU, SHIP_N, SHIP_P = 2.5, 25_000, 225

#: Correction channels: columns whose scale is the Monte-Carlo error itself.
CH: tuple[str, ...] = (
    "cv1", "cv2", "cv3", "cv1mf", "cv1mfg",
    "mfm", "mfv", "mfv2", "mfmg", "mfvg", "mfv2g",
    "relu1", "q2", "dpilot",
)
#: Shape channels: per-neuron state, O(1), used to modulate the corrections.
SH: tuple[str, ...] = (
    "one", "s", "Phi", "phi", "alpha", "sd_mc", "gam1", "gam2", "alpha_p",
    "wn1", "w43", "vbar", "arms", "keep_frac", "u1", "u2", "lam1", "lam2",
)

#: POOLED modulators.  A per-neuron head is pointwise and cannot represent
#: ``u1_j <u1, c>`` -- a rank-one interaction across neurons -- however many
#: parameters it has.  ``Cov(h^32)/N`` IS the covariance of the residual being
#: predicted, so its top eigendirection is where that residual lives; giving
#: the head the projection of each correction channel onto it, scattered back
#: through the same eigenvector, is the cheapest possible non-pointwise term
#: and it stays exactly permutation-equivariant.
POOL: tuple[str, ...] = ("u1", "u2")
#: Modulators the linear design crosses every correction channel with.
MOD: tuple[str, ...] = ("one", "Phi", "alpha")

#: Feature groups, for leave-one-group-out.
GROUPS: dict[str, tuple[str, ...]] = {
    "cv1": ("cv1",),
    "cv2": ("cv2",),
    "cv3": ("cv3",),
    "mf_pilot": ("cv1mf", "mfm"),
    "mf_gated": ("cv1mfg", "mfmg"),
    "mf_var_diag": ("mfv", "mfvg"),
    "mf_var_exact": ("mfv2", "mfv2g"),
    "relu1": ("relu1",),
    "q2": ("q2",),
    "dpilot": ("dpilot",),
}


def artifacts() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


# ---------------------------------------------------------------------------
# The score model, calibrated on the GRADER
# ---------------------------------------------------------------------------
# With ``raw = b^2 + v/N`` and ``C = F0 + cN``, the graded objective
# ``adjusted = raw x max(0.1, C/B)`` has an interior minimum, because
#
#     adjusted(N) = [b^2 F0 + b^2 c N + v F0 / N + v c] / B
#
# is convex in N with  d/dN = 0  at  N* = sqrt(v F0 / (b^2 c)),  giving the
# closed form this repository did not previously have:
#
#     adjusted(N*) = ( sqrt(b^2 F0) + sqrt(v c) )^2 / B .
#
# Constants from the GRADED sweeps in submission/estimator.py's docstrings:
# c and F0 from C/B = 0.1001 at N=8500 and 0.2287 at N=22000; b^2 = 1.2e-7 at
# P = 150 from raw = 4.846e-7 at N = 67000; both rescaled to the shipped
# P = 225 (the pilot's own FLOPs are exact; its bias floor is the error of a
# P-sample mean, so b^2 ~ 1/P).
#
# CHECK, and the reason this is trusted: at the shipped point the formula
# returns adjusted 2.470e-07 and N* = 24,150, against the GRADED 2.4646e-07 at
# the shipped N = 25,000.  It reproduces both the level (0.2%) and the argmin.
SCORE_C = 2.591e6          # billed FLOPs per scored sample
SCORE_F0_P150 = 5.21e9     # everything not proportional to N, at P = 150
SCORE_B2_P150 = 1.2e-7     # pilot bias floor at P = 150
SCORE_V = 0.0219           # residual per-sample variance of the CURRENT ship
#: Extra billed FLOPs the new channels add per MLP, from ``--mode cost``:
#: two-channel transport 8.1e6, W.^2 tables 2.1e6, top-2 modes by power
#: iteration 4.09e8 (against 3.28e9 for the Gram it replaces), scored-pass
#: gates 1.01e8, the mfv2 layer-2 second moment 1.92e7, the exact diag
#: Cov(z^2) 6.91e7, and a 2048-feature head 3.51e7.  Total 0.237% of B, which
#: is 0.91% of C at the operating point.
NEW_CHANNEL_FLOPS = 6.44e8
DENSE_FWD = 4_198_656      # FLOPs of one dense forward sample


def score_model(p_pilot: int = SHIP_P):
    f0 = SCORE_F0_P150 + (p_pilot - 150) * DENSE_FWD
    return f0, SCORE_B2_P150 * 150.0 / p_pilot


def project(r_v: float, d_flops: float = 0.0, p_pilot: int = SHIP_P,
            b2_scale: float = 1.0):
    """``(adjusted, N*)`` for a corrector that divides ``v_eff`` by ``r_v``."""
    from whestfloor.contract import FLOP_BUDGET  # noqa: PLC0415
    f0, b2 = score_model(p_pilot)
    f0 += d_flops
    b2 *= b2_scale
    v = SCORE_V / r_v
    adj = (math.sqrt(b2 * f0) + math.sqrt(v * SCORE_C)) ** 2 / FLOP_BUDGET
    return adj, math.sqrt(v * f0 / (b2 * SCORE_C))


#: Sub-directory of the artifact tree the current invocation reads and writes.
#: Runs with different channel sets live in different directories, because a
#: column that is real in one block and structurally zero in another is the one
#: way a pooled fit can silently learn nonsense.
SUBDIR = "bigcorr"


def data_dir() -> Path:
    p = artifacts() / SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# mode: refcal
# ---------------------------------------------------------------------------
def mode_refcal(n_mlps: int, n_big: int, reps: int) -> None:
    grid = (16384, 32768, 65536)
    res: dict = {}
    t0 = time.time()
    for mi in range(n_mlps):
        W = make_mlp(WIDTH, DEPTH, 990_000 + mi)
        big = 0.5 * (BC.reference(W, n_big // 2, 900 + mi, chunk=8192)
                     + BC.reference(W, n_big // 2, 950 + mi, chunk=8192))
        for n in grid:
            for cv in (False, True):
                e = [np.sqrt(np.mean((BC.reference(
                    W, n, 7000 + mi * 100 + r * 13, chunk=8192, use_cv=cv)
                    - big) ** 2)) for r in range(reps)]
                res.setdefault((n, cv), []).append(float(np.mean(e)))
        print(f"  mlp {mi + 1}/{n_mlps}  {time.time() - t0:.0f}s", flush=True)
    print(f"\n# reference calibration: {n_mlps} MLPs x {reps} reps, "
          f"{n_big:,}-sample yardstick")
    print(f"{'n_gt':>8} {'rms no CV':>11} {'rms + k=1 CV':>13} "
          f"{'var ratio':>10} {'eff. n_gt':>11}")
    for n in grid:
        a = float(np.mean(res[(n, False)]))
        b = float(np.mean(res[(n, True)]))
        print(f"{n:8d} {a:11.4e} {b:13.4e} {(a / b) ** 2:10.3f} "
              f"{n * (a / b) ** 2:11.0f}")
    (data_dir() / "refcal.json").write_text(json.dumps(
        {f"{n}_{int(cv)}": v for (n, cv), v in res.items()}, indent=1))


# ---------------------------------------------------------------------------
# mode: data
# ---------------------------------------------------------------------------
def mode_data(n_mlps: int, seed_base: int, n_gt: int, n_seeds: int,
              shard: int, n_shards: int, block: int, n_samples: int,
              n_pilot: int, tau: float, want_q2: bool, want_relu1: bool,
              gt_seed_base: int = 7_000_000) -> None:
    idx = list(range(shard, n_mlps, n_shards))
    acc: dict = {k: [] for k in BC.PER_SEED + BC.PER_MLP}
    acc.update({k: [] for k in BC.PER_SEED_SCALAR})
    ga, gb, mseeds, eseeds = [], [], [], []
    nblk = 0
    t0 = time.time()

    def flush():
        nonlocal acc, ga, gb, mseeds, eseeds, nblk
        if not mseeds:
            return
        out = data_dir() / f"blk_s{shard}_{nblk:04d}.npz"
        np.savez(out,
                 mlp_seeds=np.asarray(mseeds, dtype=np.int64),
                 est_seeds=np.asarray(eseeds, dtype=np.int64),
                 gt_a=np.asarray(ga, dtype=np.float64),
                 gt_b=np.asarray(gb, dtype=np.float64),
                 n_gt=n_gt, n_samples=n_samples, n_pilot=n_pilot, tau=tau,
                 **{k: np.asarray(v, dtype=(np.float64 if k == "mu"
                                            else np.float32))
                    for k, v in acc.items()})
        print(f"  wrote {out.name}  ({len(mseeds)} rows, "
              f"{time.time() - t0:.0f}s)", flush=True)
        nblk += 1
        acc = {k: [] for k in acc}
        ga, gb, mseeds, eseeds = [], [], [], []

    for c, i in enumerate(idx):
        ms = seed_base + i
        W = make_mlp(WIDTH, DEPTH, ms)
        a = BC.reference(W, n_gt, gt_seed_base + 2 * i, chunk=8192)
        b = BC.reference(W, n_gt, gt_seed_base + 2 * i + 1, chunk=8192)
        for k in range(n_seeds):
            es = 1_000_000 + 977 * i + k
            f = BC.extract(W, es, tau=tau, n_samples=n_samples,
                           n_pilot=n_pilot, kmax=3, want_relu1=want_relu1,
                           want_q2=want_q2)
            for key in acc:
                acc[key].append(f[key])
            ga.append(a)
            gb.append(b)
            mseeds.append(ms)
            eseeds.append(es)
        del W
        if (c + 1) % block == 0:
            flush()
    flush()


def load_blocks(limit_mlps: int | None = None, pattern: str = "blk_*.npz"):
    files = sorted(data_dir().glob(pattern))
    if not files:
        raise SystemExit(f"no data blocks in {data_dir()}")
    parts = []
    for f in files:
        with np.load(f) as z:
            parts.append({k: z[k] for k in z.files})
    keys = [k for k in parts[0] if parts[0][k].ndim >= 1]
    d = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    d["meta"] = {k: float(parts[0][k]) for k in parts[0]
                 if parts[0][k].ndim == 0}
    order = np.lexsort((d["est_seeds"], d["mlp_seeds"]))
    for k in keys:
        d[k] = d[k][order]
    if limit_mlps is not None:
        uniq = np.unique(d["mlp_seeds"])[:limit_mlps]
        sel = np.isin(d["mlp_seeds"], uniq)
        for k in keys:
            d[k] = d[k][sel]
    return d


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------
def primitives(d: dict, rows) -> dict:
    """Named per-neuron primitive arrays for the selected rows."""
    p = {k: d[k][rows].astype(np.float64) for k in d
         if k not in ("meta",) and getattr(d[k], "ndim", 0) == 2}
    one = np.ones_like(p["alpha"])
    p["one"] = one
    for k in BC.PER_SEED_SCALAR:
        p[k] = d[k][rows].astype(np.float64)[:, None] * one
    p["lam1"] = p["lam1"] - 4.0
    p["lam2"] = p["lam2"] - 0.4
    p["vbar"] = p["vbar"] - 0.025
    p["arms"] = p["arms"] - 3.1
    p["keep_frac"] = p["keep_frac"] - 0.83
    p["wn1"] = p["wn"] - 1.0
    p["w43"] = p["w4"] - 3.0
    for k in ("cv3", "relu1", "q2"):
        p.setdefault(k, np.zeros_like(one))
    # A neuron that never fires in the scored pass has ``Var(relu z) = 0`` and
    # therefore ``sd_mc = 0``; it also has an exactly zero residual, so the
    # scale-free parameterisation divides 0 by 0.  Floor at 1e-3 of the median,
    # which leaves every live neuron untouched.
    sd = p["sd_mc"]
    p["sd_mc_f"] = np.maximum(sd, 1e-3 * float(np.median(sd[sd > 0])))
    return p


def design(p: dict, ch: tuple[str, ...], sh: tuple[str, ...],
           mod: tuple[str, ...], pool: tuple[str, ...] = ()):
    """``(X, names, tags)``; ``tags[i] = (channel_or_None, "point"|"pool")``."""
    cols, names, tags = [], [], []
    for k in sh:
        cols.append(p[k])
        names.append(k)
        tags.append((None, "shape"))
    for k in ch:
        for m in mod:
            cols.append(p[k] if m == "one" else p[k] * p[m])
            names.append(k if m == "one" else f"{k}*{m}")
            tags.append((k, "point"))
    for u in pool:
        for k in ch:
            cols.append(p[u] * np.sum(p[u] * p[k], axis=-1, keepdims=True))
            names.append(f"<{u},{k}>{u}")
            tags.append((k, "pool"))
    return np.stack(cols, axis=-1), names, tags


def umse(pred, a, b) -> float:
    """Paired unbiased MSE: ``E[(p-a)(p-b)] = (p - mu_true)^2`` for a _|_ b."""
    return float(np.mean((pred - a) * (pred - b)))


def split_by_mlp(d: dict, seed: int = 20260808):
    uniq = np.unique(d["mlp_seeds"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n = len(uniq)
    n_v = max(1, int(round(0.2 * n)))
    tst, val, trn = uniq[perm[:n_v]], uniq[perm[n_v:2 * n_v]], uniq[perm[2 * n_v:]]
    m = d["mlp_seeds"]
    return (np.flatnonzero(np.isin(m, trn)), np.flatnonzero(np.isin(m, val)),
            np.flatnonzero(np.isin(m, tst)))


def fit_eval(Xt, yt, Xv, muv, av, bv, lams, scale=None):
    """Ridge over a lambda grid; returns ``(best_lam, best_umse, best_beta)``."""
    if scale is None:
        scale = np.sqrt(np.mean(Xt * Xt, axis=0))
        scale = np.where(scale > 0, scale, 1.0)
    Xs = Xt / scale
    G = Xs.T @ Xs
    bb = Xs.T @ yt
    nrow = len(Xs)
    best = (None, np.inf, None)
    for lam in lams:
        beta = BC.ridge_solve(G, bb, lam, nrow) / scale
        v = umse(muv + Xv @ beta, av, bv)
        if v < best[1]:
            best = (lam, v, beta)
    return best


# ---------------------------------------------------------------------------
# mode: fit
# ---------------------------------------------------------------------------
def mode_fit(lams, limit_mlps, rf_feats, rf_scale, rf_seed, out_name,
             do_groups: bool = True, sgd_hidden=(), sgd_epochs: int = 25,
             sgd_lr: float = 3e-3) -> None:
    d = load_blocks(limit_mlps)
    trn, val, tst = split_by_mlp(d)
    nm = len(np.unique(d["mlp_seeds"]))
    print(f"{len(d['mlp_seeds'])} rows over {nm} MLPs "
          f"({len(d['est_seeds']) / max(nm, 1):.1f} seeds each) at "
          f"N={d['meta'].get('n_samples')} P={d['meta'].get('n_pilot')} "
          f"n_gt={d['meta'].get('n_gt')}")
    print(f"  train {len(trn)} / val {len(val)} / test {len(tst)} rows "
          f"({len(trn) * WIDTH:,} training neurons)")

    A, B, MU = d["gt_a"], d["gt_b"], d["mu"]
    base = {k: umse(MU[s], A[s], B[s]) for k, s in
            (("t", trn), ("v", val), ("x", tst))}
    print(f"\nbaseline sparse-MC unbiased true MSE:  train {base['t']:.4e}  "
          f"val {base['v']:.4e}  test {base['x']:.4e}")

    pt, pv, px = primitives(d, trn), primitives(d, val), primitives(d, tst)
    yt = (0.5 * (A[trn] + B[trn]) - MU[trn]).ravel()

    # ---- 0. the SHIPPED head, evaluated as-is at the deployed point -------
    ship_beta = None
    sp = Path(__file__).resolve().parent.parent / "submission" / "corrector.npz"
    if sp.is_file():
        ship_beta = np.load(sp)["beta"].astype(np.float64)
    SHIP_CH = ("cv1", "cv2", "cv1mf")
    SHIP_SH = ("s", "Phi", "phi", "alpha")

    def ship_design(p):
        cols = [p["one"]]
        for k in SHIP_CH:
            cols += [p[k], p[k] * p["Phi"], p[k] * p["alpha"]]
        cols += [p[k] for k in SHIP_SH] + [p["dpilot"]]
        return np.stack(cols, axis=-1)

    print("\n=== 1. the shipped 15-float head at the deployed operating point ===")
    Xt15, Xv15, Xx15 = ship_design(pt), ship_design(pv), ship_design(px)
    x_ship = None
    if ship_beta is not None:
        for s, X, nmn in (("val", Xv15, val), ("test", Xx15, tst)):
            v = umse(MU[nmn] + X @ ship_beta, A[nmn], B[nmn])
            if s == "test":
                x_ship = v
            print(f"  as shipped (fitted at N=8500,P=150)   {s:>4} "
                  f"{v:11.4e}   {base[s[0]] / v:6.3f}x")
    lam15, v15, b15 = fit_eval(Xt15.reshape(-1, Xt15.shape[-1]), yt, Xv15,
                               MU[val], A[val], B[val], lams)
    x15 = umse(MU[tst] + Xx15 @ b15, A[tst], B[tst])
    print(f"  REFITTED here (lam {lam15:.0e})            val "
          f"{v15:11.4e}   {base['v'] / v15:6.3f}x")
    print(f"  {'':<38}test {x15:11.4e}   {base['x'] / x15:6.3f}x")

    # ---- 2. raw channels at unit coefficient ------------------------------
    print("\n=== 2. each channel alone, coefficient fixed at 1 (no fitting) ===")
    for k in CH:
        if k not in pv:
            continue
        v = umse(MU[val] - pv[k], A[val], B[val])
        print(f"  - {k:<8} {v:11.4e}   {base['v'] / v:6.3f}x")

    ch = tuple(k for k in CH if float(np.abs(pt[k]).max()) > 0)

    # ---- 2b. the CEILING of the whole feature set -------------------------
    # Fit the channel coefficients SEPARATELY FOR EVERY (MLP, seed) row, on
    # reference half ``a``, and score against half ``b``.  ``p_a`` never sees
    # ``b``, so ``E[(p_a - b)^2] = E[(p_a - mu_true)^2] + Var(b)`` exactly and
    # ``Var(b) = mean((a-b)^2)/2``; subtracting it leaves an honest MSE with no
    # in-sample credit at all.
    #
    # This bounds EVERY head built on these channels -- pointwise or not,
    # linear or not, at any parameter count -- because a per-MLP least squares
    # is the best any of them could do if it knew that MLP's optimal
    # coefficients exactly.  Whatever gap remains between the fitted head and
    # this row is the only thing more data or more capacity could ever buy.
    print("\n=== 2b. ceiling: per-MLP-optimal coefficients, same columns ===")
    vb = float(np.mean((A[tst] - B[tst]) ** 2)) / 2.0
    print(f"  reference half-noise Var(b) = {vb:.4e}; the correction below is "
          f"vb (1 + p/n), p = columns, n = 256 neurons")
    ceil_res = {}
    for nm_, cols_, mod_ in (
            ("shipped 3 channels", ("cv1", "cv2", "cv1mf"), ("one",)),
            ("shipped 3, x modulators", ("cv1", "cv2", "cv1mf"), MOD),
            ("all channels", ch, ("one",)),
            ("all channels x modulators", ch, MOD)):
        cols_ = tuple(k for k in cols_ if float(np.abs(px[k]).max()) > 0)
        G_, nm2, _ = design(px, cols_, ("one",), mod_)
        p_ = G_.shape[-1]
        acc = 0.0
        for src, dst in ((A, B), (B, A)):
            y_ = src[tst] - MU[tst]
            gt_ = np.einsum("mnp,mnq->mpq", G_, G_)
            bt_ = np.einsum("mnp,mn->mp", G_, y_)
            di = np.arange(p_)
            gt_[:, di, di] += 1e-9 * np.trace(gt_, axis1=1, axis2=2)[
                :, None] / p_
            th = np.linalg.solve(gt_, bt_[..., None])[..., 0]
            pa = MU[tst] + np.einsum("mnp,mp->mn", G_, th)
            acc += 0.5 * float(np.mean((pa - dst[tst]) ** 2))
        cm = acc - vb * (1.0 + p_ / WIDTH)
        ceil_res[nm_] = cm
        print(f"  {nm_:<26} {cm:11.4e}   {base['x'] / cm:6.3f}x  "
              f"({p_} coefficients per MLP)")

    # ---- 3. the rich design ----------------------------------------------
    Xt, names, tags = design(pt, ch, SH, MOD, POOL)
    Xv, _, _ = design(pv, ch, SH, MOD, POOL)
    Xx, _, _ = design(px, ch, SH, MOD, POOL)
    nf = Xt.shape[-1]
    print(f"\n=== 3. rich linear design, {nf} columns "
          f"({len(ch)} correction channels x {len(MOD)} modulators "
          f"+ {len(SH)} shape) ===")
    Xt2 = Xt.reshape(-1, nf)
    scale = np.sqrt(np.mean(Xt2 * Xt2, axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    lam, vv, beta = fit_eval(Xt2, yt, Xv, MU[val], A[val], B[val], lams, scale)
    xx = umse(MU[tst] + Xx @ beta, A[tst], B[tst])
    print(f"  lambda {lam:.0e}   val {vv:11.4e} {base['v'] / vv:6.3f}x   "
          f"test {xx:11.4e} {base['x'] / xx:6.3f}x")

    if do_groups:
        print("\n=== 4. leave-one-group-out / only-one-group, on validation ===")
        print(f"  {'group':<12} {'without':>11} {'x':>8} {'only':>11} {'x':>8}")
        sel = dict(GROUPS)
        sel["POOLED (all)"] = None
        for gname, gcols in sel.items():
            if gcols is not None and not any(g in ch for g in gcols):
                continue
            if gcols is None:
                inn = np.array([t[1] == "pool" for t in tags])
            else:
                inn = np.array([t[0] in gcols for t in tags])
            keep = np.flatnonzero(~inn)
            _, v1, _ = fit_eval(Xt2[:, keep], yt, Xv[:, :, keep], MU[val],
                                A[val], B[val], lams, scale[keep])
            only = np.flatnonzero(inn | np.array([t[1] == "shape"
                                                  for t in tags]))
            _, v2, _ = fit_eval(Xt2[:, only], yt, Xv[:, :, only], MU[val],
                                A[val], B[val], lams, scale[only])
            print(f"  {gname:<13} {v1:11.4e} {base['v'] / v1:8.3f} "
                  f"{v2:11.4e} {base['v'] / v2:8.3f}", flush=True)

    # ---- 5. the random-feature head ---------------------------------------
    # Inputs are made SCALE-FREE first: every correction column is divided by
    # the per-neuron Monte-Carlo noise scale ``sd_mc`` and the prediction is
    # multiplied back by it.  The optimal shrinkage of a control variate is a
    # ratio of signal to noise, so this is the parameterisation in which the
    # map the head has to learn is a bounded function of bounded inputs -- and
    # it is why ``sd_mc``, which a linear head measured at exactly 1.000x, is
    # worth carrying now.
    print("\n=== 5. random-feature head (tanh), scaled by sd_mc ===")
    Zt, n_lin, sct = rf_inputs(pt, ch, SH)
    Zv, _, _ = rf_inputs(pv, ch, SH, sct)
    Zx, _, _ = rf_inputs(px, ch, SH, sct)
    nin = Zt.shape[-1]
    Zt2, Zv2, Zx2 = (Z.reshape(-1, nin) for Z in (Zt, Zv, Zx))
    sdt, sdv, sdx = (q["sd_mc_f"].ravel() for q in (pt, pv, px))

    def sc_umse(flat, p, rows):
        return umse(MU[rows] + flat.reshape(p["alpha"].shape),
                    A[rows], B[rows])

    res_rf = []
    for nfeat in rf_feats:
        t0 = time.time()
        mdl = BC.RFRidge(nin, nfeat, seed=rf_seed, scale=rf_scale,
                         n_lin=n_lin)
        betas = mdl.fit(Zt2, yt, lams, row_scale=sdt)
        cand = [(sc_umse(mdl.predict(Zv2, w, row_scale=sdv), pv, val), lm, w)
                for lm, w in betas.items()]
        v, lm, w = min(cand, key=lambda t: t[0])
        xv = sc_umse(mdl.predict(Zx2, w, row_scale=sdx), px, tst)
        npar = mdl.n_params()
        print(f"  {nfeat:6d} features ({npar:9,d} params) lam {lm:.0e}   "
              f"val {v:11.4e} {base['v'] / v:6.3f}x   "
              f"test {xv:11.4e} {base['x'] / xv:6.3f}x   "
              f"[{time.time() - t0:.0f}s]", flush=True)
        res_rf.append({"n_feat": nfeat, "n_params": int(npar),
                       "val": v, "test": xv,
                       "val_x": base['v'] / v, "test_x": base['x'] / xv})

    # ---- 6. the SGD head ---------------------------------------------------
    res_sgd = []
    if sgd_hidden:
        print("\n=== 6. per-neuron MLP head, Adam in numpy, scaled by sd_mc ===")
        # BOOSTED on the ridge residual, for the same nesting reason: the
        # linear part is already optimal, so anything the network adds is
        # measured against it rather than instead of it.
        lin0 = BC.RFRidge(nin, 1, seed=0, n_lin=n_lin)
        lin0.A[:] = 0.0
        blin = lin0.fit(Zt2, yt, lams, row_scale=sdt)
        cand = [(sc_umse(lin0.predict(Zv2, w, row_scale=sdv), pv, val), lm, w)
                for lm, w in blin.items()]
        _, lm0, w0 = min(cand, key=lambda t: t[0])
        rt = yt - lin0.predict(Zt2, w0, row_scale=sdt)
        pv0 = lin0.predict(Zv2, w0, row_scale=sdv)
        px0 = lin0.predict(Zx2, w0, row_scale=sdx)
        for hid in sgd_hidden:
            t0 = time.time()
            m = BC.SGDHead(nin, hid, seed=rf_seed)
            m.fit(Zt2, rt, epochs=sgd_epochs, lr=sgd_lr, batch=8192,
                  seed=rf_seed, out_scale=sdt)
            v = sc_umse(pv0 + m.predict(Zv2, out_scale=sdv), pv, val)
            xv = sc_umse(px0 + m.predict(Zx2, out_scale=sdx), px, tst)
            print(f"  hidden {str(hid):<12} ({m.n_params():9,d} params)  "
                  f"val {v:11.4e} {base['v'] / v:6.3f}x   "
                  f"test {xv:11.4e} {base['x'] / xv:6.3f}x   "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            res_sgd.append({"hidden": list(hid), "n_params": m.n_params(),
                            "val": v, "test": xv, "val_x": base['v'] / v,
                            "test_x": base['x'] / xv})

    # ---- 7. what it is worth on the grader --------------------------------
    # Only RATIOS transfer: the local harness runs ~1.45x above the grader on
    # identical seeds, and two changes have already measured better locally and
    # graded WORSE.  So the ratio measured here against the SHIPPED head in the
    # SAME run is applied to the grader's calibrated v_eff, and nothing else
    # from this box enters.
    print("\n=== 7. projected graded score (ratios only; score model in "
          "SCORE_*) ===")
    a0, n0 = project(1.0)
    print(f"  current ship (calibration)            {a0:.4e}  at N* = {n0:,.0f}"
          f"   [graded 2.4646e-07 at N = 25,000]")
    ship_x = base["x"] / x_ship if x_ship else None
    proj = []
    for lbl, val_ in (("refit 15", x15), ("rich linear", xx),
                      *[(f"RF {r['n_feat']}", r["test"]) for r in res_rf],
                      *[(f"SGD {tuple(r['hidden'])}", r["test"])
                        for r in res_sgd]):
        if x_ship is None:
            break
        rv = x_ship / val_
        a, nn = project(rv, d_flops=NEW_CHANNEL_FLOPS)
        proj.append({"head": lbl, "test_x_over_ship": rv, "adjusted": a,
                     "n_star": nn, "adjusted_x": a0 / a})
        print(f"  {lbl:<36} {a:.4e}  at N* = {nn:,.0f}   "
              f"({rv:.3f}x on held-out TEST over the shipped head, "
              f"{a0 / a:.3f}x adjusted)")

    payload = {
        "sgd": res_sgd, "projected": proj,
        "n_mlps": int(nm), "n_rows": int(len(d["mlp_seeds"])),
        "base_val": base["v"], "base_test": base["x"],
        "ship_asis_test": x_ship,
        "ship15_refit": {"lam": lam15, "val": v15, "test": x15,
                         "val_x": base["v"] / v15, "test_x": base["x"] / x15},
        "rich": {"lam": lam, "val": vv, "test": xx, "n_cols": int(nf),
                 "val_x": base["v"] / vv, "test_x": base["x"] / xx,
                 "names": names, "beta": beta.tolist()},
        "rf": res_rf,
    }
    (data_dir() / out_name).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {data_dir() / out_name}")


def rf_inputs(p: dict, ch, sh, sc=None):
    """``(Z, n_lin, sc)`` -- the nonlinear head's inputs, in ``sd_mc`` units.

    The first ``n_lin`` columns are the FULL rich linear design divided by
    ``sd_mc``.  Because the head predicts ``y / sd_mc``, a linear map on those
    columns is *identically* the rich linear model on ``y`` -- so ridge on this
    block reproduces section 3 exactly and the random features are strictly an
    addition.  Without that nesting a nonlinear head that scores worse tells
    you nothing about capacity, only that you changed two things at once.

    The remaining columns are the bounded, dimensionless inputs the tanh layer
    sees: each correction in units of the per-neuron Monte-Carlo noise, and the
    shape columns.  Corrections are clipped at +-8 sigma so a single dead
    neuron cannot saturate every random feature at once.
    """
    sd = p["sd_mc_f"]
    lin, _, _ = design(p, ch, sh, MOD, POOL)
    lin = lin / sd[..., None]
    nl = [np.clip(p[k] / sd, -8.0, 8.0) for k in ch]
    nl += [p[k] for k in sh if k != "one"]
    Z = np.concatenate([lin, np.stack(nl, axis=-1)], axis=-1)
    if sc is None:
        sc = np.sqrt(np.mean(Z.reshape(-1, Z.shape[-1]) ** 2, axis=0))
        sc = np.where(sc > 0, sc, 1.0)
    return (Z / sc).astype(np.float32), lin.shape[-1], sc


# ---------------------------------------------------------------------------
# mode: curve
# ---------------------------------------------------------------------------
def mode_curve(lams, rf_feats, rf_scale, rf_seed, grid) -> None:
    d_all = load_blocks()
    uniq = np.unique(d_all["mlp_seeds"])
    print(f"# learning curve, {len(uniq)} MLPs available\n")
    hdr = (f"{'train MLPs':>11} {'rows':>9} {'ridge15':>9} {'rich':>9} "
           + " ".join(f"{'rf' + str(k):>9}" for k in rf_feats))
    print(hdr)
    print("-" * len(hdr))
    trn_all, val, tst = split_by_mlp(d_all)
    A, B, MU = d_all["gt_a"], d_all["gt_b"], d_all["mu"]
    base_x = umse(MU[tst], A[tst], B[tst])
    base_v = umse(MU[val], A[val], B[val])
    pv, px = primitives(d_all, val), primitives(d_all, tst)
    trn_m = np.unique(d_all["mlp_seeds"][trn_all])
    ch = tuple(k for k in CH if float(np.abs(pv[k]).max()) > 0)
    for g in grid:
        if g > len(trn_m):
            continue
        sub = trn_m[:g]
        trn = np.flatnonzero(np.isin(d_all["mlp_seeds"], sub))
        pt = primitives(d_all, trn)
        yt = (0.5 * (A[trn] + B[trn]) - MU[trn]).ravel()
        out = []
        cols15 = ("cv1", "cv2", "cv1mf")
        SH15 = ("one", "s", "Phi", "phi", "alpha")
        X15t, _, _ = design(pt, cols15, SH15, MOD)
        X15v, _, _ = design(pv, cols15, SH15, MOD)
        X15x, _, _ = design(px, cols15, SH15, MOD)
        _, _, b = fit_eval(X15t.reshape(-1, X15t.shape[-1]), yt, X15v,
                           MU[val], A[val], B[val], lams)
        out.append(base_x / umse(MU[tst] + X15x @ b, A[tst], B[tst]))
        Xt, _, _ = design(pt, ch, SH, MOD, POOL)
        Xv, _, _ = design(pv, ch, SH, MOD, POOL)
        Xx, _, _ = design(px, ch, SH, MOD, POOL)
        _, _, b = fit_eval(Xt.reshape(-1, Xt.shape[-1]), yt, Xv, MU[val],
                           A[val], B[val], lams)
        out.append(base_x / umse(MU[tst] + Xx @ b, A[tst], B[tst]))
        Zt, n_lin, sct = rf_inputs(pt, ch, SH)
        Zv, _, _ = rf_inputs(pv, ch, SH, sct)
        Zx, _, _ = rf_inputs(px, ch, SH, sct)
        nin = Zt.shape[-1]
        Zt2, Zv2, Zx2 = (Z.reshape(-1, nin) for Z in (Zt, Zv, Zx))
        sdt, sdv, sdx = (q["sd_mc_f"].ravel() for q in (pt, pv, px))
        for nfeat in rf_feats:
            mdl = BC.RFRidge(nin, nfeat, seed=rf_seed, scale=rf_scale,
                             n_lin=n_lin)
            betas = mdl.fit(Zt2, yt, lams, row_scale=sdt)
            bst = (np.inf, None)
            for lm, w in betas.items():
                v = umse(MU[val] + mdl.predict(Zv2, w, row_scale=sdv).reshape(
                    pv["alpha"].shape), A[val], B[val])
                if v < bst[0]:
                    bst = (v, w)
            w = bst[1]
            out.append(base_x / umse(
                MU[tst] + mdl.predict(Zx2, w, row_scale=sdx).reshape(
                    px["alpha"].shape), A[tst], B[tst]))
        print(f"{g:11d} {len(trn) * WIDTH:9,d} "
              + " ".join(f"{v:9.3f}" for v in out), flush=True)
    print(f"\n(all columns are held-out TEST gain over the uncorrected sparse "
          f"pass, base {base_x:.4e}; validation base {base_v:.4e})")


# ---------------------------------------------------------------------------
# mode: noise -- the MLPs-versus-label-precision tradeoff
# ---------------------------------------------------------------------------
def mode_noise(lams, grid_mlps, infl) -> None:
    """Held-out gain on a (training MLPs) x (reference precision) grid.

    Generating a second dataset at a different ``n_gt`` is not necessary and
    would be confounded by using different networks.  The reference error is
    Monte-Carlo noise on a mean, i.e. independent, zero-mean and of KNOWN
    per-neuron variance ``vh_j / (n_gt r)`` -- and ``vh_j = sd_mc_j^2 N`` is
    already a stored feature.  So a reference of ``n_gt / f`` samples is
    simulated EXACTLY by adding independent noise of variance ``(f - 1)
    vh_j/(n_gt r)`` to each of the two halves, on the same networks, with the
    paired unbiased estimator staying unbiased because the two additions are
    independent.  One dataset therefore answers the whole tradeoff.
    """
    d = load_blocks()
    trn_all, val, tst = split_by_mlp(d)
    A0, B0, MU = d["gt_a"], d["gt_b"], d["mu"]
    meta = d["meta"]
    n_gt = meta.get("n_samples") and meta["n_gt"]
    vh = (d["sd_mc"].astype(np.float64) ** 2) * meta["n_samples"]
    lab_var = vh / (n_gt * BC_REF_CV_GAIN)
    trn_m = np.unique(d["mlp_seeds"][trn_all])
    pv, px = primitives(d, val), primitives(d, tst)
    ch = tuple(k for k in CH if float(np.abs(pv[k]).max()) > 0)
    Xv, _, _ = design(pv, ch, SH, MOD, POOL)
    Xx, _, _ = design(px, ch, SH, MOD, POOL)
    print(f"# held-out TEST gain: training MLPs x simulated reference size\n"
          f"# base n_gt = {n_gt:,} per half (x2 halves), rms label noise "
          f"{np.sqrt(np.mean(lab_var)):.3e}\n")
    hdr = (f"{'train MLPs':>11} " +
           " ".join(f"{'n_gt/' + str(f):>10}" for f in infl))
    print(hdr)
    print("-" * len(hdr))
    base_x = umse(MU[tst], A0[tst], B0[tst])
    for g in grid_mlps:
        if g > len(trn_m):
            continue
        trn = np.flatnonzero(np.isin(d["mlp_seeds"], trn_m[:g]))
        pt = primitives(d, trn)
        Xt, _, _ = design(pt, ch, SH, MOD, POOL)
        Xt2 = Xt.reshape(-1, Xt.shape[-1])
        row = []
        for f in infl:
            rng = np.random.default_rng(4242 + int(f * 97))
            sd = np.sqrt(np.maximum(f - 1.0, 0.0) * lab_var[trn])
            a = A0[trn] + sd * rng.standard_normal(sd.shape)
            b = B0[trn] + sd * rng.standard_normal(sd.shape)
            yt = (0.5 * (a + b) - MU[trn]).ravel()
            _, _, beta = fit_eval(Xt2, yt, Xv, MU[val], A0[val], B0[val], lams)
            row.append(base_x / umse(MU[tst] + Xx @ beta, A0[tst], B0[tst]))
        print(f"{g:11d} " + " ".join(f"{v:10.3f}" for v in row), flush=True)
    print("\n(evaluation labels are never perturbed; only the TRAINING labels "
          "are.\n rows are iso-#MLPs, columns iso-precision; equal reference "
          "compute runs\n down-right along  n_mlps x n_gt = const.)")


#: Variance reduction the k=1 control variate buys on the reference at the
#: shipped ``n_gt``.  MEASURED by ``--mode refcal``: 1.139 / 1.333 / 1.585 at
#: n_gt = 16384 / 32768 / 65536.
BC_REF_CV_GAIN = 1.333


# ---------------------------------------------------------------------------
# mode: export -- the shippable artifact and its loader contract
# ---------------------------------------------------------------------------
def mode_export(lams, out_name: str, pool=POOL, drop=()) -> None:
    """Write ``bigcorr_head.npz`` and print the predict-time contract.

    The penalty is selected on VALIDATION and the honest generalisation number
    is the TEST split of the train-only fit; the SHIPPED coefficients are then
    refitted on train+validation at that penalty, because a coefficient vector
    estimated from 1.6x the data at a penalty chosen honestly is strictly
    better than one estimated from 1.0x, and the test split is still untouched
    by the selection.  Both numbers are written into the artifact.
    """
    d = load_blocks()
    trn, val, tst = split_by_mlp(d)
    A, B, MU = d["gt_a"], d["gt_b"], d["mu"]
    pt, pv, px = (primitives(d, s) for s in (trn, val, tst))
    avail = tuple(k for k in CH
                  if float(np.abs(pt[k]).max()) > 0 and k not in drop)
    yt = (0.5 * (A[trn] + B[trn]) - MU[trn]).ravel()
    base_v = umse(MU[val], A[val], B[val])
    base_x = umse(MU[tst], A[tst], B[tst])

    def try_set(cs, sh, pl):
        Xt_, nm_, _ = design(pt, cs, sh, MOD, pl)
        Xv_, _, _ = design(pv, cs, sh, MOD, pl)
        lam_, v_, b_ = fit_eval(Xt_.reshape(-1, len(nm_)), yt, Xv_, MU[val],
                                A[val], B[val], lams)
        return v_, lam_, b_, nm_

    # ---- selection, on VALIDATION only ------------------------------------
    # Greedy forward selection over CHANNELS, then the two optional blocks.
    # Every candidate is scored on the validation split; the test split is read
    # ONCE, at the end, for the winner.  A channel is only kept if it improves
    # validation by more than 0.2%, which is the threshold this repository has
    # used for overriding a simpler design throughout.
    print(f"# greedy forward selection over {len(avail)} channels, on "
          f"validation ({len(np.unique(d['mlp_seeds'][val]))} MLPs)")
    cur: tuple[str, ...] = ()
    path: list = []
    best_v = umse(MU[val] + design(pv, (), SH, MOD, ())[0]
                  @ try_set((), SH, ())[2], A[val], B[val])
    while True:
        cands = [(try_set(cur + (k,), SH, ())[0], k) for k in avail
                 if k not in cur]
        if not cands:
            break
        v_, k_ = min(cands)
        # Add while the step helps AT ALL, and pick the best PREFIX at the
        # end.  A per-step improvement threshold stops far too early here:
        # after two channels every single addition is worth under 0.2% and
        # the next four together are worth 4%, because the channels are
        # strongly collinear and each one only frees part of the next.
        if v_ >= best_v:
            break
        cur, best_v = cur + (k_,), v_
        path.append((v_, cur))
        print(f"  + {k_:<8} val {best_v:11.4e}  {base_v / best_v:6.3f}x")
    best_v, cur = min(path, key=lambda t: t[0])
    print(f"  best prefix: {len(cur)} channels at val {best_v:11.4e}")
    sh, pool = SH, ()
    SH_NOEIG = tuple(k for k in SH
                     if k not in ("u1", "u2", "lam1", "lam2"))
    # Shape BLOCKS, dropped one at a time.  These columns were carried into
    # the design wholesale and never ablated individually; each one that
    # survives costs dispatches in the shipped kernel, and dispatch time is
    # billed at lambda, so the prior is to drop.
    SH_BLOCKS = {"eigen": ("u1", "u2", "lam1", "lam2"),
                 "cumulants": ("gam1", "gam2"),
                 "weightcols": ("wn1", "w43"),
                 "suite": ("vbar", "arms", "keep_frac"),
                 "pilot alpha": ("alpha_p",),
                 "sd_mc": ("sd_mc",)}
    # An ADDITION must beat the incumbent by 0.2%; a REMOVAL is taken unless
    # it loses by more than 0.2%.  Both thresholds point the same way -- toward
    # the cheaper design -- which is the right prior when the expensive block
    # (the power iteration) is 0.15% of the whole FLOP budget and the
    # difference it makes is 0.1% of a validation number.
    v_ = try_set(cur, SH, POOL)[0]
    keep = v_ < best_v * 0.998
    print(f"  {'+ pooled u1/u2 terms':<22} val {v_:11.4e}  "
          f"{base_v / v_:6.3f}x   {'ADOPTED' if keep else 'rejected'}")
    if keep:
        best_v, pool = v_, POOL
    for bname, bcols in SH_BLOCKS.items():
        cand = tuple(k for k in sh if k not in bcols)
        if cand == sh:
            continue
        v_ = try_set(cur, cand, pool)[0]
        drop_it = v_ < best_v * 1.002
        print(f"  - shape {bname:<14} val {v_:11.4e}  {base_v / v_:6.3f}x   "
              f"{'DROPPED' if drop_it else 'kept'}")
        if drop_it:
            best_v, sh = v_, cand
    ch = cur
    vv, lam, beta, names = try_set(ch, sh, pool)
    Xx, _, _ = design(px, ch, sh, MOD, pool)
    xx = umse(MU[tst] + Xx @ beta, A[tst], B[tst])
    print(f"\nselected channels: {', '.join(ch)}")

    tv = np.concatenate([trn, val])
    ptv = primitives(d, tv)
    Xtv, _, _ = design(ptv, ch, sh, MOD, pool)
    ytv = (0.5 * (A[tv] + B[tv]) - MU[tv]).ravel()
    sc = np.sqrt(np.mean(Xtv.reshape(-1, len(names)) ** 2, axis=0))
    sc = np.where(sc > 0, sc, 1.0)
    Xs = Xtv.reshape(-1, len(names)) / sc
    beta_ship = (BC.ridge_solve(Xs.T @ Xs, Xs.T @ ytv, lam, len(Xs)) / sc)

    # NUMERIC ONLY.  ``fnp.load`` refuses any non-numeric dtype outright
    # ("object dtype would require pickle"), so a string column-name array in
    # the shipped npz is a hard failure at setup, not a warning.  The names go
    # in a sidecar JSON that the submission never reads.
    p = data_dir() / out_name
    np.savez(p, beta=beta_ship.astype(np.float32))
    meta = {"features": list(names), "channels": list(ch),
            "shape_cols": list(sh), "modulators": list(MOD),
            "pool": list(pool), "lam": float(lam),
            "n_train_mlps": int(len(np.unique(d["mlp_seeds"][trn]))),
            "n_fit_mlps": int(len(np.unique(d["mlp_seeds"][tv]))),
            "test_gain": float(base_x / xx), "val_gain": float(base_v / vv),
            "tau": SHIP_TAU, "n_samples": SHIP_N, "n_pilot": SHIP_P,
            "beta": beta_ship.tolist()}
    (data_dir() / (out_name + ".json")).write_text(json.dumps(meta, indent=1))
    print(f"wrote {p}  ({p.stat().st_size} bytes, {len(names)} coefficients) "
          f"+ {out_name}.json")
    print(f"  penalty {lam:.0e} selected on validation; TEST gain of the "
          f"train-only fit {base_x / xx:.3f}x")
    sp = Path(__file__).resolve().parent.parent / "submission" / "corrector.npz"
    if sp.is_file():
        sb = np.load(sp)["beta"].astype(np.float64)
        cols = [px["one"]]
        for k in ("cv1", "cv2", "cv1mf"):
            cols += [px[k], px[k] * px["Phi"], px[k] * px["alpha"]]
        cols += [px[k] for k in ("s", "Phi", "phi", "alpha")] + [px["dpilot"]]
        xs = umse(MU[tst] + np.stack(cols, axis=-1) @ sb, A[tst], B[tst])
        dflop = (NEW_CHANNEL_FLOPS - 4.09e8) if not pool and sh is SH_NOEIG \
            else NEW_CHANNEL_FLOPS
        a0, _ = project(1.0)
        a1, n1 = project(xs / xx, d_flops=dflop)
        print(f"  shipped head on the SAME test split {xs:.4e}; this head "
              f"{xx:.4e}  ->  {xs / xx:.3f}x")
        print(f"  projected graded {a1:.4e} at N* = {n1:,.0f}  "
              f"({a0 / a1:.3f}x over the shipped 2.4646e-07), "
              f"new-channel cost {dflop:.2e} FLOPs")
    print(f"  shipped vector refitted on train+validation "
          f"({int(len(np.unique(d['mlp_seeds'][tv])))} MLPs)")
    print("\n--- loader contract -------------------------------------------")
    print("  beta = fnp.load(path)['beta']            # (n_cols,) float32")
    print("  corr = fnp.stack(cols, axis=1) @ beta    # (width,)")
    print("  return ... (mu + corr)[None, :] ...")
    print("  cols, in this exact order:")
    for i, n_ in enumerate(names):
        print(f"    {i:3d}  {n_}")


# ---------------------------------------------------------------------------
# mode: cost
# ---------------------------------------------------------------------------
def mode_cost(n_params: int) -> None:
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    from whestfloor.contract import FLOP_BUDGET  # noqa: PLC0415

    W = [np.asarray(w) for w in make_mlp(WIDTH, DEPTH, 123)]
    print("# billed cost of the new channels, in a real BudgetContext\n")

    def bill(fn, *a):
        with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as c:
            fn(*a)
        return int(c.flops_used), float(c.residual_wall_time_s)

    fw = [fnp.asarray(w) for w in W]
    sq = None

    def prop_mean(weights, gates, d0):
        p = d0
        for l in range(1, len(weights)):
            p = p @ weights[l]
            if l < len(weights) - 1:
                p = p * gates[l]
        return p

    def prop_two(weights, gates, gv, d0, v0):
        nonlocal sq
        p, q = d0, v0
        for l in range(1, len(weights)):
            p = p @ weights[l]
            q = q @ sq[l]
            if l < len(weights) - 1:
                p = p * gates[l] + q * gv[l]
                q = q * gv[l]
        return p, q

    d0 = fnp.asarray(np.random.default_rng(0).standard_normal(WIDTH)
                     .astype(np.float32))
    gates = fnp.asarray(np.full((DEPTH, WIDTH), 0.5, dtype=np.float32))
    f1, r1 = bill(prop_mean, fw, gates, d0)
    with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True):
        pass
    sq = [w * w for w in fw]
    f2, r2 = bill(prop_two, fw, gates, gates, d0, d0)
    print(f"  cv1mf mean propagation (shipped)   {f1:12,d} FLOPs  "
          f"{f1 / FLOP_BUDGET * 100:7.4f}% of B   {r1 * 1e3:6.2f} ms")
    print(f"  mfm + mfv two-channel propagation  {f2:12,d} FLOPs  "
          f"{f2 / FLOP_BUDGET * 100:7.4f}% of B   {r2 * 1e3:6.2f} ms")
    print(f"  the W .^ 2 tables, once per MLP    "
          f"{DEPTH * WIDTH * WIDTH:12,d} FLOPs  "
          f"{DEPTH * WIDTH * WIDTH / FLOP_BUDGET * 100:7.4f}% of B")

    # head evaluation
    for npar_in, nfeat in ((24, 256), (24, 2048), (24, 16384)):
        Ain = fnp.asarray(np.zeros((npar_in, nfeat), dtype=np.float32))
        cin = fnp.asarray(np.zeros(nfeat, dtype=np.float32))
        w = fnp.asarray(np.zeros(nfeat + npar_in, dtype=np.float32))
        Z = fnp.asarray(np.zeros((WIDTH, npar_in), dtype=np.float32))

        def head(Z, Ain, cin, w):
            h = fnp.tanh(Z @ Ain + cin)
            return fnp.concatenate([Z, h], axis=1) @ w

        f, r = bill(head, Z, Ain, cin, w)
        print(f"  RF head {nfeat:6d} features ({npar_in * nfeat + nfeat:9,d} "
              f"params)  {f:12,d} FLOPs  {f / FLOP_BUDGET * 100:7.4f}% of B  "
              f"{r * 1e3:6.2f} ms")

    # ---- the rest of the new block, at N = 25000 -------------------------
    N = SHIP_N
    rng = np.random.default_rng(1)
    xf = fnp.asarray(rng.standard_normal((N, WIDTH)).astype(np.float32))
    muf = fnp.asarray(rng.standard_normal(WIDTH).astype(np.float32))
    V0 = fnp.asarray(np.linalg.qr(rng.standard_normal((WIDTH, 2)))[0]
                     .astype(np.float32))

    def power2(x, mu, V):
        for _ in range(8):
            Y = x @ V
            V = (x.T @ Y) / np.float32(N) - fnp.outer(mu, mu @ V)
            V = fnp.linalg.qr(V)[0]
        return V

    f, r = bill(power2, xf, muf, V0)
    print(f"  top-2 modes, 8 power sweeps at N={N:,}   {f:12,d} FLOPs  "
          f"{f / FLOP_BUDGET * 100:7.4f}% of B   {r * 1e3:6.2f} ms")

    def gram_ref(x):
        return x.T @ x

    f2, _ = bill(gram_ref, xf)
    print(f"  (the Gram it replaces)                   {f2:12,d} FLOPs  "
          f"{f2 / FLOP_BUDGET * 100:7.4f}% of B")

    zg = fnp.asarray(rng.standard_normal((BC.GATE_ROWS, WIDTH))
                     .astype(np.float32))

    def gates(z):
        m = fnp.mean(z, axis=0)
        v = fnp.maximum(fnp.mean(z * z, axis=0) - m * m, 1e-12)
        return flops.stats.norm.cdf(m / fnp.sqrt(v))

    f, r = bill(lambda: [gates(zg) for _ in range(DEPTH)])
    print(f"  scored-pass gates, {BC.GATE_ROWS} rows x {DEPTH}      "
          f"{f:12,d} FLOPs  {f / FLOP_BUDGET * 100:7.4f}% of B   "
          f"{r * 1e3:6.2f} ms")

    def mfv2_stat(z):
        return fnp.mean(z * z, axis=0), fnp.mean(z, axis=0)

    f, r = bill(mfv2_stat, xf)
    print(f"  mfv2 layer-2 second moment at N={N:,}    {f:12,d} FLOPs  "
          f"{f / FLOP_BUDGET * 100:7.4f}% of B   {r * 1e3:6.2f} ms")

    W1 = fnp.asarray(W[0])
    W2 = fnp.asarray(W[1])

    def l12(w1, w2):
        S = w1.T @ w1
        s = fnp.sqrt(fnp.maximum(fnp.diagonal(S), 1e-12))
        rho = fnp.clip(S / fnp.outer(s, s), -1.0, 1.0)
        Ch = (fnp.outer(s, s) / (2.0 * np.pi)) * (
            fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0))
            + rho * (np.pi / 2.0 + fnp.arcsin(rho)))
        Ch = Ch - fnp.outer(s, s) / (2.0 * np.pi)
        return fnp.sum((w2.T @ Ch) * w2.T, axis=1)

    f, r = bill(l12, W1, W2)
    print(f"  exact diag Cov(z^2) (arc-cosine kernel)  {f:12,d} FLOPs  "
          f"{f / FLOP_BUDGET * 100:7.4f}% of B   {r * 1e3:6.2f} ms")

    # npz load at size, and the 5 s setup window
    print()
    for mb in (1, 8, 32):
        n = int(mb * 1024 * 1024 / 4)
        p = artifacts() / f"_loadprobe_{mb}.npz"
        np.savez(p, W=np.zeros(n, dtype=np.float32))
        t = time.time()
        with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as c:
            arr = fnp.load(str(p))["W"]
            _ = arr.shape
        dt = time.time() - t
        print(f"  fnp.load {mb:3d} MiB ({n:,} float32): {int(c.flops_used)} "
              f"FLOPs, {dt * 1e3:.0f} ms wall "
              f"({dt / 5.0 * 100:.1f}% of the setup window)")
        p.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("refcal", "data", "fit", "curve", "noise",
                             "export", "cost"))
    ap.add_argument("--n-mlps", type=int, default=2000)
    ap.add_argument("--mlp-seed-base", type=int, default=400_000)
    ap.add_argument("--n-gt", type=int, default=65_536)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=SHIP_N)
    ap.add_argument("--n-pilot", type=int, default=SHIP_P)
    ap.add_argument("--tau", type=float, default=SHIP_TAU)
    ap.add_argument("--q2", action="store_true")
    ap.add_argument("--no-relu1", action="store_true")
    ap.add_argument("--limit-mlps", type=int, default=None)
    ap.add_argument("--lams", type=str,
                    default="1e-6,1e-4,1e-3,1e-2,1e-1,3e-1,1,3,10,30,100")
    ap.add_argument("--rf-feats", type=str, default="256,1024,4096")
    ap.add_argument("--rf-scale", type=float, default=1.0)
    ap.add_argument("--rf-seed", type=int, default=0)
    ap.add_argument("--curve-grid", type=str, default="25,50,100,200,400,800")
    ap.add_argument("--out", type=str, default="fit.json")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n-big", type=int, default=1_000_000)
    ap.add_argument("--no-groups", action="store_true")
    ap.add_argument("--sub", type=str, default="bigcorr")
    ap.add_argument("--sgd-hidden", type=str, default="",
                    help="e.g. '64;256;512,256' -- semicolon-separated shapes")
    ap.add_argument("--sgd-epochs", type=int, default=25)
    ap.add_argument("--sgd-lr", type=float, default=3e-3)
    ap.add_argument("--infl", type=str, default="1,2,4,8,16")
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--drop", type=str, default="")
    args = ap.parse_args()

    global SUBDIR
    SUBDIR = args.sub
    lams = [float(v) for v in args.lams.split(",")]
    rf = [int(v) for v in args.rf_feats.split(",") if int(v) > 0]
    if args.mode == "refcal":
        mode_refcal(args.n_mlps, args.n_big, args.reps)
    elif args.mode == "data":
        mode_data(args.n_mlps, args.mlp_seed_base, args.n_gt, args.n_seeds,
                  args.shard, args.n_shards, args.block, args.n_samples,
                  args.n_pilot, args.tau, args.q2, not args.no_relu1)
    elif args.mode == "fit":
        hid = tuple(tuple(int(x) for x in h.split(","))
                    for h in args.sgd_hidden.split(";") if h)
        mode_fit(lams, args.limit_mlps, rf, args.rf_scale, args.rf_seed,
                 args.out, not args.no_groups, hid, args.sgd_epochs,
                 args.sgd_lr)
    elif args.mode == "export":
        mode_export(lams, args.out if args.out != "fit.json"
                    else "bigcorr_head.npz",
                    () if args.no_pool else POOL,
                    tuple(k for k in args.drop.split(",") if k))
    elif args.mode == "noise":
        mode_noise(lams, [int(v) for v in args.curve_grid.split(",")],
                   [float(v) for v in args.infl.split(",")])
    elif args.mode == "curve":
        mode_curve(lams, rf, args.rf_scale, args.rf_seed,
                   [int(v) for v in args.curve_grid.split(",")])
    else:
        mode_cost(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
