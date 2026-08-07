#!/usr/bin/env python
"""Re-sweep the sparsity threshold ``tau`` against the RIGHT objective.

Why every earlier sweep answered the wrong question
---------------------------------------------------
``adjusted = raw x max(0.1, C/B)`` with ``C = F_fix + c(tau) N + lambda R``.
``N`` is not a constant of the estimator -- it is an inner variable, and the
sweep has to re-optimise it at every ``tau``:

    adjusted(tau) = min_N [ b(tau)^2 + v_eff(tau)/N ]
                         * max(0.1, (F0 + c(tau) N) / B)

Holding ``N`` fixed prices ``c`` at zero.  Pinning ``N`` to the 0.1 clamp
prices it at the clamp's slope.  Both are wrong, in opposite directions.  The
graded 11-point N-sweep (325597-325608) showed the optimum is INTERIOR and
sits at ``C/B ~ 0.22``, 2.2x past the clamp: above the clamp the objective is
``(1/B)[F0 b^2 + c v_eff + F0 v_eff/N + c b^2 N]``, so the fixed compute
``F0`` amortising is what pulls ``N`` up and the bias floor is what pushes it
back.  In the zero-bias limit the whole score is ``v_eff * c / B`` -- the
product this sweep exists to minimise.

How ``b`` and ``v`` are separated
---------------------------------
Two independent sample streams per MLP, and for each stream BOTH the dense
pass (``tau=None``, unbiased) and the pruned pass, driven by the SAME input
draw.  Common random numbers make ``e_i = mu_sparse_i - mu_dense_i`` a
low-variance estimate of the pruning bias, and

    b^2  =  mean_j  e_1j e_2j          (unbiased: e_1 and e_2 independent)
    v/N  =  mean_j (mu_1j - mu_2j)^2 / 2

The pilot -- hence the mask -- is drawn from the same generator before the
scored draw, exactly as the shipped kernel does, so the pairing changes
nothing about the estimator being measured.  ``c(tau)`` is flopscope's own
``dF/dN`` on the real kernel.

The answer, and it is a refutation
----------------------------------
The premise was that the bias budget is underspent, so a much lower ``tau``
should win.  It does not, and the reason is not bias at all: **pruning a
marginal neuron injects variance faster than it saves compute.**  Measured on
48 local MLPs, ``v_raw`` against ``c``:

    tau    keep    c/sample   v_raw     v_raw*c    b^2        adjusted x2.5
    4.00   0.924   3.63e6     0.05168   1.876e5    -1.7e-14   0.808
    3.00   0.848   3.10e6     0.05192   1.610e5     4.4e-11   0.934
    2.75   0.827   2.96e6     0.05179   1.534e5     5.8e-10   0.978
    2.50   0.806   2.82e6     0.05296   1.496e5     3.9e-09   1.000  <- min
    2.25   0.784   2.68e6     0.05641   1.513e5     2.9e-08   0.978
    2.00   0.760   2.53e6     0.06198   1.571e5     2.5e-07   0.883
    1.50   0.709   2.23e6     0.09079   2.023e5     8.7e-06   0.272
    1.00   0.653   1.91e6     0.24160   4.603e5     2.0e-04   0.018

``v_raw * c`` is minimised AT tau = 2.5, and it is minimised there by the
variance term alone -- the bias only makes low ``tau`` worse on top.  Below
2.5, ``v_raw`` rises 17% (2.0), 71% (1.5), 356% (1.0) while ``c`` falls only
10%, 21%, 33%.  Above 2.5 the trade reverses and ``c`` dominates.  The optimum
is broad: 2.25 and 2.75 are both 0.978x, so nothing here is delicately tuned.

The pruning bias is also an order of magnitude smaller than the repository
believed: ``b^2(2.5) = 3.9e-09``, i.e. an RMS sign error of 6.3e-05, against
the docstring's claim of "2-3e-7 in raw-MSE terms".  It is NOT what limits
``N``.  What limits ``N`` is ``b2_head`` -- see ``--mode objective``.

Selection data
--------------
LOCAL MLPs only (``--seed-base``, default 700000), disjoint from the official
suite and from the corrector's training seeds.  ``official_mini.npz`` is never
read by this script; the graded constants it consumes (``--f0``, ``--c-ref``)
come from the submitted N-sweep, not from any local suite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import (  # noqa: E402
    DEPTH,
    FLOP_BUDGET,
    MULTIPLIER_FLOOR,
    WIDTH,
)
from whestfloor.mc import make_mlp  # noqa: E402

VAR_FLOOR = 1e-12


def artifacts() -> Path:
    p = Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve() / "tau"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Numpy mirror of whestfloor.kernels._pilot_stats / _sparse_plan.
# ---------------------------------------------------------------------------
def pilot_stats(weights, rng, n_pilot, n, closure=False):
    """``(alpha, mean_h)``; with ``closure``, ``mean_h`` is the RECTIFIED
    GAUSSIAN mean ``m Phi(a) + s phi(a)`` instead of the pilot's sample mean.

    Why that can matter enormously for the frozen dead-neuron constants: a
    neuron with ``alpha < -2.5`` fires on ~0.6% of draws, so the P = 150
    sample mean of ``relu(z)`` is a mean over ~1 nonzero observation and its
    relative error is of order 100%.  ``m`` and ``s`` are estimated from all P
    draws at ~6% relative error, and ``E[relu] = m Phi(a) + s phi(a)`` is
    exact for a Gaussian marginal, so the closure trades a huge sampling error
    for a small non-Gaussianity error -- on quantities that are tiny to begin
    with.
    """
    from whestfloor import corrector as C  # noqa: PLC0415

    x = rng.standard_normal((n_pilot, n), dtype=np.float32)
    ms, e2s, mhs = [], [], []
    for w in weights:
        z = x @ w
        ms.append(np.mean(z, axis=0))
        e2s.append(np.mean(z * z, axis=0))
        x = np.maximum(z, 0.0)
        mhs.append(np.mean(x, axis=0))
    m = np.stack(ms, axis=0)
    v = np.maximum(np.stack(e2s, axis=0) - m * m, VAR_FLOOR)
    sig = np.sqrt(v)
    a = m / sig
    if closure:
        return a, (m * C.norm_cdf(a) + sig * C.norm_pdf(a)).astype(np.float32)
    return a, np.stack(mhs, axis=0)


def sparse_plan(weights, alpha, mean_h, tau):
    depth = len(weights)
    keeps = None if tau is None else (alpha > -tau)
    subs, biases, sizes = [], [], []
    keep_prev = None
    for l, w in enumerate(weights):
        keep = None if (keeps is None or l == depth - 1) else keeps[l]
        wc = w if keep is None else w[:, keep]
        subs.append(wc if keep_prev is None else wc[keep_prev, :])
        biases.append(None if keep_prev is None
                      else np.where(keep_prev, np.float32(0.0),
                                    mean_h[l - 1]) @ wc)
        sizes.append(subs[-1].shape)
        keep_prev = keep
    return subs, biases, (sizes, [b is not None for b in biases])


def scored_mu(x, subs, biases):
    for s, b in zip(subs, biases):
        z = x @ s
        if b is not None:
            z = z + b
        x = np.maximum(z, 0.0)
    return np.mean(x, axis=0, dtype=np.float64)


def cost_per_sample(sizes, has_bias) -> float:
    """Billed FLOPs per scored sample: flopscope's matmul + add + relu prices.

    A ``(1, p) @ (p, q)`` row costs ``q * (2p - 1)``, the frozen-bias add
    ``q``, and the rectifier ``q``; the standard-normal draw is 16 per input
    coordinate.  ``--mode check`` pins this against flopscope's own
    ``dF/dN`` on the real kernel (agrees to <1e-6 relative).
    """
    tot = 16.0 * sizes[0][0]
    for (p, q), b in zip(sizes, has_bias):
        tot += q * (2.0 * p - 1.0) + q + (q if b else 0)
    return tot


# ---------------------------------------------------------------------------
def mode_sweep(taus, n_mlps: int, seed_base: int, n_samples: int,
               n_pilot: int, out: str) -> None:
    print(f"# tau sweep on {n_mlps} LOCAL MLPs (seeds {seed_base}..), "
          f"N={n_samples}, P={n_pilot}")
    print("# b^2 from the cross-product of two independent common-random-"
          "number bias estimates;\n# v/N from the same two streams.  "
          "official_mini.npz is NOT read.\n")

    acc = {t: {"bb": [], "vn": [], "c": [], "keep": []} for t in taus}
    t0 = time.time()
    for i in range(n_mlps):
        W = make_mlp(WIDTH, DEPTH, seed_base + i)
        es = {t: [] for t in taus}
        mus = {t: [] for t in taus}
        for rep in range(2):
            rng = np.random.default_rng(seed_base + i + 7919 * (rep + 1))
            alpha, mean_h = pilot_stats(W, rng, n_pilot, WIDTH)
            x0 = rng.standard_normal((n_samples, WIDTH), dtype=np.float32)
            ds, db, _ = sparse_plan(W, alpha, mean_h, None)
            mu_dense = scored_mu(x0, ds, db)
            for t in taus:
                ss, sb, sz = sparse_plan(W, alpha, mean_h, t)
                mu = scored_mu(x0, ss, sb)
                if rep == 0:
                    acc[t]["c"].append(cost_per_sample(*sz))
                    acc[t]["keep"].append(float(np.mean(alpha[:-1] > -t)))
                es[t].append(mu - mu_dense)
                mus[t].append(mu)
        for t in taus:
            acc[t]["bb"].append(float(np.mean(es[t][0] * es[t][1])))
            acc[t]["vn"].append(
                float(np.mean((mus[t][0] - mus[t][1]) ** 2) / 2.0))
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{n_mlps}  {time.time()-t0:.0f}s", flush=True)

    rows = []
    hdr = (f"{'tau':>6} {'keep':>7} {'c/sample':>10} {'c rel':>7} "
           f"{'b^2':>11} {'+-':>10} {'v_raw':>9} {'rms bias':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    c25 = None
    for t in taus:
        a = acc[t]
        bb = np.asarray(a["bb"])
        vn = np.asarray(a["vn"])
        c = float(np.mean(a["c"]))
        if abs(t - 2.5) < 1e-9:
            c25 = c
        rows.append({
            "tau": t, "b2": float(np.mean(bb)),
            "b2_se": float(np.std(bb, ddof=1) / np.sqrt(len(bb))),
            "vN": float(np.mean(vn)),
            "v_raw": float(np.mean(vn)) * n_samples,
            "c": c, "keep_frac": float(np.mean(a["keep"])),
        })
    for r in rows:
        r["c_rel"] = r["c"] / (c25 or r["c"])
        print(f"{r['tau']:6.2f} {r['keep_frac']:7.4f} {r['c']:10.4g} "
              f"{r['c_rel']:7.4f} {r['b2']:11.4e} {r['b2_se']:10.2e} "
              f"{r['v_raw']:9.5f} {np.sqrt(max(r['b2'],0)):10.3e}")

    p = artifacts() / out
    p.write_text(json.dumps({"n_mlps": n_mlps, "seed_base": seed_base,
                             "n_samples": n_samples, "n_pilot": n_pilot,
                             "rows": rows}, indent=1))
    print(f"\nwrote {p}")


# ---------------------------------------------------------------------------
def mode_objective(path: str, f0: float, head_gain: float, b2_head: float,
                   c_ref: float) -> None:
    """Score each ``tau`` at ITS OWN optimal ``N``.

    ``N`` is an inner variable, not a constant of the estimator, and pinning
    it -- to 8500, or to the clamp -- misprices ``c`` in opposite directions.
    The graded 11-point N-sweep settled where the optimum actually is:

        adjusted(tau) = min_N [ b(tau)^2 + v_eff(tau)/N ]
                             * max(0.1, (F0 + c(tau) N) / B)

    with ``F0 = F_fix + lambda R`` the N-independent compute.  Above the clamp
    this is ``(1/B)[F0 b^2 + c v_eff + F0 v_eff/N + c b^2 N]``, whose interior
    minimum is ``N* = sqrt(v_eff F0 / (b^2 c))``.  The fixed residual is
    exactly what makes the optimum interior; the bias is exactly what stops it
    running away.  Both branches are evaluated on a grid so the clamped branch
    wins wherever it should.

    ``b^2 = b_prune^2(tau) + b2_head``, the second being the N-INDEPENDENT
    error the offline head injects -- 88% of it is ``dpilot``, which carries
    the 150-sample pilot's own Monte-Carlo error and therefore does not shrink
    with ``N``.
    """
    d = json.loads((artifacts() / path).read_text())
    grid = np.geomspace(4.0e3, 4.0e5, 1500)
    print("# adjusted(tau) = min_N [b^2 + v_eff/N] * max(0.1, (F0 + c N)/B)")
    print(f"# F0 = {f0:.4g} ({f0/FLOP_BUDGET:.5f} of B),  "
          f"v_eff = v_raw/{head_gain:.3f},  b^2 = b_prune^2 + b_head^2 "
          f"({b2_head:.3e})")
    print(f"# c(tau) rescaled so c(2.5) = {c_ref:,.0f}, the graded value\n")
    hdr = (f"{'tau':>6} {'c/sample':>10} {'b^2':>10} {'v_eff':>8} "
           f"{'v_eff*c':>10} {'N*':>8} {'C/B':>7} {'raw':>11} "
           f"{'adjusted':>11} {'x tau2.5':>9}")
    print(hdr)
    print("-" * len(hdr))
    c25 = next(r["c"] for r in d["rows"] if abs(r["tau"] - 2.5) < 1e-9)
    out, base = [], None
    for r in d["rows"]:
        c = r["c"] / c25 * c_ref
        v = r["v_raw"] / head_gain
        b2 = max(r["b2"], 0.0) + b2_head
        raw = b2 + v / grid
        mult = np.maximum(MULTIPLIER_FLOOR, (f0 + c * grid) / FLOP_BUDGET)
        adj = raw * mult
        i = int(np.argmin(adj))
        if abs(r["tau"] - 2.5) < 1e-9:
            base = float(adj[i])
        out.append((r["tau"], c, b2, v, float(grid[i]), float(adj[i]),
                    float(raw[i]), float(mult[i])))
    for t, c, b2, v, N, a, raw, m in out:
        print(f"{t:6.2f} {c:10.4g} {b2:10.3e} {v:8.5f} {v*c:10.4g} "
              f"{N:8.0f} {m:7.4f} {raw:11.4e} {a:11.4e} "
              f"{(base / a) if base else 0:9.3f}")
    print("\n(x tau2.5 > 1 is better.  `v_eff * c` is the entire objective in "
          "the zero-bias limit --\n the score asymptotes to it at large N -- "
          "so it is printed separately.)")


# ---------------------------------------------------------------------------
def mode_check(taus, seed: int, n_pilot: int) -> None:
    """Validate ``cost_per_sample`` against flopscope on the real kernel."""
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    from whestfloor import kernels  # noqa: PLC0415

    Wn = make_mlp(WIDTH, DEPTH, seed)
    W = [fnp.asarray(w) for w in Wn]
    print(f"{'tau':>6} {'flopscope dF/dN':>16} {'analytic c':>12} "
          f"{'F at N=8500':>14} {'F/B':>7}")
    for t in taus:
        fl = []
        for nn in (2000, 4000):
            with flops.BudgetContext(flop_budget=int(1e13), quiet=True) as c:
                kernels.sparse_mc_kernel(W, tau=t, n_samples=nn,
                                         n_pilot=n_pilot, seed=1, safe=False)
            fl.append(int(c.flops_used))
        dF = (fl[1] - fl[0]) / 2000.0
        rng = np.random.default_rng(1)
        alpha, mean_h = pilot_stats(Wn, rng, n_pilot, WIDTH)
        _, _, sz = sparse_plan(Wn, alpha, mean_h, t)
        c_ana = cost_per_sample(*sz)
        with flops.BudgetContext(flop_budget=int(1e13), quiet=True) as c:
            kernels.sparse_mc_kernel(W, tau=t, n_samples=8500,
                                     n_pilot=n_pilot, seed=1, safe=False)
        F = int(c.flops_used)
        print(f"{t:6.2f} {dF:16.1f} {c_ana:12.1f} "
              f"{F:14,d} {F/FLOP_BUDGET:7.4f}")
        print(f"       F_fix = F - 8500*dF = {F - 8500*dF:,.0f}")


# ---------------------------------------------------------------------------
# mode: head -- what pins N, and what it costs to unpin it
# ---------------------------------------------------------------------------
CLOSURE = [False]
SHIP_FEATURES = ("one", "cv1", "cv1_Phi", "cv1_a", "cv2", "cv2_Phi", "cv2_a",
                 "cv1mf", "cv1mf_Phi", "cv1mf_a", "s", "Phi", "phi", "a",
                 "dpilot")
INV_SQRT_2PI = 0.3989422804014327


def shipped_features(W, seed, n_samples, n_pilot):
    """``(mu, design)`` -- the numpy mirror of what ``predict`` builds.

    Same generator order, same pilot, same mask, same sliced matmuls, so the
    design here is the design the grader would see at this ``n_samples``.
    """
    from whestfloor import corrector as C  # noqa: PLC0415

    depth, n = len(W), W[0].shape[0]
    rng = np.random.default_rng(seed)
    alpha, mean_h = pilot_stats(W, rng, n_pilot, n, CLOSURE[0])
    subs, biases, _ = sparse_plan(W, alpha, mean_h, 2.5)

    x0 = rng.standard_normal((n_samples, n), dtype=np.float32)
    x, z1, h1m = x0, None, None
    for l in range(depth):
        z = x @ subs[l]
        if biases[l] is not None:
            z = z + biases[l]
        if l == 0:
            z1 = z
        x = np.maximum(z, 0.0)
        if l == 0:
            h1m = np.mean(x, axis=0)
    mu = np.mean(x, axis=0, dtype=np.float64)

    cvs = C.hermite_cv(x0, z1, x, W[0], kmax=2, split=True)
    cv1, cv2 = cvs[0], cvs[1]
    m = np.mean(z, axis=0)
    v = np.maximum(np.mean(z * z, axis=0) - m * m, VAR_FLOOR)
    s = np.sqrt(v)
    a = m / s
    Ph = C.norm_cdf(a).astype(np.float32)
    ph = C.norm_pdf(a).astype(np.float32)
    gates = C.norm_cdf(alpha).astype(np.float32)
    sig1 = np.sqrt(np.maximum(np.sum(W[0] * W[0], axis=0), VAR_FLOOR))
    prop = h1m - sig1 * np.float32(INV_SQRT_2PI)
    for l in range(1, depth):
        prop = prop @ W[l]
        if l < depth - 1:
            prop = prop * gates[l]
    mf = prop * Ph
    one = np.ones_like(a)
    cols = [one, cv1, cv1 * Ph, cv1 * a, cv2, cv2 * Ph, cv2 * a,
            mf, mf * Ph, mf * a, s, Ph, ph, a, mu - mean_h[-1]]
    return mu, np.stack([np.asarray(c, dtype=np.float64) for c in cols],
                        axis=1)


def _umse(pred, a, b):
    return float(np.mean((pred - a) * (pred - b)))


def mode_head(n_mlps: int, n_list, n_pilot: int, f0: float, c_ref: float,
              out: str) -> None:
    """Is ``dpilot`` the thing pinning N, and what does removing it cost?

    ``dpilot = mu - mean_h[-1]`` is an inverse-variance blend with a SECOND,
    independent estimate of the same quantity -- the pilot's own final-layer
    mean.  The blend is unbiased for any coefficient, so with the OPTIMAL
    coefficient it can only help; the trouble is that the shipped coefficient
    is frozen at its N = 8500 value while the pilot's variance ``v/P`` does
    not shrink with N.  Frozen, it injects ``beta^2 v / P`` of N-INDEPENDENT
    error -- and that is what stops N going past ~20k.

    Everything here is fitted and selected on the GENERATED training MLPs
    (their own two independent Monte-Carlo reference halves), split by MLP.
    """
    from whestfloor import corrector as C  # noqa: PLC0415

    src = (Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()
           / "corrector")
    parts = [np.load(f) for f in sorted(src.glob("train_s*.npz"))]
    seeds = np.concatenate([p["mlp_seeds"] for p in parts])
    A = np.concatenate([p["gt_a"] for p in parts])
    Bg = np.concatenate([p["gt_b"] for p in parts])
    o = np.argsort(seeds)
    seeds, A, Bg = seeds[o], A[o], Bg[o]
    sel = np.linspace(0, len(seeds) - 1, n_mlps).astype(int)
    seeds, A, Bg = seeds[sel], A[sel], Bg[sel]

    rng = np.random.default_rng(20260807)
    perm = rng.permutation(n_mlps)
    tr, te = perm[: n_mlps // 2], perm[n_mlps // 2:]
    beta_ship = np.load(Path(__file__).resolve().parent.parent / "submission"
                        / "corrector.npz")["beta"].astype(np.float64)
    keep14 = [i for i, f in enumerate(SHIP_FEATURES) if f != "dpilot"]

    res = {}
    for N in n_list:
        t0 = time.time()
        MU, X = [], []
        for k, sd in enumerate(seeds):
            W = make_mlp(WIDTH, DEPTH, int(sd))
            mu, d = shipped_features(W, int(sd) + 1, N, n_pilot)
            MU.append(mu)
            X.append(d)
            del W
        MU, X = np.asarray(MU), np.asarray(X)
        base = _umse(MU[te], A[te], Bg[te])
        # Zero-refit-noise variants of the SHIPPED head: the only thing
        # that changes is the dpilot coefficient, whose N-dependence is
        # analytic (an inverse-variance blend of two independent unbiased
        # estimates weights the pilot at -P/(P+N)).  Rescaling it is not a
        # re-fit, it is applying the known law.
        b_zero = beta_ship.copy()
        b_zero[14] = 0.0
        b_scale = beta_ship.copy()
        b_scale[14] *= (n_pilot + 8500.0) / (n_pilot + N)
        rows = [("no head at all", MU[te], None)]
        rows.append(("shipped beta (frozen at N=8500)",
                     MU[te] + X[te] @ beta_ship, beta_ship))
        rows.append(("shipped, dpilot ZEROED",
                     MU[te] + X[te] @ b_zero, b_zero))
        rows.append(("shipped, dpilot rescaled -P/(P+N)",
                     MU[te] + X[te] @ b_scale, b_scale))
        for lbl, idx in (("refit 15 col at this N", list(range(15))),
                         ("refit 14 col, dpilot DROPPED", keep14)):
            Xt = X[tr][:, :, idx].reshape(-1, len(idx))
            yt = (0.5 * (A[tr] + Bg[tr]) - MU[tr]).reshape(-1)
            sc = C.design_scale(Xt)
            best = (None, np.inf)
            for lm in (1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
                bb = C.ridge_fit(Xt, yt, lm, sc)
                q = _umse(MU[tr] + X[tr][:, :, idx] @ bb, A[tr], Bg[tr])
                if q < best[1]:
                    best = (bb, q)
            bb = best[0]
            full = np.zeros(15)
            full[idx] = bb
            rows.append((lbl, MU[te] + X[te][:, :, idx] @ bb, full))
        print(f"\n=== N = {N:,}  ({n_mlps} generated MLPs, "
              f"{len(te)} held out, {time.time()-t0:.0f}s) ===")
        print(f"{'head':<34}{'held-out umse':>14}{'x no-head':>10}"
              f"{'beta_dpilot':>13}")
        for lbl, pred, bb in rows:
            q = _umse(pred, A[te], Bg[te])
            bd = "-" if bb is None else f"{bb[14]:.5f}"
            print(f"{lbl:<34}{q:14.4e}{base/q:10.3f}{bd:>13}")
            res.setdefault(lbl, {})[N] = q
        print(f"  optimal dpilot weight -P/(P+N) = "
              f"{-n_pilot/(n_pilot+N):.5f}")

    # ---- what each variant implies for the optimal N -------------------
    print("\n=== implied bias floor and re-optimised N ===")
    ns = np.asarray(sorted(n_list), dtype=float)
    grid = np.geomspace(4e3, 4e5, 1500)
    print(f"{'head':<34}{'b^2':>11}{'v_eff':>9}{'N*':>8}{'C/B':>7}"
          f"{'adjusted':>12}")
    out_rows = []
    for lbl, dd in res.items():
        y = np.asarray([dd[int(n)] for n in ns])
        M = np.stack([np.ones_like(ns), 1.0 / ns], 1)
        w = 1.0 / y
        (b2, v), *_ = np.linalg.lstsq(M * w[:, None], y * w, rcond=None)
        raw = max(b2, 0.0) + v / grid
        mult = np.maximum(MULTIPLIER_FLOOR, (f0 + c_ref * grid) / FLOP_BUDGET)
        adj = raw * mult
        i = int(np.argmin(adj))
        print(f"{lbl:<34}{b2:11.3e}{v:9.5f}{grid[i]:8.0f}{mult[i]:7.4f}"
              f"{adj[i]:12.4e}")
        out_rows.append({"head": lbl, "b2": float(b2), "v_eff": float(v),
                         "N_star": float(grid[i]), "cb": float(mult[i]),
                         "adjusted": float(adj[i]),
                         "umse": {int(n): dd[int(n)] for n in ns}})
    (artifacts() / out).write_text(json.dumps(
        {"n_mlps": n_mlps, "n_list": list(n_list), "n_pilot": n_pilot,
         "rows": out_rows}, indent=1))
    print(f"\nwrote {artifacts() / out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("sweep", "objective", "check", "head"))
    ap.add_argument("--taus", default="3.0,2.5,2.0,1.5,1.25,1.0,0.75,0.5")
    ap.add_argument("--n-mlps", type=int, default=48)
    ap.add_argument("--seed-base", type=int, default=700_000)
    ap.add_argument("--n-samples", type=int, default=8500)
    ap.add_argument("--n-pilot", type=int, default=150)
    ap.add_argument("--out", default="tau_curve.json")
    # Defaults are the GRADED constants: F0 and c come from regressing the
    # 11 submitted C/B values on N (residual < 0.4% of B); b2_head is the
    # N-independent error the head injects, dominated by ``dpilot``.
    ap.add_argument("--f0", type=float, default=2.437e9)
    ap.add_argument("--c-ref", type=float, default=2.790e6)
    ap.add_argument("--head-gain", type=float, default=1.60)
    ap.add_argument("--b2-head", type=float, default=1.029e-7)
    ap.add_argument("--n-grid", default="8500,22000,45000")
    ap.add_argument("--closure", action="store_true",
                    help="frozen dead-neuron constants from the "
                         "rectified-Gaussian closure, not the "
                         "pilot sample mean")
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(",") if x]
    CLOSURE[0] = args.closure
    if args.mode == "head":
        mode_head(args.n_mlps, [int(v) for v in args.n_grid.split(",")],
                  args.n_pilot, args.f0, args.c_ref, args.out)
        return 0
    if args.mode == "sweep":
        mode_sweep(taus, args.n_mlps, args.seed_base, args.n_samples,
                   args.n_pilot, args.out)
    elif args.mode == "check":
        mode_check(taus, args.seed_base, args.n_pilot)
    else:
        mode_objective(args.out, args.f0, args.head_gain,
                       args.b2_head, args.c_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
