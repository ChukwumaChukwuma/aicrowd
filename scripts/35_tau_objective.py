#!/usr/bin/env python
"""Re-sweep the sparsity threshold ``tau`` against the CLAMPED objective.

Why the old sweep answered the wrong question
---------------------------------------------
``adjusted = raw x max(0.1, C/B)``.  The graded run (submission 325593) came
back at ``C/B = 0.10012`` -- we sit ON the clamp.  There, ``N`` is not a free
parameter: it is pinned at ``N* = (0.1 B - F_fix - lambda R) / c(tau)``, and

    adjusted(tau) = 0.1 * [ b(tau)^2  +  v_eff(tau) / N*(tau) ]

so a cheaper ``tau`` does not pay a multiplier penalty -- it buys samples.  The
whole trade is ``bias^2`` against ``v_eff * c``.  Every previous sweep
minimised raw MSE at FIXED ``N``, or adjusted score above the clamp, and both
of those price ``c`` far too dearly.  The bias budget is much larger than we
were treating it as: an RMS bias of 1.8e-3 costs one whole current score.

How ``b`` and ``v`` are separated
---------------------------------
Two independent sample streams per MLP, and for each stream BOTH the dense
pass (``tau=None``, unbiased) and the pruned pass, driven by the SAME input
draw.  Common random numbers make ``e_i = mu_sparse_i - mu_dense_i`` a
low-variance estimate of the pruning bias, and

    b^2  =  mean_j  e_1j e_2j          (unbiased: e_1 and e_2 independent)
    v/N  =  mean_j (mu_1j - mu_2j)^2 / 2

The pilot -- hence the mask -- is drawn from the same generator before the
scored draw, exactly as the shipped kernel does, so nothing about the pairing
changes the estimator being measured.

``c(tau)`` is read from flopscope directly, by differencing the billed FLOPs
of the real kernel at two sample counts.

Selection data
--------------
LOCAL MLPs only (``--seed-base``, default 700000), disjoint from the official
suite and from the corrector's training seeds.  ``official_mini.npz`` is never
read by this script.
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
def pilot_stats(weights, rng, n_pilot, n):
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
    return m / np.sqrt(v), np.stack(mhs, axis=0)


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
def mode_objective(path: str, f_fix: float, resid_s: float, head_gain: float,
                   lam: float = 1.0e11) -> None:
    """Turn the measured ``(b^2, v_raw, c)`` curve into the clamped score."""
    d = json.loads((artifacts() / path).read_text())
    free = MULTIPLIER_FLOOR * FLOP_BUDGET - f_fix - lam * resid_s
    print(f"# clamped objective   adjusted = 0.1 * [ b^2 + v_eff / N* ]")
    print(f"# N*(tau) = ({MULTIPLIER_FLOOR*FLOP_BUDGET:.4g} - F_fix "
          f"{f_fix:.4g} - 1e11*R {lam*resid_s:.4g}) / c(tau) = "
          f"{free:.5g} / c")
    print(f"# v_eff = v_raw / {head_gain:.3f}  (head's measured variance gain)")
    hdr = (f"{'tau':>6} {'c/sample':>10} {'N*':>7} {'0.1 b^2':>11} "
           f"{'0.1 v/N*':>11} {'adjusted':>11} {'x ship':>7}")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for r in d["rows"]:
        N = free / r["c"]
        bias_term = MULTIPLIER_FLOOR * r["b2"]
        var_term = MULTIPLIER_FLOOR * (r["v_raw"] / head_gain) / N
        adj = bias_term + var_term
        if abs(r["tau"] - 2.5) < 1e-9:
            base = adj
        print(f"{r['tau']:6.2f} {r['c']:10.4g} {N:7.0f} {bias_term:11.4e} "
              f"{var_term:11.4e} {adj:11.4e} "
              + (f"{base/adj:7.3f}" if base else "      -"))
    print("\n(x ship > 1 is better; the tau=2.5 row is the reference)")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("sweep", "objective", "check"))
    ap.add_argument("--taus", default="3.0,2.5,2.0,1.5,1.25,1.0,0.75,0.5")
    ap.add_argument("--n-mlps", type=int, default=48)
    ap.add_argument("--seed-base", type=int, default=700_000)
    ap.add_argument("--n-samples", type=int, default=8500)
    ap.add_argument("--n-pilot", type=int, default=150)
    ap.add_argument("--out", default="tau_curve.json")
    ap.add_argument("--f-fix", type=float, default=2.05e8)
    ap.add_argument("--resid", type=float, default=0.0209)
    ap.add_argument("--head-gain", type=float, default=1.866)
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(",") if x]
    if args.mode == "sweep":
        mode_sweep(taus, args.n_mlps, args.seed_base, args.n_samples,
                   args.n_pilot, args.out)
    elif args.mode == "check":
        mode_check(taus, args.seed_base, args.n_pilot)
    else:
        mode_objective(args.out, args.f_fix, args.resid, args.head_gain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
