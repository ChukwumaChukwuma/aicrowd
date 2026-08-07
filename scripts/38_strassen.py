#!/usr/bin/env python
"""Strassen in the scored pass: 1.13x fewer BILLED FLOPs for the same answer.

Why this is not the accounting exploit that ``docs/graded.md`` refuted
---------------------------------------------------------------------
That audit showed every *equivalent contraction* -- ``matmul``, ``dot``,
``einsum ij,jk->ik``, ``tensordot``, ``multi_dot`` -- bills exactly
``n w (2w-1)``, so there is no mispriced op to arbitrage.  Strassen is not an
equivalent contraction.  It is a different ALGORITHM: it returns the same
product from 7 half-size multiplications plus 18 additions instead of 8
multiplications, and 7/8 of half the work is less work.  The bill is honest;
the arithmetic really is cheaper.  Recursion depth 1 on a 256-wide layer is

    7 x (N/2 x 128) @ (128 x 128)  +  13 elementwise (N/2 x 128) adds
    = 115,072 FLOPs/sample   against   130,816 direct    = 1.137x

and the weight-side combinations are per-MLP, not per-sample, so they are
priced in the plan and not in ``dF/dN``.

What actually decides it: RESIDUAL, not FLOPs
---------------------------------------------
The saving is bounded (depth 2 gives 1.27x, the limit is ~1.7x) and it is paid
for in DISPATCHES: 7 matmuls + 13 adds + 4 relu + 4 bias-adds a layer instead
of 3 ops.  A flopscope dispatch costs ~26 us (elementwise) to ~106 us (matmul)
of BILLED residual on this box -- ``wall - backend - overhead``, charged at
1e11 FLOP/s -- so depth 1 adds ~26 ms = 2.6e9 effective FLOPs.  That is a FIXED
cost while the 1.137x is per sample, so the trade improves with ``N`` and the
gain tends to the full 1.137x in the large-``N`` limit.  ``--mode price``
measures both, at several ``N``, with a real ``BudgetContext``.

Depth 2 is measured and REJECTED here: 1.27x billed against +216 ms of
residual, which is a net 1.00x.

``--mode score`` runs the official 100-MLP suite so the accuracy claim is not
taken on faith: the answer moves by 3.0e-07 rms from Strassen's round-off, and
by 1.5e-04 rms from rounding the kept sets up to even (which Strassen needs to
split the contraction, and which is a strictly WEAKER approximation -- the
least-dead pruned neuron is put back).  Both are reported separately.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import (  # noqa: E402
    DEPTH,
    FLOP_BUDGET,
    LAMBDA_FLOPS_PER_SECOND,
    MULTIPLIER_FLOOR,
    WIDTH,
)
from whestfloor.official_seeds import (  # noqa: E402
    derive_estimator_seed,
    make_official_mlp,
)
from whestfloor.suite import Suite  # noqa: E402

VARIANTS = (
    ("ship: direct, mask as shipped", dict(strassen=False, even=False)),
    ("direct, kept sets rounded to even", dict(strassen=False, even=True)),
    ("STRASSEN depth 1 (+ even sets)", dict(strassen=True, even=True)),
)


def artifacts() -> Path:
    p = Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve() / "cost"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    return (Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()
            / "corrector")


# ---------------------------------------------------------------------------
def mode_price(n_list, tau, n_pilot, seed, coef) -> None:
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    from whestfloor import kernels  # noqa: PLC0415
    from whestfloor.mc import make_mlp  # noqa: PLC0415

    beta = np.load(Path(__file__).resolve().parent.parent / "submission"
                   / coef)["beta"]
    W = [fnp.asarray(w) for w in make_mlp(WIDTH, DEPTH, seed)]

    def run(N, kw):
        with flops.BudgetContext(flop_budget=int(1e13), quiet=True) as c:
            o = kernels.corrected_sparse_kernel(
                W, tau=tau, n_samples=N, n_pilot=n_pilot, seed=seed + 1,
                beta=beta, damp=1.0, kmax=2, safe=False, **kw)
        return (int(c.flops_used), float(c.residual_wall_time_s),
                float(c.wall_time_s), np.asarray(o)[-1])

    print(f"# billed price, one LOCAL MLP (seed {seed}), tau={tau}, "
          f"P={n_pilot}.  dF/dN from a 2-point difference.\n")
    hdr = (f"{'variant':<36}{'N':>7}{'F/B':>8}{'dF/dN':>11}{'x c':>7}"
           f"{'resid ms':>10}{'C/B':>8}{'x C':>7}{'wall ms':>9}")
    print(hdr)
    print("-" * len(hdr))
    ref, rows = {}, []
    for N in n_list:
        base = None
        for name, kw in VARIANTS:
            f1, r1, w1, o1 = run(N, kw)
            f2, _, _, _ = run(N + 2000, kw)
            c = (f2 - f1) / 2000.0
            C = f1 + LAMBDA_FLOPS_PER_SECOND * r1
            if base is None:
                base = (c, C)
                ref[N] = o1
            rows.append({"N": N, "variant": name, "F": f1, "c": c,
                         "residual_s": r1, "C": C, "wall_s": w1})
            print(f"{name:<36}{N:>7}{f1/FLOP_BUDGET:>8.4f}{c:>11,.0f}"
                  f"{base[0]/c:>7.4f}{r1*1e3:>10.1f}{C/FLOP_BUDGET:>8.4f}"
                  f"{base[1]/C:>7.4f}{w1*1e3:>9.0f}")
            if name != VARIANTS[0][0]:
                d = o1 - ref[N]
                print(f"{'':<36}{'':>7}   vs ship: rms |d mu| "
                      f"{np.sqrt(np.mean(d ** 2)):.3e}  max "
                      f"{np.abs(d).max():.3e}  (|mu| max "
                      f"{np.abs(ref[N]).max():.3f})")
        print()
    (artifacts() / "strassen_price.json").write_text(
        json.dumps({"tau": tau, "n_pilot": n_pilot, "rows": rows}, indent=1))
    print(f"wrote {artifacts() / 'strassen_price.json'}")


# ---------------------------------------------------------------------------
def mode_score(suite, n_mlps, n_samples, tau, n_pilot, coef) -> None:
    from whestfloor import kernels  # noqa: PLC0415
    from whestfloor.harness import run_billed  # noqa: PLC0415

    beta = np.load(data_dir() / coef)["beta"]
    gt = suite.gt[:, -1, :]
    print(f"# official suite, {n_mlps} MLPs, N=1e9 reference -> raw_mse is "
          f"leaderboard-comparable.  n_samples={n_samples}\n")
    hdr = (f"{'variant':<36}{'raw_mse':>12}{'F/B':>8}{'C/B':>8}{'adj@1x':>11}"
           f"{'adj@2x':>11}{'adj@3x':>11}{'raise':>6}")
    print(hdr)
    print("-" * len(hdr))
    out, preds, base = [], {}, None
    for name, kw in VARIANTS:
        mses, fls, rss, nfail, P = [], [], [], 0, []
        for i in range(n_mlps):
            Wn = make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])
            fn = functools.partial(
                kernels.corrected_sparse_kernel, tau=tau, n_samples=n_samples,
                n_pilot=n_pilot, seed=derive_estimator_seed(suite.mlp_seeds[i]),
                beta=beta, damp=1.0, kmax=2, **kw)
            try:
                pred, fl, rs = run_billed(fn, Wn)
            except Exception as e:  # noqa: BLE001
                print(f"  !! RAISED on mlp {i}: {type(e).__name__}: {e}")
                nfail += 1
                pred, fl, rs = np.zeros((DEPTH, WIDTH)), FLOP_BUDGET, 0.0
            P.append(pred[-1])
            mses.append(float(np.mean((pred[-1] - gt[i]) ** 2)))
            fls.append(fl)
            rss.append(rs)
        raw, F, R = float(np.mean(mses)), float(np.mean(fls)), float(np.mean(rss))
        adj = [raw * max(MULTIPLIER_FLOOR,
                         (F + k * LAMBDA_FLOPS_PER_SECOND * R) / FLOP_BUDGET)
               for k in (1, 2, 3)]
        C = F + LAMBDA_FLOPS_PER_SECOND * R
        print(f"{name:<36}{raw:>12.4e}{F/FLOP_BUDGET:>8.4f}{C/FLOP_BUDGET:>8.4f}"
              + "".join(f"{v:>11.4e}" for v in adj) + f"{nfail:>6d}")
        preds[name] = np.asarray(P)
        if base is None:
            base = adj[0]
        out.append({"variant": name, "raw_final_layer_mse": raw,
                    "flop_ratio": F / FLOP_BUDGET, "compute_ratio": C / FLOP_BUDGET,
                    "adj1": adj[0], "adj2": adj[1], "adj3": adj[2],
                    "raises": nfail, "gain_vs_ship": base / adj[0]})
    b = preds[VARIANTS[0][0]]
    print("\n# how far the answer moved, paired on the identical stream")
    for name, _ in VARIANTS[1:]:
        d = preds[name] - b
        print(f"  {name:<36} rms {np.sqrt(np.mean(d ** 2)):.3e}  "
              f"max {np.abs(d).max():.3e}")
    d = preds[VARIANTS[2][0]] - preds[VARIANTS[1][0]]
    print(f"  {'strassen alone (vs even-mask direct)':<36} "
          f"rms {np.sqrt(np.mean(d ** 2)):.3e}  max {np.abs(d).max():.3e}")
    print("\n# adjusted gain over the ship")
    for r in out:
        print(f"  {r['variant']:<36} {r['gain_vs_ship']:.4f}x")
    (artifacts() / "strassen_score.json").write_text(
        json.dumps({"n_mlps": n_mlps, "n_samples": n_samples, "rows": out},
                   indent=1))
    print(f"\nwrote {artifacts() / 'strassen_score.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("price", "score"))
    ap.add_argument("--n", default="8500,22000,45000")
    ap.add_argument("--n-samples", type=int, default=22000)
    ap.add_argument("--n-mlps", type=int, default=100)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--n-pilot", type=int, default=150)
    ap.add_argument("--seed", type=int, default=700_000)
    ap.add_argument("--coef", default="corrector.npz")
    ap.add_argument("--suite", default="")
    a = ap.parse_args()
    if a.mode == "price":
        mode_price([int(v) for v in a.n.split(",")], a.tau, a.n_pilot,
                   a.seed, a.coef)
    else:
        p = (Path(a.suite) if a.suite
             else Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()
             / "suites" / "official_mini.npz")
        mode_score(Suite.load(p), a.n_mlps, a.n_samples, a.tau, a.n_pilot,
                   a.coef)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
