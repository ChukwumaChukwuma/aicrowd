#!/usr/bin/env python
"""ADVERSARIAL: is the Edgeworth gain a mechanism, or a fitted constant?

Two probes:

1. **damp sweep.**  ``cov_prop_edgeworth`` scales its correction by ``damp``.
   A term derived from first principles must be best at its derived
   coefficient, damp = 1.0.  If the optimum is far from 1.0 the sign/shape of
   the term may be right but its magnitude is being fitted, and the "analytic"
   claim is weaker than advertised.

2. **trivial-constant baseline.**  Decompose the mehler_k4 final-layer error
   into a coherent bias (mean over neurons) and a scatter.  If the error is
   bias-dominated, a *single fitted scalar* -- multiplicative or additive,
   fitted on the very suite it is scored on, i.e. cheating -- reproduces the
   gain and the "mechanism" is a constant.

Both are scored with the unbiased two-half MSE, which is the honest metric.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import kernels  # noqa: E402
from whestfloor.harness import run_billed  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402


def unbiased(pred_f, a_f, b_f):
    return float(np.mean((pred_f - a_f) * (pred_f - b_f)))


def collect(kernel, suite):
    """Final-layer predictions for every MLP in the suite."""
    out = []
    for i in range(suite.n_mlps):
        w = suite.weights(i)
        pred, _fl, _r = run_billed(kernel, w)
        out.append(pred[-1].copy())
        del w
    return np.asarray(out)


def per_mlp_T(P, suite):
    return np.array([unbiased(P[i], suite.gt_a[i][-1], suite.gt_b[i][-1])
                     for i in range(suite.n_mlps)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--umax", type=int, default=1)
    ap.add_argument("--damps", type=str,
                    default="0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0")
    args = ap.parse_args()

    s = Suite.load(args.suite)
    G = np.asarray([0.5 * (s.gt_a[i][-1] + s.gt_b[i][-1])
                    for i in range(s.n_mlps)])

    print(f"# suite={s.name} n_mlps={s.n_mlps} umax={args.umax} "
          f"n_per_half={s.n_per_half:,}")
    print()

    # ---------------- 1. damp sweep -------------------------------------
    damps = sorted({float(x) for x in args.damps.split(",")} | {0.0, 1.0})
    print("damp     unbiased_true_mse    per-MLP min .. max        vs damp=1")
    print("-" * 74)
    results = {}
    for d in damps:
        k = functools.partial(kernels.cov_prop_edgeworth, kmax=4,
                              umax=args.umax, damp=d)
        P = collect(k, s)
        T = per_mlp_T(P, s)
        results[d] = (float(T.mean()), T, P)
    base1 = results[1.0][0]
    best_d, best_v = min(((d, v[0]) for d, v in results.items()),
                         key=lambda t: t[1])
    for d in damps:
        m, T, _ = results[d]
        print(f"{d:5.2f}   {m:.6e}      {T.min():.3e} .. {T.max():.3e}   "
              f"{m / base1:6.3f}x")
    print()
    print(f"OPTIMUM damp = {best_d}  ({best_v:.6e});  damp=1.0 gives "
          f"{base1:.6e};  ratio {base1 / best_v:.4f}")

    # parabola through the three points around the optimum, for a continuous
    # argmin -- the grid is coarse and the true optimum lies between points.
    ds = np.array(damps)
    vs = np.array([results[d][0] for d in damps])
    c = np.polyfit(ds, vs, 2)
    if c[0] > 0:
        print(f"quadratic fit over the whole sweep: argmin damp = "
              f"{-c[1] / (2 * c[0]):.4f}")

    # ---------------- 2. bias vs scatter --------------------------------
    print()
    print("Error decomposition of the FINAL layer (per MLP, vs the "
          "two-half mean):")
    print("kernel        bias(mean err)     scatter(sd err)    "
          "bias^2/MSE")
    print("-" * 70)
    P0 = results[0.0][2]
    P1 = results[1.0][2]
    for nm, P in (("mehler(d=0)", P0), ("edgeworth", P1)):
        err = P - G
        bias = err.mean(axis=1)
        sc = err.std(axis=1)
        mse = (err ** 2).mean(axis=1)
        frac = (bias ** 2) / np.maximum(mse, 1e-30)
        print(f"{nm:12s}  {bias.mean():+.4e}      {sc.mean():.4e}     "
              f"{frac.mean():.3f}   (per-MLP bias "
              f"{bias.min():+.2e} .. {bias.max():+.2e})")

    # ---------------- 3. fitted single scalar ---------------------------
    print()
    print("Trivial baselines on top of mehler_k4 (damp=0), fitted ON THE "
          "SCORED SUITE (cheating upper bound):")
    T0 = per_mlp_T(P0, s)
    print(f"  mehler_k4 alone                       {T0.mean():.6e}")

    # global additive: p + k, single k over the whole suite
    k_add = float(G.mean() - P0.mean())
    Tadd = per_mlp_T(P0 + k_add, s)
    print(f"  + global additive k={k_add:+.5f}         {Tadd.mean():.6e}"
          f"   ({T0.mean() / Tadd.mean():.3f}x)")

    # global multiplicative: c*p
    c_mul = float((P0 * G).sum() / (P0 * P0).sum())
    Tmul = per_mlp_T(c_mul * P0, s)
    print(f"  x global multiplicative c={c_mul:.5f}     {Tmul.mean():.6e}"
          f"   ({T0.mean() / Tmul.mean():.3f}x)")

    # global affine: a*p + b
    A = np.stack([P0.ravel(), np.ones(P0.size)], axis=1)
    coef, *_ = np.linalg.lstsq(A, G.ravel(), rcond=None)
    Taff = per_mlp_T(coef[0] * P0 + coef[1], s)
    print(f"  affine a={coef[0]:.5f} b={coef[1]:+.5f}      "
          f"{Taff.mean():.6e}   ({T0.mean() / Taff.mean():.3f}x)")

    # per-MLP oracle affine (strictly stronger than any fixed constant)
    Tpm = []
    for i in range(s.n_mlps):
        Ai = np.stack([P0[i], np.ones(P0.shape[1])], axis=1)
        ci, *_ = np.linalg.lstsq(Ai, G[i], rcond=None)
        Tpm.append(unbiased(ci[0] * P0[i] + ci[1], s.gt_a[i][-1],
                            s.gt_b[i][-1]))
    Tpm = np.array(Tpm)
    print(f"  per-MLP ORACLE affine                 {Tpm.mean():.6e}"
          f"   ({T0.mean() / Tpm.mean():.3f}x)")
    print()
    print(f"  edgeworth damp=1 (the claim)          {base1:.6e}"
          f"   ({T0.mean() / base1:.3f}x)")
    print()
    print("VERDICT INPUT: if the fitted-scalar rows reach the edgeworth row, "
          "the mechanism is a constant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
