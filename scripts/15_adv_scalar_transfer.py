#!/usr/bin/env python
"""ADVERSARIAL: can a single hard-coded scalar replace the "mechanism"?

The Edgeworth term subtracts ``(kappa_3/6)(m/s^3) phi(a)``, which is
predominantly positive-valued and therefore mostly *shrinks* an over-estimated
mean.  If the error covariance-propagation makes is dominated by a coherent
multiplicative offset, then one number -- fitted once, offline, hard-coded --
does the same job, and no cumulant expansion is needed.

Design so this cannot be dismissed as in-sample fitting:

  * the scalar is fitted on suite A and scored on suite B, and vice versa;
  * two shapes of scalar are tried: a final-layer output scale ``c`` (trivially
    shippable: one multiply) and a per-layer mean shrink ``g`` applied inside
    the propagation (also one number, but it compounds through the depth);
  * the correction is then stacked ON TOP of the fitted scalar.  If
    ``edgeworth + c`` is no better than ``mehler + c``, the correction and the
    constant are doing the same job and the mechanism claim is empty.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

from whestfloor import kernels  # noqa: E402
from whestfloor.harness import run_billed  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402


def mehler_shrink(weights, ctx=None, kmax: int = 4, g: float = 1.0):
    """mehler_k4 with a single scalar shrink ``g`` on the rectified mean.

    One fitted number, applied identically at every layer, compounding through
    the depth exactly the way the Edgeworth correction's feedback does.
    """
    n = weights[0].shape[0]
    mu = fnp.zeros(n, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))
    rows = []
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sig = fnp.sqrt(var_pre)
        mu, var_post, alpha, ph, Ph = kernels._relu_gauss(mu_pre, var_pre, sig)
        mu = mu * g
        inv_sig = 1.0 / sig
        rho = cov_pre * fnp.outer(inv_sig, inv_sig)
        rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)
        a = kernels._hermite_coeffs(alpha, sig, ph, Ph, kmax)
        rho_k = rho
        acc = fnp.outer(a[1], a[1]) * rho
        fact = 1.0
        for k in range(2, kmax + 1):
            rho_k = rho_k * rho
            fact *= k
            acc = acc + fnp.outer(a[k], a[k]) * (rho_k * (1.0 / fact))
        cov = acc
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        rows.append(mu)
    return fnp.stack(rows, axis=0)


def collect(kernel, s):
    out = []
    for i in range(s.n_mlps):
        w = s.weights(i)
        pred, _f, _r = run_billed(kernel, w)
        out.append(pred[-1].copy())
        del w
    return np.asarray(out)


def T_of(P, s):
    return float(np.mean([np.mean((P[i] - s.gt_a[i][-1]) *
                                  (P[i] - s.gt_b[i][-1]))
                          for i in range(s.n_mlps)]))


def fit_c(P, s):
    """LS output scale minimising the unbiased MSE (== LS against (a+b)/2)."""
    G = np.asarray([0.5 * (s.gt_a[i][-1] + s.gt_b[i][-1])
                    for i in range(s.n_mlps)])
    return float((P * G).sum() / (P * P).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-a", required=True)
    ap.add_argument("--suite-b", required=True)
    ap.add_argument("--shrink-grid", type=str,
                    default="1.0,0.9999,0.99985,0.9998,0.99977,0.99974,"
                            "0.9997,0.9996")
    ap.add_argument("--only-shrink", action="store_true")
    args = ap.parse_args()

    S = {"A": Suite.load(args.suite_a), "B": Suite.load(args.suite_b)}
    for k, s in S.items():
        print(f"# suite {k}: {s.name}  seeds {s.mlp_seeds[0]}.."
              f"{s.mlp_seeds[-1]}  n_half {s.n_per_half:,}")
    print()

    meh = functools.partial(kernels.cov_prop_mehler, kmax=4)
    edg = functools.partial(kernels.cov_prop_edgeworth, kmax=4, umax=1,
                            damp=1.0)
    gain = kernels.cov_prop_gain

    P = {}
    for k, s in S.items():
        P[("gain", k)] = collect(gain, s)
        P[("mehler", k)] = collect(meh, s)
        P[("edge", k)] = collect(edg, s)
    tedgA = T_of(P[("edge", "A")], S["A"])
    tedgB = T_of(P[("edge", "B")], S["B"])
    if args.only_shrink:
        print(f"edgeworth reference: A {tedgA:.4e}   B {tedgB:.4e}\n")
        shrink_table(S, [float(x) for x in args.shrink_grid.split(",")],
                     tedgA, tedgB)
        return 0

    # ---- fitted output scale, cross-validated -------------------------
    print("Output scale c fitted on one suite, SCORED ON THE OTHER "
          "(out-of-sample):")
    print("base       c(fit A)  c(fit B)    T(A) base     T(A)+c_B      "
          "T(B) base     T(B)+c_A")
    print("-" * 96)
    cs = {}
    for base in ("gain", "mehler", "edge"):
        cA = fit_c(P[(base, "A")], S["A"])
        cB = fit_c(P[(base, "B")], S["B"])
        cs[base] = (cA, cB)
        tA = T_of(P[(base, "A")], S["A"])
        tAx = T_of(cB * P[(base, "A")], S["A"])   # scale from B -> scored A
        tB = T_of(P[(base, "B")], S["B"])
        tBx = T_of(cA * P[(base, "B")], S["B"])   # scale from A -> scored B
        print(f"{base:9s}  {cA:.5f}   {cB:.5f}   {tA:.4e}    {tAx:.4e}    "
              f"{tB:.4e}    {tBx:.4e}")
    print()

    tmehA, tmehB = T_of(P[("mehler", "A")], S["A"]), T_of(P[("mehler", "B")], S["B"])
    tedgA, tedgB = T_of(P[("edge", "A")], S["A"]), T_of(P[("edge", "B")], S["B"])
    cmA, cmB = cs["mehler"]
    tmehA_x = T_of(cmB * P[("mehler", "A")], S["A"])
    tmehB_x = T_of(cmA * P[("mehler", "B")], S["B"])
    ceA, ceB = cs["edge"]
    tedgA_x = T_of(ceB * P[("edge", "A")], S["A"])
    tedgB_x = T_of(ceA * P[("edge", "B")], S["B"])

    print("HEAD TO HEAD (all out-of-sample where a constant is involved):")
    print(f"  suite A: mehler {tmehA:.4e} | mehler*c_B {tmehA_x:.4e} | "
          f"edgeworth {tedgA:.4e} | edgeworth*c_B {tedgA_x:.4e}")
    print(f"  suite B: mehler {tmehB:.4e} | mehler*c_A {tmehB_x:.4e} | "
          f"edgeworth {tedgB:.4e} | edgeworth*c_A {tedgB_x:.4e}")
    print()
    print(f"  one hard-coded scalar beats the cumulant expansion on A? "
          f"{tmehA_x < tedgA}   (ratio {tedgA / tmehA_x:.3f})")
    print(f"  one hard-coded scalar beats the cumulant expansion on B? "
          f"{tmehB_x < tedgB}   (ratio {tedgB / tmehB_x:.3f})")
    print(f"  does the correction still add on top of the scalar?  "
          f"A: {tmehA_x / tedgA_x:.3f}x   B: {tmehB_x / tedgB_x:.3f}x")
    print()

    shrink_table(S, [float(x) for x in args.shrink_grid.split(",")],
                 tedgA, tedgB)
    return 0


def shrink_table(S, grid, tedgA, tedgB):
    # ---- per-layer shrink, cross-validated ----------------------------
    print("Per-layer mean shrink g (one number, applied at every layer):")
    tab = {}
    for k, s in S.items():
        row = []
        for g in grid:
            row.append(T_of(collect(functools.partial(mehler_shrink, kmax=4,
                                                      g=g), s), s))
        tab[k] = np.array(row)
    print("     g        T(A)         T(B)")
    for i, g in enumerate(grid):
        print(f"  {g:.6f}  {tab['A'][i]:.4e}   {tab['B'][i]:.4e}")
    gA = grid[int(np.argmin(tab["A"]))]
    gB = grid[int(np.argmin(tab["B"]))]
    print(f"  argmin on A: g={gA}   argmin on B: g={gB}")
    print(f"  OUT-OF-SAMPLE: g fitted on A ({gA}) scored on B -> "
          f"{tab['B'][grid.index(gA)]:.4e}  vs edgeworth on B {tedgB:.4e}")
    print(f"  OUT-OF-SAMPLE: g fitted on B ({gB}) scored on A -> "
          f"{tab['A'][grid.index(gB)]:.4e}  vs edgeworth on A {tedgA:.4e}")


if __name__ == "__main__":
    raise SystemExit(main())
