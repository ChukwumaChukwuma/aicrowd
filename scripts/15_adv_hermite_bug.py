#!/usr/bin/env python
"""ADVERSARIAL: the shipped ``_hermite_coeffs`` is wrong from a_4 up.

``submission/estimator.py`` (the frozen graded artefact) builds the ReLU
Hermite coefficients with

    a_k = (-1)^k s He_{k-2}(alpha) phi(alpha),   k >= 2

by the recurrence ``He_{j+1} = alpha He_j - j He_{j-1}``, representing
``He_0 == 1`` by the sentinel ``None``.  The recurrence then reads

    hj = base if h_prev is None else base - (j-1) * h_prev

but ``h_prev is None`` means ``He_{j-1} == 1``, NOT ``He_{j-1} == 0``.  At
``j = 2`` that drops the ``-1``:

    He_2 computed as alpha^2      (true value alpha^2 - 1)
    He_3 computed as alpha^3-2a   (true value alpha^3 - 3 alpha)

so ``a_4`` and every higher coefficient are wrong.  ``KMAX = 4`` means the
shipped estimator *uses* the wrong ``a_4``: the "exact post-ReLU covariance via
Mehler" is not exact at the one order beyond k=2 that it pays for, and the
"Mehler saturates at k=4 / k=8 / k=16" measurements in the docstring were taken
with corrupted coefficients at every one of those orders.

This script:
  1. checks the shipped coefficients against Gauss-Hermite quadrature;
  2. rebuilds ``cov_prop_mehler`` / ``cov_prop_edgeworth`` with correct
     coefficients and re-scores, so the size of the error is a number.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

from whestfloor.harness import run_billed  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402


def load_shipped():
    if "whestbench" not in sys.modules:
        wb = types.ModuleType("whestbench")
        wb.BaseEstimator = type("BaseEstimator", (), {})
        sys.modules["whestbench"] = wb
    spec = importlib.util.spec_from_file_location(
        "shipped_estimator", ROOT / "submission" / "estimator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hermite_fixed(alpha, sig, ph, Ph, kmax):
    """Correct ``a_k``: He_{j} = alpha He_{j-1} - (j-1) He_{j-2}, He_0 = 1."""
    out = [None, sig * Ph]
    if kmax >= 2:
        s_phi = sig * ph
        h_pp = None    # He_{j-2}, None == the constant 1
        h_p = None     # He_{j-1}
        for k in range(2, kmax + 1):
            j = k - 2
            if j == 0:
                hj = None
                h_pp, h_p = None, None
            elif j == 1:
                hj = alpha
                h_pp, h_p = None, hj          # He_0 == 1 (implicit), He_1
            else:
                base = alpha * h_p
                hj = (base - float(j - 1) if h_pp is None
                      else base - float(j - 1) * h_pp)
                h_pp, h_p = h_p, hj
            term = s_phi if hj is None else s_phi * hj
            out.append(term if k % 2 == 0 else -term)
    return out


def quadrature_a(m, s, kmax, half=14.0, n=2_000_001):
    """a_k = E[relu(m + s t) He_k(t)] by direct Simpson integration.

    Independent of any closed form: only relu, He_k (numpy's own recurrence)
    and the standard normal density are used.  The integrand decays like
    exp(-t^2/2) times a polynomial, so a +-14 sigma window at 1.4e-5 spacing is
    exact to well past float64 noise.
    """
    t = np.linspace(-half, half, n)
    dt = t[1] - t[0]
    phi = np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)
    wts = np.ones(n)
    wts[1:-1:2] = 4.0
    wts[2:-1:2] = 2.0
    wts *= dt / 3.0
    f = np.maximum(m[:, None] + s[:, None] * t[None, :], 0.0)
    out = [None]
    for k in range(1, kmax + 1):
        c = np.zeros(k + 1)
        c[k] = 1.0
        He = np.polynomial.hermite_e.hermeval(t, c)
        out.append((f * (wts * phi * He)[None, :]).sum(axis=1))
    return out


def make_kernel(hermite, kmax=4, umax=1, damp=1.0, edge=True):
    from whestfloor import kernels as K

    def k(weights, ctx=None):
        n = weights[0].shape[0]
        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32),
                                 symmetry=(0, 1))
        prev = None
        rows = []
        for w in weights:
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sig = fnp.sqrt(var_pre)
            alpha = mu_pre / sig
            ph = flops.stats.norm.pdf(alpha)
            Ph = flops.stats.norm.cdf(alpha)
            mu = mu_pre * Ph + sig * ph
            ez2 = (mu_pre * mu_pre + var_pre) * Ph + mu_pre * sig * ph
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)
            if edge and prev is not None:
                k3 = K._kappa3_star(w, prev[0], prev[1], umax)
                mu = mu - (damp / 6.0) * k3 * (mu_pre / (var_pre * sig)) * ph
            inv_sig = 1.0 / sig
            rho = cov_pre * fnp.outer(inv_sig, inv_sig)
            rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)
            a = hermite(alpha, sig, ph, Ph, max(kmax, 2 * umax))
            acc = fnp.outer(a[1], a[1]) * rho
            rho_k = rho
            fact = 1.0
            for kk in range(2, kmax + 1):
                rho_k = rho_k * rho
                fact *= kk
                acc = acc + fnp.outer(a[kk], a[kk]) * (rho_k * (1.0 / fact))
            cov = acc
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))
            prev = (a, rho)
            rows.append(mu)
        return fnp.stack(rows, axis=0)
    return k


def T_of(kernel, s):
    t = []
    for i in range(s.n_mlps):
        w = s.weights(i)
        p, _f, _r = run_billed(kernel, w)
        t.append(float(np.mean((p[-1] - s.gt_a[i][-1]) *
                               (p[-1] - s.gt_b[i][-1]))))
        del w
    return float(np.mean(t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*", default=[])
    ap.add_argument("--kmax", type=str, default="2,4,6,8")
    args = ap.parse_args()

    sub = load_shipped()
    rng = np.random.default_rng(0)
    m = rng.standard_normal(6) * 0.5
    s = np.abs(rng.standard_normal(6)) + 0.3
    KM = 8
    ref = quadrature_a(m, s, KM)
    alpha = m / s
    ph = np.exp(-0.5 * alpha ** 2) / np.sqrt(2 * np.pi)
    import math
    Ph = 0.5 * (1 + np.vectorize(math.erf)(alpha / np.sqrt(2)))
    got = sub._hermite_coeffs(alpha, s, ph, Ph, KM)
    fix = hermite_fixed(alpha, s, ph, Ph, KM)

    print("Hermite coefficients a_k vs Gauss-Hermite quadrature "
          "(max relative error over 6 random neurons):")
    print("  k   shipped submission/estimator.py     corrected recurrence")
    for k in range(1, KM + 1):
        e1 = np.max(np.abs(np.asarray(got[k]) - ref[k]) /
                    np.maximum(np.abs(ref[k]), 1e-30))
        e2 = np.max(np.abs(np.asarray(fix[k]) - ref[k]) /
                    np.maximum(np.abs(ref[k]), 1e-30))
        flag = "   <-- WRONG" if e1 > 1e-6 else ""
        print(f"  {k}   {e1:.3e}                          {e2:.3e}{flag}")
    print()
    print("  He_2 shipped vs true (alpha^2 vs alpha^2-1):")
    print(f"    alpha        = {np.array2string(alpha, precision=4)}")
    print(f"    a_4 shipped  = {np.array2string(np.asarray(got[4]), precision=6)}")
    print(f"    a_4 exact    = {np.array2string(ref[4], precision=6)}")
    print()

    for path in args.suites:
        su = Suite.load(path)
        print(f"suite {su.name}: effect of fixing the coefficients "
              f"(unbiased true MSE)")
        print("  kmax   mehler shipped   mehler fixed    edgeworth shipped"
              "   edgeworth fixed")
        for kmax in [int(x) for x in args.kmax.split(",")]:
            a = T_of(make_kernel(sub._hermite_coeffs, kmax=kmax, edge=False), su)
            b = T_of(make_kernel(hermite_fixed, kmax=kmax, edge=False), su)
            c = T_of(make_kernel(sub._hermite_coeffs, kmax=kmax, umax=1,
                                 edge=True), su)
            d = T_of(make_kernel(hermite_fixed, kmax=kmax, umax=1,
                                 edge=True), su)
            print(f"  {kmax:4d}   {a:.6e}    {b:.6e}    {c:.6e}"
                  f"    {d:.6e}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
