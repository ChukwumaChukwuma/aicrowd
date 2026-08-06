#!/usr/bin/env python
"""End-to-end: the cumulant-corrected recursion for the whole 32-layer network,
measured against Monte-Carlo ground truth, plus the FLOP model.

LAYER UPDATE  (state carried between layers:  m, s, R, k3, k4, K21, K31, K22)
============================================================================
  m, s, R          mean / sd / correlation of the pre-activation z^l
  k3[i], k4[i]     marginal 3rd and 4th cumulants of z^l_i
  K21[i,k]         kappa(z_i, z_i, z_k)      (n x n)
  K31[i,k]         kappa(z_i, z_i, z_i, z_k)
  K22[i,k]         kappa(z_i, z_i, z_k, z_k)

  1  a_k, I_p, beta_{b,d}, single-site cumulants of x        O(K n)
  2  mu  = a_0 + Delta E[x]              (Edgeworth, sec 7)  O(n)
     var = relu_var + Delta Var(x)                            O(n)
  3  C   = Mehler(a, rhohat) + Delta Cov(x)                   O(K n^2)
  4  m'  = W^T mu                                             O(n^2)     [exact]
     S'  = W^T C W                                            2 matmuls  [exact]
  5  k3', k4' from the diagram catalogue                      ~6K matmuls
  6  K21', K31', K22' at coincident-block order               3 matmuls
  7  s' = sqrt(diag S'),  R' = S'/(s' s'^T)

Run:  python scripts/03_theory_scheme.py            # recursion vs MC, n=256
      python scripts/03_theory_scheme.py cost       # FLOP model
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/aicrowd")
_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(_here, fn))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


ords = _load("thord", "03_theory_orders.py")
selfc = _load("thself", "03_theory_selfconsistency.py")

from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import (  # noqa: E402
    hermite_prob, phi, relu_hermite_coeffs, relu_mean, relu_var,
)


# ---------------------------------------------------------------------------
def mixed_cumulants_leading(W, k3x, k4x):
    """kappa_{i^a k^b}(z') at coincident-block (rhohat^0) order.
    3 matmuls: (W^o2)^T diag(k3x) W, (W^o3)^T diag(k4x) W, (W^o2)^T diag(k4x) W^o2."""
    W2, W3 = W * W, W * W * W
    K21 = (W2 * k3x[:, None]).T @ W
    K31 = (W3 * k4x[:, None]).T @ W
    K22 = (W2 * k4x[:, None]).T @ W2
    return K21, K31, K22


def layer_step(m, s, R, W, k3, k4, K21, K31, K22, emax, kmax, use_nongauss,
               max_cycles=0):
    n = W.shape[0]
    rhohat = R - np.eye(n)
    a = relu_hermite_coeffs(m, s, kmax + 6)
    mu = a[0].copy()
    var = relu_var(m, s)
    C = np.zeros((n, n))
    pw = np.ones((n, n))
    for p in range(1, kmax + 1):
        pw = pw * rhohat
        C += (pw / math.factorial(p)) * np.outer(a[p], a[p])
    if use_nongauss:
        K3d = {"iii": k3, "iik": K21}
        K4d = {"iiii": k4, "iiik": K31, "iikk": K22}
        dmu, dvar, dcov = selfc.nongauss_corrections(m, s, rhohat, K3d, K4d, kmax)
        mu = mu + dmu
        var = var + dvar
        C = C + dcov
    np.fill_diagonal(C, var)

    mp = W.T @ mu
    Sp = W.T @ C @ W
    sp = np.sqrt(np.maximum(np.diag(Sp), 1e-14))
    Rp = Sp / np.outer(sp, sp)
    np.clip(Rp, -1.0, 1.0, out=Rp)

    per, ctx = ords.layer_cumulants(m, s, R, W, emax, max_cycles=max_cycles)
    k3p, k4p = ords.totals(per)
    K21p, K31p, K22p = mixed_cumulants_leading(W, ctx["k3"], ctx["k4"])
    return mp, sp, Rp, k3p, k4p, K21p, K31p, K22p, mu


def run(n=256, depth=32, seed=0, emax=None, kmax=14, use_nongauss=True,
        use_cumulants=True, max_cycles=0):
    emax = emax or {2: 8, 3: 4, 4: 3}
    W = [w.astype(np.float64) for w in make_mlp(n, depth, seed=seed)]
    m = np.zeros(n)
    S = W[0].T @ W[0]
    s = np.sqrt(np.diag(S))
    R = S / np.outer(s, s)
    z = np.zeros(n)
    k3 = k4 = z
    K21 = K31 = K22 = np.zeros((n, n))
    finals = None
    for li in range(depth):
        last = li == depth - 1
        if last:
            al = m / s
            out = relu_mean(m, s)
            if use_cumulants:
                out = out - (k3 / 6.0) * s ** -2 * hermite_prob(1, al) * phi(al) \
                    + (k4 / 24.0) * s ** -3 * hermite_prob(2, al) * phi(al)
            finals = out
            break
        m, s, R, k3n, k4n, K21, K31, K22, _ = layer_step(
            m, s, R, W[li + 1], k3, k4, K21, K31, K22, emax, kmax,
            use_nongauss and use_cumulants, max_cycles)
        if use_cumulants:
            k3, k4 = k3n, k4n
    return finals, k3, k4, s, m


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cost":
        ords.cost()
        sys.exit(0)
    d = np.load("/tmp/claude-0/-home-user-aicrowd/373914b3-787b-5580-866d-92a576ded1af"
                "/scratchpad/mccov256.npz")
    tgt = d["XS1"][31]
    N = int(d["N"])
    m1 = d["S1"]
    sdz = np.sqrt(d["S2"] - m1 ** 2)
    mu3 = d["S3"] - 3 * m1 * d["S2"] + 2 * m1 ** 3
    k4t = (d["S4"] - 4 * m1 * d["S3"] + 6 * m1 ** 2 * d["S2"] - 3 * m1 ** 4) \
        - 3 * (d["S2"] - m1 ** 2) ** 2
    print(f"reference: N={N:,} MC, se on the final mean ~ {math.sqrt(0.15/N):.1e}")
    for label, kw in (
        ("gaussian baseline (Mehler cov)   ", dict(use_cumulants=False, use_nongauss=False)),
        ("+ kappa_3,4 (marginal only)      ", dict(use_cumulants=True, use_nongauss=False)),
        ("+ kappa_3,4 + self-consistency   ", dict(use_cumulants=True, use_nongauss=True)),
    ):
        t0 = time.time()
        pred, k3, k4, s, m = run(**kw)
        e = pred - tgt
        print(f"  {label} final rms {np.sqrt((e**2).mean()):.4e}  "
              f"mse {np.mean(e**2):.3e}   [{time.time()-t0:.0f}s]")
        print(f"      layer-32  d(m) rms {np.sqrt(((m-m1[31])**2).mean()):.3e}   "
              f"d(sigma) rms {np.sqrt(((s-sdz[31])**2).mean()):.3e}   "
              f"d(k3) rms {np.sqrt(((k3-mu3[31])**2).mean()):.3e} "
              f"(|k3|={np.sqrt((mu3[31]**2).mean()):.3e})   "
              f"d(k4) rms {np.sqrt(((k4-k4t[31])**2).mean()):.3e} "
              f"(|k4|={np.sqrt((k4t[31]**2).mean()):.3e})")
