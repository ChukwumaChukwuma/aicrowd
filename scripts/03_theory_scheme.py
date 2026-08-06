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

THE TERM THAT IS MISSING FROM STEP 5, AND WHY IT MATTERS
=======================================================
The diagram catalogue computes the cumulants of z^{l+1} *assuming z^l is
Gaussian*.  That is only the SOURCE term.  Applying the same first-order
Edgeworth operator of sec 7 to the joint cumulants of x themselves gives, for
distinct neurons,

    delta kappa(x_i1,x_i2,x_i3) = kappa^{(z)}_{i1 i2 i3} Phi_i1 Phi_i2 Phi_i3
                                  + O(rhohat)

(the three derivatives land on three different factors and relu' = 1{.}), so

    kappa^(3)(z^{l+1}) = S^(3)   +   (Wt^T)^{ox3} kappa^(3)(z^l),    Wt = diag(Phi) W
    kappa^(4)(z^{l+1}) = S^(4)   +   (Wt^T)^{ox4} kappa^(4)(z^l)

-- a linear TRANSPORT plus a source.  It is obvious in hindsight: a neuron with
|alpha| >> 0 is an affine map, and affine maps carry cumulants through exactly.

Consequence: **the recursion does not close on the marginals.**  Transport mixes
off-diagonal entries of the 3-index tensor into the diagonal, so kappa_3(z_j)
alone is not enough state.  Dropping transport is what makes the source-only
scheme undershoot the true skewness by ~7x at depth 32 (measured below).

Run:  python scripts/03_theory_scheme.py            # recursion vs MC, n=256
      python scripts/03_theory_scheme.py transport  # the transport deficit
      python scripts/03_theory_scheme.py oracle     # ceiling with exact cumulants
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
#: Monte-Carlo reference (per-neuron pre-activation moments 1..4 for every
#: layer, plus E[relu(z^32)]).  Cached because it costs a few minutes; the
#: cache path is a scratch file, so regenerate it if it is not there.
MC_CACHE = os.environ.get(
    "WHEST_MC_CACHE",
    "/tmp/claude-0/-home-user-aicrowd/373914b3-787b-5580-866d-92a576ded1af"
    "/scratchpad/mccov256.npz")


def mc_reference(n=256, depth=32, seed=0, nsamp=2_000_000, chunk=1024):
    """Load the cached reference, or build it (a few minutes at n=256)."""
    if os.path.exists(MC_CACHE):
        return np.load(MC_CACHE)
    print(f"building the Monte-Carlo reference ({nsamp:,} samples) -> {MC_CACHE}",
          flush=True)
    W = make_mlp(n, depth, seed=seed)
    rng = np.random.default_rng(999 + seed)
    S = [np.zeros((depth, n)) for _ in range(4)]
    XS1 = np.zeros((depth, n))
    done = 0
    while done < nsamp:
        nb = min(chunk, nsamp - done)
        x = rng.standard_normal((nb, n), dtype=np.float32)
        for li in range(depth):
            z = x @ W[li]
            zz = z.astype(np.float64)
            S[0][li] += zz.sum(0)
            z2 = zz * zz
            S[1][li] += z2.sum(0)
            S[2][li] += (z2 * zz).sum(0)
            S[3][li] += (z2 * z2).sum(0)
            x = np.maximum(z, np.float32(0.0))
            XS1[li] += x.sum(0, dtype=np.float64)
        done += nb
    np.savez(MC_CACHE, n=n, N=done, seed=seed, S1=S[0] / done, S2=S[1] / done,
             S3=S[2] / done, S4=S[3] / done, XS1=XS1 / done)
    return np.load(MC_CACHE)


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


def transport_demo(n=256, depth=32, seed=0):
    """Carry kappa^(3), kappa^(4) in the CP form sum_v c_v G_v^{ox r} with the
    COINCIDENT-BLOCK source only, transported by Wt = diag(Phi) W, and compare
    with the truth.  Shows both that transport is real and that a
    diagonal-only source does not capture it."""
    from whestfloor.relu_moments import Phi, relu_cov_mehler
    W = [w.astype(np.float64) for w in make_mlp(n, depth, seed=seed)]
    d = mc_reference()
    m1 = d["S1"]; sdt = np.sqrt(d["S2"] - m1 ** 2)
    k3t = d["S3"] - 3 * m1 * d["S2"] + 2 * m1 ** 3
    mu, C = np.zeros(n), np.eye(n)
    G, c = np.zeros((0, n)), np.zeros(0)
    print(f"{'layer':>5s} {'g3 source+transport':>20s} {'g3 true':>9s}  rank")
    for li in range(depth):
        w = W[li]
        m = w.T @ mu
        S = w.T @ C @ w
        s = np.sqrt(np.maximum(np.diag(S), 1e-12))
        k3 = (c[:, None] * G ** 3).sum(0) if len(c) else np.zeros(n)
        if li in (3, 7, 15, 23, 31):
            print(f"{li+1:5d} {np.sqrt(((k3/s**3)**2).mean()):20.4f} "
                  f"{np.sqrt(((k3t[li]/sdt[li]**3)**2).mean()):9.4f}  {len(c)}")
        al = m / s
        Ph = Phi(al)
        I = [Ph, al * Ph + phi(al)]
        for p in range(2, 4):
            I.append(al * I[p - 1] + (p - 1) * I[p - 2])
        A = [s ** p * I[p] for p in range(4)]
        k3x = A[3] - 3 * A[1] * A[2] + 2 * A[1] ** 3
        if li + 1 < depth:
            Wt = Ph[:, None] * W[li + 1]
            G = np.vstack([G @ Wt, W[li + 1]]) if len(c) else W[li + 1].copy()
            c = np.concatenate([c, k3x])
        mu, C = A[1], relu_cov_mehler(S, m, s, kmax=16)


def oracle(n=256, depth=32, seed=0):
    """Upper bound on what the first-order scheme can buy: inject the EXACT
    marginal cumulants of every layer (from Monte Carlo) and keep the mixed
    cumulants at coincident-block order."""
    d = mc_reference()
    m1 = d["S1"]
    K3T = d["S3"] - 3 * m1 * d["S2"] + 2 * m1 ** 3
    K4T = (d["S4"] - 4 * m1 * d["S3"] + 6 * m1 ** 2 * d["S2"] - 3 * m1 ** 4) \
        - 3 * (d["S2"] - m1 ** 2) ** 2
    tgt = d["XS1"][31]
    W = [w.astype(np.float64) for w in make_mlp(n, depth, seed=seed)]
    emax = {2: 8, 3: 4, 4: 3}
    for nm, inject, ng in (("gaussian                   ", False, False),
                           ("exact marginal k3,k4 at all", True, True)):
        m = np.zeros(n)
        S = W[0].T @ W[0]
        s = np.sqrt(np.diag(S))
        R = S / np.outer(s, s)
        K21 = K31 = K22 = np.zeros((n, n))
        for li in range(depth):
            k3 = K3T[li] if inject else np.zeros(n)
            k4 = K4T[li] if inject else np.zeros(n)
            if li == depth - 1:
                al = m / s
                p = relu_mean(m, s)
                if inject:
                    p = p - (k3 / 6.0) * s ** -2 * hermite_prob(1, al) * phi(al) \
                        + (k4 / 24.0) * s ** -3 * hermite_prob(2, al) * phi(al)
                e = p - tgt
                print(f"  {nm}  final rms {np.sqrt((e**2).mean()):.4e}  "
                      f"mse {np.mean(e**2):.3e}")
                break
            m, s, R, _, _, K21, K31, K22, _ = layer_step(
                m, s, R, W[li + 1], k3, k4, K21, K31, K22, emax, 14, ng, 0)
    print("  (compare: exact mu and C at every layer would leave only the final")
    print("   rectification error, 1.5e-4 rms -- see docs/cumulant_expansion.md sec 9)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cost":
        ords.cost()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "transport":
        transport_demo()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "oracle":
        oracle()
        sys.exit(0)
    d = mc_reference()
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
