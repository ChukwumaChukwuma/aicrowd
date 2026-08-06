#!/usr/bin/env python
"""Randomised quasi-Monte Carlo for the final-layer sampler: measure, then build.

The order of the modes is the order the decision was taken, and the first one
is meant to make the rest unnecessary if it comes out small.

``--mode anova``
    **The predictor.**  RQMC only beats plain Monte Carlo on the part of the
    integrand's variance that lives in LOW-ORDER ANOVA terms of the input
    coordinates, because that is the only part a lattice's low-dimensional
    projections integrate better than random points.  So measure the ANOVA
    order distribution of ``x -> relu(z^32_j(x))`` directly, three ways that
    check each other:

      * ``f_1``, the first-order (best additive approximation) share, from
        Hermite projections ``C_{mjk} = E[f_m He_k(x_j)]`` with a split-half
        product so the estimate of ``C^2`` is unbiased rather than inflated by
        its own sampling variance;
      * the **mean dimension** ``d_M = sum_j T_j`` (sum of total Sobol'
        indices) from the Jansen one-coordinate-resample estimator, which is
        the moment ``sum_d d f_d`` of the same order distribution;
      * the whole **order-generating function** ``g(p) = sum_d f_d p^d``, from
        pick-and-freeze on Bernoulli(p) random coordinate subsets -- because
        ``E_A[Var(E[f|x_A])] = sum_B V_B p^{|B|}``, one estimator gives every
        moment at once.  ``g`` anchors ``f_2`` and cross-checks both of the
        above (``g'(1) = d_M``).

``--mode lattice``
    Quality control on the point sets themselves, with no network involved:
    projection uniformity and the exact FLOP bill of a randomly-shifted rank-1
    lattice draw against ``rng.standard_normal``.

``--mode probe``
    Head-to-head RQMC vs iid at MATCHED EFFECTIVE COMPUTE (matched C/B, not
    matched N) on real MLPs, repeated over independent randomisations so the
    ratio has an error bar.  Includes the dimension-ordering variants.

``--mode score``
    End-to-end on the official 100-MLP suite through the shipped sparse
    kernel, with the pseudorandom draw as the ablation through the identical
    code path.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.official_seeds import make_official_mlp  # noqa: E402
from whestfloor.suite import Suite  # noqa: E402

WIDTH, DEPTH = 256, 32


def artifacts() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


def load_suite(path: str | None) -> Suite:
    p = Path(path) if path else artifacts() / "suites" / "official_mini.npz"
    return Suite.load(p)


def forward(W, x):
    """Final-layer activations ``h^32`` for a batch of inputs."""
    for w in W:
        x = np.maximum(x @ w, 0.0)
    return x


# ---------------------------------------------------------------------------
# mode: anova
# ---------------------------------------------------------------------------
def _hermite_stack(x, kmax):
    """``He_1..He_kmax`` evaluated at ``x``, float64, as a list of (B, n)."""
    xd = x.astype(np.float64)
    out = [xd]
    hm1 = np.ones_like(xd)
    for k in range(1, kmax):
        nxt = xd * out[-1] - k * hm1
        hm1 = out[-1]
        out.append(nxt)
    return out


PS_DEFAULT = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375)

#: Ornstein-Uhlenbeck correlations for the CHAOS (total Hermite degree)
#: decomposition.  ``rho = -1`` is the antithetic pair and needs no separate
#: experiment: ``G(-1) = corr(f(x), f(-x))`` fixes the odd/even split exactly.
RHO_DEFAULT = (-1.0, -0.5, 0.25, 0.5, 0.75, 0.9)


def anova_probe(fn, dim, n_samples, kmax, chunk, rng, ps=PS_DEFAULT):
    """ANOVA order structure of a vector-valued ``fn(x) -> (B, m)``.

    Returns a dict with the aggregated variance ``V = sum_m Var(f_m)``, the
    first-order share ``f1``, the mean dimension ``dM``, and the
    order-generating function ``g(p)``.  Three independent estimators of the
    same object, so they check each other.
    """
    fact = np.array([math.factorial(k) for k in range(1, kmax + 1)], float)
    nchunk = max(1, n_samples // chunk)
    # FOUR accumulators, not two: quarters (0,1) and (2,3) each give an
    # independent split-half estimate, so the per-coordinate profile has a
    # measurable reliability instead of an assumed one.
    Q = [None] * 4
    nQ = [0] * 4
    s1 = s2 = None
    nf = 0
    pf_num = {p: [] for p in ps}
    ch_num = {r: [] for r in RHO_DEFAULT}
    jansen = []
    for c in range(nchunk):
        x = rng.standard_normal((chunk, dim), dtype=np.float32)
        xp = rng.standard_normal((chunk, dim), dtype=np.float32)
        fx = np.asarray(fn(x), dtype=np.float64)
        fxp = np.asarray(fn(xp), dtype=np.float64)
        if Q[0] is None:
            m = fx.shape[1]
            for q in range(4):
                Q[q] = np.zeros((kmax, dim, m))  # [order k, coord j, output m]
            s1 = np.zeros(m)
            s2 = np.zeros(m)
        s1 += fx.sum(0)
        s2 += (fx * fx).sum(0)
        nf += chunk

        # C_{kjm} = E[f_m He_k(x_j)]; a product of two DISJOINT halves is an
        # unbiased estimate of C^2 (squaring one estimate would inflate it by
        # its own variance -- here that bias is ~3% of V1, small but real).
        He = _hermite_stack(x, kmax)
        q = c % 4
        for k in range(kmax):
            Q[q][k] += He[k].T @ fx
        nQ[q] += chunk

        # pick-and-freeze on Bernoulli(p) subsets:
        # E[f(x)(f(y)-f(x'))] = sum_B V_B p^{|B|} = g(p) V, A redrawn per sample
        dd = float((fx * fxp).sum())
        for p in ps:
            msk = rng.random((chunk, dim), dtype=np.float32) < p
            y = np.where(msk, x, xp)
            fy = np.asarray(fn(y), dtype=np.float64)
            pf_num[p].append(float((fx * fy).sum()) - dd)

        # Chaos decomposition by the Ornstein-Uhlenbeck / Mehler identity:
        # with y = rho x + sqrt(1-rho^2) x',
        #   E[f(x) f(y)] - (E f)^2 = sum_k rho^k Var(f_k),
        # f_k being the projection on the k-th Wiener chaos (TOTAL Hermite
        # degree).  Degree, not ANOVA order, is what a lattice responds to:
        # a randomly shifted lattice annihilates every Fourier mode outside
        # its dual, and low degree means low frequency.
        for r in RHO_DEFAULT:
            if r == -1.0:
                y = -x
            else:
                y = (np.float32(r) * x
                     + np.float32(math.sqrt(1.0 - r * r)) * xp)
            fy = np.asarray(fn(y), dtype=np.float64)
            ch_num[r].append(float((fx * fy).sum()) - dd)

        # Jansen with one random coordinate resampled:
        # (1/2)E[(f(x)-f(x^(j)))^2] = T_j V; averaged over uniform j and
        # multiplied by dim it is sum_j T_j V = d_M V.
        j = rng.integers(0, dim, size=chunk)
        y = x.copy()
        y[np.arange(chunk), j] = xp[np.arange(chunk), j]
        fy = np.asarray(fn(y), dtype=np.float64)
        jansen.append(0.5 * float(((fx - fy) ** 2).sum()))

    V = float((s2 / nf - (s1 / nf) ** 2).sum())
    Qn = [Q[q] / max(nQ[q], 1) for q in range(4)]
    C2 = (0.5 * (Qn[0] + Qn[2])) * (0.5 * (Qn[1] + Qn[3]))   # half(0,2) x half(1,3)
    per_k = np.einsum("kjm->k", C2) / fact
    per_j = np.einsum("kjm,k->j", C2, 1.0 / fact)
    V1 = float(per_j.sum())
    # two INDEPENDENT split-half estimates of the k=1 per-coordinate profile,
    # so its across-coordinate spread can be separated from estimator noise.
    pj1 = np.einsum("jm->j", Qn[0][0] * Qn[1][0])
    pj2 = np.einsum("jm->j", Qn[2][0] * Qn[3][0])
    nc = len(jansen)
    jm = np.array(jansen) / chunk
    gp = {}
    for p in ps:
        a = np.array(pf_num[p]) / chunk
        gp[p] = (a.mean() / V, a.std(ddof=1) / math.sqrt(nc) / V)
    ch = {}
    for r in RHO_DEFAULT:
        a = np.array(ch_num[r]) / chunk
        ch[r] = (a.mean() / V, a.std(ddof=1) / math.sqrt(nc) / V)
    return dict(V=V, V1=V1, f1=V1 / V, per_k=per_k / V, per_j=per_j / V,
                pj1=pj1 / V, pj2=pj2 / V,
                dM=dim * jm.mean() / V,
                dM_se=dim * jm.std(ddof=1) / math.sqrt(nc) / V, gp=gp, ch=ch)


def mode_anova(suite, n_mlps, n_samples, kmax, chunk, seed):
    """ANOVA order structure of ``x -> relu(z^32(x))`` on real official MLPs."""
    ps = PS_DEFAULT
    rows = []
    t0 = time.time()
    for i in range(n_mlps):
        W = make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])
        rng = np.random.default_rng(0xA0A0 + 7919 * seed + i)
        r = anova_probe(lambda z: forward(W, z), WIDTH, n_samples, kmax,
                        chunk, rng, ps)
        rows.append(r)
        print(f"  mlp {i + 1}/{n_mlps}  V={r['V']:.4f}  f_1={r['f1']:8.5f}  "
              f"d_M={r['dM']:6.2f}+-{r['dM_se']:.2f}  "
              f"g(0.5)={r['gp'][0.5][0]:.4f}  [{time.time() - t0:.0f}s]",
              flush=True)
    nchunk = max(1, n_samples // chunk)

    f1 = np.array([r["f1"] for r in rows])
    dM = np.array([r["dM"] for r in rows])
    Vs = np.array([r["V"] for r in rows])
    print(f"\n=== ANOVA structure of x -> relu(z^32_j(x)), {n_mlps} official "
          f"MLPs x {nchunk * chunk:,} samples ===")
    print(f"  Var aggregated over the 256 outputs   {Vs.mean():.4f} "
          f"(avg per neuron {Vs.mean() / WIDTH:.5f})")
    print(f"\n  FIRST-ORDER SHARE  f_1 = {f1.mean() * 100:.3f}% "
          f"+- {f1.std(ddof=1) / math.sqrt(n_mlps) * 100:.3f}%  "
          f"(per-MLP range {f1.min() * 100:.2f}-{f1.max() * 100:.2f}%)")
    print(f"  MEAN DIMENSION     d_M = {dM.mean():.2f} "
          f"+- {dM.std(ddof=1) / math.sqrt(n_mlps):.2f}   "
          f"(1.0 = purely additive)")

    pk = np.stack([r["per_k"] for r in rows]).mean(0)
    print("\n  first-order variance by Hermite order of the single coordinate")
    print("    " + "  ".join(f"k={k + 1}: {100 * v:7.4f}%"
                             for k, v in enumerate(pk)))

    # Per-coordinate importance V_j.  By Stein, E[f_m x_j] = E[df_m/dx_j], so
    # the k=1 profile IS the mean-Jacobian ordering the brief asks about.  Two
    # independent split-half estimates separate real spread from noise.
    pj1 = np.stack([r["pj1"] for r in rows])
    pj2 = np.stack([r["pj2"] for r in rows])
    rr = []
    slope = []
    for i in range(n_mlps):
        a, b = pj1[i], pj2[i]
        rr.append(float(np.corrcoef(a, b)[0, 1]))
        w1 = make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])[0]
        rn = (w1 ** 2).sum(1)
        slope.append(float(np.corrcoef(rn, 0.5 * (a + b))[0, 1]))
    pj = 0.5 * (pj1 + pj2)
    rel = pj / pj.mean(1, keepdims=True)
    # observed spread^2 = true spread^2 + noise^2; noise^2 from the half-diff
    obs = rel.std(axis=1).mean()
    noi = ((pj1 - pj2) / (2 * pj.mean(1, keepdims=True))).std(axis=1).mean()
    true = math.sqrt(max(obs ** 2 - noi ** 2, 0.0))
    print(f"\n  per-coordinate importance V_j (k=1) = mean-Jacobian ordering")
    print(f"    observed spread/mean {obs:.3f}, estimator noise {noi:.3f} "
          f"=> TRUE spread/mean {true:.3f}")
    print(f"    split-half reliability r = {np.mean(rr):+.3f}; "
          f"corr with ||W^1 row_j||^2 = {np.mean(slope):+.3f}")
    # Honest top/bottom ratio: RANK on one half, MEASURE on the other, so the
    # selection noise does not inflate the spread.
    tb = []
    for i in range(n_mlps):
        o = np.argsort(pj1[i])
        lo, hi = pj2[i][o[:64]].mean(), pj2[i][o[-64:]].mean()
        tb.append(hi / lo if lo > 0 else float("nan"))
    print(f"    out-of-sample top-64 / bottom-64 importance ratio: "
          f"{np.nanmean(tb):.2f}")
    print("    (the ranking is real but ||W^1 row_j||^2 does not carry it, and"
          "\n     the only unbiased predict-time estimate of it is Cov(f,x_j)"
          " itself)")

    print("\n  order-generating function g(p) = sum_d f_d p^d  (pick-and-freeze)")
    print(f"    {'p':>7} {'g(p)':>9} {'+-':>8} {'g(p)/p':>9}   "
          f"{'implied sum_{{d<=3}} f_d <= g/p^3':>10}")
    gmean = {}
    for p in ps:
        a = np.array([r["gp"][p][0] for r in rows])
        se = math.sqrt(sum(r["gp"][p][1] ** 2 for r in rows)) / n_mlps
        gmean[p] = a.mean()
        print(f"    {p:7.4f} {a.mean():9.5f} {se:8.5f} {a.mean() / p:9.4f}   "
              f"{a.mean() / p ** 3:10.3f}")

    # Non-negative least squares fit of the order distribution, with f_1
    # pinned to the Hermite value and sum_d f_d = 1 enforced.
    # Chaos (total Hermite degree) decomposition -- the quantity the lattice
    # actually responds to, and the one that separates "high effective
    # dimension" from "high frequency".
    print("\n  chaos-generating function G(rho) = sum_k rho^k Var(f_k)/V")
    print(f"    {'rho':>7} {'G(rho)':>9} {'+-':>8}")
    gch = {}
    for r in RHO_DEFAULT:
        a = np.array([w["ch"][r][0] for w in rows])
        se = math.sqrt(sum(w["ch"][r][1] ** 2 for w in rows)) / n_mlps
        gch[r] = a.mean()
        print(f"    {r:7.2f} {a.mean():9.5f} {se:8.5f}")
    Dk = 40
    rv = np.array(RHO_DEFAULT)
    Ak = np.stack([rv ** k for k in range(1, Dk + 1)], 1)
    Ak = np.vstack([Ak, np.ones(Dk), np.eye(Dk)[0]])
    yk = np.concatenate([[gch[r] for r in RHO_DEFAULT], [1.0], [pk[0]]])
    wk = np.concatenate([np.full(len(RHO_DEFAULT), 1.0), [30.0], [10.0]])
    ck = _nnls(Ak * wk[:, None], yk * wk, iters=8000)
    ck = ck / max(ck.sum(), 1e-12)
    cck = np.cumsum(ck)
    print("    " + "  ".join(f"c_{k + 1}={ck[k]:.4f}" for k in range(6)))
    print(f"    cumulative degree:  <=1 {cck[0]:.4f}  <=2 {cck[1]:.4f}  "
          f"<=3 {cck[2]:.4f}  <=4 {cck[3]:.4f}  <=6 {cck[5]:.4f}")
    print(f"    mean Hermite degree {float((np.arange(1, Dk + 1) * ck).sum()):.2f}"
          f"   (mean ANOVA dimension {dM.mean():.2f})")
    odd = 0.5 * (1.0 - gch[-1.0])
    print(f"\n    odd-chaos share (1-G(-1))/2 = {odd:.4f}  =>  antithetic "
          f"pairing is worth 1/(1+G(-1)) = {1.0 / (1.0 + gch[-1.0]):.3f}x, "
          f"predicted before it is run")

    # Hard bounds first: f_d >= 0 gives sum_{d<=D} f_d <= g(p)/p^D for p<1,
    # and with f_1 known, f_2 <= (g(p) - f_1 p)/p^2.  No fit, no assumptions.
    dd = np.arange(1, 5)
    print("\n  hard bounds from g (no fit): f_d >= 0 alone gives")
    for D in (2, 3):
        bnd = min(gmean[p] / p ** D for p in ps)
        print(f"    sum_(d<={D}) f_d <= {bnd:.4f}")
    f2b = min((gmean[p] - f1.mean() * p) / p ** 2 for p in ps if p <= 0.5)
    print(f"    f_2 <= {f2b:.4f}   (so f_1 + f_2 <= {f1.mean() + f2b:.4f})")

    # Constrained fit: sum f_d = 1, f_1 pinned to Hermite, sum d f_d pinned to
    # the Jansen mean dimension.  Three independent measurements, one
    # distribution -- the fit is a reconciliation, not a source of new numbers.
    D = 64
    pv = np.array(ps)
    A = np.stack([pv ** d for d in range(1, D + 1)], 1)
    y = np.array([gmean[p] for p in ps])
    A = np.vstack([A, np.ones(D), np.eye(D)[0], np.arange(1, D + 1) / dM.mean()])
    y = np.concatenate([y, [1.0], [f1.mean()], [1.0]])
    wts = np.concatenate([np.full(len(ps), 1.0), [30.0], [30.0], [30.0]])
    f = _nnls(A * wts[:, None], y * wts, iters=8000)
    f = f / max(f.sum(), 1e-12)
    cum = np.cumsum(f)
    print("\n  reconciled ANOVA order distribution (NNLS; f_1, sum f_d and "
          "sum d f_d all pinned)")
    print("    " + "  ".join(f"f_{d + 1}={f[d]:.4f}" for d in range(6)))
    print(f"    cumulative:  <=1 {cum[0]:.4f}   <=2 {cum[1]:.4f}   "
          f"<=3 {cum[2]:.4f}   <=5 {cum[4]:.4f}   <=10 {cum[9]:.4f}")
    print(f"    mean dimension of the fit "
          f"{float((np.arange(1, D + 1) * f).sum()):.2f} "
          f"(Jansen {dM.mean():.2f}); g residual rms "
          f"{float(np.sqrt(np.mean((A[:len(ps)] @ f - y[:len(ps)]) ** 2))):.4f}")
    print("\n  effective dimension in the superposition sense: "
          f"d_S(0.99) = {int(np.searchsorted(cum, 0.99) + 1)}, "
          f"d_S(0.95) = {int(np.searchsorted(cum, 0.95) + 1)}, "
          f"d_S(0.50) = {int(np.searchsorted(cum, 0.50) + 1)}")
    print(f"\n  CEILING IMPLIED FOR RQMC if it annihilated every ANOVA term of "
          f"order <= d:\n    d=1: {1 / (1 - cum[0]):.2f}x   "
          f"d=2: {1 / (1 - cum[1]):.2f}x   d=3: {1 / (1 - cum[2]):.2f}x   "
          f"d=5: {1 / (1 - cum[4]):.2f}x")
    _ = dd

    out = artifacts() / "rqmc" / "anova.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, f1=f1, dM=dM, V=Vs, per_k=pk,
             per_j=np.stack([r["per_j"] for r in rows]), pj1=pj1, pj2=pj2,
             ps=np.array(ps), g=np.array([gmean[p] for p in ps]), fit=f,
             gp_all=np.array([[w["gp"][p][0] for p in ps] for w in rows]),
             rhos=np.array(RHO_DEFAULT), chaos=np.array(
                 [[w["ch"][r][0] for r in RHO_DEFAULT] for w in rows]),
             chaos_fit=ck)
    print(f"\nsaved {out}")


# ---------------------------------------------------------------------------
# Rank-1 lattice construction.  Offline, unbilled: a generating vector is a
# property of the point set, not of the data, exactly like Sobol' direction
# numbers -- but it is committed to the repo so the shipped code carries no
# search.
# ---------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_at_most(n: int) -> int:
    while not is_prime(n):
        n -= 1
    return n


def primitive_root(p: int) -> int:
    fac, m = set(), p - 1
    d = 2
    while d * d <= m:
        while m % d == 0:
            fac.add(d)
            m //= d
        d += 1
    if m > 1:
        fac.add(m)
    for g in range(2, p):
        if all(pow(g, (p - 1) // f, p) != 1 for f in fac):
            return g
    raise ValueError("no primitive root")


def _b2(x):
    """Bernoulli ``B_2({x}) = x^2 - x + 1/6`` -- the shift-averaged kernel of
    the unanchored Sobolev space, so ``(1/N) sum_i prod_{j in u} B_2(x_ij)`` is
    exactly the squared worst-case error contributed by projection ``u``."""
    return x * x - x + 1.0 / 6.0


def cbc_order2(N: int, d: int, verbose: bool = False):
    """Component-by-component lattice, order-2 (pairwise) weights, exact and fast.

    The criterion is ``S2(z) = sum_{j<k} (1/N) sum_i B_2(i z_j/N) B_2(i z_k/N)``
    -- the whole two-dimensional part of the shift-averaged worst-case error.
    Adding dimension ``s`` adds ``(1/N) sum_i B_i B_2(i z_s / N)`` with
    ``B_i = sum_{j<s} B_2(i z_j / N)``, and with ``N`` prime and ``i = g^k`` that
    sum is a cyclic correlation, so ONE FFT scores every candidate ``z_s`` at
    once.  Full CBC in ``O(d N log N)``, not the ``O(d N^2)`` of a naive search.

    One-dimensional projections need no search at all: for any ``z_j`` coprime
    to ``N`` the projection is the exact ``N``-point grid, and its term is
    ``1/(6 N^2)`` -- which is why a lattice annihilates first-order ANOVA terms
    whatever the generating vector, and why the search only buys the pairs.
    """
    g = primitive_root(N)
    k = np.arange(N - 1)
    gpow = np.ones(N - 1, dtype=np.int64)
    for t in range(1, N - 1):
        gpow[t] = gpow[t - 1] * g % N
    b = _b2(np.arange(N) / N)                    # b[m] = B_2(m/N)
    v = b[gpow]                                  # v_k = B_2(g^k / N)
    Fv = np.fft.rfft(v)
    B = np.zeros(N)                              # running sum over chosen dims
    z = np.zeros(d, dtype=np.int64)
    for s in range(d):
        if s == 0:
            z[s] = 1                             # first dim is a free choice
        else:
            u = B[gpow]
            corr = np.fft.irfft(np.conj(np.fft.rfft(u)) * Fv, n=N - 1)
            z[s] = int(gpow[int(np.argmin(corr))])
        B += b[(np.arange(N) * z[s]) % N]
        if verbose and (s + 1) % 64 == 0:
            print(f"    cbc dim {s + 1}/{d}", flush=True)
    _ = k
    return z


def lattice_quality(N: int, z: np.ndarray):
    """Exact order-1 and order-2 worst-case-error terms of a rank-1 lattice."""
    d = len(z)
    b = _b2(np.arange(N) / N)
    Bm = b[(np.arange(N)[:, None] * z[None, :]) % N]          # (N, d)
    t1 = Bm.mean(0)                                            # per dim
    tot = Bm.sum(1)
    s2 = 0.5 * ((tot * tot).mean() - (Bm * Bm).sum(1).mean())  # sum_{j<k} T_jk
    # worst pair, without materialising all d^2 of them
    G = (Bm.T @ Bm) / N
    np.fill_diagonal(G, 0.0)
    return dict(t1_mean=float(t1.mean()), t1_ref=1.0 / (6.0 * N * N),
                s2=float(s2), s2_per_pair=float(s2 / (d * (d - 1) / 2)),
                tmax=float(G.max()), tmin=float(G.min()),
                trms=float(np.sqrt((G ** 2).sum() / (d * (d - 1)))))


def roberts_z(N: int, d: int):
    """Kronecker / Roberts lattice: ``alpha_j = phi_d^{-(j+1)}``, rationalised."""
    phi = 2.0
    for _ in range(200):
        phi = (1.0 + phi) ** (1.0 / (d + 1.0))
    a = np.array([phi ** -(j + 1) for j in range(d)])
    return np.round(a * N).astype(np.int64) % N


def korobov_z(N: int, d: int, a: int):
    z = np.ones(d, dtype=np.int64)
    for j in range(1, d):
        z[j] = z[j - 1] * a % N
    return z


def mode_lattice(N, d, seed):
    print(f"# rank-1 lattice generating vectors, N={N} (prime: {is_prime(N)}), "
          f"d={d}\n")
    t0 = time.time()
    zc = cbc_order2(N, d, verbose=True)
    print(f"  CBC done in {time.time() - t0:.1f}s")
    rng = np.random.default_rng(seed)
    cands = {
        "CBC order-2": zc,
        "random z": rng.integers(1, N, size=d),
        "Roberts/Kronecker": roberts_z(N, d),
        "Korobov a=3": korobov_z(N, d, 3),
        "Korobov a=1571": korobov_z(N, d, 1571),
    }
    print(f"\n{'vector':<20} {'1-D term':>11} {'1/(6N^2)':>11} "
          f"{'sum_{j<k}T':>12} {'per pair':>11} {'max T':>11} {'rms T':>11}")
    best = None
    for k, z in cands.items():
        q = lattice_quality(N, z)
        print(f"{k:<20} {q['t1_mean']:11.3e} {q['t1_ref']:11.3e} "
              f"{q['s2']:12.4e} {q['s2_per_pair']:11.3e} {q['tmax']:11.3e} "
              f"{q['trms']:11.3e}")
        if best is None:
            best = (k, z, q)
    print("\n  The 1-D term equals 1/(6N^2) for EVERY vector: all first-order "
          "ANOVA\n  terms are integrated at O(N^-2) instead of MC's O(N^-1), "
          f"a factor ~N/6 = {N // 6:,}x,\n  regardless of the search.  The "
          "search only improves the PAIRS.")
    out = artifacts() / "rqmc" / f"z_cbc_{N}_{d}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, zc)
    print(f"\nsaved {out}")
    print("\nZ_CBC = (" + ", ".join(str(int(v)) for v in zc) + ")")


# ---------------------------------------------------------------------------
# mode: probe -- head to head at MATCHED EFFECTIVE COMPUTE
# ---------------------------------------------------------------------------
def _bill_n(fn, W, n):
    import flopscope as flops  # noqa: PLC0415
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as c:
        fn(W, n)
    return int(c.flops_used)


def mode_probe(suite, n_mlps, n_reps, N, seed, tau):
    """RQMC vs iid at matched billed FLOPs, paired by MLP, over ``n_reps``
    independent randomisations so the ratio carries an error bar."""
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    from whestfloor import kernels as K  # noqa: PLC0415
    from whestfloor.contract import FLOP_BUDGET  # noqa: PLC0415

    Wc = [fnp.asarray(w) for w in make_official_mlp(WIDTH, DEPTH,
                                                    suite.mlp_seeds[0])]
    base = K.lattice_base(N, K.RQMC_Z)          # as setup would build it

    def iid(W, n):
        return K.rqmc_sparse_kernel(W, tau=tau, n_samples=n, rqmc=False,
                                    safe=False)

    # match compute: solve F_iid(n) = F_rqmc(N) on the exactly linear bill
    f_lo, f_hi = _bill_n(iid, Wc, 4000), _bill_n(iid, Wc, 8000)
    per = (f_hi - f_lo) / 4000.0
    setup = f_lo - per * 4000.0
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as c:
        K.rqmc_sparse_kernel(Wc, tau=tau, n_samples=N, rqmc=True, base=base,
                             safe=False)
    f_rq = int(c.flops_used)
    n_iid = int(round((f_rq - setup) / per))
    f_iid = _bill_n(iid, Wc, n_iid)
    print(f"# matched compute: RQMC N={N} bills {f_rq:,} (F/B "
          f"{f_rq / FLOP_BUDGET:.4f});  iid gets n={n_iid} -> {f_iid:,} "
          f"({100 * (f_iid / f_rq - 1):+.3f}%)")
    print(f"# per-sample marginal cost {per:,.0f}; RQMC draw overhead "
          f"{f_rq - _bill_n(iid, Wc, N):,} FLOPs = "
          f"{100 * (f_rq / _bill_n(iid, Wc, N) - 1):.2f}% of the pass")

    arms = {
        "iid (ablation)": lambda W, s: K.rqmc_sparse_kernel(
            W, tau=tau, n_samples=n_iid, rqmc=False, seed=s, safe=False),
        "RQMC lattice": lambda W, s: K.rqmc_sparse_kernel(
            W, tau=tau, n_samples=N, rqmc=True, base=base, seed=s,
            safe=False),
        "RQMC + Jacobian order": lambda W, s: K.rqmc_sparse_kernel(
            W, tau=tau, n_samples=N, rqmc=True, base=base, order=True,
            seed=s, safe=False),
        "antithetic (not RQMC)": lambda W, s: K.rqmc_sparse_kernel(
            W, tau=tau, n_samples=n_iid - (n_iid % 2), rqmc="anti", seed=s,
            safe=False),
    }
    gt = suite.gt[:, -1, :]
    mse = {k: np.zeros((n_mlps, n_reps)) for k in arms}
    psum = {k: np.zeros((n_mlps, WIDTH)) for k in arms}
    t0 = time.time()
    for i in range(n_mlps):
        W = [fnp.asarray(w) for w in make_official_mlp(WIDTH, DEPTH,
                                                       suite.mlp_seeds[i])]
        for r in range(n_reps):
            s = 1_000_003 * (seed + r) + 17
            for k, fn in arms.items():
                with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
                    p = np.asarray(fn(W, s), dtype=np.float64)
                mse[k][i, r] = float(np.mean((p[-1] - gt[i]) ** 2))
                psum[k][i] += p[-1]
        print(f"  mlp {i + 1}/{n_mlps}  [{time.time() - t0:.0f}s]", flush=True)

    base_m = mse["iid (ablation)"]
    print(f"\n=== matched-compute probe, {n_mlps} official MLPs x {n_reps} "
          f"randomisations, tau={tau} ===")
    print(f"{'arm':<24} {'raw MSE':>11} {'+- se':>10} {'gain vs iid':>12} "
          f"{'+- se':>8}   {'>=1.5x bar':>10}")
    out = {}
    for k in arms:
        m = mse[k]
        # paired over (mlp, rep): the ratio of suite means, with a bootstrap
        # SE over MLPs (reps within an MLP are averaged first)
        per_mlp_a = base_m.mean(1)
        per_mlp_b = m.mean(1)
        g = per_mlp_a.sum() / per_mlp_b.sum()
        rng = np.random.default_rng(0)
        bs = [float(per_mlp_a[idx].sum() / per_mlp_b[idx].sum())
              for idx in rng.integers(0, n_mlps, size=(2000, n_mlps))]
        se_g = float(np.std(bs))
        se_m = float(m.mean(1).std(ddof=1) / math.sqrt(n_mlps))
        out[k] = (float(m.mean()), se_m, g, se_g)
        print(f"{k:<24} {m.mean():11.4e} {se_m:10.3e} {g:12.3f} "
              f"{se_g:8.3f}   {'PASS' if g - se_g >= 1.5 else 'fail':>10}")

    # Unbiasedness.  Averaging R independent randomisations must divide the
    # MSE by R exactly if the estimator is unbiased; any residue is bias^2.
    # A biased estimator would quietly void the floor argument in
    # docs/floor_theorem.md, so this is checked, not assumed.
    print(f"\n  unbiasedness: MSE of the {n_reps}-randomisation mean, "
          f"x{n_reps}, against the single-draw MSE (1.00 = unbiased)")
    for k in arms:
        bm = float(np.mean((psum[k] / n_reps - gt[:n_mlps]) ** 2))
        print(f"    {k:<24} {bm:11.4e} x{n_reps} = {bm * n_reps:11.4e}  "
              f"ratio {bm * n_reps / mse[k].mean():6.3f}")
    np.savez(artifacts() / "rqmc" / "probe.npz",
             **{k.replace(" ", "_"): v for k, v in mse.items()})
    return out


# ---------------------------------------------------------------------------
# mode: score -- end to end on the official suite
# ---------------------------------------------------------------------------
def mode_score(suite, n_mlps, grid, seed):
    import functools  # noqa: PLC0415

    import flopscope.numpy as fnp  # noqa: PLC0415

    from whestfloor import kernels as K  # noqa: PLC0415
    from whestfloor.contract import (  # noqa: PLC0415
        FLOP_BUDGET, LAMBDA_FLOPS_PER_SECOND, MULTIPLIER_FLOOR,
        effective_compute,
    )
    from whestfloor.harness import run_billed  # noqa: PLC0415

    gt = suite.gt[:, -1, :]
    variants = []
    for spec in grid.split(";"):
        if not spec:
            continue
        kw = dict(kv.split("=") for kv in spec.split(","))
        N = int(kw.get("n", 8501))
        mode = kw.get("draw", "rqmc")
        tau = None if kw.get("tau", "2.5") == "None" else float(kw["tau"])
        order = kw.get("order", "0") == "1"
        sd = int(kw.get("seed", seed))
        P = int(kw.get("P", 150))
        if mode == "rqmc":
            zc = (K.RQMC_Z if N == K.RQMC_N
                  else cbc_order2(N, WIDTH))
            base = K.lattice_base(N, zc)     # as setup would, unbilled
            rq = True
        else:
            base, rq = None, ("anti" if mode == "anti" else False)
        variants.append((
            f"{mode:<5} n={N:<5} tau={kw.get('tau','2.5'):>4} P={P:<4} "
            f"ord={int(order)} s={sd}",
            functools.partial(K.rqmc_sparse_kernel, tau=tau, n_samples=N,
                              n_pilot=P, rqmc=rq, base=base, order=order,
                              seed=sd)))

    print(f"# official suite, {n_mlps} MLPs, N=1e9 reference -> raw_mse is "
          "leaderboard-comparable")
    print("# adj@kx = adjusted score if the grader's residual is k times this "
          "machine's\n")
    hdr = (f"{'variant':<47} {'raw_mse':>11} {'F/B':>7} {'C/B':>7} "
           f"{'adj@1x':>10} {'adj@2x':>10} {'adj@3x':>10} {'raise':>6} "
           f"{'worst':>10}")
    print(hdr)
    print("-" * len(hdr))
    res = {}
    for name, fn in variants:
        mses, fls, rss, nfail, worst = [], [], [], 0, 0.0
        for i in range(n_mlps):
            Wn = make_official_mlp(WIDTH, DEPTH, suite.mlp_seeds[i])
            try:
                pred, fl, rs = run_billed(fn, Wn)
                assert pred.shape == (DEPTH, WIDTH) and np.isfinite(pred).all()
            except Exception as e:  # noqa: BLE001
                print(f"  !! RAISED on mlp {i}: {type(e).__name__}: {e}")
                nfail += 1
                pred, fl, rs = np.zeros((DEPTH, WIDTH)), FLOP_BUDGET, 0.0
            m = float(np.mean((pred[-1] - gt[i]) ** 2))
            worst = max(worst, m)
            mses.append(m)
            fls.append(fl)
            rss.append(rs)
        raw = float(np.mean(mses))
        F, R = float(np.mean(fls)), float(np.mean(rss))
        adj = [raw * max(MULTIPLIER_FLOOR,
                         effective_compute(F, k * R, LAMBDA_FLOPS_PER_SECOND)
                         / FLOP_BUDGET) for k in (1, 2, 3)]
        C1 = effective_compute(F, R, LAMBDA_FLOPS_PER_SECOND)
        print(f"{name:<47} {raw:11.4e} {F / FLOP_BUDGET:7.4f} "
              f"{C1 / FLOP_BUDGET:7.4f} " + " ".join(f"{v:10.4e}" for v in adj)
              + f" {nfail:6d} {worst:10.3e}", flush=True)
        res[name] = (raw, adj[0])
    _ = fnp
    return res


def _nnls(A, b, iters=2000):
    """Projected-gradient NNLS.  No scipy in this sandbox."""
    AtA = A.T @ A
    Atb = A.T @ b
    L = float(np.linalg.eigvalsh(AtA).max())
    x = np.zeros(A.shape[1])
    for _ in range(iters):
        x = np.maximum(x - (AtA @ x - Atb) / L, 0.0)
    return x


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("anova", "lattice", "probe", "score"))
    ap.add_argument("--suite", default=None)
    ap.add_argument("--n-mlps", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=40960)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=8501)
    ap.add_argument("--n-reps", type=int, default=8)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--grid", type=str,
                    default="draw=iid,n=8623;draw=rqmc,n=8501")
    args = ap.parse_args()
    if args.mode == "lattice":
        mode_lattice(args.N, WIDTH, args.seed)
        return 0
    suite = load_suite(args.suite)
    if args.mode == "anova":
        mode_anova(suite, args.n_mlps, args.n_samples, args.kmax, args.chunk,
                   args.seed)
    elif args.mode == "probe":
        mode_probe(suite, args.n_mlps, args.n_reps, args.N, args.seed,
                   None if args.tau < 0 else args.tau)
    elif args.mode == "score":
        mode_score(suite, args.n_mlps, args.grid, args.seed)
    else:
        raise SystemExit(f"mode {args.mode} not implemented yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
