#!/usr/bin/env python
"""High-degree Hermite control variates on network-adapted directions.

The reframe this script tests
-----------------------------
``docs/floor_theorem.md`` bounds a surrogate whose ANOVA content sits at order
``<= k`` by ``1/(1 - sum_{d<=k} f_d)``, and ``docs/learned_corrector.md`` ships
the ``k <= 2`` instance.  But ``He_d`` of a layer-1 pre-activation is a
*degree-d* object, so the ``k <= 2`` instance of the barrier does not bound
degree-3+ Hermite control variates at all.  What killed ``k = 3`` in the ledger
was ``p/N``: each extra coordinate block is 256 more coefficients against
``N = 8500`` samples.

That is an estimation-noise problem, not an approximation-power problem, and
the cure is a *better-chosen, smaller* basis rather than a bigger one.  The 256
coordinate directions ``W^1[:,i]`` are an arbitrary basis for the degree-1
chaos; the network supplies its own.

The algebra, all exact
----------------------
``x ~ N(0, I_n)``.  For any unit vector ``a``, ``s_a = <a, x>`` is exactly
standard normal, so with the L2-normalised Hermite ``h_d = He_d / sqrt(d!)``

    E[h_d(s_a)] = 0                                    exactly, every d >= 1
    Cov(h_d(s_a), h_e(s_b)) = delta_de <a,b>^d                       (Mehler)

Every feature is therefore an exactly-mean-zero control variate with an
**analytic** Gram, block diagonal in degree -- nothing is ever estimated except
the covariance with the target.  Two further facts make the whole dictionary
analytic:

* ``h_d(<a,x>)`` is the unit vector ``a^{(x)d}`` of the degree-``d`` chaos, i.e.
  ``h_d(<a,x>) = sum_{|alpha|=d} sqrt(d!/alpha!) a^alpha H_alpha(x)`` in ANY
  orthonormal frame.  Hence for an orthonormal set ``A`` the tensor products
  ``H_alpha = prod_r h_{alpha_r}(<a_r,x>)`` are exactly orthonormal, and their
  overlap with a pure power on an arbitrary unit ``c`` is
  ``delta_{|alpha|,d} sqrt(d!/alpha!) prod_r (A^T c)_r^{alpha_r}``.
* The layer-1 coordinate family is the special case ``a = W^1[:,i]/|W^1[:,i]|``,
  whose degree-``d`` Gram is the Hadamard power ``rho^{o d}``.

So a single design matrix can hold the shipped coordinate family, an adapted
family with all its cross products, and their unions, with a Gram written down
from the weights alone.

Modes
-----
``--mode dirs``    direction diagnostics: how concentrated are the network's
                   own directions, and how well can they be reached from the
                   weights alone at predict time.
``--mode span``    **the R^2(p, d) surface.**  Coefficients fitted on one half
                   of the sample and explained variance read on the other, so
                   no ``p/N`` in-sample inflation.  This is the measurement
                   that decides everything.
``--mode ceiling`` the degree-free ceiling: the closed span of the Hermite
                   family on ``m`` directions is exactly ``L^2(sigma(A^T x))``,
                   so ``R^2 <= sum_j Var(E[y_j | A^T x]) / sum_j Var(y_j)`` for
                   *every* degree and *every* product at once.  Measured
                   nonparametrically, plus the dictionary-free top-p bound.

All MLPs here are LOCAL (``make_mlp``, seed base 900000), disjoint from the
official suite, which this script never opens.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import DEPTH, WIDTH  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.official_seeds import make_official_mlp  # noqa: E402

#: Weight source.  ``local`` is ``make_mlp`` on seeds 900000+, disjoint from
#: every suite.  ``official`` is seed protocol 3.0 on the held-out suite's own
#: seeds -- used ONLY to confirm that a measured ceiling transfers to the
#: graded networks.  Nothing is ever fitted or selected from it: every
#: dictionary, degree and direction rule in this file is fixed before the
#: official weights are touched, and the official run reports the same table.
_SOURCE = {"mode": "local"}


def mlp_weights(k: int, seed0: int):
    """``(label, weights)`` for the k-th MLP of the configured source."""
    if _SOURCE["mode"] == "official":
        seeds = _SOURCE["seeds"]
        return int(seeds[k]), make_official_mlp(WIDTH, DEPTH, int(seeds[k]))
    return seed0 + k, make_mlp(WIDTH, DEPTH, seed0 + k)


def load_official_seeds():
    import os  # noqa: PLC0415
    from whestfloor.suite import Suite  # noqa: PLC0415
    root = Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()
    return list(Suite.load(root / "suites" / "official_mini.npz").mlp_seeds)

#: Chunk that keeps the working set in cache (see whestfloor/mc.py).
CHUNK = 1024


def norm_cdf(v):
    from flopscope.stats._erf import _erf  # noqa: PLC0415
    return 0.5 * (1.0 + _erf(v / np.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Hermite
# ---------------------------------------------------------------------------
def herm_table(s, dmax):
    """``(dmax, B, m)`` with ``out[d-1] = h_d(s) = He_d(s)/sqrt(d!)``.

    The normalised three-term recurrence
    ``h_d = (s h_{d-1} - sqrt(d-1) h_{d-2}) / sqrt(d)`` keeps every entry O(1)
    for |s| <= 5; the raw ``He_d`` overflows float32 by d ~ 12.
    """
    out = np.empty((dmax,) + s.shape, dtype=np.float32)
    hm1 = np.ones_like(s)
    h = s.copy()
    out[0] = h
    for d in range(2, dmax + 1):
        hn = (s * h - np.float32(math.sqrt(d - 1.0)) * hm1) * np.float32(
            1.0 / math.sqrt(float(d)))
        hm1, h = h, hn
        out[d - 1] = h
    return out


# ---------------------------------------------------------------------------
# Network-adapted directions
# ---------------------------------------------------------------------------
def pilot_stats(W, seed, npil):
    """``(alpha per layer, mean-Jacobian G = E[x^T ybar], Var(y), E[y])``.

    ``G`` uses the CENTRED target.  ``E[x] = 0`` exactly in population but not
    in sample, and the sample term ``xbar (x) ybar`` is a rank-one contaminant
    whose norm at n = 256, |E y| ~ 0.6 is comparable to the whole signal -- it
    would otherwise dominate the top singular direction outright.
    """
    n = W[0].shape[0]
    G = np.zeros((n, n))
    sy = np.zeros(n)
    sy2 = np.zeros(n)
    sx = np.zeros(n)
    am = [np.zeros(n) for _ in range(len(W))]
    a2 = [np.zeros(n) for _ in range(len(W))]
    rng = np.random.default_rng(seed)
    done = 0
    while done < npil:
        B = min(CHUNK, npil - done)
        x = rng.standard_normal((B, n), dtype=np.float32)
        h = x
        for li, w in enumerate(W):
            z = h @ w
            am[li] += z.sum(0)
            a2[li] += (z.astype(np.float64) ** 2).sum(0)
            h = np.maximum(z, 0.0)
        xd = x.astype(np.float64)
        yd = h.astype(np.float64)
        G += xd.T @ yd
        sx += xd.sum(0)
        sy += yd.sum(0)
        sy2 += (yd * yd).sum(0)
        done += B
    G = G / npil - np.outer(sx / npil, sy / npil)
    alpha = []
    for li in range(len(W)):
        m = am[li] / npil
        v = np.maximum(a2[li] / npil - m * m, 1e-12)
        alpha.append(m / np.sqrt(v))
    return alpha, G, sy2 / npil - (sy / npil) ** 2, sy / npil


def meanfield_path(W, alpha):
    """``W^1 diag(Phi(a^1)) W^2 ... diag(Phi(a^31)) W^32``: the mean-field
    Jacobian of ``z^32`` w.r.t. ``x``, from the weights and the pilot alone."""
    A = W[0].astype(np.float64)
    for li in range(1, len(W)):
        A = (A * norm_cdf(alpha[li - 1])) @ W[li]
    return A


def direction_sets(W, alpha, G, M, seed):
    """Candidate orthonormal direction frames, ``(n, M)`` each."""
    n = W[0].shape[0]
    out = {}
    out["jac"] = np.linalg.svd(G)[0][:, :M]                  # oracle
    out["mf"] = np.linalg.svd(meanfield_path(W, alpha))[0][:, :M]
    An = W[0].astype(np.float64)
    for li in range(1, len(W)):
        An = An @ W[li]
    out["nogate"] = np.linalg.svd(An)[0][:, :M]
    rr = np.random.default_rng(seed + 7)
    out["rand"] = np.linalg.qr(rr.standard_normal((n, M)))[0]
    return {k: np.ascontiguousarray(v.astype(np.float32)) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Design specification.  Every feature is one of
#   ("pure", set_name, direction index, degree)   -- unit direction, pure power
#   ("tens", multi-index over the orthonormal adapted frame)
# and the whole Gram follows from the two Mehler identities in the docstring.
# ---------------------------------------------------------------------------
#: Adapted tensor block: ``(total degree, how many leading directions)``.
TENS_PLAN = ((1, 24), (2, 16), (3, 8), (4, 5))
#: Adapted pure powers beyond the tensor plan: degrees 5..PURE_DMAX on the top
#: PURE_M directions.
PURE_DMAX, PURE_M = 16, 8
#: Coordinate (layer-1) pure powers.  Degree 1 is carried by the input basis.
COORD_DMAX = 8


def tensor_indices(nmax: int = 10 ** 9):
    """Multi-indices over the adapted frame, as tuples of (coord, degree).

    ``nmax`` clamps every entry of :data:`TENS_PLAN` and :data:`PURE_M` to the
    number of directions actually available, so a smaller ``--n-dirs`` degrades
    the plan instead of indexing past the frame.  At the published
    ``--n-dirs 24`` every clamp is a no-op and the layout is unchanged.
    """
    seen, out = set(), []
    for deg, m in TENS_PLAN:
        for comb in itertools.combinations_with_replacement(
                range(min(m, nmax)), deg):
            a = tuple(sorted((r, comb.count(r)) for r in set(comb)))
            if a not in seen:
                seen.add(a)
                out.append(a)
    for r in range(min(PURE_M, nmax)):
        for d in range(5, PURE_DMAX + 1):
            a = ((r, d),)
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out


def multinomial_sqrt(alpha):
    """``sqrt(d! / alpha!)`` for a multi-index given as ((coord, deg), ...)."""
    d = sum(k for _, k in alpha)
    v = math.factorial(d)
    for _, k in alpha:
        v //= math.factorial(k)
    return math.sqrt(float(v))


class Design:
    """Feature layout, analytic Gram, and per-chunk construction."""

    def __init__(self, A, Wn):
        self.A = A                      # (n, M) orthonormal adapted frame
        self.Wn = Wn                    # (n, 256) unit layer-1 directions
        self.M = A.shape[1]
        n = Wn.shape[0]
        self.tens = tensor_indices(self.M)
        self.blocks = {}
        p = 0
        self.blocks["IN1"] = (p, p + n)          # x_i  (== coordinate deg 1)
        p += n
        for d in range(2, COORD_DMAX + 1):
            self.blocks[f"CO{d}"] = (p, p + n)
            p += n
        self.tens_off = p
        for a in self.tens:
            self.blocks[f"AD{a}"] = (p, p + 1)
            p += 1
        self.p = p

    # -- Gram ------------------------------------------------------------
    def gram(self):
        n = self.Wn.shape[0]
        A64 = self.A.astype(np.float64)
        Wn64 = self.Wn.astype(np.float64)
        rho = Wn64.T @ Wn64
        np.fill_diagonal(rho, 1.0)
        G = np.zeros((self.p, self.p))
        # pure-power sets: (matrix of unit directions, degree, slice)
        pure = [(np.eye(n), 1, self.blocks["IN1"])]
        for d in range(2, COORD_DMAX + 1):
            pure.append((Wn64, d, self.blocks[f"CO{d}"]))
        for (C1, d1, (a0, a1)) in pure:
            for (C2, d2, (b0, b1)) in pure:
                if d1 != d2:
                    continue
                G[a0:a1, b0:b1] = (C1.T @ C2) ** d1
        # adapted tensor block: exactly orthonormal
        t0 = self.tens_off
        G[t0:self.p, t0:self.p] = np.eye(self.p - t0)
        # cross: <H_alpha, h_d(<c,x>)> = d_{|a|,d} sqrt(d!/a!) prod (A^T c)^a
        for (C, d, (a0, a1)) in pure:
            Q = A64.T @ C                       # (M, q)
            for i, alpha in enumerate(self.tens):
                if sum(k for _, k in alpha) != d:
                    continue
                v = multinomial_sqrt(alpha) * np.ones(C.shape[1])
                for (r, k) in alpha:
                    v = v * Q[r] ** k
                G[t0 + i, a0:a1] = v
                G[a0:a1, t0 + i] = v
        return 0.5 * (G + G.T)

    # -- per-chunk features ----------------------------------------------
    def build(self, x):
        B = x.shape[0]
        D = np.empty((B, self.p), dtype=np.float32)
        D[:, :x.shape[1]] = x
        t = x @ self.Wn
        Hc = herm_table(t, COORD_DMAX)
        for d in range(2, COORD_DMAX + 1):
            a0, a1 = self.blocks[f"CO{d}"]
            D[:, a0:a1] = Hc[d - 1]
        del Hc
        s = x @ self.A
        Ha = herm_table(s, PURE_DMAX)
        t0 = self.tens_off
        for i, alpha in enumerate(self.tens):
            col = Ha[alpha[0][1] - 1][:, alpha[0][0]]
            for (r, k) in alpha[1:]:
                col = col * Ha[k - 1][:, r]
            D[:, t0 + i] = col
        return D


# ---------------------------------------------------------------------------
# Accumulation and held-out R^2
# ---------------------------------------------------------------------------
class Half:
    def __init__(self, p, n):
        self.N = 0
        self.Dy = np.zeros((p, n))
        self.Dm = np.zeros(p)
        self.sy = np.zeros(n)
        self.sy2 = np.zeros(n)

    def add(self, D, y):
        self.N += len(y)
        self.Dy += (D.T @ y).astype(np.float64)
        self.Dm += D.sum(0, dtype=np.float64)
        self.sy += y.sum(0, dtype=np.float64)
        self.sy2 += np.einsum("ij,ij->j", y, y, dtype=np.float64)

    def cov(self):
        my = self.sy / self.N
        return self.Dy / self.N - np.outer(self.Dm / self.N, my)


RIDGE_GRID = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def r2_holdout(G, cA, cB, T, idx):
    """``(population span R^2, best held-out R^2)``.

    ``G`` is the ANALYTIC population Gram, so ``2 b'c_B - b' G b`` with ``b``
    fitted on half A is an unbiased estimate of the *population* explained
    variance of that predictor -- no empirical Gram is ever inverted and there
    is no ``p/N`` in-sample inflation to subtract.

    The first return value is the cross-half quadratic form
    ``c_A' G^+ c_B``, which is unbiased for the POPULATION span R^2 (the
    approximation power of the dictionary, i.e. its ceiling as N -> inf).  The
    second sweeps a ridge grid and reports the best held-out value, i.e. what
    the dictionary actually achieves when its coefficients are fitted on N/2
    samples.  The gap between them is the ``p/N`` cost, measured rather than
    modelled.
    """
    if len(idx) == 0:
        return 0.0, 0.0
    Gi = G[np.ix_(idx, idx)]
    p = len(idx)
    tau = float(np.trace(Gi)) / p
    g, V = np.linalg.eigh(Gi)
    hA, hB = V.T @ cA[idx], V.T @ cB[idx]
    keep = g > 1e-9 * max(float(g.max()), 1e-300)
    pop = float(np.sum(hA[keep] * hB[keep] / g[keep, None])) / T
    g = np.maximum(g, 0.0)
    best = -np.inf
    for lam in RIDGE_GRID:
        dn = g + lam * tau
        dn = np.where(dn > 1e-300, dn, np.inf)
        acc = 0.0
        for hs, hd in ((hA, hB), (hB, hA)):
            b = hs / dn[:, None]
            acc += float(2.0 * np.sum(b * hd) - np.sum(g[:, None] * b * b))
        best = max(best, 0.5 * acc / T)
    return pop, best


# ---------------------------------------------------------------------------
# mode: dirs
# ---------------------------------------------------------------------------
def mode_dirs(n_mlps, npil):
    print("# How concentrated are the network's own directions, and can they "
          "be reached\n# from the weights alone?  Energy shares of the mean "
          "Jacobian E[x^T ybar].\n")
    for k in range(n_mlps):
        seed, W = mlp_weights(k, 900_000)
        alpha, G, vy, _ = pilot_stats(W, abs(seed) % 100_000 + 555, npil)
        Amf = meanfield_path(W, alpha)
        uG, sG, _ = np.linalg.svd(G)
        uM = np.linalg.svd(Amf)[0]
        e = sG ** 2 / np.sum(sG ** 2)
        f1 = np.sum(sG ** 2) / np.sum(vy)
        print(f"=== MLP {seed}   f_1 = {f1:.4f}   Var tot {np.sum(vy):.4f} ===")
        print("  E[x^T ybar] energy: " + "  ".join(
            f"top{m}={np.sum(e[:m]):.4f}" for m in (1, 2, 4, 8, 16, 32)))
        for m in (1, 2, 4, 8, 16, 32):
            c = uG[:, :m].T @ uM[:, :m]
            print(f"  mean-field frame recovers top-{m:<2d} jac subspace: "
                  f"captured {np.sum(c ** 2) / m:.4f}"
                  + (f"   |cos| top1 {abs(c[0, 0]):.4f}" if m == 1 else ""))
        print()


# ---------------------------------------------------------------------------
# mode: span   -- the R^2(p, d) surface
# ---------------------------------------------------------------------------
def mode_span(n_mlps, n_samples, npil, dirname, M, seed0):
    n = WIDTH
    print("# R^2 of each dictionary against relu(z^32_j), total-variance "
          "weighted.\n# Gram is ANALYTIC (Mehler); only Cov(feature, y) is "
          "estimated.  Coefficients\n# are fitted on one half of the sample "
          "and the explained variance read on the\n# other, both ways, best "
          "over a ridge grid -- so no p/N in-sample inflation.\n"
          f"# adapted frame = '{dirname}', M = {M}\n")
    rows: dict[str, list[float]] = {}
    rowp: dict[str, list[float]] = {}
    psize: dict[str, int] = {}
    PN_D = 8
    for k in range(n_mlps):
        seed, W = mlp_weights(k, seed0)
        sd = abs(seed) % 100_000
        alpha, G0, _, _ = pilot_stats(W, sd + 555, npil)
        A = direction_sets(W, alpha, G0, M, sd)[dirname]
        w1 = W[0]
        Wn = (w1 / np.sqrt(np.sum(w1 * w1, axis=0))).astype(np.float32)
        des = Design(A, Wn)
        Gm = des.gram()
        # per-output-neuron directions: a different a for each j, from the
        # mean-field path matrix (weights only) and from the sampled mean
        # Jacobian (oracle).  Their Gram is the identity WITHIN a neuron, so
        # the estimation cost is PN_D coefficients per neuron, not 256.
        Amf = meanfield_path(W, alpha)
        PN = {"mf": Amf, "jac": G0}
        PN = {kk: np.ascontiguousarray(
            (v / np.maximum(np.linalg.norm(v, axis=0), 1e-30)).astype(
                np.float32)) for kk, v in PN.items()}
        pnc = {kk: [np.zeros((PN_D, n)), np.zeros((PN_D, n))] for kk in PN}
        HA, HB = Half(des.p, n), Half(des.p, n)
        rng = np.random.default_rng(sd + 12345)
        t0, done, flip = time.time(), 0, 0
        while done < n_samples:
            B = min(CHUNK, n_samples - done)
            x = rng.standard_normal((B, n), dtype=np.float32)
            h = x
            for w in W:
                h = np.maximum(h @ w, 0.0)
            (HA if flip == 0 else HB).add(des.build(x), h)
            for kk, Ad in PN.items():
                Hp = herm_table(x @ Ad, PN_D)
                pnc[kk][flip] += np.einsum("dbj,bj->dj", Hp, h,
                                           dtype=np.float64)
            flip ^= 1
            done += B
        cA, cB = HA.cov(), HB.cov()
        vy = 0.5 * ((HA.sy2 / HA.N - (HA.sy / HA.N) ** 2)
                    + (HB.sy2 / HB.N - (HB.sy / HB.N) ** 2))
        T = float(np.sum(vy))
        dt = time.time() - t0

        bl = des.blocks
        rng_ = np.arange
        co = {d: rng_(*bl[f"CO{d}"]) for d in range(2, COORD_DMAX + 1)}
        in1 = rng_(*bl["IN1"])
        tens = des.tens

        def tsel(pred):
            return np.array([des.tens_off + i for i, a in enumerate(tens)
                             if pred(a)], dtype=int)

        def deg(a):
            return sum(kk for _, kk in a)

        def top(a):
            return max(r for r, _ in a)

        dicts: dict[str, np.ndarray] = {}
        # --- coordinate (layer-1) family, the shipped basis ---------------
        cum = in1
        dicts["coord d<=1  (H1, shipped)"] = cum
        for d in range(2, COORD_DMAX + 1):
            cum = np.concatenate([cum, co[d]])
            nm = "coord d<=2  (H1+H2, SHIPPED)" if d == 2 else f"coord d<={d}"
            dicts[nm] = cum.copy()
        for d in range(2, COORD_DMAX + 1):
            dicts[f"  coord deg {d} alone"] = co[d]
        # --- adapted, pure powers only ------------------------------------
        for m in (1, 2, 4, 8, 16, 24):
            if m > M:
                continue
            for dm in (1, 2, 4, 8, 16):
                if dm > PURE_DMAX:
                    continue
                s = tsel(lambda a, m=m, dm=dm: len(a) == 1 and a[0][0] < m
                         and a[0][1] <= dm)
                if len(s):
                    dicts[f"adapt pure m={m:<2d} d<={dm:<2d}"] = s
        # --- adapted, full tensor products --------------------------------
        for (dd, mm) in ((2, 4), (2, 8), (2, 16), (3, 8), (4, 5)):
            s = tsel(lambda a, dd=dd, mm=mm: deg(a) <= dd and top(a) < mm)
            dicts[f"adapt tensor m={mm:<2d} deg<={dd}"] = s
        dicts["adapt ALL"] = rng_(des.tens_off, des.p)
        # --- unions --------------------------------------------------------
        ship = np.concatenate([in1, co[2]])
        dicts["SHIP + adapt ALL"] = np.concatenate([ship, dicts["adapt ALL"]])
        dicts["SHIP + adapt tensor m=16 deg<=2"] = np.concatenate(
            [ship, dicts["adapt tensor m=16 deg<=2"]])
        dicts["SHIP + coord d<=4"] = np.concatenate(
            [in1, co[2], co[3], co[4]])
        dicts["EVERYTHING"] = rng_(0, des.p)

        print(f"=== local MLP {seed}   {n_samples:,} samples   {dt:.0f}s   "
              f"p_total {des.p}   Var tot {T:.4f} ===")
        hdr = (f"  {'dictionary':<34} {'p':>5} {'p/N@8500':>9} "
               f"{'R2_pop':>9} {'R2_fit':>9} {'1/(1-R2)':>9} "
               f"{'eff R2':>8} {'eff x':>7}")
        print(hdr)
        for nm, ii in dicts.items():
            pop, v = r2_holdout(Gm, cA, cB, T, ii)
            eff = pop - 2.0 * len(ii) / 8500.0
            vv, ee = min(pop, 0.999), min(eff, 0.999)
            print(f"  {nm:<34} {len(ii):5d} {len(ii)/8500:9.4f} "
                  f"{pop * 100:8.3f}% {v * 100:8.3f}% {1/(1-vv):9.3f} "
                  f"{ee*100:7.2f}% {1/(1-ee):7.3f}")
            rows.setdefault(nm, []).append(v)
            rowp.setdefault(nm, []).append(pop)
            psize[nm] = len(ii)
        # per-neuron directions: R^2 = sum_j sum_d cA cB / T, Gram = I
        for kk in PN:
            a0, a1 = pnc[kk]
            a0 = a0 / HA.N - np.outer(np.zeros(PN_D), np.zeros(n))
            a0 = pnc[kk][0] / HA.N
            a1 = pnc[kk][1] / HB.N
            cum = np.cumsum(np.sum(a0 * a1, axis=1)) / T
            for d in (1, 2, 3, 4, 6, 8):
                nm = f"per-neuron dir={kk} d<={d}"
                rows.setdefault(nm, []).append(float(cum[d - 1]))
                rowp.setdefault(nm, []).append(float(cum[d - 1]))
                psize[nm] = d
                print(f"  {nm:<34} {d:5d} {d/8500:9.4f} "
                      f"{cum[d-1]*100:8.3f}% {'':>9} "
                      f"{1/(1-min(cum[d-1],0.999)):9.3f} "
                      f"{(cum[d-1]-2*d/8500)*100:7.2f}% "
                      f"{1/(1-min(cum[d-1]-2*d/8500,0.999)):7.3f}")
        print()

    print("=== mean over MLPs ===")
    print(f"  {'dictionary':<34} {'p':>5} {'R2_pop':>9} {'R2_fit':>9} "
          f"{'1/(1-R2)':>9} {'eff R2':>8} {'eff x':>7}")
    best = (None, -9.9)
    for nm in rows:
        v = float(np.mean(rows[nm]))
        pop = float(np.mean(rowp[nm]))
        eff = pop - 2.0 * psize[nm] / 8500.0
        vv, ee = min(pop, 0.999), min(eff, 0.999)
        print(f"  {nm:<34} {psize[nm]:5d} {pop * 100:8.3f}% {v * 100:8.3f}% "
              f"{1/(1-vv):9.3f} {ee*100:7.2f}% {1/(1-ee):7.3f}")
        if eff > best[1] and not nm.startswith("  "):
            best = (nm, eff)
    ship = float(np.mean(rowp["coord d<=2  (H1+H2, SHIPPED)"]))
    print("\nBAR (fixed before the run): held-out R^2 > 0.75 (> 4x) at "
          "p/N < 0.15, i.e. p < 1275.")
    ok = [nm for nm in rows
          if float(np.mean(rowp[nm])) > 0.75 and psize[nm] < 1275]
    print(f"  shipped H1+H2                     {ship * 100:7.3f}%  "
          f"(p = 512)")
    print(f"  best effective-R2 dictionary      {best[0]}  "
          f"eff {best[1]*100:.2f}%  -> {1/(1-min(best[1],0.999)):.3f}x")
    print(f"  VERDICT: {'PASS ' + str(ok) if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# The degree-2 reachability spectrum -- an EXACT rank bound
# ---------------------------------------------------------------------------
def degree2_rank(W, seed, B, T):
    """Spectrum of ``P = sum_j S_j^2``, ``S_j = E[ybar_j (x x' - I)]``.

    ``h_2(<a,x>)`` is the unit RANK-ONE symmetric tensor ``a (x) a`` of the
    degree-2 chaos, and a dictionary on an ``m``-dimensional direction
    subspace ``A`` (with all its cross products) spans exactly
    ``Sym^2(A)``.  Its capture of the degree-2 chaos is
    ``(1/2) sum_j ||Pi_A S_j Pi_A||_F^2 <= (1/2) sum_j ||Pi_A S_j||_F^2``,
    whose maximum over all ``m``-dimensional ``A`` is
    ``(1/2) sum_{k<=m} lambda_k(P)``.  So the eigenvalue profile of the
    256x256 matrix ``P`` is an EXACT ceiling on what any set of ``m``
    directions can reach at degree 2, over every choice of directions at once.

    ``P`` is estimated from a cross-half pair block, using
    ``(x x' - I)(x' x'' - I) = <x,x'> x x'' - x x' - x' x'' + I``, so it is
    unbiased -- a same-half estimate is inflated by the noise of 65,536
    entries.  ``tr(P) = 2 sum_j ||T_j^(2)||^2 = 2 f_2 sum_j Var(y_j)``, which
    checks against the OU degree spectrum from ``--mode chaos``.
    """
    n = W[0].shape[0]
    rng = np.random.default_rng(seed)
    XA = rng.standard_normal((B, n), dtype=np.float32)
    XB = rng.standard_normal((B, n), dtype=np.float32)
    YA = fwd(W, XA).astype(np.float64)
    YB = fwd(W, XB).astype(np.float64)
    YA -= YA.mean(0)
    YB -= YB.mean(0)
    xa, xb = XA.astype(np.float64), XB.astype(np.float64)
    Wm = YA @ YB.T
    U = xa @ xb.T
    P = xa.T @ ((Wm * U) @ xb)
    P -= xa.T @ (Wm.sum(1)[:, None] * xa)
    P -= xb.T @ (Wm.sum(0)[:, None] * xb)
    P += float(Wm.sum()) * np.eye(n)
    P /= B * B
    P = 0.5 * (P + P.T)
    ev = np.sort(np.linalg.eigvalsh(P))[::-1]
    f2 = 0.5 * float(np.sum(ev)) / T
    pos = np.maximum(ev, 0.0)
    cum = np.cumsum(pos) / np.sum(pos)
    return np.concatenate([[f2], cum[[0, 1, 3, 7, 15, 31, 63, 127, 255]]])


# ---------------------------------------------------------------------------
# mode: ceiling
# ---------------------------------------------------------------------------
def mode_ceiling(n_mlps, n_samples, npil, M, nbin, seed0):
    """Degree-free ceilings.

    (1) **The family ceiling.**  The closed linear span of
    ``{prod_r He_{d_r}(<a_r, x>)}`` over all degrees and all products is
    exactly ``L^2(sigma(A^T x))``, so for ANY such dictionary

        R^2  <=  sum_j Var(E[y_j | A^T x]) / sum_j Var(y_j)

    with no truncation in degree anywhere.  Measured by binning, with the
    per-bin means taken from independent halves so the estimate is unbiased
    (a same-half bin variance is inflated by nbin/N).

    (2) **The basis-size ceiling.**  For ANY dictionary of p functions
    whatsoever, the best achievable is the top-p eigenvalues of the 256x256
    output covariance ``Cov(y)``, because the p-dimensional subspace of
    ``L^2_0`` maximising ``sum_j ||Pi ybar_j||^2`` is spanned by the top-p
    eigenfunctions of ``sum_j ybar_j (x) ybar_j``, whose non-zero spectrum is
    that of ``Cov(y)``.  This is dictionary-free: no Hermite, no network.
    """
    n = WIDTH
    print("# Degree-free ceilings on the whole Hermite family.\n")
    agg = {}
    for k in range(n_mlps):
        seed, W = mlp_weights(k, seed0)
        sd = abs(seed) % 100_000
        alpha, G0, _, _ = pilot_stats(W, sd + 555, npil)
        DS = direction_sets(W, alpha, G0, M, sd)
        cands = {"jac": DS["jac"], "mf": DS["mf"], "rand": DS["rand"]}
        # binning accumulators, per candidate frame and per m
        MS = (1, 2, 3)
        binsz = {1: nbin, 2: int(round(nbin ** 0.55)), 3: int(round(nbin ** 0.4))}
        acc = {(cn, m): [np.zeros((binsz[m] ** m, n)) for _ in range(2)]
               for cn in cands for m in MS}
        cnt = {(cn, m): [np.zeros(binsz[m] ** m) for _ in range(2)]
               for cn in cands for m in MS}
        sy = [np.zeros(n), np.zeros(n)]
        sy2 = [np.zeros(n), np.zeros(n)]
        gram = [np.zeros((n, n)), np.zeros((n, n))]
        nn = [0, 0]
        rng = np.random.default_rng(sd + 12345)
        done, flip = 0, 0
        t0 = time.time()
        while done < n_samples:
            B = min(CHUNK, n_samples - done)
            x = rng.standard_normal((B, n), dtype=np.float32)
            h = x
            for w in W:
                h = np.maximum(h @ w, 0.0)
            y = h.astype(np.float64)
            for cn, A in cands.items():
                s = x @ A[:, :max(MS)]
                for m in MS:
                    q = binsz[m]
                    # equiprobable bins of the standard normal
                    u = np.clip(((norm_cdf(s[:, :m].astype(np.float64))
                                  * q).astype(np.int64)), 0, q - 1)
                    idx = np.zeros(B, dtype=np.int64)
                    for r in range(m):
                        idx = idx * q + u[:, r]
                    np.add.at(acc[(cn, m)][flip], idx, y)
                    np.add.at(cnt[(cn, m)][flip], idx, 1.0)
            sy[flip] += y.sum(0)
            sy2[flip] += np.einsum("ij,ij->j", y, y)
            gram[flip] += y.T @ y
            nn[flip] += B
            flip ^= 1
            done += B
        vy = 0.5 * sum(sy2[f] / nn[f] - (sy[f] / nn[f]) ** 2 for f in (0, 1))
        T = float(np.sum(vy))
        print(f"=== local MLP {seed}  {n_samples:,} samples  "
              f"{time.time()-t0:.0f}s  Var tot {T:.4f} ===")
        for cn in cands:
            out = []
            for m in MS:
                a0, a1 = acc[(cn, m)]
                c0, c1 = cnt[(cn, m)]
                live = (c0 > 0) & (c1 > 0)
                m0 = a0[live] / c0[live, None] - (sy[0] / nn[0])
                m1 = a1[live] / c1[live, None] - (sy[1] / nn[1])
                wgt = (c0[live] + c1[live]) / (nn[0] + nn[1])
                v = float(np.sum(wgt[:, None] * m0 * m1))
                out.append((m, binsz[m] ** m, v / T))
            print("  " + f"{cn:<5} " + "   ".join(
                f"m={m}: R2<= {100*v:6.2f}% ({b} bins)" for m, b, v in out))
            agg.setdefault(cn, []).append([v for _, _, v in out])
        # basis-size ceiling from Cov(y)
        C = 0.5 * sum(gram[f] / nn[f] - np.outer(sy[f] / nn[f], sy[f] / nn[f])
                      for f in (0, 1))
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        cum = np.cumsum(ev) / np.sum(ev)
        print("  ANY dictionary of p features (top-p eigenvalues of Cov(y)): "
              + "  ".join(f"p={p}: {100*cum[p-1]:.2f}%"
                          for p in (1, 2, 4, 8, 16, 32, 64)))
        agg.setdefault("_covy", []).append(cum[[0, 1, 3, 7, 15, 31, 63]])
        agg.setdefault("_rank2", []).append(
            degree2_rank(W, sd + 777, 4096, T))
        print()
    print("=== mean over MLPs ===")
    for cn, vs in agg.items():
        v = np.mean(vs, axis=0)
        if cn == "_covy":
            print("  ANY p features <= top-p of Cov(y):  " + "  ".join(
                f"p={p}: {100*x:.2f}%"
                for p, x in zip((1, 2, 4, 8, 16, 32, 64), v)))
        elif cn == "_rank2":
            print(f"  DEGREE-2 rank ceiling (f_2 = {100*v[0]:.2f}% of Var): "
                  "fraction of f_2 reachable by an m-dim direction set")
            print("    " + "  ".join(
                f"m={m}: {100*x:.1f}%"
                for m, x in zip((1, 2, 4, 8, 16, 32, 64, 128, 256), v[1:])))
        else:
            print(f"  {cn:<5} " + "   ".join(
                f"m={m}: {100*x:6.2f}%" for m, x in zip((1, 2, 3), v)))


# ---------------------------------------------------------------------------
# mode: chaos  -- the Hermite DEGREE spectrum of the target
# ---------------------------------------------------------------------------
#: Ornstein-Uhlenbeck couplings.  Small ones resolve the low degrees (the
#: contamination from degree > D at coupling t is at most t^(D+1)); ones near 1
#: give the mean degree as (1 - C(t))/(1 - t).
T_LOW = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
T_HIGH = (0.70, 0.80, 0.90, 0.95, 0.98)
#: Highest degree carried in the non-negative inversion of ``C(t)``.
CHAOS_DMAX = 80


def nnls_spectrum(ts, C, dmax=CHAOS_DMAX, iters=300_000):
    """Non-negative ``f_d >= 0`` with ``sum_d f_d t^d = C(t)`` and
    ``sum_d f_d = 1`` (which is ``C(1) = 1``, exact by definition).

    Inverting a Hausdorff moment problem is ill-posed, so individual ``f_d``
    at large ``d`` are not identifiable; the CUMULATIVES ``sum_{d<=D} f_d``,
    which are the only thing the barrier needs, are.  Multiplicative updates
    (V >= 0, C >= 0) keep the iterate non-negative without a projection.
    """
    tt = np.concatenate([np.asarray(ts, dtype=np.float64), [1.0]])
    y = np.concatenate([np.asarray(C, dtype=np.float64), [1.0]])
    w = np.ones_like(y)
    w[-1] = 20.0                      # C(1) = 1 is exact, weight it hard
    V = tt[:, None] ** np.arange(1, dmax + 1)[None, :]
    Vw = V * w[:, None]
    yw = y * w
    f = np.full(dmax, 1.0 / dmax)
    num = Vw.T @ yw
    A = Vw.T @ Vw
    for _ in range(iters):
        f = f * num / np.maximum(A @ f, 1e-300)
    return f


def hard_upper(ts, C, D):
    """Assumption-free upper bound on ``sum_{d<=D} f_d``.

    For any ``t`` in (0, 1] and any ``d <= D``, ``t^d >= t^D``, so
    ``C(t) >= sum_{d<=D} f_d t^d >= t^D sum_{d<=D} f_d``.  Minimising over the
    measured grid needs no model of the tail at all.
    """
    return min(float(c) / (t ** D) for t, c in zip(ts, C))


def fwd(W, x):
    h = x
    for w in W:
        h = np.maximum(h @ w, 0.0)
    return h


def mode_chaos(n_mlps, n_samples, seed0):
    """The exact Hermite degree spectrum ``f_d`` of ``relu(z^32_j)``.

    The Ornstein-Uhlenbeck / Mehler semigroup multiplies the degree-``d``
    chaos by ``t^d``: with ``x_t = t x + sqrt(1-t^2) xi`` and ``xi`` an
    independent standard normal,

        C(t) := sum_j Cov(y_j(x), y_j(x_t)) / sum_j Var(y_j)
              = sum_{d>=1} f_d t^d

    where ``f_d`` is the share of ``Var(y)`` carried by the degree-``d`` Wiener
    chaos.  This is the DEGREE decomposition, not the coordinate-subset ANOVA
    decomposition that ``docs/floor_theorem.md`` measures -- ``He_2(x_1)`` has
    ANOVA order 1 and degree 2 -- and it is the one that bounds a Hermite
    dictionary.  It costs one extra forward pass per coupling and its variance
    is ``O(Var(y)^2 / N)``: no curse of dimensionality, unlike estimating the
    degree-``d`` coefficient tensor itself.

    ``C(1) = 1`` identically, so ``1 - sum_{d<=D} f_d`` is the mass strictly
    above degree ``D``, and ``(1 - C(t))/(1 - t) -> sum_d d f_d`` as ``t -> 1``.
    """
    n = WIDTH
    ts = T_LOW + T_HIGH
    print("# Hermite DEGREE spectrum of relu(z^32) by the OU/Mehler "
          "semigroup.\n# C(t) = sum_j Cov(y_j(x), y_j(x_t)) / sum_j Var(y_j) "
          "= sum_{d>=1} f_d t^d.\n")
    agg = []
    for k in range(n_mlps):
        seed, W = mlp_weights(k, seed0)
        sd = abs(seed) % 100_000
        sxy = np.zeros((len(ts), n))
        s1 = np.zeros(n)
        s2 = np.zeros(n)
        st = np.zeros((len(ts), n))
        NN = 0
        rng = np.random.default_rng(sd + 4242)
        t0, done = time.time(), 0
        while done < n_samples:
            B = min(CHUNK, n_samples - done)
            x = rng.standard_normal((B, n), dtype=np.float32)
            xi = rng.standard_normal((B, n), dtype=np.float32)
            y0 = fwd(W, x).astype(np.float64)
            s1 += y0.sum(0)
            s2 += np.einsum("ij,ij->j", y0, y0)
            for a, t in enumerate(ts):
                xt = (np.float32(t) * x
                      + np.float32(math.sqrt(1.0 - t * t)) * xi)
                yt = fwd(W, xt).astype(np.float64)
                sxy[a] += np.einsum("ij,ij->j", y0, yt)
                st[a] += yt.sum(0)
            NN += B
            done += B
        m0 = s1 / NN
        v0 = s2 / NN - m0 * m0
        T = float(np.sum(v0))
        C = np.array([float(np.sum(sxy[a] / NN - m0 * (st[a] / NN))) / T
                      for a in range(len(ts))])
        f = nnls_spectrum(ts, C)
        cum = np.cumsum(f)
        md = [(1.0 - c) / (1.0 - t) for t, c in zip(ts, C) if t >= 0.7]
        print(f"=== local MLP {seed}  {n_samples:,} samples  "
              f"{time.time()-t0:.0f}s  Var tot {T:.4f} ===")
        print("  C(t): " + "  ".join(f"{t:.2f}->{c:.4f}" for t, c in zip(ts, C)))
        print("  f_d : " + "  ".join(f"d={d}: {100*f[d-1]:5.2f}%"
                                     for d in range(1, 9)))
        print("  cumulative sum_{d<=D} f_d      : " + "  ".join(
            f"D={D}: {100*cum[D-1]:6.2f}%" for D in (1, 2, 3, 4, 6, 8, 12, 16)))
        print("  assumption-free UPPER bound     : " + "  ".join(
            f"D={D}: {100*min(hard_upper(ts, C, D),1.0):6.2f}%"
            for D in (1, 2, 3, 4, 6, 8, 12, 16)))
        print("  mean degree (1-C(t))/(1-t): " + "  ".join(
            f"t={t}: {v:.2f}" for t, v in zip(T_HIGH, md))
            + f"   | sum_d d f_d = {float(np.sum(f*np.arange(1,len(f)+1))):.2f}")
        agg.append(np.concatenate([f, C]))
        print()
    a = np.mean(agg, axis=0)
    f, C = a[:CHAOS_DMAX], a[CHAOS_DMAX:]
    cum = np.cumsum(f)
    print("=== mean over MLPs ===")
    print("  C(t): " + "  ".join(f"{t:.2f}->{c:.4f}" for t, c in zip(ts, C)))
    print("  f_d : " + "  ".join(f"d={d}: {100*f[d-1]:5.2f}%"
                                 for d in range(1, 9)))
    print("  cumulative sum_{d<=D} f_d, and the barrier 1/(1 - sum):")
    for D in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        ub = min(hard_upper(ts, C, D), 1.0)
        print(f"    D={D:<3d} fitted {100*cum[D-1]:6.2f}%  -> "
              f"{1/max(1-cum[D-1],1e-9):6.2f}x     assumption-free upper "
              f"{100*ub:6.2f}%  -> {1/max(1-ub,1e-9):7.2f}x")
    print(f"  mean Hermite degree sum_d d f_d = "
          f"{float(np.sum(f*np.arange(1,len(f)+1))):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("dirs", "span", "ceiling", "chaos"))
    ap.add_argument("--n-mlps", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--n-pilot", type=int, default=100_000)
    ap.add_argument("--dirs", default="mf")
    ap.add_argument("--n-dirs", type=int, default=24)
    ap.add_argument("--n-bins", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--official", action="store_true",
                    help="read weights from the held-out official suite's "
                         "seeds.  CONFIRMATION ONLY -- every dictionary, "
                         "degree and direction rule is fixed beforehand and "
                         "nothing is fitted or selected from the result.")
    a = ap.parse_args()
    if a.official:
        _SOURCE["mode"] = "official"
        _SOURCE["seeds"] = load_official_seeds()
        print("# WEIGHTS: official suite (seed protocol 3.0), "
              f"{a.n_mlps} of {len(_SOURCE['seeds'])} MLPs.  "
              "Confirmation only: nothing is fitted or selected here.\n")
    if a.mode == "dirs":
        mode_dirs(a.n_mlps, a.n_pilot)
    elif a.mode == "span":
        mode_span(a.n_mlps, a.n_samples, a.n_pilot, a.dirs, a.n_dirs, a.seed0)
    elif a.mode == "chaos":
        mode_chaos(a.n_mlps, a.n_samples, a.seed0)
    else:
        mode_ceiling(a.n_mlps, a.n_samples, a.n_pilot, a.n_dirs, a.n_bins,
                     a.seed0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
