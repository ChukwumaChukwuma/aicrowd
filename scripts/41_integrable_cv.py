#!/usr/bin/env python
"""Expressive AND analytically integrable control variates: the trade curve.

The question
------------
``docs/hermite_rank_ceiling.md`` sec 5.3 records a dictionary-free bound: the
top-8 eigenfunctions of ``Cov(y)`` explain 90.1% of ``sum_j Var(y_j)``, against
the 39.2% the shipped layer-1 Hermite basis reaches.  That looks like a 10x
lever on ``v_eff``, hence on ``adjusted = v_eff c / B``.

A control variate is only usable if ``E[g]`` is known EXACTLY:

    mu_hat = E[g]_exact + (1/N) sum_s [ f(x_s) - g(x_s) ]

is unbiased for any ``g``, and a wrong claimed mean enters the score as
``0.1 b^2``.  So the question this script answers is whether expressiveness and
exact integrability can be had at the same time.

Modes
-----
``--mode eig``     What the top eigenfunctions ARE.  They are, by construction,
                   linear combinations of the centred outputs themselves; this
                   mode measures their Hermite-degree spectrum exactly at
                   degrees 1 and 2 (closed forms, cross-half unbiased) and
                   through the OU/Mehler semigroup at every degree, and their
                   span against the shipped dictionary.
``--mode ladder``  The trade curve.  For each layer ``L``, the held-out ``R^2``
                   of a linear control variate in the layer-``L`` activations,
                   and the BIAS its analytically propagated mean carries.
                   ``adjusted(L) = 0.1 b(L)^2 + (1 - R(L)) V0`` is the real
                   objective and this mode is its argmin.
``--mode quad``    The exactly-integrable frontier: degree-2 polynomials in the
                   layer-1 activations.  ``E[relu(z^1_i) relu(z^1_j)]`` is the
                   arc-cosine kernel, exact in closed form, and ``z^2 = h^1
                   W^2`` is already computed by the forward pass -- so
                   ``{z^2_i z^2_j}`` is a zero-cost, exactly-mean-known,
                   genuinely-not-rank-one dictionary.  How far does it reach?
``--mode anti``    Symmetry-derived variates: ``x -> -x`` (kills every odd
                   Hermite degree exactly) and the network's exact positive
                   homogeneity ``y(x) = ||x|| y(x/||x||)`` (Rao-Blackwellises
                   the radius against an exactly known ``E[chi_n]``).

All MLPs are LOCAL (``make_mlp``, seed base 950000) unless ``--official`` is
given; the official suite is opened only to confirm a frozen conclusion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import DEPTH, WIDTH  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.official_seeds import make_official_mlp  # noqa: E402
from whestfloor.relu_moments import (  # noqa: E402
    Phi,
    phi,
    relu_cov_mehler,
    relu_mean,
    relu_var,
)

ART = Path(os.environ.get("WHEST_ARTIFACTS", "artifacts"))
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
CHUNK = 2048


def mlp_weights(k: int, base: int, official: bool = False):
    if official:
        from whestfloor.official_seeds import OFFICIAL_SEEDS  # noqa: PLC0415
        seed = OFFICIAL_SEEDS[k]
        return seed, make_official_mlp(WIDTH, DEPTH, seed)
    seed = base + k
    return seed, make_mlp(WIDTH, DEPTH, seed)


# ---------------------------------------------------------------------------
# Analytic moment propagation.  Layers 1 and 2 are EXACT.
# ---------------------------------------------------------------------------
def analytic_moments(W, kmax: int = 8):
    """Gaussian-closure means and covariances for every layer.

    Returns ``(mz, Cz, mh, Ch)``, lists of length ``depth``:
    ``mz[l]``/``Cz[l]`` the pre-activation mean and covariance of layer ``l``
    (0-indexed), ``mh[l]``/``Ch[l]`` the post-ReLU ones.

    Layer 0 (``z^1 = x W^1``) is EXACT: ``x ~ N(0, I)`` exactly, so
    ``mz = 0`` and ``Cz = W^1' W^1`` with no approximation.  ``mh[0]`` and
    ``Ch[0]`` are then exact too -- ``E[relu]`` in closed form and the
    arc-cosine kernel for the pair moments -- which makes ``mz[1] = W^2' mh[0]``
    and ``Cz[1] = W^2' Ch[0] W^2`` **exact as well**.  From layer 2 onwards the
    closure assumes ``z^l`` jointly Gaussian, which it is not, and the error
    this injects is exactly what ``--mode ladder`` prices.
    """
    mz, Cz, mh, Ch = [], [], [], []
    m = np.zeros(W[0].shape[1], dtype=np.float64)
    C = (W[0].astype(np.float64).T @ W[0].astype(np.float64))
    for l in range(len(W)):
        C = 0.5 * (C + C.T)
        s = np.sqrt(np.maximum(np.diag(C), 1e-30))
        mz.append(m.copy())
        Cz.append(C.copy())
        hm = relu_mean(m, s)
        hC = relu_cov_mehler(C, m, s, kmax=kmax)
        np.fill_diagonal(hC, relu_var(m, s))
        mh.append(hm)
        Ch.append(hC)
        if l + 1 < len(W):
            Wl = W[l + 1].astype(np.float64)
            m = Wl.T @ hm
            C = Wl.T @ hC @ Wl
    return mz, Cz, mh, Ch


# ---------------------------------------------------------------------------
# Forward pass that hands back selected internals.
# ---------------------------------------------------------------------------
def forward(W, x, want=()):
    """Return ``(y, {l: h^l}, {('z', l): z^l})`` for the requested layers."""
    out_h, out_z = {}, {}
    h = x
    for l, w in enumerate(W):
        z = h @ w
        if ("z", l) in want:
            out_z[("z", l)] = z
        h = np.maximum(z, 0.0)
        if l in want:
            out_h[l] = h
    return h, out_h, out_z


# ---------------------------------------------------------------------------
# Streaming cross-moment accumulator, split into two independent halves.
# ---------------------------------------------------------------------------
class Halves:
    """Accumulate ``D'D``, ``D'Y``, ``sum D``, ``sum Y``, ``sum Y*Y`` per half."""

    def __init__(self, p, q):
        self.p, self.q = p, q
        self.G = [np.zeros((p, p)) for _ in range(2)]
        self.DY = [np.zeros((p, q)) for _ in range(2)]
        self.sD = [np.zeros(p) for _ in range(2)]
        self.sY = [np.zeros(q) for _ in range(2)]
        self.sY2 = [np.zeros(q) for _ in range(2)]
        self.n = [0, 0]

    def add(self, half, D, Y):
        D = np.asarray(D, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self.G[half] += D.T @ D
        self.DY[half] += D.T @ Y
        self.sD[half] += D.sum(0)
        self.sY[half] += Y.sum(0)
        self.sY2[half] += np.einsum("ij,ij->j", Y, Y)
        self.n[half] += len(D)

    def cov(self, half):
        n = self.n[half]
        mD, mY = self.sD[half] / n, self.sY[half] / n
        G = self.G[half] / n - np.outer(mD, mD)
        c = self.DY[half] / n - np.outer(mD, mY)
        vy = self.sY2[half] / n - mY * mY
        return G, c, vy, mD, mY


RIDGE_GRID = (1e-8, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def span_r2(H: Halves, idx, n_ship: int | None = None):
    """``(R2_pop, R2_holdout, best_lambda, coefficients)`` for a sub-dictionary.

    ``R2_pop`` is the cross-half quadratic form ``c_A' G^-1 c_B / T``: both
    cross-moment vectors come from independent halves, so the product has no
    ``p/N`` inflation and estimates the dictionary's POPULATION approximation
    power.  ``R2_holdout`` fits ridge coefficients on one half and reads the
    explained variance on the other, both ways -- what the dictionary actually
    delivers at ``N = n_half`` samples.
    """
    idx = np.asarray(idx, dtype=np.int64)
    p = len(idx)
    GA, cA, vyA, _, _ = H.cov(0)
    GB, cB, vyB, _, _ = H.cov(1)
    GA, cA = GA[np.ix_(idx, idx)], cA[idx]
    GB, cB = GB[np.ix_(idx, idx)], cB[idx]
    T = 0.5 * (float(vyA.sum()) + float(vyB.sum()))
    G = 0.5 * (GA + GB)
    tau = float(np.trace(G)) / p
    ev, V = np.linalg.eigh(G)
    keep = ev > 1e-10 * max(float(ev.max()), 1e-300)
    hA, hB = V.T @ cA, V.T @ cB
    pop = float(np.sum(hA[keep] * hB[keep] / ev[keep, None])) / T

    best, best_lam, best_b = -np.inf, None, None
    for lam in RIDGE_GRID:
        dn = np.maximum(ev, 0.0) + lam * tau
        acc, bs = 0.0, []
        for hs, hd, Gd in ((hA, hB, GB), (hB, hA, GA)):
            b = hs / dn[:, None]
            bb = V @ b
            bs.append(bb)
            acc += float(2.0 * np.sum(bb * (V @ hd)) - np.sum(bb * (Gd @ bb)))
        val = 0.5 * acc / T
        if val > best:
            best, best_lam, best_b = val, lam, 0.5 * (bs[0] + bs[1])
    eff = None
    if n_ship:
        eff = pop - p / float(n_ship)
    return pop, best, best_lam, best_b, eff, T


# ---------------------------------------------------------------------------
# mode: eig
# ---------------------------------------------------------------------------
def mode_eig(n_mlps, n_samples, seed0, official):
    n = WIDTH
    print("# What the top eigenfunctions of Cov(y) ARE.\n"
          "# The operator sum_j ybar_j (x) ybar_j maps every function into\n"
          "# span{ybar_1..ybar_256}, so ITS EIGENFUNCTIONS ARE LINEAR\n"
          "# COMBINATIONS OF THE CENTRED OUTPUTS, and their non-zero spectrum\n"
          "# is that of the 256x256 matrix Cov(y).  90.1% at p=8 is therefore\n"
          "# the statement 'Cov(y) has effective rank 8', not the existence of\n"
          "# any dictionary we do not already have.\n")
    ts = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
    rows = []
    for k in range(n_mlps):
        seed, W = mlp_weights(k, seed0, official)
        rng = np.random.default_rng(abs(seed) % 100000 + 7717)
        t0 = time.time()
        # --- streaming pass: Cov(y), and the coupled-y cross products -------
        sy = np.zeros(n)
        syy = np.zeros((n, n))
        sxY = None
        done = 0
        Cnum = {t: np.zeros(n) for t in ts}
        syt = {t: np.zeros(n) for t in ts}
        # We need the eigenvectors before we can project, so do two passes.
        while done < n_samples:
            m = min(CHUNK, n_samples - done)
            x = rng.standard_normal((m, n), dtype=np.float32)
            y, _, _ = forward(W, x)
            y64 = y.astype(np.float64)
            sy += y64.sum(0)
            syy += y64.T @ y64
            done += m
        my = sy / n_samples
        Cy = syy / n_samples - np.outer(my, my)
        ev, V = np.linalg.eigh(Cy)
        order = np.argsort(ev)[::-1]
        ev, V = ev[order], V[:, order]
        tot = float(ev.sum())
        shares = np.cumsum(ev) / tot
        print(f"=== MLP {seed}   sum_j Var(y_j) = {tot:.5f}   "
              f"[{time.time()-t0:.0f}s] ===")
        print("  top-p share of sum_j Var(y_j): " + "  ".join(
            f"p={p}:{shares[p-1]*100:6.2f}%" for p in (1, 2, 4, 8, 16, 32, 64)))

        # --- second pass: degree content of the top eigenfunctions ----------
        K = 8
        Vk = V[:, :K]
        rng = np.random.default_rng(abs(seed) % 100000 + 7717)
        acc = {
            "n": 0,
            "sY": np.zeros(K), "sY2": np.zeros(K),
            "d1": [np.zeros((n, K)) for _ in range(2)],
            "d2": [np.zeros((n, n)) for _ in range(K)],  # half A only
            "d2b": [np.zeros((n, n)) for _ in range(K)],
            "nh": [0, 0],
            "Ct": {t: np.zeros(K) for t in ts},
            "sYt": {t: np.zeros(K) for t in ts},
        }
        done = 0
        while done < n_samples:
            m = min(CHUNK, n_samples - done)
            x = rng.standard_normal((m, n), dtype=np.float32)
            y, _, _ = forward(W, x)
            Y = (y.astype(np.float64) - my) @ Vk           # (m, K)
            half = 0 if (done // CHUNK) % 2 == 0 else 1
            x64 = x.astype(np.float64)
            acc["n"] += m
            acc["sY"] += Y.sum(0)
            acc["sY2"] += np.einsum("ij,ij->j", Y, Y)
            acc["d1"][half] += x64.T @ Y
            acc["nh"][half] += m
            for j in range(K):
                w = Y[:, j]
                M = (x64 * w[:, None]).T @ x64
                M -= np.eye(n) * w.sum()
                (acc["d2"] if half == 0 else acc["d2b"])[j] += M
            # OU couplings share the same x
            for t in ts:
                xi = rng.standard_normal((m, n), dtype=np.float32)
                xt = (np.float32(t) * x
                      + np.float32(math.sqrt(1.0 - t * t)) * xi)
                yt, _, _ = forward(W, xt)
                Yt = (yt.astype(np.float64) - my) @ Vk
                acc["Ct"][t] += np.einsum("ij,ij->j", Y, Yt)
                acc["sYt"][t] += Yt.sum(0)
            done += m
        N = acc["n"]
        varY = acc["sY2"] / N - (acc["sY"] / N) ** 2
        nA, nB = acc["nh"]
        f1 = np.einsum("ik,ik->k", acc["d1"][0] / nA, acc["d1"][1] / nB) / varY
        f2 = np.array([
            0.5 * float(np.sum((acc["d2"][j] / nA) * (acc["d2b"][j] / nB)))
            / varY[j] for j in range(K)])
        print(f"  {'k':>2} {'lam_k/tot':>10} {'Var':>10} "
              f"{'deg1':>8} {'deg2':>8} {'deg<=2':>8}   "
              + "  ".join(f"C({t})" for t in ts))
        for j in range(K):
            ctj = [acc["Ct"][t][j] / N
                   - (acc["sY"][j] / N) * (acc["sYt"][t][j] / N)
                   for t in ts]
            ctj = [c / varY[j] for c in ctj]
            print(f"  {j+1:>2} {ev[j]/tot:10.4f} {varY[j]:10.5f} "
                  f"{f1[j]*100:7.2f}% {f2[j]*100:7.2f}% "
                  f"{(f1[j]+f2[j])*100:7.2f}%   "
                  + "  ".join(f"{c:6.4f}" for c in ctj))
            rows.append({"mlp": int(seed), "k": j + 1,
                         "share": float(ev[j] / tot), "deg1": float(f1[j]),
                         "deg2": float(f2[j]),
                         "C": {str(t): float(c) for t, c in zip(ts, ctj)}})
        # variance-weighted totals over the whole output
        w = ev[:K] / tot
        print(f"  variance-weighted over top-{K}: deg1 "
              f"{float(np.sum(w*f1)/np.sum(w))*100:.2f}%  deg2 "
              f"{float(np.sum(w*f2)/np.sum(w))*100:.2f}%  deg<=2 "
              f"{float(np.sum(w*(f1+f2))/np.sum(w))*100:.2f}%")
        print(f"  assumption-free bound on deg<=D from min_t C(t)/t^D "
              "(top eigenfunction):")
        c1 = [acc["Ct"][t][0] / N - (acc["sY"][0] / N) * (acc["sYt"][t][0] / N)
              for t in ts]
        c1 = [c / varY[0] for c in c1]
        for D in (1, 2, 3, 4):
            b = min(c / t ** D for c, t in zip(c1, ts))
            print(f"    D={D}: deg<=D <= {b*100:6.2f}%")
        print()
    (ART / "cv").mkdir(parents=True, exist_ok=True)
    (ART / "cv" / "eig.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------
# mode: ladder
# ---------------------------------------------------------------------------
def mode_ladder(n_mlps, n_samples, n_ref, seed0, official, layers, n_ship,
                v0=4.06e-07):
    """R^2 of a linear CV in layer-L activations vs the bias of its analytic mean.

    ``v0`` is the shipped ``v_eff c / B`` divided by its own residual share, so
    ``(1 - R2) v0`` is the sampling term of ``adjusted`` a dictionary with that
    ``R2`` would deliver at the shipped operating point.  ``0.1 b^2`` is the
    bias term of the SAME objective (``docs/graded.md`` sec 3).
    """
    n = WIDTH
    print("# The expressiveness / integrability trade, along depth.\n"
          "# Features: hbar^L = relu(z^L) - mtilde^L, 256 of them, mtilde^L\n"
          "# the ANALYTIC (Gaussian-closure) mean.  R^2 is held out; the bias\n"
          "# is c_j'(mtilde^L - m^L_true) with c_j the ridge coefficients that\n"
          "# would actually be used, and m^L_true a large independent MC.\n"
          f"# objective: adjusted(L) = 0.1 b^2 + (1 - R2_eff) V0, "
          f"V0 = {v0:.3e},  N_ship = {n_ship}\n"
          "# shipped basis reaches R2_eff ~ 0.392 -> adjusted 2.47e-07.\n")
    allrows = []
    for k in range(n_mlps):
        seed, W = mlp_weights(k, seed0, official)
        t0 = time.time()
        mz, Cz, mh, Ch = analytic_moments(W)
        print(f"=== MLP {seed}   [analytic moments {time.time()-t0:.1f}s] ===")

        # ---- true layer means, big independent MC -------------------------
        t0 = time.time()
        rng = np.random.default_rng(abs(seed) % 100000 + 4242)
        sh = np.zeros((len(W), n))
        done = 0
        while done < n_ref:
            m = min(4096, n_ref - done)
            x = rng.standard_normal((m, n), dtype=np.float32)
            h = x
            for l, w in enumerate(W):
                h = np.maximum(h @ w, 0.0)
                sh[l] += h.sum(0, dtype=np.float64)
            done += m
        m_true = sh / n_ref
        print(f"  reference means: {n_ref} samples [{time.time()-t0:.0f}s]")

        # ---- one pass, every layer at once --------------------------------
        t0 = time.time()
        rng = np.random.default_rng(abs(seed) % 100000 + 9191)
        Hs = {L: Halves(n, n) for L in layers}
        done = 0
        while done < n_samples:
            m = min(CHUNK, n_samples - done)
            x = rng.standard_normal((m, n), dtype=np.float32)
            keep = {}
            h = x
            for l, w in enumerate(W):
                h = np.maximum(h @ w, 0.0)
                if l + 1 in Hs:
                    keep[l + 1] = h.astype(np.float64)
            y = h.astype(np.float64)
            half = 0 if (done // CHUNK) % 2 == 0 else 1
            for L in layers:
                Hs[L].add(half, keep[L] - mh[L - 1], y)
            done += m
        print(f"  span pass: {n_samples} samples [{time.time()-t0:.0f}s]")
        print(f"  {'L':>3} {'rms(mt-m)':>10} {'/sd(h)':>9} "
              f"{'R2_pop':>8} {'R2_eff':>8} "
              f"{'rms bias':>10} {'0.1b^2':>10} {'(1-R2)V0':>10} "
              f"{'adjusted':>10} {'x ship':>7}")
        for L in layers:
            pop, hold, lam, b, eff, T = span_r2(Hs[L], np.arange(n),
                                                n_ship=n_ship)
            dm = mh[L - 1] - m_true[L - 1]
            sdh = np.sqrt(np.maximum(np.diag(Ch[L - 1]), 1e-30))
            bias = b.T @ dm                       # (n_outputs,)
            b2 = float(np.mean(bias * bias))
            samp = (1.0 - eff) * v0
            adj = 0.1 * b2 + samp
            allrows.append({
                "mlp": int(seed), "L": L, "p": n, "R2_pop": pop,
                "R2_hold": hold, "R2_eff": eff, "rms_bias": math.sqrt(b2),
                "rms_dm": float(np.sqrt(np.mean(dm * dm))),
                "rel_dm": float(np.sqrt(np.mean((dm / sdh) ** 2))),
                "bias_term": 0.1 * b2, "samp_term": samp, "adjusted": adj,
                "T": T, "lam": lam,
            })
            print(f"  {L:>3} {np.sqrt(np.mean(dm*dm)):10.3e} "
                  f"{np.sqrt(np.mean((dm/sdh)**2)):9.2e} "
                  f"{pop*100:7.2f}% {eff*100:7.2f}% "
                  f"{math.sqrt(b2):10.3e} {0.1*b2:10.3e} "
                  f"{samp:10.3e} {adj:10.3e} {2.47e-07/adj:7.3f}")
        print()
    (ART / "cv").mkdir(parents=True, exist_ok=True)
    (ART / "cv" / "ladder.json").write_text(json.dumps(allrows, indent=1))


# ---------------------------------------------------------------------------
# mode: quad
# ---------------------------------------------------------------------------
def build_blocks(W, x, mh0, mz1, Cz1, Cz0, sk1, sk2, kq):
    """Design blocks with EXACTLY known means, from one forward pass.

    * ``t``       : ``z^1_i / sigma_i``          E = 0 exactly (z^1 Gaussian)
    * ``he2``     : ``t_i^2 - 1``                E = 0 exactly
    * ``h1``      : ``relu(z^1_i) - sigma_i/sqrt(2pi)``   E = 0 exactly
    * ``q1``      : ``u_a u_b - Cov``, ``u = z^1 S1``     E = 0 exactly (Wick)
    * ``q2``      : ``v_a v_b - Cov``, ``v = z^2 S2``     E = 0 exactly
                    (arc-cosine kernel gives Cov(h^1) in closed form, hence
                    Cov(z^2) = W^2' Cov(h^1) W^2 exactly)
    """
    n = WIDTH
    z1 = x @ W[0]
    h1 = np.maximum(z1, 0.0)
    z2 = h1 @ W[1]
    sig = np.sqrt(np.maximum(np.diag(Cz0), 1e-30))
    t = z1.astype(np.float64) / sig
    blocks = {
        "t": t,
        "he2": t * t - 1.0,
        "h1": h1.astype(np.float64) - mh0,
    }
    if kq:
        u = z1.astype(np.float64) @ sk1                 # (m, kq)
        v = (z2.astype(np.float64) - mz1) @ sk2
        Cu = sk1.T @ Cz0 @ sk1
        Cv = sk2.T @ Cz1 @ sk2
        iu, ju = np.triu_indices(kq)
        blocks["q1"] = u[:, iu] * u[:, ju] - Cu[iu, ju]
        blocks["q2"] = v[:, iu] * v[:, ju] - Cv[iu, ju]
        blocks["l2"] = v
    h = h1
    for w in W[1:]:
        h = np.maximum(h @ w, 0.0)
    return blocks, h


def mode_quad(n_mlps, n_samples, seed0, official, kq, n_ship, sketch):
    n = WIDTH
    print("# The exactly-integrable frontier: degree-2 polynomials in the\n"
          "# layer-1 ACTIVATIONS.  E[relu(z_i) relu(z_j)] is the arc-cosine\n"
          "# kernel -- closed form, exact -- so Cov(z^2) = W^2' Cov(h^1) W^2\n"
          "# is exact and every feature below has an exactly known mean.\n"
          "# z^2 is already computed by the forward pass, so the marginal cost\n"
          "# of the q2 block is k^2/2 multiplies a sample.\n"
          f"# sketch = {sketch}, k = {kq}, N_ship = {n_ship}\n")
    rows = []
    for kk in range(n_mlps):
        seed, W = mlp_weights(kk, seed0, official)
        t0 = time.time()
        mz, Cz, mh, Ch = analytic_moments(W)
        # sketch directions
        if sketch == "first":
            S1 = np.eye(n)[:, :kq]
            S2 = np.eye(n)[:, :kq]
        elif sketch == "mf":
            # mean-field Jacobian from each layer to the output
            g = [Phi(mz[l] / np.sqrt(np.maximum(np.diag(Cz[l]), 1e-30)))
                 for l in range(len(W))]
            J1 = np.eye(n)
            M = W[-1].astype(np.float64)
            for l in range(len(W) - 1, 0, -1):
                M = (g[l - 1][:, None] * M)
                if l - 1 > 0:
                    M = W[l - 1].astype(np.float64) @ M
                else:
                    J1 = M
            # J1: layer-1 pre-activation -> output, (n, n)
            S1 = np.linalg.svd(J1, full_matrices=False)[0][:, :kq]
            M2 = W[-1].astype(np.float64)
            for l in range(len(W) - 1, 1, -1):
                M2 = W[l - 1].astype(np.float64) @ (g[l - 1][:, None] * M2)
            S2 = np.linalg.svd(g[1][:, None] * M2, full_matrices=False)[0][:, :kq]
        else:
            rg = np.random.default_rng(12345)
            S1 = np.linalg.qr(rg.standard_normal((n, kq)))[0]
            S2 = np.linalg.qr(rg.standard_normal((n, kq)))[0]
        nq = kq * (kq + 1) // 2
        names = ["t", "he2", "h1", "l2", "q1", "q2"]
        widths = {"t": n, "he2": n, "h1": n, "l2": kq, "q1": nq, "q2": nq}
        offs, o = {}, 0
        for nm in names:
            offs[nm] = np.arange(o, o + widths[nm])
            o += widths[nm]
        p = o
        H = Halves(p, n)
        done = 0
        rng = np.random.default_rng(abs(seed) % 100000 + 3131)
        while done < n_samples:
            m = min(CHUNK, n_samples - done)
            x = rng.standard_normal((m, n), dtype=np.float32)
            bl, y = build_blocks(W, x, mh[0], mz[1], Cz[1], Cz[0], S1, S2, kq)
            D = np.concatenate([bl[nm] for nm in names], axis=1)
            half = 0 if (done // CHUNK) % 2 == 0 else 1
            H.add(half, D, y.astype(np.float64))
            done += m
        print(f"=== MLP {seed}   p = {p}  N = {n_samples} "
              f"[{time.time()-t0:.0f}s] ===")
        # sanity: every block's sample mean should be zero to MC error
        _, _, _, mDA, _ = H.cov(0)
        gA, _, vyA, _, _ = H.cov(0)
        sdD = np.sqrt(np.maximum(np.diag(gA), 1e-300))
        zsc = mDA / (sdD / math.sqrt(H.n[0]))
        for nm in names:
            i = offs[nm]
            print(f"    mean-zero check {nm:>4}: max |z| = "
                  f"{np.max(np.abs(zsc[i])):6.2f}  "
                  f"rms |mean|/sd = {np.sqrt(np.mean((mDA[i]/sdD[i])**2)):.2e}")
        dicts = [
            ("SHIP  t+he2", ["t", "he2"]),
            ("h1", ["h1"]),
            ("t", ["t"]),
            ("t+he2+h1", ["t", "he2", "h1"]),
            (f"q1 (deg2 chaos, k={kq})", ["q1"]),
            (f"q2 (deg2 in z^2, k={kq})", ["q2"]),
            (f"SHIP+q1", ["t", "he2", "q1"]),
            (f"SHIP+q2", ["t", "he2", "q2"]),
            (f"SHIP+h1+q2", ["t", "he2", "h1", "q2"]),
            (f"SHIP+h1+q1+q2", ["t", "he2", "h1", "q1", "q2"]),
            ("ALL", names),
        ]
        print(f"  {'dictionary':<28} {'p':>6} {'R2_pop':>9} {'R2_hold':>9} "
              f"{'R2_eff':>9} {'1/(1-eff)':>10}")
        for nm, bl in dicts:
            idx = np.concatenate([offs[b] for b in bl])
            pop, hold, lam, b, eff, T = span_r2(H, idx, n_ship=n_ship)
            print(f"  {nm:<28} {len(idx):>6} {pop*100:8.2f}% {hold*100:8.2f}% "
                  f"{eff*100:8.2f}% {1.0/max(1e-9,1-eff):10.3f}")
            rows.append({"mlp": int(seed), "dict": nm, "p": int(len(idx)),
                         "R2_pop": pop, "R2_hold": hold, "R2_eff": eff,
                         "kq": kq, "sketch": sketch})
        print()
    (ART / "cv").mkdir(parents=True, exist_ok=True)
    (ART / "cv" / f"quad_{sketch}_{kq}.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------
# mode: anti
# ---------------------------------------------------------------------------
def mode_anti(n_mlps, n_samples, seed0, official, n_ship):
    n = WIDTH
    print("# Symmetry-derived variates.  x -> -x is an exact distributional\n"
          "# symmetry, so (y(x)+y(-x))/2 has the same mean and NO odd Hermite\n"
          "# degree at all -- an exactly-mean-known variate with no model.\n"
          "# The network has no biases, so it is also exactly positively\n"
          "# homogeneous: y(x) = ||x|| Y(x/||x||), and E||x|| is known in\n"
          "# closed form, which Rao-Blackwellises the radius exactly.\n")
    # E[chi_n]
    lg = math.lgamma((n + 1) / 2.0) - math.lgamma(n / 2.0)
    Er = math.sqrt(2.0) * math.exp(lg)
    print(f"  E||x|| = {Er:.10f}   (n = {n}, exact)\n")
    rows = []
    for k in range(n_mlps):
        seed, W = mlp_weights(k, seed0, official)
        mz, Cz, mh, Ch = analytic_moments(W)
        sig = np.sqrt(np.maximum(np.diag(Cz[0]), 1e-30))
        rng = np.random.default_rng(abs(seed) % 100000 + 555)
        # accumulate: y+, y-, features for both, and the homogeneity variant
        blocks = ("t", "he2")
        p = 2 * n
        H = {kk: Halves(p, n) for kk in ("plain", "anti")}
        Hh = {kk: Halves(p, n) for kk in ("plain", "anti")}
        done = 0
        while done < n_samples:
            m = min(CHUNK, n_samples - done)
            x = rng.standard_normal((m, n), dtype=np.float32)
            yp, _, _ = forward(W, x)
            ym, _, _ = forward(W, -x)
            t = (x @ W[0]).astype(np.float64) / sig
            Dp = np.concatenate([t, t * t - 1.0], axis=1)
            Dm = np.concatenate([-t, t * t - 1.0], axis=1)
            r = np.linalg.norm(x.astype(np.float64), axis=1)
            half = 0 if (done // CHUNK) % 2 == 0 else 1
            H["plain"].add(half, Dp, yp.astype(np.float64))
            H["anti"].add(half, 0.5 * (Dp + Dm),
                          0.5 * (yp.astype(np.float64) + ym.astype(np.float64)))
            Hh["plain"].add(half, Dp, Er * yp.astype(np.float64) / r[:, None])
            Hh["anti"].add(
                half, 0.5 * (Dp + Dm),
                0.5 * Er * (yp.astype(np.float64) + ym.astype(np.float64))
                / r[:, None])
            done += m
        print(f"=== MLP {seed} ===")
        print(f"  {'scheme':<26} {'var/sample':>12} {'R2':>9} "
              f"{'resid/sample':>13} {'x ship':>8}")
        base = None
        for nm, acc, cost in (("plain MC", H["plain"], 1.0),
                              ("antithetic pair", H["anti"], 2.0),
                              ("homogeneity RB", Hh["plain"], 1.0),
                              ("anti + homog", Hh["anti"], 2.0)):
            pop, hold, lam, b, eff, T = span_r2(acc, np.arange(p),
                                                n_ship=n_ship)
            # residual variance per unit of forward-pass cost
            resid = T * (1.0 - eff) * cost
            if base is None:
                base = resid
            print(f"  {nm:<26} {T*cost:12.6f} {eff*100:8.2f}% "
                  f"{resid:13.6f} {base/resid:8.4f}")
            rows.append({"mlp": int(seed), "scheme": nm, "T": T,
                         "R2_eff": eff, "resid_per_cost": resid,
                         "x_ship": base / resid})
        print()
    (ART / "cv").mkdir(parents=True, exist_ok=True)
    (ART / "cv" / "anti.json").write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("eig", "ladder", "quad", "anti"))
    ap.add_argument("--mlps", type=int, default=3)
    ap.add_argument("--samples", type=int, default=120_000)
    ap.add_argument("--ref-samples", type=int, default=2_000_000)
    ap.add_argument("--seed0", type=int, default=950_000)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--kq", type=int, default=48)
    ap.add_argument("--sketch", default="first",
                    choices=("first", "mf", "rand"))
    ap.add_argument("--n-ship", type=int, default=27_000)
    ap.add_argument("--layers", default="1,2,3,4,6,8,12,16,20,24,28,30,31,32")
    a = ap.parse_args()
    if a.mode == "eig":
        mode_eig(a.mlps, a.samples, a.seed0, a.official)
    elif a.mode == "ladder":
        layers = [int(v) for v in a.layers.split(",")]
        mode_ladder(a.mlps, a.samples, a.ref_samples, a.seed0, a.official,
                    layers, a.n_ship)
    elif a.mode == "quad":
        mode_quad(a.mlps, a.samples, a.seed0, a.official, a.kq, a.n_ship,
                  a.sketch)
    elif a.mode == "anti":
        mode_anti(a.mlps, a.samples, a.seed0, a.official, a.n_ship)


if __name__ == "__main__":
    main()
