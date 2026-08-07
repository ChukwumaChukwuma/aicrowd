#!/usr/bin/env python
"""Stein control variates driven by the network's own gradient.

The construction
----------------
For ``x ~ N(0, I_n)`` and any weakly-differentiable vector field
``phi: R^n -> R^n`` of at most polynomial growth, Gaussian integration by parts
gives Stein's identity::

    E[ div phi(x) ]  =  E[ x . phi(x) ]

which needs no Gaussianity anywhere *inside* the network -- only at the input.
Taking ``phi(x) = psi(x) c`` for a fixed direction ``c`` and a scalar field
``psi``::

    h(x) = c . grad psi(x)  -  (c . x) psi(x)        E[h] = 0  EXACTLY

``psi`` is chosen to be one of the network's own internals -- ``z^32_j``,
``relu(z^32_j)``, a hidden activation ``x^l_i``, or any linear combination of
them -- so ``c . grad psi`` is one **forward-mode tangent pass**,
``v^l = (v^{l-1} W^l) * 1{z^l > 0}`` with ``v^0 = c``, which costs exactly one
forward pass and yields ``c . grad psi`` for *all* output neurons at once.
(Reverse mode would give the whole input gradient of ONE scalar; forward mode
gives one directional derivative of ALL 256 outputs, which is what a
per-output-neuron control variate needs.)

Unlike the shipped layer-1 Hermite family, nothing here is restricted to
order <= 2: ``grad psi`` is piecewise constant and switches on the same
order-15 sign structure as the network itself.

Modes
-----
``--mode verify``
    Step 1, before anything else.  Build ``h`` from several ``psi`` families
    and several ``c``, and check ``mean(h)`` is zero to Monte-Carlo error.  If
    it is not, the tangent pass or the growth condition is wrong and every
    number downstream is worthless.

``--mode span``
    Step 3.  The R^2 of the Stein span against ``relu(z^32_j)``, which is the
    whole answer: the achievable variance reduction is ``1/(1 - R^2)`` and
    ``docs/floor_theorem.md`` caps any ``k <= 2`` dictionary at 1.75x.
    Coefficients are fitted on one half of the sample and the explained
    variance is read on the other, so the reported R^2 carries no ``p/N``
    in-sample inflation (the dictionaries here have p = 256 to 2560).

    Also reports an EXACT, Jacobian-free ceiling for the entire 256-direction
    family built on ``psi = y_j``.  Writing ``a_c^+ = (c.x) - c.grad`` for the
    Hermite raising operator in direction ``c``, the CV is ``h = -a_c^+ psi``
    and, for centred ``psi``,

        Var(h)       = ||a_c psi||^2 + |c|^2 ||psi||^2  >=  |c|^2 Var(psi)
        Cov(h, ybar) = -<a_c ybar_j, psi>

    so with ``psi = ybar_j`` the covariance is ``-c . k_j`` with

        k_j = E[ybar_j grad y_j] = (1/2) E[x ybar_j^2]        (Stein again)

    -- no Jacobian needed -- and hence, over ALL 256 directions at once and
    for every rescaling of ``c`` (including ``c = W^32[:,j]``),

        R^2  <=  |k_j|^2 / Var(y_j)^2 .

All MLPs used here are LOCAL (``make_mlp``, seed base 900000), disjoint from
the official suite, which this script never opens.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import DEPTH, WIDTH  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402

#: Chunk that keeps the working set in cache (see whestfloor/mc.py).
CHUNK = 1024



# ---------------------------------------------------------------------------
# forward pass with tangents
# ---------------------------------------------------------------------------
def forward_tangent(x, W, Cd, want_layers=()):
    """One forward pass plus ``M`` forward-mode tangent passes.

    ``Cd`` is ``(M, n)``.  Returns ``(y, z1, zL, VzL, keep)`` where ``keep``
    maps a layer index to ``(x^l, V^l)`` with ``V^l[r] = c_r . grad x^l``.
    The tangent recursion is the exact derivative of the ReLU MLP away from
    the measure-zero kink set: ``v <- (v W) * 1{z > 0}``.
    """
    M, n = Cd.shape
    B = x.shape[0]
    h = x
    V = np.repeat(Cd[:, None, :], B, axis=1)
    z1 = zL = VzL = None
    keep = {}
    for l, w in enumerate(W, start=1):
        z = h @ w
        Vz = (V.reshape(M * B, n) @ w).reshape(M, B, n)
        mk = z > 0
        h = np.where(mk, z, np.float32(0.0))
        V = Vz * mk[None]
        if l == 1:
            z1 = z
        if l == len(W):
            zL, VzL = z, Vz
        if l in want_layers:
            keep[l] = (h, V)
    return h, z1, zL, VzL, keep


# ---------------------------------------------------------------------------
# mode: verify
# ---------------------------------------------------------------------------
def mode_verify(n_mlps: int, n_samples: int, n_dirs: int) -> None:
    n, M = WIDTH, n_dirs
    print("# Stein identity  h = c.grad psi - (c.x) psi,  E[h] = 0\n"
          f"# {n_mlps} local MLPs x {n_samples:,} samples x {M} directions "
          f"x {n} neurons\n")
    hdr = (f"{'mlp':>8} {'psi':<16} {'rms mean(h)':>12} {'rms MC se':>12} "
           f"{'ratio':>7} {'rms z':>7} {'max|z|':>7} {'chi2/dof':>9}")
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    for k in range(n_mlps):
        W = make_mlp(WIDTH, DEPTH, 900_000 + k)
        rng = np.random.default_rng(4242 + k)
        Cd = rng.standard_normal((M, n)).astype(np.float32)
        Cd /= np.linalg.norm(Cd, axis=1, keepdims=True)
        fams = ("relu(z^32_j)", "z^32_j", "x^16_i", "x^1_i")
        s1 = {f: np.zeros((M, n)) for f in fams}
        s2 = {f: np.zeros((M, n)) for f in fams}
        rng = np.random.default_rng(999_000 + k)
        done = 0
        while done < n_samples:
            B = min(CHUNK, n_samples - done)
            x = rng.standard_normal((B, n), dtype=np.float32)
            y, _, zL, VzL, keep = forward_tangent(x, W, Cd, (1, 16))
            cx = (x @ Cd.T).T                             # (M, B)
            for f, tan, val in (("relu(z^32_j)", None, y),
                                ("z^32_j", VzL, zL),
                                ("x^16_i", keep[16][1], keep[16][0]),
                                ("x^1_i", keep[1][1], keep[1][0])):
                if tan is None:                           # tangent of relu(zL)
                    tan = VzL * (zL > 0)[None]
                hh = tan - cx[:, :, None] * val[None]
                s1[f] += hh.sum(1)
                s2[f] += (hh.astype(np.float64) ** 2).sum(1)
            done += B
        for f in fams:
            mh = s1[f] / n_samples
            sd = np.sqrt(np.maximum(s2[f] / n_samples - mh * mh, 0.0))
            live = sd > 0
            se = np.where(live, sd, 1.0) / np.sqrt(n_samples)
            z = np.where(live, mh / se, 0.0)
            dof = int(live.sum())
            print(f"{900_000+k:>8} {f:<16} {np.sqrt(np.mean(mh**2)):12.4e} "
                  f"{np.sqrt(np.mean(se[live]**2)):12.4e} "
                  f"{np.sqrt(np.mean(mh**2))/np.sqrt(np.mean(se[live]**2)):7.3f} "
                  f"{np.sqrt(np.sum(z**2)/dof):7.3f} {np.abs(z).max():7.3f} "
                  f"{np.sum(z**2)/dof:9.3f}")
            worst = max(worst, np.sqrt(np.sum(z ** 2) / dof))
    print(f"\nworst rms z-score over all families and MLPs: {worst:.3f}   "
          "(1.0 = exactly Monte-Carlo error, i.e. the identity holds)")


# ---------------------------------------------------------------------------
# mode: span
# ---------------------------------------------------------------------------
#: Depths at which a hidden-activation Stein block is measured, plus the
#: layer-32 PRE-activation.  Every block is 256 features wide (one psi per
#: source neuron), so by linearity of the Stein operator in psi it already
#: contains ``sum_i w_i x^l_i`` for every weighting w, random or aligned.
SPAN_BLOCKS = (1, 16, 32, "z32")


class Acc:
    """Cross-moment accumulator for one half of the sample.

    The whole design ``D = [P | Q]`` -- shared layer-1 Hermite block ``P`` and
    every Stein block ``Q`` concatenated -- is accumulated as ONE Gram, so any
    sub-dictionary is a submatrix and no cross-block term has to be
    special-cased.
    """

    def __init__(self, p, n):
        self.N = 0
        self.DD = np.zeros((p, p))
        self.Dm = np.zeros(p)
        self.Dy = np.zeros((p, n))
        self.sy1 = np.zeros(n)
        self.sy2 = np.zeros(n)
        self.xy = np.zeros((n, n))
        self.xyy = np.zeros((n, n))

    def add(self, D, y, x):
        self.N += len(y)
        self.DD += D.T @ D
        self.Dm += D.sum(0)
        self.Dy += D.T @ y
        self.sy1 += y.sum(0)
        self.sy2 += (y * y).sum(0)
        self.xy += x.T @ y
        self.xyy += x.T @ (y * y)

    def cov(self, idx):
        """Centred ``(Cov(D_idx), Cov(D_idx, y), Var(y), E[y])``."""
        N = self.N
        my = self.sy1 / N
        vy = self.sy2 / N - my * my
        Dm = self.Dm[idx] / N
        G = self.DD[np.ix_(idx, idx)] / N - np.outer(Dm, Dm)
        c = self.Dy[idx] / N - np.outer(Dm, my)
        return G, c, vy, my


#: Ridge grid for the held-out projection.  Several Stein blocks are close to
#: rank-deficient (at depth the network Jacobian is near rank one, so the
#: tangent ``c . grad x^l`` collapses onto a few directions), and an
#: unregularised solve there reports a hugely NEGATIVE held-out R^2 that says
#: nothing about the population span.  The grid is swept and the best held-out
#: value reported, which is generous to the mechanism on purpose.
RIDGE_GRID = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def r2_holdout(accA, accB, idx):
    """Fit the projection on one half, read explained variance on the other.

    Returns ``(best_heldout_R2, in_sample_R2)``, both total-variance weighted:

        R2 = sum_j explained_j / sum_j Var(y_j)

    The held-out form carries no ``p/N`` inflation, which matters here because
    the dictionaries are 256 to 2048 features wide.
    """
    if len(idx) == 0:
        return 0.0, 0.0
    p = len(idx)
    GA, cA, vA, _ = accA.cov(idx)
    GB, cB, vB, _ = accB.cov(idx)
    tA, tB = float(np.sum(vA)), float(np.sum(vB))
    ins = 0.5 * (float(np.sum(cA * np.linalg.solve(
        GA + 1e-10 * np.trace(GA) / p * np.eye(p), cA))) / tA
        + float(np.sum(cB * np.linalg.solve(
            GB + 1e-10 * np.trace(GB) / p * np.eye(p), cB))) / tB)
    best = -np.inf
    for lam in RIDGE_GRID:
        acc = 0.0
        for (Gs, cs, Gd, cd, td) in ((GA, cA, GB, cB, tB), (GB, cB, GA, cA, tA)):
            beta = np.linalg.solve(Gs + lam * np.trace(Gs) / p * np.eye(p), cs)
            expl = 2.0 * np.sum(beta * cd, axis=0) - np.einsum(
                "pj,pq,qj->j", beta, Gd, beta, optimize=True)
            acc += float(np.sum(expl)) / td
        best = max(best, 0.5 * acc)
    return best, ins


def mode_span(n_mlps: int, n_samples: int, n_dirs: int) -> None:
    n = WIDTH
    print("# R^2 of each dictionary against relu(z^32_j), total-variance "
          "weighted.\n# Coefficients are fitted on one half of the sample and "
          "the explained variance\n# read on the other (both ways, best over a "
          "ridge grid), so the number carries\n# no p/N in-sample inflation.  "
          "In-sample R^2 is quoted beside it.\n")
    rows: dict[str, list[float]] = {}
    for k in range(n_mlps):
        seed = 900_000 + k
        W = make_mlp(WIDTH, DEPTH, seed)
        M = n_dirs
        # --- pilot: means, and the mean Jacobian E[grad y] = E[x^T y] -----
        rng = np.random.default_rng(seed + 555)
        npil, Gb = 20_000, np.zeros((n, n))
        hm = {l: np.zeros(n) for l in SPAN_BLOCKS if l != "z32"}
        done = 0
        while done < npil:
            B = min(CHUNK, npil - done)
            x = rng.standard_normal((B, n), dtype=np.float32)
            h = x
            for l, w in enumerate(W, start=1):
                h = np.maximum(h @ w, 0.0)
                if l in hm:
                    hm[l] += h.sum(0)
            Gb += x.T.astype(np.float64) @ h
            done += B
        hm = {l: (v / npil).astype(np.float32) for l, v in hm.items()}
        u, sv, _ = np.linalg.svd(Gb / npil)
        rr = np.random.default_rng(seed + 999)
        Rr = rr.standard_normal((max(M - 1, 0), n))
        Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
        Cd = np.ascontiguousarray(
            np.concatenate([u[:, :1].T, Rr]).astype(np.float32))
        dnames = ["jac"] + [f"rnd{i}" for i in range(1, M)]

        qkeys = [(r, l) for r in range(M) for l in SPAN_BLOCKS]
        nq = len(qkeys)
        p = 2 * n + nq * n
        sig1 = np.sqrt(np.sum(W[0] * W[0], axis=0)).astype(np.float32)
        inv1 = (1.0 / sig1).astype(np.float32)
        A, Bc = Acc(p, n), Acc(p, n)
        want = tuple(l for l in SPAN_BLOCKS if l != "z32")
        rng = np.random.default_rng(seed + 12345)
        t0, done, flip = time.time(), 0, 0
        while done < n_samples:
            B = min(CHUNK, n_samples - done)
            x = rng.standard_normal((B, n), dtype=np.float32)
            y, z1, zL, VzL, keep = forward_tangent(x, W, Cd, want)
            cx = x @ Cd.T
            t = z1 * inv1
            cols = [x, t * t - np.float32(1.0)]
            for (r, l) in qkeys:
                if l == "z32":
                    val, tan, ctr = zL, VzL[r], np.float32(0.0)
                else:
                    val, tan, ctr = keep[l][0], keep[l][1][r], hm[l]
                cols.append(tan - cx[:, r:r + 1] * (val - ctr))
            D = np.concatenate(cols, axis=1).astype(np.float64)
            (A if flip == 0 else Bc).add(D, y.astype(np.float64),
                                         x.astype(np.float64))
            flip ^= 1
            done += B
        dt = time.time() - t0

        # --- exact ceiling, psi = y_j, ALL 256 directions ------------------
        # |k_j|^2 estimated as k_j^A . k_j^B across the two independent halves,
        # so it is UNBIASED -- a same-sample |k|^2 would be inflated by the
        # noise of 256 estimated components.
        def kmat(a):
            my = a.sy1 / a.N
            return 0.5 * (a.xyy / a.N - 2.0 * my * (a.xy / a.N))
        kA, kB = kmat(A), kmat(Bc)
        vy = 0.5 * ((A.sy2 / A.N - (A.sy1 / A.N) ** 2)
                    + (Bc.sy2 / Bc.N - (Bc.sy1 / Bc.N) ** 2))
        tot = float(np.sum(vy))
        live = vy > 1e-14
        ceil = np.zeros(n)
        ceil[live] = np.sum(kA[:, live] * kB[:, live], axis=0) / vy[live] ** 2
        cw = float(np.sum(np.clip(ceil, 0.0, 1.0) * vy) / tot)

        base = 2 * n
        idx = {"H1  (linear CV == Hermite k=1)": np.arange(n),
               "H1+H2  (shipped, layer-1 k<=2)": np.arange(base)}
        for a, (r, l) in enumerate(qkeys):
            idx[f"Stein psi={l if l == 'z32' else f'x^{l}_i'}  dir={dnames[r]}"] \
                = np.arange(base + a * n, base + (a + 1) * n)
        idx["Stein ALL blocks x all dirs"] = np.arange(base, p)
        idx["H1+H2 + Stein ALL"] = np.arange(p)

        print(f"=== local MLP {seed}   {n_samples:,} samples, {M} tangents, "
              f"{dt:.0f}s   mean-Jacobian rank-1 share "
              f"{sv[0] ** 2 / np.sum(sv ** 2):.3f} ===")
        print(f"  {'dictionary':<34} {'p':>5} {'held-out R2':>12} "
              f"{'x':>8}   {'in-sample':>10}")
        for nm, ii in idx.items():
            v, ins = r2_holdout(A, Bc, ii)
            vv = min(v, 0.999999)
            print(f"  {nm:<34} {len(ii):5d} {v * 100:11.3f}% "
                  f"{1 / (1 - vv):8.3f}   {ins * 100:9.3f}%")
            rows.setdefault(nm, []).append(v)
        print(f"  {'CEILING psi=y_j, ALL 256 dirs':<34} {256:5d} "
              f"{cw * 100:11.3f}% {1 / (1 - cw):8.3f}   "
              f"(exact, any c incl. W^32[:,j])")
        rows.setdefault("CEILING psi=y_j, ALL 256 dirs", []).append(cw)
        print()

    print("=== mean over MLPs ===")
    for nm, vs in rows.items():
        v = float(np.mean(vs))
        vv = min(v, 0.999999)
        print(f"  {nm:<34} R2 = {v * 100:8.3f}%   -> {1 / (1 - vv):7.3f}x")
    both = float(np.mean(rows["H1+H2 + Stein ALL"]))
    ship = float(np.mean(rows["H1+H2  (shipped, layer-1 k<=2)"]))
    st = float(np.mean(rows["Stein ALL blocks x all dirs"]))
    print("\nBAR (fixed before the run): the Stein dictionary must reach "
          "R^2 > 0.75, i.e. > 4x,\n     materially above the k<=2 ceiling of "
          "1.75x, to justify the 2x per-sample\n     cost of the extra "
          "gradient pass.")
    print(f"  Stein alone                {st * 100:7.3f}%")
    print(f"  shipped k<=2 alone         {ship * 100:7.3f}%")
    print(f"  both together              {both * 100:7.3f}%   "
          f"(Stein increment {100 * (both - ship):+.3f} points)")
    print(f"  VERDICT: {'PASS' if st > 0.75 else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("verify", "span"))
    ap.add_argument("--n-mlps", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--n-dirs", type=int, default=2)
    a = ap.parse_args()
    if a.mode == "verify":
        mode_verify(a.n_mlps, a.n_samples, a.n_dirs)
    else:
        mode_span(a.n_mlps, a.n_samples, a.n_dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
