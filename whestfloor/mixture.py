"""Gaussian-mixture state for a deep ReLU MLP: the ceiling, and the propagator.

Why a mixture, and what it would have to beat
=============================================

`docs/cost_floor.md` shows the top of the leaderboard is not sampling, and
`docs/traj_closure.md` closes the moment-closure route with a bound (`r = 2.0`
deployable, `r ~ 5-7` with exact cumulants, against a 4.4 break-even).  What is
left that is deterministic, costs about six covariance propagations and has
~4e-9 model error?  The obvious candidate is a **K-component Gaussian mixture**:
each component is propagated by the exact rectified-Gaussian machinery in
:mod:`whestfloor.relu_moments`, so there is no *moment* closure per component,
and the answer is

    E[relu(z^L_j)]  ~=  sum_k w_k * relu_mean(m_kj, s_kj).                 (*)

That form is the whole idea and it is also its own ceiling.  Whatever route
produces the components, the final answer of **any** mixture method has shape
(*), so the best it can possibly do is the mixture whose components are the
*true* conditional laws of `z^L` given some K-cell partition of the sample
space — evaluated as if each conditional were Gaussian.  This module measures
that ceiling exactly (:func:`cell_oracle`) before building anything.

The partition is the only free choice, and it is indexed by two numbers:

``r``   the dimension of the subspace whose coordinates the cells are cut along
        (the top-`r` eigendirections of `Cov(z^L)` in the version
        `docs/state_of_play.md` records, but the code takes any frame);
``K``   the number of cells = the number of mixture components = the number of
        covariance propagations the method has to pay for.

`scripts/06_conditional_gaussianity.py` measured a single row of this table
(top-`k` eigendirections, product bins, one MLP) and found 2.9x at `k = 3`.
This module generalises it in the three ways that matter: Lloyd cells instead
of product bins (so `K` means components, not bins-per-axis), frames chosen for
non-Gaussianity and from the input side rather than only by variance, and an
error estimate with the reference's own Monte-Carlo variance removed.

Layout
======

``lloyd``            k-means++ / Lloyd quantiser on the pilot coefficients.
``CellAccumulator``  streaming per-cell ``(count, sum z, sum z^2)``.
``cell_oracle``      the prediction (*) from those accumulators.
``frames``           candidate conditioning frames (eig / kurt / input / raw).
``MixtureState``     the deployable propagator: K Gaussians, exact per-component
                     rectification, no sampling anywhere.
"""

from __future__ import annotations

import numpy as np

from .relu_moments import relu_mean, relu_var
from .trajclosure import closure_step

VAR_FLOOR = 1e-30


# ---------------------------------------------------------------------------
# Quantiser
# ---------------------------------------------------------------------------
def lloyd(T: np.ndarray, K: int, *, iters: int = 25, seed: int = 0,
          tol: float = 1e-7) -> np.ndarray:
    """Lloyd (k-means) centroids of ``T`` (n, r) with k-means++ seeding.

    For ``r = 1`` this is the Lloyd-Max scalar quantiser, i.e. exactly the
    optimal ``K``-node placement along the conditioning direction, which is the
    honest form of "Gauss-Hermite nodes along the top eigendirection": Lloyd
    can only beat a fixed quadrature rule, so a ceiling measured with Lloyd
    cells bounds one measured with GH nodes.
    """
    n = T.shape[0]
    rng = np.random.default_rng(seed)
    K = min(K, n)
    C = np.empty((K, T.shape[1]), dtype=np.float64)
    C[0] = T[rng.integers(n)]
    d2 = ((T - C[0]) ** 2).sum(axis=1)
    for k in range(1, K):
        tot = float(d2.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            C[k] = T[rng.integers(n)]
        else:
            C[k] = T[np.searchsorted(np.cumsum(d2), rng.random() * tot)]
        np.minimum(d2, ((T - C[k]) ** 2).sum(axis=1), out=d2)
    prev = np.inf
    for _ in range(iters):
        lab = assign(T, C)
        cnt = np.bincount(lab, minlength=K).astype(np.float64)
        S = np.zeros_like(C)
        np.add.at(S, lab, T)
        live = cnt > 0
        C[live] = S[live] / cnt[live, None]
        if not live.all():  # respawn empties on the worst-served point
            far = np.argsort(((T - C[lab]) ** 2).sum(axis=1))[::-1]
            C[~live] = T[far[: int((~live).sum())]]
        obj = float(((T - C[lab]) ** 2).sum())
        if abs(prev - obj) <= tol * max(obj, 1e-30):
            break
        prev = obj
    return C


def assign(T: np.ndarray, C: np.ndarray, chunk: int = 65_536) -> np.ndarray:
    """Nearest-centroid labels, ``argmin_k |T_i - C_k|^2``, chunked."""
    C = np.asarray(C, dtype=T.dtype)
    cn = (C * C).sum(axis=1)
    out = np.empty(T.shape[0], dtype=np.int64)
    for lo in range(0, T.shape[0], chunk):
        hi = min(lo + chunk, T.shape[0])
        out[lo:hi] = np.argmin(cn[None, :] - 2.0 * (T[lo:hi] @ C.T), axis=1)
    return out


# ---------------------------------------------------------------------------
# Streaming per-cell sufficient statistics
# ---------------------------------------------------------------------------
class CellAccumulator:
    """``count``, ``sum z``, ``sum z^2`` per cell, folded chunk by chunk.

    Cells come from centroids ``C`` (K, r) in whatever frame the caller
    projects into.  Everything is float64 and the group-by is a stable sort
    plus one ``add.reduceat`` over the *concatenated* ``[z, z^2]`` block, which
    is the formulation that stays memory-bound rather than scatter-bound at
    ``K ~ 1e3``: one gather and one reduce instead of two of each.
    """

    #: Cells below which a dense one-hot ``M.T @ ZZ`` is used instead of
    #: sort-and-reduce.  Measured on this box (19 GB/s copy bandwidth) the
    #: matmul route is 22 ms vs 26 ms at K = 6 and 313 ms vs 29 ms at K = 64 —
    #: skinny multithreaded sgemm loses badly — so the default is off.
    MATMUL_K = 0

    __slots__ = ("C", "K", "n", "cnt", "s", "_oh")

    def __init__(self, C: np.ndarray, n: int):
        self.C = np.ascontiguousarray(C, dtype=np.float32)
        self.K = C.shape[0]
        self.n = int(n)
        self.cnt = np.zeros(self.K, dtype=np.float64)
        self.s = np.zeros((self.K, 2 * self.n), dtype=np.float64)
        self._oh = None

    def add(self, ZZ: np.ndarray, T: np.ndarray) -> None:
        """Fold one chunk.  ``ZZ`` is ``(B, 2n)`` = ``[z | z*z]`` float32,
        built once per chunk and shared by every configuration.

        float32 for the per-chunk partial sums is worth 2x and costs nothing
        measurable: pairwise summation over <= 8192 terms leaves ~1e-5 relative
        on each cell's ``E[z^2] - E[z]^2``, which moves ``relu_mean`` by ~6e-6
        per chunk and averages away over the stream, three orders below the
        1e-8 errors being resolved.  The accumulators themselves are float64.
        """
        B = ZZ.shape[0]
        lab = assign(T, self.C)
        self.cnt += np.bincount(lab, minlength=self.K)
        if self.K <= self.MATMUL_K:
            if self._oh is None or self._oh.shape[0] < B:
                self._oh = np.zeros((B, self.K), dtype=np.float32)
            M = self._oh[:B]
            M[...] = 0.0
            M[np.arange(B), lab] = 1.0
            self.s += (M.T @ ZZ).astype(np.float64)
            return
        order = np.argsort(lab, kind="stable")
        b = np.searchsorted(lab[order], np.arange(self.K + 1))
        nz = np.nonzero(b[1:] > b[:-1])[0]
        if nz.size:
            self.s[nz] += np.add.reduceat(ZZ[order], b[nz], axis=0)

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.cnt.copy(), self.s.copy()


def predict_state(cnt: np.ndarray, s: np.ndarray, n: int) -> np.ndarray:
    """The mixture answer (*): ``sum_k w_k relu_mean(m_k, s_k)``.

    Cells holding a single sample carry no variance estimate, so they
    contribute ``relu`` of the point itself — which is the ``K -> infinity``
    limit of the same formula, and exact.
    """
    tot = float(cnt.sum())
    live = cnt > 0
    c = cnt[live][:, None]
    m = s[live, :n] / c
    v = np.maximum(s[live, n:] / c - m * m, 0.0)
    out = np.where(c < 2.0, np.maximum(m, 0.0),
                   relu_mean(m, np.sqrt(np.maximum(v, VAR_FLOOR))))
    return (c * out).sum(axis=0) / tot


def pooled_moments(cnt: np.ndarray, s: np.ndarray, n: int):
    """Pooled mean and per-neuron variance — the ``K = 1`` Gaussian oracle."""
    tot = float(cnt.sum())
    m = s[:, :n].sum(axis=0) / tot
    v = s[:, n:].sum(axis=0) / tot - m * m
    return m, np.maximum(v, VAR_FLOOR)


# ---------------------------------------------------------------------------
# Conditioning frames
# ---------------------------------------------------------------------------
def frame_eig(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigenframe of ``Cov(z)``, descending.  Returns ``(m0, U, evals)``."""
    Zd = np.asarray(Z, dtype=np.float64)
    m0 = Zd.mean(axis=0)
    Zc = Zd - m0
    S = (Zc.T @ Zc) / Zd.shape[0]
    w, V = np.linalg.eigh(S)
    o = np.argsort(w)[::-1]
    return m0, np.ascontiguousarray(V[:, o]), w[o]


def frame_kurt(Z: np.ndarray, r: int, *, seed: int = 0, iters: int = 40,
               ridge: float = 1e-9) -> np.ndarray:
    """FastICA-style directions of maximal |excess kurtosis|, deflated.

    Variance says where the fluctuation is; it does not say where the
    *non-Gaussianity* is, and `scripts/06`'s note ("the fluctuation is low-rank
    but the non-Gaussianity is not") is exactly that distinction.  This frame is
    the fair test of the mixture hypothesis with the strongest possible choice
    of directions: it searches for them.

    Returns ``U`` (n, r) in the original coordinates, orthonormal under the
    whitening metric (hence a valid conditioning frame).
    """
    Zd = np.asarray(Z, dtype=np.float64)
    m0 = Zd.mean(axis=0)
    Zc = Zd - m0
    S = (Zc.T @ Zc) / Zd.shape[0]
    w, V = np.linalg.eigh(S)
    keep = w > ridge * float(w.max())
    Wh = V[:, keep] / np.sqrt(w[keep])          # (n, p) whitening
    Y = Zc @ Wh                                  # (N, p), identity covariance
    p = Y.shape[1]
    rng = np.random.default_rng(seed)
    B = np.zeros((p, 0))
    cols = []
    for _ in range(r):
        u = rng.standard_normal(p)
        u -= B @ (B.T @ u) if B.shape[1] else 0.0
        u /= np.linalg.norm(u)
        for _ in range(iters):
            g = Y @ u
            un = (Y.T @ (g ** 3)) / Y.shape[0] - 3.0 * u
            if B.shape[1]:
                un -= B @ (B.T @ un)
            nrm = np.linalg.norm(un)
            if nrm < 1e-12:
                un = rng.standard_normal(p)
                if B.shape[1]:
                    un -= B @ (B.T @ un)
                nrm = np.linalg.norm(un)
            un /= nrm
            if abs(abs(float(u @ un)) - 1.0) < 1e-9:
                u = un
                break
            u = un
        cols.append(u)
        B = np.stack(cols, axis=1)
    return np.ascontiguousarray(Wh @ B)


def frame_input(X: np.ndarray, Z: np.ndarray, r: int) -> np.ndarray:
    """Top-``r`` input directions by linear response, as an ``n``-frame.

    A mixture that splits at layer 1 and propagates conditions on linear
    functionals of ``x``, not of ``z^L``.  This returns the top-``r`` left
    singular directions of ``E[x z^T]`` so the caller can bin on ``x @ A``.
    """
    Xd = np.asarray(X, dtype=np.float64)
    Zd = np.asarray(Z, dtype=np.float64)
    B = (Xd.T @ (Zd - Zd.mean(axis=0))) / Xd.shape[0]
    U, _, _ = np.linalg.svd(B, full_matrices=False)
    return np.ascontiguousarray(U[:, :r])


# ---------------------------------------------------------------------------
# The deployable propagator
# ---------------------------------------------------------------------------
def gauss_hermite(K: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilists' Gauss-Hermite nodes/weights: ``sum w_k f(x_k) ~ E f(N(0,1))``."""
    x, w = np.polynomial.hermite_e.hermegauss(K)
    return x, w / w.sum()


class MixtureState:
    """``K`` Gaussians over ``z^l``, propagated by exact rectified moments.

    ``split`` re-represents the current mixture along a direction: each
    component is replaced by ``K`` children displaced along ``u`` with the
    variance in that direction deflated, which is a quadrature of the parent.
    Splitting a *single* Gaussian is a pure quadrature identity and buys
    nothing on its own — the gain, if any, comes from the components then
    being rectified separately, so the mixture stops being a Gaussian the
    moment it passes through one ReLU.

    ``reduce_to`` merges back to ``K`` components (Runnalls' moment-preserving
    pairwise merge, restricted to the dominant direction so the cost stays
    ``O(K^2 n)``), which is what keeps the cost at ``K`` propagations rather
    than ``K^depth``.
    """

    def __init__(self, w: np.ndarray, m: np.ndarray, C: np.ndarray):
        self.w = np.asarray(w, dtype=np.float64)
        self.m = np.asarray(m, dtype=np.float64)          # (K, n)
        self.C = np.asarray(C, dtype=np.float64)          # (K, n, n)

    @property
    def K(self) -> int:
        return self.m.shape[0]

    @classmethod
    def gaussian(cls, m: np.ndarray, C: np.ndarray) -> "MixtureState":
        return cls(np.ones(1), m[None, :].copy(), C[None, :, :].copy())

    def collapse(self) -> tuple[np.ndarray, np.ndarray]:
        M = self.w @ self.m
        D = self.m - M
        C = np.einsum("k,kij->ij", self.w, self.C)
        C += np.einsum("k,ki,kj->ij", self.w, D, D)
        return M, C

    def top_direction(self) -> np.ndarray:
        _, C = self.collapse()
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        return np.ascontiguousarray(V[:, int(np.argmax(w))])

    def split(self, u: np.ndarray, nodes: int) -> "MixtureState":
        """Split every component along ``u`` into ``nodes`` children."""
        x, wq = gauss_hermite(nodes)
        K, n = self.m.shape
        wn = (self.w[:, None] * wq[None, :]).reshape(-1)
        Cu = self.C @ u                                    # (K, n)
        s2 = np.einsum("ki,i->k", Cu, u)                   # (K,)
        s = np.sqrt(np.maximum(s2, VAR_FLOOR))
        # child means: m_k + x_q * (C_k u)/s_k   (the conditional-mean shift)
        mn = (self.m[:, None, :]
              + (x[None, :, None] / s[:, None, None]) * Cu[:, None, :])
        Cn = self.C - (Cu[:, :, None] * Cu[:, None, :]) / s2[:, None, None]
        Cn = np.repeat(Cn, nodes, axis=0)
        return MixtureState(wn, mn.reshape(K * nodes, n), Cn)

    def relu_step(self, Wn: np.ndarray, kmax: int = 8,
                  exact_acos: bool = False) -> "MixtureState":
        """Rectify every component exactly, then push through ``Wn``."""
        K, n = self.m.shape
        mo = np.empty_like(self.m)
        Co = np.empty_like(self.C)
        for k in range(K):
            mu, Ch, *_ = closure_step(self.m[k], self.C[k], kmax=kmax,
                                      exact_acos=exact_acos)
            mo[k] = Wn.T @ mu
            Co[k] = Wn.T @ Ch @ Wn
        return MixtureState(self.w.copy(), mo, Co)

    def relu_mean_mixture(self) -> np.ndarray:
        """``sum_k w_k E[relu(z_j) | k]`` — the answer, form (*)."""
        s = np.sqrt(np.maximum(np.diagonal(self.C, axis1=1, axis2=2),
                               VAR_FLOOR))
        return self.w @ relu_mean(self.m, s)

    def reduce_to(self, K: int) -> "MixtureState":
        """Runnalls moment-preserving merge down to ``K`` components."""
        w, m, C = self.w.copy(), self.m.copy(), self.C.copy()
        while w.shape[0] > K:
            best, bi, bj = np.inf, 0, 1
            for i in range(w.shape[0]):
                for j in range(i + 1, w.shape[0]):
                    wij = w[i] + w[j]
                    d = m[i] - m[j]
                    # Runnalls' upper bound on the KL cost of merging (i, j),
                    # evaluated on the diagonal so it stays O(K^2 n).
                    ci = np.diagonal(C[i])
                    cj = np.diagonal(C[j])
                    cm = (w[i] * ci + w[j] * cj) / wij + (w[i] * w[j] / wij**2) * d * d
                    cost = float(0.5 * (wij * np.log(cm).sum()
                                        - w[i] * np.log(ci).sum()
                                        - w[j] * np.log(cj).sum()))
                    if cost < best:
                        best, bi, bj = cost, i, j
            wij = w[bi] + w[bj]
            d = m[bi] - m[bj]
            mm = (w[bi] * m[bi] + w[bj] * m[bj]) / wij
            CC = ((w[bi] * C[bi] + w[bj] * C[bj]) / wij
                  + (w[bi] * w[bj] / wij ** 2) * np.outer(d, d))
            keep = [t for t in range(w.shape[0]) if t not in (bi, bj)]
            w = np.concatenate([w[keep], [wij]])
            m = np.concatenate([m[keep], mm[None, :]])
            C = np.concatenate([C[keep], CC[None, :, :]])
        return MixtureState(w, m, C)
