"""Randomised quasi-Monte Carlo: lattice construction, orderings, and kernels.

Credit where it is due.  The construction measured here is the one **evaaaz**
published (forum topic 18053): a randomly-shifted rank-1 lattice with a
Cranley-Patterson rotation and inverse-CDF mapping to Gaussians, unbiased at
every ``N``.  **radiant-allomancer** (18085) published the two facts that shape
how it is measured here — that the gain collapses to 1.40x once covariance
shrinkage is in the pipeline because the two levers are partly redundant
(§2.1), and that *antithetic* RQMC is a different animal that loses (§4.1) —
and the float64 inverse-CDF warning is jamesrahenry/mohanty's (18097/18125).

Why a separate module from :mod:`whestfloor.kernels`
----------------------------------------------------
``kernels.rqmc_sparse_kernel`` already exists and already works; what it does
not carry is (a) a generating vector at any ``N`` but 8501, (b) any dimension
ordering beyond a row-norm permutation, and (c) the layer-1 Hermite control
variates the shipped estimator uses.  All three are needed to answer the
question the ledger actually asks — *what is the convergence exponent* — and
the third is needed to answer whether the lattice and the control variates are
redundant.  This module is additive: nothing here imports from or mutates the
shipped path.

The unbiasedness argument, in full
----------------------------------
Let ``P = {p_i}_{i<N} ⊂ [0,1)^d`` be any fixed point set and ``U ~ U[0,1)^d``.
Define ``u_i = frac(p_i + U)``.  For each fixed ``i`` and each coordinate
``j``, the map ``t -> frac(p_ij + t)`` pushes ``U[0,1)`` forward to
``U[0,1)`` — it is a measure-preserving rotation of the circle — and the
coordinates of ``U`` are independent, so ``u_i ~ U[0,1)^d`` **exactly**, for
every ``i`` and every ``N``.  Hence ``x_i = Phi^{-1}(u_i) ~ N(0, I_d)`` exactly
and

    E[(1/N) sum_i f(x_i)] = E[f(x)]

for any integrable ``f``, with no asymptotics and no lattice property used.
Only the *dependence between* the ``x_i`` is structured, and dependence does
not move a mean.  This is what keeps ``docs/floor_theorem.md``'s independence
argument applicable to the lattice arm: the estimator is still unbiased, so
its MSE is still variance, and the ``vbar/N`` floor still binds — what changes
is the *rate* at which the estimator's own variance falls, not the floor.

Composing with an orthogonal ``Q`` costs nothing extra: ``x ~ N(0, I)`` implies
``Q x ~ N(0, I)``, so any rotation of the lattice dimensions is also exactly
unbiased, and a rotation can be folded into ``W^1`` for free.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Primes, primitive roots, CBC
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
    """``B_2({x}) = x^2 - x + 1/6``, the shift-averaged kernel of the
    unanchored Sobolev space: ``(1/N) sum_i prod_{j in u} B_2(x_ij)`` is exactly
    the squared worst-case error contributed by projection ``u``."""
    return x * x - x + 1.0 / 6.0


def cbc_order2(N: int, d: int, verbose: bool = False) -> np.ndarray:
    """Component-by-component rank-1 lattice, order-2 (pairwise) criterion.

    ``O(d N log N)`` by the primitive-root/FFT trick.  ONE-dimensional
    projections need no search: for any ``z_j`` coprime to ``N`` the projection
    is the exact ``N``-point grid with term ``1/(6N^2)``.  That is the whole
    reason a lattice changes the *rate* — every first-order ANOVA term is
    integrated at ``O(N^-2)`` instead of ``O(N^-1)``, whatever the vector.
    """
    if not is_prime(N):
        raise ValueError(f"cbc_order2 needs prime N, got {N}")
    g = primitive_root(N)
    gpow = np.ones(N - 1, dtype=np.int64)
    for t in range(1, N - 1):
        gpow[t] = gpow[t - 1] * g % N
    b = _b2(np.arange(N) / N)
    v = b[gpow]
    Fv = np.fft.rfft(v)
    B = np.zeros(N)
    z = np.zeros(d, dtype=np.int64)
    idx = np.arange(N)
    for s in range(d):
        if s == 0:
            z[s] = 1
        else:
            u = B[gpow]
            corr = np.fft.irfft(np.conj(np.fft.rfft(u)) * Fv, n=N - 1)
            z[s] = int(gpow[int(np.argmin(corr))])
        B += b[(idx * z[s]) % N]
        if verbose and (s + 1) % 64 == 0:
            print(f"    cbc dim {s + 1}/{d}", flush=True)
    return z


def roberts_z(N: int, d: int) -> np.ndarray:
    """Kronecker/Roberts lattice, ``alpha_j = phi_d^{-(j+1)}``, rationalised.

    This is evaaaz's construction.  No search, valid at any ``N``, and
    dimension-*balanced* — which is exactly why radiant-allomancer's B12 found
    no ordering gradient to exploit on it.
    """
    phi = 2.0
    for _ in range(200):
        phi = (1.0 + phi) ** (1.0 / (d + 1.0))
    a = np.array([phi ** -(j + 1) for j in range(d)])
    return np.round(a * N).astype(np.int64) % N


def lattice_quality(N: int, z: np.ndarray) -> dict:
    """Exact order-1 and order-2 worst-case-error terms of a rank-1 lattice."""
    d = len(z)
    b = _b2(np.arange(N) / N)
    Bm = b[(np.arange(N)[:, None] * np.asarray(z)[None, :]) % N]
    t1 = Bm.mean(0)
    tot = Bm.sum(1)
    s2 = 0.5 * ((tot * tot).mean() - (Bm * Bm).sum(1).mean())
    G = (Bm.T @ Bm) / N
    np.fill_diagonal(G, 0.0)
    return dict(t1_mean=float(t1.mean()), t1_ref=1.0 / (6.0 * N * N),
                s2=float(s2), s2_per_pair=float(s2 / (d * (d - 1) / 2)),
                tmax=float(G.max()), trms=float(np.sqrt((G ** 2).sum()
                                                        / (d * (d - 1)))))


# ---------------------------------------------------------------------------
# Cached generating vectors
# ---------------------------------------------------------------------------


def _art() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


def get_z(N: int, d: int = 256, kind: str = "cbc",
          verbose: bool = False) -> np.ndarray:
    """Generating vector for ``N`` points in ``d`` dimensions, disk-cached.

    ``kind='cbc'`` runs the order-2 CBC search (prime ``N`` only);
    ``kind='roberts'`` is the search-free Kronecker vector.
    """
    if kind == "roberts":
        return roberts_z(N, d)
    cache = _art() / "rqmc" / f"z_{kind}_{N}_{d}.npy"
    if cache.is_file():
        return np.load(cache)
    z = cbc_order2(N, d, verbose=verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, z)
    return z


# ---------------------------------------------------------------------------
# Point sets and the Gaussian map (offline / research; plain numpy)
# ---------------------------------------------------------------------------

#: float32 has 24 mantissa bits, so a uniform can round to exactly 0.0 or 1.0
#: and send ``ndtri`` to +-inf.  A8 (jamesrahenry 18097 erratum, mohanty 18125):
#: this has already NaN'd another team's tail branch.  Clamp, and do the ppf in
#: float64 -- but cast straight back to float32, because a promoted array
#: reprices the entire downstream chain at 2x.
U_EPS = 6.0e-8


def _ndtri(u):
    from flopscope.stats._ndtri import _ndtri as f  # noqa: PLC0415
    return f(u)


def lattice_rows(i0: int, i1: int, N: int, z: np.ndarray) -> np.ndarray:
    """``frac(i * z_j / N)`` for ``i in [i0, i1)``, float64, exact modulo."""
    i = np.arange(i0, i1, dtype=np.int64)
    p = (i[:, None] * np.asarray(z, dtype=np.int64)[None, :]) % N
    return p / float(N)


def shifted_normals(base: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Cranley-Patterson: ``Phi^{-1}(frac(base + shift))`` in float64 -> float32.

    ``base`` and ``shift`` are float64; the return is float32 so the 32 layers
    downstream stay float32.  See :data:`U_EPS`.
    """
    u = base + shift
    u -= np.floor(u)
    np.clip(u, U_EPS, 1.0 - U_EPS, out=u)
    return _ndtri(u).astype(np.float32)


# ---------------------------------------------------------------------------
# Dimension ordering
# ---------------------------------------------------------------------------


def rownorm_perm(W1: np.ndarray) -> np.ndarray:
    """Input coordinates ordered by ``||W^1[j, :]||`` (descending).

    By Stein's lemma ``Cov(f, x_j) = E[df/dx_j]``, so the first-order influence
    of input coordinate ``j`` is its total drive into layer 1; the row norm is
    the only predict-time-affordable proxy for it.
    """
    imp = (W1.astype(np.float64) ** 2).sum(1)
    return np.argsort(-imp)


def meanfield_jacobian(weights, alpha) -> np.ndarray:
    """``R = W^1 diag(g^1) W^2 ... diag(g^{L-1}) W^L``, ``g^l_i = Phi(alpha^l_i)``.

    The mean-field Jacobian of ``z^L`` in the input: the expected gate of a
    ReLU is exactly ``Phi(alpha)``, so this is ``E[dz^L/dx]`` to first order in
    the gate fluctuations.  31 matmuls of ``256^3`` = 1.0e9 FLOPs = 0.4% of B.
    """
    from math import erf, sqrt  # noqa: PLC0415
    _ = erf, sqrt
    R = weights[0].astype(np.float64)
    for l in range(1, len(weights)):
        g = 0.5 * (1.0 + np.vectorize(math.erf)(alpha[l - 1] / math.sqrt(2.0)))
        R = (R * g[None, :]) @ weights[l].astype(np.float64)
    return R


def active_subspace_rotation(R: np.ndarray) -> np.ndarray:
    """Orthogonal ``Q`` whose rows are the active-subspace directions of ``R``,
    ordered by decreasing importance.

    ``C = R R^T`` is the (mean-field) first-order sensitivity matrix of the
    scored row in the input; its top eigenvectors are the input directions the
    network actually responds to.  Feeding lattice dimension ``k`` with
    ``(Q x)_k`` therefore puts the most important direction in the best-searched
    lattice dimension.  ``Q`` is orthogonal, so ``Q x ~ N(0, I)`` and the
    estimator stays exactly unbiased; and ``x -> Q^T`` folds into ``W^1`` at
    zero per-sample cost.
    """
    C = R @ R.T
    w, V = np.linalg.eigh(C)
    order = np.argsort(-w)
    return V[:, order].T  # rows = directions, most important first


# ---------------------------------------------------------------------------
# The deployable draw: billed, float32-clean, chunked
# ---------------------------------------------------------------------------


def billed_lattice_base(n_points: int, z, chunk: int = 32768):
    """``(n_points, d)`` float32 base points ``frac(i z_j / N)``, via flopscope.

    Data-independent, so the submission builds it once in ``setup`` where it is
    free.  Chunked because the intermediate ``i * z_j`` is float64 and at
    ``N = 1e5 x 256`` that is 205 MB in one allocation.

    The modulo is exact as long as ``(N-1) * max(z) < 2^53``; at ``N = 131071``
    that product is 1.7e10, eleven orders inside float64's integer range.
    """
    import flopscope.numpy as fnp  # noqa: PLC0415

    zz = fnp.asarray([float(v) for v in z])
    parts = []
    for a in range(0, n_points, chunk):
        b = min(a + chunk, n_points)
        i = fnp.arange(a, b, dtype=fnp.float64)
        p = fnp.outer(i, zz)
        p = p - fnp.floor(p * (1.0 / n_points)) * float(n_points)
        parts.append((p * (1.0 / n_points)).astype(fnp.float32))
    return parts[0] if len(parts) == 1 else fnp.concatenate(parts, axis=0)


def billed_lattice_normals(base, rng, chunk: int = 32768):
    """Cranley-Patterson shift + inverse CDF, float64 ppf, float32 out.

    **A8, and it has already cost another team 2x.**  ``flopscope.stats.*``
    promotes float32 to float64 to match scipy, and a single promoted array
    reprices the whole downstream chain at the float64 rate.  So the ppf runs
    in float64 (which is what stops a uniform of exactly 1.0 from returning
    ``inf``) and the result is cast back to float32 *immediately*, before
    anything touches it.  ``scripts/51_rqmc_deploy.py --mode cost`` asserts the
    32 scored matmuls still bill at the float32 rate in a real BudgetContext.

    Measured: 173 FLOPs/element, of which ``norm.ppf`` is 166.  Against 16 for
    ``standard_normal`` that is +157/element -- 1.0% of a scored pass.
    """
    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415

    d = base.shape[1]
    n = base.shape[0]
    shift = rng.random(d, dtype=fnp.float32)
    parts = []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        u = base[a:b] + shift
        u = u - fnp.floor(u)
        u = fnp.minimum(fnp.maximum(u, U_EPS), 1.0 - U_EPS)
        parts.append(flops.stats.norm.ppf(u).astype(fnp.float32))
    return parts[0] if len(parts) == 1 else fnp.concatenate(parts, axis=0)


def lattice_x0_fn(base, chunk: int = 32768):
    """An ``x0_fn`` for ``kernels.corrected_sparse_kernel``.

    Signature ``(rng, n_samples, width) -> (n_samples, width) float32``.  The
    ``rng`` is the kernel's own, which descends from ``mlp.seed``, so the
    Cranley-Patterson shift is reproducible and per-MLP.
    """
    def fn(rng, n_samples, width):
        if base.shape[0] != n_samples or base.shape[1] != width:
            raise ValueError(
                f"lattice base {tuple(base.shape)} does not match "
                f"({n_samples}, {width})")
        return billed_lattice_normals(base, rng, chunk)
    return fn


def rotate_first_layer(weights, Q):
    """``[Q @ W^1] + W[1:]`` — fold a lattice-dimension rotation in for free.

    ``z^1 = x_orig^T W^1`` and ``x_orig = Q^T x_rot``, so feeding the kernel
    ``x_rot`` with ``W^1 -> Q W^1`` is the identical network evaluated on the
    identical distribution.  One ``256^3`` matmul per MLP (3.4e7 FLOPs, 0.01%
    of B) and nothing per sample.
    """
    import flopscope.numpy as fnp  # noqa: PLC0415

    return [fnp.asarray(np.asarray(Q, dtype=np.float32)) @ weights[0]] + \
        list(weights[1:])
