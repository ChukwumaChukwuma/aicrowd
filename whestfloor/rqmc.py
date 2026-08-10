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


# ---------------------------------------------------------------------------
# The shipping vectors, as literals
#
# The grader sandbox has no numpy, so ``cbc_order2`` cannot run at predict time
# and the generating vector has to travel as source.  These are the two operating
# points ``docs/rqmc.md`` section 7 leaves standing.  Both are PRIME, which is
# what makes every ``z_j`` automatically coprime to ``N`` and therefore makes
# every one-dimensional projection the exact ``N``-point grid -- the property the
# whole method rests on.
#
# ``N = 24989`` is the recommended ship (the prime just under the shipped 25,000).
# ``N = 11987`` scores the same on the official suite at HALF the billed compute
# and ``C/B = 0.139`` instead of 0.282, i.e. much closer to the ``max(0.1, C/B)``
# clamp; it is the safer choice if the grader's residual is worse than the box
# these were measured on.
#
# Quality of the CBC search against the search-free Roberts/Kronecker vector
# evaaaz used -- the 1-D term is identical by construction (that part needs no
# search), and the search buys the pairs:

#: N = 24989: 1-D term 2.6690e-10 == 1/(6N^2) 2.6690e-10;
#: order-2 sum_{j<k}T = 1.0132e-03 against Roberts' 2.1273e-02 (21.0x better), worst pair 4.288e-06 vs 1.389e-03.
RQMC_Z_24989 = (
    1, 9664, 10561, 11442, 15862, 17147, 14783, 13957, 17740, 19081, 5340,
    16980, 3155, 10906, 16095, 2239, 6896, 4887, 1705, 7057, 5689, 13463,
    7926, 2875, 8874, 21826, 16579, 24378, 16121, 1053, 18422, 17677,
    5656, 19427, 7710, 20103, 19243, 20052, 2729, 16889, 20720, 14881,
    5224, 15059, 17349, 14869, 15055, 14405, 18953, 14449, 17454, 5132,
    3546, 15501, 14300, 8375, 4173, 7884, 7080, 4973, 3469, 24279, 22138,
    23523, 14828, 3773, 16389, 1179, 19050, 9713, 13980, 716, 1624, 4913,
    5972, 16813, 17104, 1246, 18837, 7948, 8714, 1631, 9405, 18004, 1181,
    23313, 14635, 20644, 17798, 10402, 6697, 4392, 16344, 13222, 902,
    18682, 19932, 8772, 13267, 12227, 6014, 3404, 8573, 2984, 22511, 581,
    23746, 17799, 20415, 16636, 1128, 15391, 21472, 3724, 8918, 891,
    12331, 2504, 15379, 15746, 1230, 20991, 16702, 19539, 3870, 17207,
    11865, 21353, 3747, 11789, 10105, 24253, 24070, 10868, 8716, 1917,
    15479, 19780, 12896, 18737, 24209, 9794, 23203, 3465, 23586, 13294,
    7987, 14911, 19792, 19208, 23561, 614, 22520, 19232, 915, 10417, 4014,
    15882, 21249, 13117, 22757, 11549, 14668, 14036, 7098, 1000, 954,
    8670, 20655, 13259, 21069, 633, 577, 7109, 8396, 23045, 15665, 11546,
    8265, 21213, 9627, 6451, 16145, 4384, 23183, 7538, 16809, 15744, 5181,
    10571, 14584, 3369, 4535, 6071, 7316, 7985, 24609, 7234, 13133, 18673,
    8955, 5099, 728, 3514, 1610, 11321, 7787, 4951, 20972, 20397, 8792,
    19551, 21018, 4330, 16463, 13907, 11209, 24673, 12451, 12965, 4289,
    20935, 23932, 2015, 12822, 9312, 4355, 20378, 6362, 17611, 7531,
    21894, 21125, 658, 4482, 16112, 6417, 2900, 1926, 10274, 17704, 17459,
    22287, 2277, 8293, 7838, 15232, 23418, 11112, 6795, 23385, 17384,
    4081, 16673, 22022, 10361
)

#: N = 11987: 1-D term 1.1599e-09 == 1/(6N^2) 1.1599e-09;
#: order-2 sum_{j<k}T = 4.3148e-03 against Roberts' 3.7514e-02 (8.7x better), worst pair 1.716e-05 vs 1.389e-03.
RQMC_Z_11987 = (
    1, 4964, 7437, 6948, 6800, 5478, 9400, 3373, 10870, 8122, 6267, 9493,
    5099, 3333, 5529, 3394, 4599, 10865, 9334, 652, 2645, 5644, 1979,
    5486, 8224, 9755, 2729, 642, 6926, 6271, 1675, 6599, 8582, 2613, 7333,
    2478, 2574, 6755, 7592, 3013, 10184, 563, 2910, 11300, 11281, 7161,
    8034, 5699, 9462, 6665, 1827, 4653, 9261, 5566, 1098, 7431, 7662,
    9434, 697, 2272, 5167, 590, 9495, 9778, 636, 10206, 11882, 7063, 7735,
    3847, 5744, 10352, 6329, 5089, 11723, 2109, 11319, 7037, 5028, 11759,
    6143, 8675, 10450, 3022, 809, 2099, 11717, 7154, 4901, 7631, 2638,
    2791, 7461, 10795, 369, 7007, 11729, 2780, 4424, 9533, 2632, 1622,
    1021, 4396, 1096, 7167, 4771, 3365, 5911, 11817, 11338, 9483, 11748,
    4144, 6338, 2414, 1859, 6792, 1011, 459, 10054, 7940, 6423, 5518, 787,
    4098, 1468, 11836, 1910, 11895, 10815, 10691, 6524, 8855, 4068, 11947,
    5248, 7074, 5969, 9183, 1795, 4288, 4020, 7495, 9367, 6892, 5354,
    9176, 1839, 7123, 2877, 10646, 6428, 8794, 11444, 9049, 11854, 7584,
    5935, 3146, 4608, 5057, 2866, 1186, 3283, 10223, 4268, 5029, 9883,
    8920, 10205, 5933, 11533, 311, 3573, 2513, 3553, 1185, 9704, 7948,
    503, 5659, 10456, 465, 4726, 2496, 11862, 10567, 8285, 8501, 2828,
    11552, 1693, 5939, 11111, 11175, 4274, 8553, 4680, 9063, 5526, 10758,
    3424, 1481, 9580, 11846, 8762, 5506, 8789, 9253, 2701, 8537, 9060,
    2292, 3850, 4338, 8351, 2886, 4030, 7474, 1563, 8323, 11797, 11875,
    6088, 5048, 9270, 3228, 1504, 1230, 9405, 11099, 1601, 6120, 11040,
    7971, 1798, 1447, 10336, 4142, 4034, 3729, 4097, 7698, 3052, 11225,
    7090, 2467, 3882, 2864, 10866, 10462, 3536, 3099, 1306, 10426
)

#: What to use.  ``get_z`` is for research; these two are for the submission.
RQMC_Z_SHIP = RQMC_Z_24989
RQMC_N_SHIP = 24989


def ship_lattice_base(n_points: int = RQMC_N_SHIP, chunk: int = 32768):
    """The recommended point set, billed, as ``setup`` would build it.

    ``docs/rqmc.md`` section 9 is the whole handover; this is its one line.
    Measured on an official-protocol MLP disjoint from the suite: setup bills
    76,866,676 FLOPs (0.00028 of B, and free in ``setup``), and

        corrected_sparse_kernel(W, tau=2.5, n_samples=24989, n_pilot=225,
                                beta=None, damp=0.0,
                                x0_fn=lattice_x0_fn(base))

    bills ``F/B = 0.2799`` with the scored row finite on every MLP.
    """
    z = RQMC_Z_SHIP if n_points == RQMC_N_SHIP else RQMC_Z_11987
    if n_points not in (RQMC_N_SHIP, 11987):
        raise ValueError(f"no literal generating vector for N={n_points}; "
                         "use get_z() offline and add it here")
    return billed_lattice_base(n_points, z, chunk)
