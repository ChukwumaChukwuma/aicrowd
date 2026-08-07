"""The algebra behind ``scripts/32_adapted_hermite.py``, checked numerically.

Every number in ``docs/hermite_rank_ceiling.md`` rests on four identities that
are asserted here rather than assumed:

* the normalised Hermite recurrence is ``He_d / sqrt(d!)``;
* ``Cov(h_d(<a,x>), h_e(<b,x>)) = delta_de <a,b>^d`` (Mehler), which is what
  makes the Gram analytic and every feature exactly mean zero;
* ``h_d(<a,x>)`` is the unit RANK-ONE tensor ``a^(x)d`` of the degree-d chaos,
  i.e. ``h_d(<a,x>) = sum_{|alpha|=d} sqrt(d!/alpha!) a^alpha H_alpha`` --
  the identity the whole ceiling argument turns on;
* the OU/Mehler semigroup reads the degree spectrum:
  ``Cov(y(x), y(x_t)) = sum_d f_d t^d``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "adapted_hermite", ROOT / "scripts" / "32_adapted_hermite.py")
AH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AH)


def _herm_scalar(s, d):
    """``He_d(s)`` by the textbook recurrence, unnormalised."""
    hm1, h = np.ones_like(s), s
    for k in range(1, d):
        hm1, h = h, s * h - k * hm1
    return h


def test_herm_table_is_normalised_hermite():
    s = np.linspace(-4.0, 4.0, 41).astype(np.float32)[:, None]
    tab = AH.herm_table(s, 10)
    for d in range(1, 11):
        want = _herm_scalar(s.astype(np.float64), d) / math.sqrt(
            float(math.factorial(d)))
        assert np.allclose(tab[d - 1], want, rtol=2e-4, atol=2e-4), d


def test_mehler_covariance_and_exact_mean_zero():
    """``E[h_d] = 0`` and ``Cov(h_d(s_a), h_e(s_b)) = delta_de <a,b>^d``.

    Only degrees 1-3 are checked tightly: ``h_d`` has a fourth moment that
    grows fast with ``d`` (``E[h_5^4] ~ 1.4e4``), so a Monte-Carlo check of the
    degree-5 diagonal at 4e5 samples carries +-0.2 of noise from the estimator,
    not from the identity.  The identity itself is exact.
    """
    rng = np.random.default_rng(0)
    n, N, dm = 24, 400_000, 3
    A = rng.standard_normal((n, 3))
    A /= np.linalg.norm(A, axis=0)
    x = rng.standard_normal((N, n), dtype=np.float32)
    tab = AH.herm_table(x @ A.astype(np.float32), dm)
    F = tab.transpose(1, 0, 2).reshape(N, -1).astype(np.float64)
    assert np.abs(F.mean(0)).max() < 12.0 / math.sqrt(N)
    emp = F.T @ F / N
    want = np.zeros_like(emp)
    ip = A.T @ A
    for d in range(1, dm + 1):
        for r in range(3):
            for q in range(3):
                want[(d - 1) * 3 + r, (d - 1) * 3 + q] = ip[r, q] ** d
    assert np.abs(emp - want).max() < 0.03


def test_pure_power_is_the_rank_one_tensor():
    """``h_d(<a,x>) = sum_{|alpha|=d} sqrt(d!/alpha!) a^alpha H_alpha``.

    Checked in a small dimension by expanding both sides on the same sample:
    this is the identity that says a one-direction Hermite feature is exactly
    a RANK-ONE symmetric tensor, hence the ceiling.
    """
    rng = np.random.default_rng(1)
    n, N, d = 4, 200_000, 3
    a = rng.standard_normal(n)
    a /= np.linalg.norm(a)
    x = rng.standard_normal((N, n), dtype=np.float32)
    lhs = AH.herm_table((x @ a.astype(np.float32))[:, None], d)[d - 1][:, 0]
    coord = AH.herm_table(x, d)                              # (d, N, n)
    rhs = np.zeros(N, dtype=np.float64)
    import itertools
    for comb in itertools.combinations_with_replacement(range(n), d):
        alpha = [comb.count(i) for i in range(n)]
        c = math.sqrt(math.factorial(d) / math.prod(
            math.factorial(k) for k in alpha))
        term = np.full(N, c, dtype=np.float64)
        for i, k in enumerate(alpha):
            if k:
                term = term * coord[k - 1][:, i] * (a[i] ** k)
        rhs += term
    assert np.abs(lhs - rhs).max() < 2e-3


def test_design_gram_matches_the_sample_covariance():
    """The analytic Gram is the population covariance of the built design."""
    n, M, N = 32, 4, 300_000
    rng = np.random.default_rng(2)
    A = np.linalg.qr(rng.standard_normal((n, M)))[0].astype(np.float32)
    Wn = rng.standard_normal((n, n)).astype(np.float32)
    Wn /= np.linalg.norm(Wn, axis=0)
    old = (AH.TENS_PLAN, AH.PURE_DMAX, AH.PURE_M, AH.COORD_DMAX)
    AH.TENS_PLAN, AH.PURE_DMAX, AH.PURE_M, AH.COORD_DMAX = (
        ((1, 4), (2, 3), (3, 2)), 5, 2, 3)
    try:
        des = AH.Design(A, Wn)
        G = des.gram()
        acc = np.zeros((des.p, des.p))
        done = 0
        while done < N:
            B = min(4096, N - done)
            D = des.build(rng.standard_normal((B, n), dtype=np.float32))
            acc += D.T.astype(np.float64) @ D.astype(np.float64)
            done += B
        emp = acc / N
    finally:
        AH.TENS_PLAN, AH.PURE_DMAX, AH.PURE_M, AH.COORD_DMAX = old
    scale = np.sqrt(np.outer(np.diag(G), np.diag(G)))
    assert np.abs((emp - G) / scale).max() < 0.05


def test_ou_semigroup_reads_the_degree_spectrum():
    """``Cov(y(x), y(x_t)) / Var(y) = sum_d f_d t^d`` for a known ``y``."""
    rng = np.random.default_rng(3)
    n, N = 16, 400_000
    a = rng.standard_normal(n)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(n)
    b /= np.linalg.norm(b)
    af, bf = a.astype(np.float32), b.astype(np.float32)

    def y(z):                       # 1 part degree 1, 3 parts degree 4
        return (AH.herm_table((z @ af)[:, None], 1)[0][:, 0]
                + math.sqrt(3.0) * AH.herm_table((z @ bf)[:, None], 4)[3][:, 0])

    x = rng.standard_normal((N, n), dtype=np.float32)
    xi = rng.standard_normal((N, n), dtype=np.float32)
    y0 = y(x).astype(np.float64)
    v = y0.var()
    for t in (0.3, 0.6, 0.9):
        xt = np.float32(t) * x + np.float32(math.sqrt(1 - t * t)) * xi
        c = float(np.mean(y0 * y(xt).astype(np.float64))
                  - y0.mean() * y(xt).astype(np.float64).mean()) / v
        assert abs(c - (0.25 * t + 0.75 * t ** 4)) < 0.02, t


def test_degree2_rank_recovers_a_known_rank_one_tensor():
    """``sum_j S_j^2`` for ``y_1 = (a.x)^2 - 1``: rank one, trace 4."""
    n, B = 16, 6000
    rng = np.random.default_rng(4)
    a = rng.standard_normal(n)
    a /= np.linalg.norm(a)
    XA = rng.standard_normal((B, n))
    XB = rng.standard_normal((B, n))

    def yy(z):
        out = np.zeros((len(z), n))
        out[:, 0] = (z @ a) ** 2 - 1.0
        return out

    YA, YB = yy(XA), yy(XB)
    YA -= YA.mean(0)
    YB -= YB.mean(0)
    Wm = YA @ YB.T
    U = XA @ XB.T
    P = XA.T @ ((Wm * U) @ XB)
    P -= XA.T @ (Wm.sum(1)[:, None] * XA)
    P -= XB.T @ (Wm.sum(0)[:, None] * XB)
    P += float(Wm.sum()) * np.eye(n)
    P = 0.5 * (P + P.T) / (B * B)
    ev = np.sort(np.linalg.eigvalsh(P))[::-1]
    assert abs(float(np.trace(P)) - 4.0) < 0.5
    assert ev[0] / float(np.sum(np.maximum(ev, 0.0))) > 0.9
    assert abs(abs(float(np.linalg.eigh(P)[1][:, -1] @ a)) - 1.0) < 0.05


def test_hard_upper_bound_is_valid_on_a_known_spectrum():
    """``hard_upper`` never under-states, on a deliberately spiky spectrum."""
    f = np.zeros(40)
    f[0], f[3], f[19] = 0.3, 0.4, 0.3
    ts = np.array(AH.T_LOW + AH.T_HIGH)
    C = np.array([float(np.sum(f * t ** np.arange(1, 41))) for t in ts])
    for D in (1, 2, 4, 8, 20):
        assert AH.hard_upper(ts, C, D) >= float(np.sum(f[:D])) - 1e-12


def test_nnls_recovers_a_smooth_spectrum():
    """Inverting ``C(t)`` is an ill-posed Hausdorff moment problem: it is
    accurate for a SMOOTH spectrum (which is what the network has -- ``f_d``
    decays geometrically) and only indicative for a spiky one.  A geometric
    spectrum is recovered to ~2 points of cumulative mass."""
    d = np.arange(1, 61)
    f = 0.62 ** d
    f /= f.sum()
    ts = np.array(AH.T_LOW + AH.T_HIGH)
    C = np.array([float(np.sum(f * t ** d)) for t in ts])
    fit = AH.nnls_spectrum(ts, C, dmax=60, iters=300_000)
    for D in (1, 2, 4, 8, 16):
        assert abs(np.sum(fit[:D]) - np.sum(f[:D])) < 0.02, D
        assert AH.hard_upper(ts, C, D) >= float(np.sum(f[:D])) - 1e-12


if __name__ == "__main__":
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {nm}")
            except AssertionError as e:  # noqa: PERF203
                fails += 1
                print(f"FAIL {nm}: {e}")
    raise SystemExit(1 if fails else 0)
