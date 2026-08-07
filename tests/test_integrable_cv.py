"""The identities ``docs/integrable_cv.md`` rests on, asserted numerically.

The page turns on one claim: **the last exactly computable objects in the
network are ``E[h^1]``, ``Cov(h^1)``, ``E[z^2]`` and ``Cov(z^2)``.**  If that is
wrong, the ``q2`` block of sec 4 is a biased control variate and the depth
ladder's ``L = 1`` reference point is wrong too.  It is pinned here two ways --
against the closed-form arc-cosine kernel, and against Monte Carlo -- together
with the two exact symmetries of sec 5 and the shrinkage algebra of sec 1.
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
    "integrable_cv", ROOT / "scripts" / "41_integrable_cv.py")
ICV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ICV)

from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import relu_cov_exact_centered  # noqa: E402

N = 64  # a small width keeps the Monte-Carlo error visible against the claim
DEPTH = 4


def _mlp(seed=3, width=N, depth=DEPTH):
    return make_mlp(width, depth, seed)


def test_layer1_mehler_matches_the_arc_cosine_kernel():
    """At layer 1 the mean is exactly zero, so Mehler must reproduce arccos."""
    W = _mlp()
    mz, Cz, mh, Ch = ICV.analytic_moments(W)
    s = np.sqrt(np.diag(Cz[0]))
    exact = relu_cov_exact_centered(Cz[0], s)
    assert np.max(np.abs(Ch[0] - exact)) < 1e-12, np.max(np.abs(Ch[0] - exact))


def test_layer1_and_layer2_moments_are_exact_against_monte_carlo():
    """``E[h^1]``, ``Cov(h^1)``, ``E[z^2]``, ``Cov(z^2)``: exact, not modelled.

    Layer 3 is included as the control: the same machinery is *wrong* there,
    by a margin far outside the Monte-Carlo error, which is the whole content
    of the depth ladder.
    """
    W = _mlp()
    mz, Cz, mh, Ch = ICV.analytic_moments(W)
    rng = np.random.default_rng(0)
    M = 2_000_000
    s1 = np.zeros(N)
    s2 = np.zeros((N, N))
    sz2 = np.zeros(N)
    szz = np.zeros((N, N))
    sz3 = np.zeros(N)
    done = 0
    while done < M:
        m = min(20000, M - done)
        x = rng.standard_normal((m, N))
        h1 = np.maximum(x @ W[0], 0.0)
        z2 = h1 @ W[1]
        z3 = np.maximum(z2, 0.0) @ W[2]
        s1 += h1.sum(0)
        s2 += h1.T @ h1
        sz2 += z2.sum(0)
        szz += z2.T @ z2
        sz3 += z3.sum(0)
        done += m
    m1 = s1 / M
    C1 = s2 / M - np.outer(m1, m1)
    m2 = sz2 / M
    C2 = szz / M - np.outer(m2, m2)
    m3 = sz3 / M

    sd1 = np.sqrt(np.diag(C1))
    se1 = sd1 / math.sqrt(M)
    assert np.max(np.abs(mh[0] - m1) / se1) < 6.0
    assert np.max(np.abs(Ch[0] - C1) / np.outer(sd1, sd1)) < 6.0 / math.sqrt(M) * 3

    sd2 = np.sqrt(np.diag(C2))
    assert np.max(np.abs(mz[1] - m2) / (sd2 / math.sqrt(M))) < 6.0
    assert np.max(np.abs(Cz[1] - C2) / np.outer(sd2, sd2)) < 6.0 / math.sqrt(M) * 3

    # ...and the control: layer 3 is NOT exact.  z^2 is not Gaussian, so the
    # closure's E[relu(z^2)] is wrong and the error shows up in E[z^3].
    z3rel = np.max(np.abs(mz[2] - m3) / (np.sqrt(np.diag(Cz[2])) / math.sqrt(M)))
    assert z3rel > 20.0, z3rel


def test_antithetic_pair_has_no_odd_hermite_degree():
    """``(y(x) + y(-x))/2`` kills every odd degree exactly, for any network."""
    W = _mlp()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((40000, N))
    fwd = lambda v: ICV.forward(W, v)[0]  # noqa: E731
    p = 0.5 * (fwd(x) + fwd(-x))
    # any odd-degree functional must have zero covariance with the pair mean;
    # the sharpest cheap probe is the degree-1 one, <x, .>
    q = fwd(x)
    c_pair = np.abs(x.T @ (p - p.mean(0))).max() / len(x)
    c_plain = np.abs(x.T @ (q - q.mean(0))).max() / len(x)
    assert c_pair < 0.08 * c_plain, (c_pair, c_plain)


def test_homogeneity_is_exact_and_E_chi_is_right():
    """No biases anywhere, so ``y(c x) = c y(x)`` for ``c > 0``, exactly."""
    W = _mlp()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((512, N))
    y1 = ICV.forward(W, x)[0]
    y3 = ICV.forward(W, 3.0 * x)[0]
    assert np.max(np.abs(y3 - 3.0 * y1)) < 1e-10 * max(1.0, np.abs(y1).max())
    # E||x|| for chi_n, against Monte Carlo
    n = 256
    er = math.sqrt(2.0) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))
    r = np.linalg.norm(rng.standard_normal((400000, n)), axis=1)
    assert abs(er - r.mean()) < 6.0 * r.std() / math.sqrt(len(r))


def test_shrinkage_optimum_matches_the_closed_form():
    """``theta* = D/(D+b^2)`` and the saving is ``D^2/(D+b^2)``."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        D, b2, base = rng.uniform(1e-8, 1e-5), rng.uniform(0, 1e-5), 1e-5
        f = lambda t: base - 2 * t * D + t * t * D + t * t * b2  # noqa: E731
        ts = np.linspace(0, 1, 200001)
        assert abs(ts[np.argmin(f(ts))] - D / (D + b2)) < 1e-4
        assert abs((base - f(D / (D + b2))) - D * D / (D + b2)) < 1e-14


def test_corrector_blocks_match_the_research_script():
    """``whestfloor.corrector``'s round-12 blocks are the measured ones.

    ``docs/integrable_cv.md`` sec 4's numbers come from
    ``scripts/41_integrable_cv.py``; the deployable copies in
    :mod:`whestfloor.corrector` must be the same objects or the page stops
    describing what an integrator would ship.
    """
    from whestfloor import corrector as C

    W = _mlp()
    mz, Cz, mh, Ch = ICV.analytic_moments(W)
    mh1, Ch1, m2, C2 = C.layer12_moments(W)
    assert np.max(np.abs(mh1 - mh[0])) < 1e-14
    assert np.max(np.abs(Ch1 - Ch[0])) < 1e-14
    assert np.max(np.abs(m2 - mz[1])) < 1e-13
    assert np.max(np.abs(C2 - Cz[1])) < 1e-13

    A = C.kink_frame(W, mz, Cz, 8)
    Ak = ICV.kink_frames(W, mz, Cz, 8)[1]
    assert np.max(np.abs(A.T @ A - np.eye(8))) < 1e-12
    # same subspace, up to the sign of each eigenvector
    assert np.max(np.abs(np.abs(A.T @ Ak) - np.eye(8))) < 1e-9


def test_quad2_features_are_exactly_mean_zero():
    """The whole point: ``E[v_a v_b] = (A2' Cov(z^2) A2)_ab`` with no model.

    A biased control variate would show up here as a systematic offset; the
    check is against Monte-Carlo error, at 2e6 samples, on every one of the
    ``k(k+1)/2`` features at once.
    """
    from whestfloor import corrector as C

    W = _mlp()
    mz, Cz, _, _ = ICV.analytic_moments(W)
    _, _, m2, C2 = C.layer12_moments(W)
    A = C.kink_frame(W, mz, Cz, 8)
    Cv = A.T @ C2 @ A
    iu, ju = np.triu_indices(8)
    rng = np.random.default_rng(7)
    M, s1, s2 = 2_000_000, 0.0, 0.0
    done = 0
    while done < M:
        m = min(20000, M - done)
        x = rng.standard_normal((m, N))
        z2 = np.maximum(x @ W[0], 0.0) @ W[1]
        v = (z2 - m2) @ A
        g = v[:, iu] * v[:, ju] - Cv[iu, ju]
        s1 += g.sum(0)
        s2 += np.einsum("ij,ij->j", g, g)
        done += m
    mean = s1 / M
    sd = np.sqrt(np.maximum(s2 / M - mean * mean, 1e-300))
    z = np.abs(mean) / (sd / math.sqrt(M))
    assert np.max(z) < 6.0, (np.max(z), np.argmax(z))


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
