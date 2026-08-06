"""Pins for the chaos-2 / conditional-independence research machinery.

These are the identities that make the measurements in
``scripts/15_chaos2_bakeoff.py`` mean something.  If any of them breaks, the
"chaos-2 beats the Gaussian model by only ~4x" claim is a claim about a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from whestfloor.chaos import (  # noqa: E402
    chaos2_relu_mean, chaos_pieces, gauss_relu_mean, normalised_hermite_coeffs,
)
from whestfloor.condindep import (  # noqa: E402
    indep_relu_mean_fft, relu_cumulants, relu_raw_moments,
)
from whestfloor.relu_moments import relu_hermite_coeffs, relu_mean  # noqa: E402


def _state(n=12, seed=3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n + 4))
    cov = A @ A.T / (n + 4) + 0.3 * np.eye(n)
    s = np.sqrt(np.diag(cov))
    R = cov / np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    m = 0.4 * rng.standard_normal(n)
    W = rng.standard_normal((n, n)) * np.sqrt(2.0 / n)
    return W, m, s, R, cov


def test_normalised_hermite_matches_closed_form():
    _, m, s, _, _ = _state()
    kmax = 8
    ah = normalised_hermite_coeffs(m, s, kmax)
    a = relu_hermite_coeffs(m, s, kmax)
    fact = np.cumprod(np.r_[1.0, np.arange(1, kmax + 1)])
    assert np.allclose(ah, a / np.sqrt(fact)[:, None], rtol=1e-12, atol=1e-14)


def test_chaos_variance_decomposition_is_the_exact_variance():
    """v_l + v_q + v_eta must be Var(S_j) -- checked against Monte Carlo."""
    W, m, s, R, cov = _state()
    pc = chaos_pieces(W, m, s, R, kmax=120)
    ev, U = np.linalg.eigh(cov)
    L = U * np.sqrt(np.clip(ev, 0.0, None))
    rng = np.random.default_rng(11)
    n_s, acc1, acc2 = 400_000, 0.0, 0.0
    for _ in range(20):
        z = rng.standard_normal((n_s // 20, len(m))) @ L.T + m
        S = np.maximum(z, 0.0) @ W
        acc1 = acc1 + S.sum(axis=0)
        acc2 = acc2 + (S * S).sum(axis=0)
    mu = acc1 / n_s
    var = acc2 / n_s - mu * mu
    assert np.max(np.abs(pc["c"] - mu)) < 8.0 * np.sqrt(var.max() / n_s)
    assert np.max(np.abs(pc["v_tot"] / var - 1.0)) < 0.02


def test_chaos2_collapses_to_the_gaussian_when_the_quadratic_form_vanishes():
    """With a_2 zeroed there is no chaos-2 term, so the CF inversion must
    reproduce the Gaussian answer to quadrature accuracy -- this is the same
    ablation discipline as ``cov_prop_edgeworth(damp=0)``."""
    W, m, s, R, _ = _state()
    pc = chaos_pieces(W, m, s, R, kmax=60)
    pc0 = dict(pc)
    pc0["G2"] = pc["G2"] * 0.0
    pc0["v_eta"] = pc["v_tot"] - pc["v_l"]
    got = chaos2_relu_mean(W, m, s, R, pc0)
    assert np.max(np.abs(got - gauss_relu_mean(pc))) < 1e-11


def test_chaos2_quadrature_is_converged():
    W, m, s, R, _ = _state()
    pc = chaos_pieces(W, m, s, R, kmax=60)
    a = chaos2_relu_mean(W, m, s, R, pc, n_panel=48, order=24)
    b = chaos2_relu_mean(W, m, s, R, pc, n_panel=96, order=32)
    assert np.max(np.abs(a - b)) < 1e-10


def test_relu_raw_moment_recurrence():
    m = np.array([-1.0, 0.0, 0.7])
    s = np.array([0.5, 1.0, 1.3])
    M = relu_raw_moments(m, s, 4)
    rng = np.random.default_rng(0)
    y = rng.standard_normal((4_000_000, 3)) * s + m
    r = np.maximum(y, 0.0)
    for p in (1, 2, 3, 4):
        emp = (r ** p).mean(axis=0)
        assert np.allclose(M[p], emp, rtol=0.02, atol=1e-3)


def test_indep_fft_matches_the_independent_sum():
    """With lam = 1 and a = 0 the renormalised model IS the independent sum,
    so the FFT evaluation can be checked against Monte Carlo directly."""
    rng = np.random.default_rng(5)
    n = 10
    m = 0.5 * rng.standard_normal(n)
    s = np.exp(0.3 * rng.standard_normal(n))
    W = rng.standard_normal((n, 3)) * np.sqrt(2.0 / n)
    mu_i, c2, _, _ = relu_cumulants(m, s)
    c_ex = W.T @ mu_i
    v_ex = (W * W).T @ c2
    p = indep_relu_mean_fft(W, m, s, c_ex, v_ex, log2m=17)
    n_s = 4_000_000
    acc = np.zeros(3)
    for _ in range(20):
        y = rng.standard_normal((n_s // 20, n)) * s + m
        acc = acc + np.maximum(np.maximum(y, 0.0) @ W, 0.0).sum(axis=0)
    mc = acc / n_s
    assert np.max(np.abs(p - mc)) < 6.0 * 0.5 / np.sqrt(n_s)


def test_gauss_model_is_the_exact_rectified_gaussian():
    W, m, s, R, _ = _state()
    pc = chaos_pieces(W, m, s, R, kmax=40)
    assert np.allclose(gauss_relu_mean(pc),
                       relu_mean(pc["c"], np.sqrt(pc["v_tot"])))
