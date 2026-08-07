"""Assert the claims ``whestfloor/trajclosure.py`` and ``docs/traj_closure.md``
rest on, rather than assuming them.

Three of these matter.  The first two layers of the closure are EXACT, so the
``r`` yardstick starts from a chain whose first error is injected at layer 3
and not before.  The analytic cumulant catalogue must agree with the one
``docs/cumulant_expansion.md`` verified against Monte Carlo, or the ``r = 1.41``
baseline of §1 is not the shipped ``kappa_3`` arm.  And ``ChainDriver`` must
produce exactly what ``chain`` produces, or the DAgger fit of §3 is not fitting
the chain that gets deployed.

No pytest in this environment: run it as ``$PY tests/test_trajclosure.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import trajclosure as tc  # noqa: E402
from whestfloor.cumulants import coeffs, kappa3_star  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import relu_cov_exact_centered  # noqa: E402

SEED = 987_654
NW, ND = 64, 6
_W = [w.astype(np.float64) for w in make_mlp(NW, ND, SEED)]


def test_the_exactness_boundary_is_where_it_is_claimed_to_be():
    """``z^1 = x W^1`` is exactly Gaussian, so ``E[relu(z^1)]``, ``Cov(relu(z^1))``
    and hence ``E[z^2]``, ``Cov(z^2)`` are closed form.  ``E[relu(z^2)]`` is
    NOT -- ``z^2`` is a sum of rectified Gaussians, not a Gaussian -- and
    ``docs/integrable_cv.md`` §3 draws the boundary in exactly that place.
    Both directions are asserted, because the ``r`` yardstick's ``L = 1``
    reference point depends on the first and its ``L >= 2`` rows on the second.
    """
    recs = list(tc.chain(_W, src="coinc"))
    rng = np.random.default_rng(7)
    n_mc, chunk = 400_000, 8192
    sh = [np.zeros(NW) for _ in range(3)]
    qh = [np.zeros(NW) for _ in range(3)]
    sz2 = np.zeros(NW)
    qz2 = np.zeros(NW)
    done = 0
    while done < n_mc:
        m = min(chunk, n_mc - done)
        h = rng.standard_normal((m, NW))
        for k in range(3):
            pre = h @ _W[k]
            if k == 1:
                sz2 += pre.sum(0)
                qz2 += (pre * pre).sum(0)
            h = np.maximum(pre, 0.0)
            sh[k] += h.sum(0)
            qh[k] += (h * h).sum(0)
        done += m

    def zscore(pred, s, q):
        mh = s / n_mc
        se = np.sqrt(np.maximum(q / n_mc - mh * mh, 1e-30) / n_mc)
        return float(np.sqrt(np.mean(((pred - mh) / se) ** 2)))

    # exact: E[relu(z^1)], and the pre-activation mean and variance of z^2
    assert zscore(recs[0]["mu"], sh[0], qh[0]) < 2.0
    assert zscore(recs[1]["m"], sz2, qz2) < 2.0
    v2 = qz2 / n_mc - (sz2 / n_mc) ** 2
    assert np.max(np.abs(np.diag(recs[1]["C"]) / v2 - 1.0)) < 0.02
    # NOT exact, and this is the first line of the closure's error budget
    assert zscore(recs[1]["mu"], sh[1], qh[1]) > 5.0
    assert zscore(recs[2]["mu"], sh[2], qh[2]) > 5.0


def test_layer1_covariance_is_the_arc_cosine_kernel():
    """Layer 1 uses the closed form, not the truncated Mehler series."""
    C = _W[0].T @ _W[0]
    s = np.sqrt(np.diag(C))
    _, Ch0, *_ = tc.closure_step(np.zeros(NW), C, exact_acos=True)
    exact = relu_cov_exact_centered(C, s)
    np.fill_diagonal(exact, np.diag(Ch0))
    assert np.max(np.abs(Ch0 - exact)) < 1e-12


def test_star_diagram_matches_the_shipped_arm():
    """``kappa3_star_np`` is ``whestfloor.cumulants.kappa3_star`` to round-off,
    which ``scripts/13`` validated against Monte Carlo."""
    rec = list(tc.chain(_W, src="coinc"))[3]
    mine = tc.kappa3_star_np(_W[4], rec["acoef"], rec["rho"], umax=1)
    theirs = kappa3_star(_W[4], rec["m"], rec["sig"], rec["rho"], umax=1)
    scale = max(float(np.max(np.abs(theirs))), 1e-300)
    assert np.max(np.abs(mine - theirs)) <= 1e-12 * scale


def test_kappa_x_matches_the_verified_coefficients():
    """Marginal ``kappa_2..4`` of ``relu(z)`` agree with ``cumulants.coeffs``."""
    rec = list(tc.chain(_W, src="coinc"))[3]
    got = tc.kappa_x(rec["acoef"], rec["mu0"], rec["sig"], rec["alpha"],
                     rec["pa"], rec["Pa"])
    want = coeffs(rec["m"], rec["sig"], 8)[4:7]
    for a, b in zip(got, want):
        assert np.max(np.abs(a - b)) <= 1e-10 * max(float(np.max(np.abs(b))),
                                                    1e-30)


def test_driver_reproduces_the_generator():
    """``ChainDriver`` is the suspendable form of ``chain``: with the same
    (zero) coefficients it must yield the same trajectory, for every cumulant
    source, or the DAgger fit is fitting a different chain."""
    for src in ("coinc", "star", "tree"):
        kw = dict(mspec=tc.MEAN_JRH, vspec=tc.VAR_JRH, ospec=tc.OFF_JRH,
                  src=src, want_off=True)
        ref = tc.run(_W, coefs=None, **kw)
        d = tc.ChainDriver(_W, **kw)
        got = []
        for _ in range(len(_W)):
            d.step()
            got.append(d.apply(np.zeros(len(tc.MEAN_JRH)), np.zeros(0),
                               np.zeros(0)))
        assert np.max(np.abs(np.stack(got) - ref)) < 1e-12, src


def test_corrections_actually_move_the_chain():
    """A zero coefficient must be a no-op; a non-zero one must move every
    later layer, so a fitted zero is a measurement and not a plumbing bug."""
    kw = dict(mspec=("sphi",), vspec=(), ospec=(), src="coinc")
    zero = [(np.zeros(1), np.zeros(0), np.zeros(0))] * len(_W)
    one = [(np.array([0.05]), np.zeros(0), np.zeros(0))] * len(_W)
    base = tc.run(_W, coefs=None, **kw)
    assert np.max(np.abs(tc.run(_W, coefs=zero, **kw) - base)) < 1e-14
    moved = np.max(np.abs(tc.run(_W, coefs=one, **kw) - base), axis=1)
    assert np.min(moved) > 0.0, moved


def test_the_deployable_pack_never_touches_a_sample():
    """The probe-free arm must be a function of the weights alone -- that is
    the entire reason it is preferred to jamesrahenry's probe-fed one."""
    kw = dict(mspec=tc.MEAN_JRH, vspec=(), ospec=(), src="star")
    coefs = [(np.arange(8) * 0.01, np.zeros(0), np.zeros(0))] * len(_W)
    np.random.seed(1)
    a = tc.run(_W, coefs=coefs, **kw)
    np.random.seed(2)
    b = tc.run(_W, coefs=coefs, **kw)
    assert np.array_equal(a, b)


def test_layer1_cumulants_are_identically_zero():
    """``z^1`` is exactly Gaussian, so every cumulant feature at layer 1 is
    zero -- which is why the per-layer fit must use a pseudo-inverse."""
    rec = list(tc.chain(_W, src="star"))[0]
    assert np.max(np.abs(rec["k3"])) == 0.0
    assert np.max(np.abs(rec["k4"])) == 0.0


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
