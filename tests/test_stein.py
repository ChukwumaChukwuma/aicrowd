"""The Stein identity, and the audit trail of the feature list it did not join.

`docs/stein_cv.md` rests on two claims that are cheap to pin and expensive to
get silently wrong:

* `h(x) = c.grad psi(x) - (c.x) psi(x)` is exactly mean-zero for `psi` taken
  from a ReLU MLP's internals, with `c.grad psi` from a hand-written
  forward-mode tangent pass.  If the tangent recursion were wrong the whole
  measurement in that document would be measuring something else.
* the shipped head's column order is the same object in three files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _tangent_forward(x, W, c):
    """`(y, c.grad y)` for `y = relu(z^L)`, by the tangent recursion."""
    h, v = x, np.broadcast_to(c, x.shape).copy()
    for w in W:
        z = h @ w
        vz = v @ w
        mk = z > 0
        h = np.where(mk, z, 0.0)
        v = np.where(mk, vz, 0.0)
    return h, v


def test_tangent_pass_is_the_directional_derivative():
    """The hand-written tangent must equal a central finite difference.

    ReLU is not differentiable on the kink set, so the check is made away from
    it: `eps` is small enough that no sign flips between `x - eps c` and
    `x + eps c` for the samples used, which the assertion on the mask verifies
    implicitly by demanding agreement to 1e-4 relative.
    """
    from whestfloor.mc import make_mlp

    rng = np.random.default_rng(0)
    W = [w.astype(np.float64) for w in make_mlp(24, 5, seed=3)]
    x = rng.standard_normal((64, 24))
    c = rng.standard_normal(24)
    c /= np.linalg.norm(c)

    _, v = _tangent_forward(x, W, c)
    eps = 1e-6
    yp, _ = _tangent_forward(x + eps * c, W, c)
    ym, _ = _tangent_forward(x - eps * c, W, c)
    fd = (yp - ym) / (2.0 * eps)
    scale = max(np.abs(v).max(), 1e-12)
    assert np.abs(v - fd).max() / scale < 1e-4


def test_stein_identity_is_exactly_mean_zero():
    """`E[c.grad psi - (c.x) psi] = 0` to Monte-Carlo error.

    Small shape and a modest sample, so this runs in a second; the production
    measurement is `scripts/30_stein_cv.py --mode verify` at 256x32 and
    200,000 samples, which lands at an rms z-score of 1.278 over 3,072 tests
    per `psi` family.
    """
    from whestfloor.mc import make_mlp

    n, depth, N = 32, 6, 400_000
    W = [w.astype(np.float64) for w in make_mlp(n, depth, seed=11)]
    rng = np.random.default_rng(5)
    c = rng.standard_normal(n)
    c /= np.linalg.norm(c)

    s1 = np.zeros(n)
    s2 = np.zeros(n)
    done = 0
    while done < N:
        b = min(4096, N - done)
        x = rng.standard_normal((b, n))
        y, v = _tangent_forward(x, W, c)
        h = v - (x @ c)[:, None] * y
        s1 += h.sum(0)
        s2 += (h * h).sum(0)
        done += b
    mean = s1 / N
    sd = np.sqrt(np.maximum(s2 / N - mean * mean, 0.0))
    live = sd > 0
    z = mean[live] / (sd[live] / np.sqrt(N))
    # 3.9 is ~0.005 two-sided over 32 tests; a broken tangent pass or a
    # violated growth condition misses by orders of magnitude, not by 4 sigma.
    assert np.abs(z).max() < 3.9, f"max |z| = {np.abs(z).max():.2f}"


def test_shipped_feature_order_is_one_object():
    """`submission`, `whestfloor.corrector` and the beta on disk must agree."""
    import importlib.util
    import types

    from whestfloor import corrector as C

    if "whestbench" not in sys.modules:
        wb = types.ModuleType("whestbench")

        class BaseEstimator:
            pass

        wb.BaseEstimator = BaseEstimator
        sys.modules["whestbench"] = wb
    spec = importlib.util.spec_from_file_location(
        "shipped_estimator_features", ROOT / "submission" / "estimator.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)

    assert tuple(sub.FEATURES) == tuple(C.FEATURES)
    # FEATURES_FULL is the frozen 28-column research design that every
    # ablation table in docs/learned_corrector.md indexes; round 9 appended
    # the adapted degree-1 block as FEATURES_V2 rather than reordering it, so
    # the old tables still mean what they say.
    assert tuple(C.FEATURES_V2[:len(C.FEATURES_FULL)]) == tuple(C.FEATURES_FULL)
    assert set(C.FEATURES) | set(C.DROPPED) == set(C.FEATURES_V2)
    assert not set(C.FEATURES) & set(C.DROPPED)
    beta = np.load(ROOT / "submission" / "corrector.npz")["beta"]
    assert beta.shape == (C.N_FEATURES,), (beta.shape, C.N_FEATURES)
