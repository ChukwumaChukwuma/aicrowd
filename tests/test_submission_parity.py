"""The shipped estimator and the research kernel must be the same algorithm.

`submission/estimator.py` is what the grader runs; `whestfloor/kernels.py` is
what every measurement in the ledger was taken with.  If they drift, every
number in the ledger becomes a claim about code that is not being submitted.
This test pins them together: identical predictions, bit for bit, and
identical FLOP counts.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_submission():
    if "whestbench" not in sys.modules:
        wb = types.ModuleType("whestbench")

        class BaseEstimator:  # minimal stand-in; the grader supplies the real one
            pass

        wb.BaseEstimator = BaseEstimator
        sys.modules["whestbench"] = wb
    spec = importlib.util.spec_from_file_location(
        "shipped_estimator", ROOT / "submission" / "estimator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _MLP:
    def __init__(self, w):
        self.width = w[0].shape[0]
        self.depth = len(w)
        self.weights = w
        self.seed = 7


def test_submission_matches_research_kernel():
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels
    from whestfloor.mc import make_mlp

    sub = _load_submission()
    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=3)]

    est = sub.Estimator()
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c1:
        a = np.asarray(est.predict(_MLP(W), int(1e12)))

    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c2:
        b = np.asarray(kernels.blend_kernel(
            W, n_samples=sub.N_SAMPLES, seed=7, wmc=sub.W_MC,
            kmax=sub.KMAX, g=sub.SHRINK, damp=sub.DAMP, umax=sub.UMAX))

    assert a.shape == b.shape == (6, 64)
    assert np.array_equal(a, b), (
        f"shipped and research kernels disagree; max |diff| = "
        f"{np.abs(a - b).max():.3e}")
    assert c1.flops_used == c2.flops_used, (
        f"FLOP counts differ: {c1.flops_used} vs {c2.flops_used}")


def test_submission_contract():
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor.contract import DEPTH, FREE_COMPUTE, LAMBDA_FLOPS_PER_SECOND, WIDTH
    from whestfloor.mc import make_mlp

    sub = _load_submission()
    est = sub.Estimator()

    class Ctx:
        seed = 0

    est.setup(Ctx())
    W = [fnp.asarray(x) for x in make_mlp(WIDTH, DEPTH, seed=0)]
    with flops.BudgetContext(flop_budget=272_000_000_000, quiet=True) as ctx:
        out = est.predict(_MLP(W), 272_000_000_000)
    o = np.asarray(out)
    assert o.shape == (DEPTH, WIDTH)
    assert np.isfinite(o).all()
    C = ctx.flops_used + LAMBDA_FLOPS_PER_SECOND * ctx.residual_wall_time_s
    # The HARD cap is the full budget: exceeding it zeroes the MLP and forces
    # the multiplier to 1.0, which is catastrophic. That is what must never
    # happen, and it is the only thing asserted here.
    assert C <= 272_000_000_000, f"effective compute {C:.3e} exceeds the budget"
    # Crossing FREE_COMPUTE is NOT a failure -- above it the multiplier grows
    # linearly while blending keeps lowering raw MSE, so the shipped estimator
    # sits deliberately just over. But FLOPs are machine-independent and
    # residual wall time is not, so the FLOP-only ratio is what we pin: it must
    # leave room for the grader's residual, which is billed at 1e11 FLOP/s on
    # one physical core.
    flop_ratio = ctx.flops_used / 272_000_000_000
    assert flop_ratio <= 0.10, (
        f"analytic FLOPs alone are {flop_ratio:.3f} of budget, leaving no "
        f"headroom for grader residual")
