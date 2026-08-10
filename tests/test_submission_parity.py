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
    def __init__(self, w, seed=7):
        self.width = w[0].shape[0]
        self.depth = len(w)
        self.weights = w
        self.seed = seed


class _Ctx:
    submission_dir = str(ROOT / "submission")
    seed = 0


def test_submission_fallback_matches_research_kernel():
    """The defensive path must be the same algorithm too.

    `predict` swallows any exception from the sparse path and falls back to a
    dense pass.  A fallback that silently differed from the research kernel
    would be an unmeasured estimator running on exactly the MLPs where things
    went wrong, so it is pinned as tightly as the main path.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels
    from whestfloor.mc import make_mlp

    sub = _load_submission()
    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=5)]
    est = sub.Estimator()

    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c1:
        a = np.asarray(est._dense(_MLP(W), sub.FALLBACK_SAMPLES, 7))
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c2:
        b = np.asarray(kernels._dense_rows(W, kernels.SPARSE_FALLBACK_SAMPLES,
                                           7))

    assert sub.FALLBACK_SAMPLES == kernels.SPARSE_FALLBACK_SAMPLES
    assert np.array_equal(a, b)
    assert c1.flops_used == c2.flops_used


def test_sparse_ablation_is_exactly_dense():
    """`tau=None` must reproduce plain MC bit for bit through the same path.

    That is what makes the reported 1.44x an ablation rather than a
    comparison between two different programs.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels
    from whestfloor.mc import make_mlp

    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=11)]
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        a = np.asarray(kernels.sparse_mc_kernel(W, tau=None, n_samples=512,
                                                n_pilot=64, seed=3))
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        b = np.asarray(kernels._sparse_mc(W, None, 512, 64, 3))
    assert np.array_equal(a, b)

    # and the pruned path must actually differ, or the ablation is vacuous
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        c = np.asarray(kernels.sparse_mc_kernel(W, tau=2.5, n_samples=512,
                                                n_pilot=64, seed=3))
    assert not np.array_equal(a[-1], c[-1])


def test_submission_matches_research_kernel():
    """Shipped corrector == research corrector, bit for bit and FLOP for FLOP."""
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels
    from whestfloor.mc import make_mlp

    sub = _load_submission()
    est = sub.Estimator()
    est.setup(_Ctx())
    assert est._beta is not None, "submission/corrector.npz did not load"
    assert est._beta2 is not None, "submission/bigcorr_head.npz did not load"
    beta = np.asarray(est._beta)
    beta2 = np.asarray(est._beta2)
    assert beta2.shape == (len(sub.FEATURES2),)

    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=3)]
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c1:
        a = np.asarray(est.predict(_MLP(W), int(1e12)))
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c2:
        b = np.asarray(kernels.corrected_sparse_kernel(
            W, tau=sub.TAU, n_samples=sub.N_SAMPLES, n_pilot=sub.N_PILOT,
            seed=7, beta=beta, beta2=beta2, damp=sub.DAMP, kmax=sub.CV_KMAX))

    assert a.shape == b.shape == (6, 64)
    assert np.array_equal(a, b), (
        f"shipped and research kernels disagree; max |diff| = "
        f"{np.abs(a - b).max():.3e}")
    assert c1.flops_used == c2.flops_used, (
        f"FLOP counts differ: {c1.flops_used} vs {c2.flops_used}")


def test_corrector_damp_zero_is_exactly_uncorrected():
    """`DAMP = 0` must reproduce the previous ship bit for bit.

    Bar 3 of the corrector work: the correction has to be ablatable through
    the identical code path, not by swapping in a different program.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels
    from whestfloor.mc import make_mlp

    sub = _load_submission()
    est = sub.Estimator()
    est.setup(_Ctx())
    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=17)]

    old = sub.DAMP
    try:
        sub.DAMP = 0.0
        with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c1:
            a = np.asarray(est.predict(_MLP(W, 21), int(1e12)))
    finally:
        sub.DAMP = old
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c2:
        b = np.asarray(kernels.sparse_mc_kernel(
            W, tau=sub.TAU, n_samples=sub.N_SAMPLES, n_pilot=sub.N_PILOT,
            seed=21))
    assert np.array_equal(a, b)
    assert c1.flops_used == c2.flops_used

    # ... and the corrected path must actually differ, or it is vacuous
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        c = np.asarray(est.predict(_MLP(W, 21), int(1e12)))
    assert not np.array_equal(a[-1], c[-1])


def test_missing_coefficient_file_degrades_not_fails():
    """A head that will not load must leave a working estimator behind."""
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels
    from whestfloor.mc import make_mlp

    sub = _load_submission()
    est = sub.Estimator()

    class Bad:
        submission_dir = "/nonexistent-path-for-this-test"
        seed = 0

    est.setup(Bad())
    assert est._beta is None
    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=23)]
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        a = np.asarray(est.predict(_MLP(W, 29), int(1e12)))
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        b = np.asarray(kernels.sparse_mc_kernel(
            W, tau=sub.TAU, n_samples=sub.N_SAMPLES, n_pilot=sub.N_PILOT,
            seed=29))
    assert np.array_equal(a, b)


def test_numpy_feature_extractor_matches_the_shipped_kernel():
    """The training features and the deployed features must be the same map.

    ``whestfloor.corrector.sparse_mc_features`` is the numpy generator used to
    build the training set; the flopscope kernel is what runs at grade time.
    They are not bitwise equal because the generator accumulates the design in
    float64 (strictly better for the fit) while the shipped path stays in
    float32, so this pins the agreement at 1e-6 absolute -- three orders below
    the ~1.7e-3 residual the head is predicting.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import corrector as C
    from whestfloor import kernels
    from whestfloor.mc import make_mlp

    rng = np.random.default_rng(0)
    beta = (rng.standard_normal(C.N_FEATURES) * 1e-3).astype(np.float32)
    Wn = make_mlp(128, 8, 4242)
    mu, f = C.sparse_mc_features(Wn, seed=4242, n_samples=1200, n_pilot=60,
                                 kmax=2)
    pred_np = mu + C.build_design(f) @ beta.astype(np.float64)

    W = [fnp.asarray(w) for w in Wn]
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        out = kernels.corrected_sparse_kernel(
            W, tau=2.5, n_samples=1200, n_pilot=60, seed=4242, beta=beta,
            kmax=2, safe=False)
    assert np.abs(pred_np - np.asarray(out)[-1]).max() < 1e-6


def test_scaled_head_absent_degrades_to_the_fifteen_float_head():
    """The ladder has THREE rungs and the middle one must be reachable.

    A corrupt or missing `bigcorr_head.npz` must leave the previously shipped
    15-float head running, not no head at all -- and it must be the SAME
    15-float estimator that every number in `docs/learned_corrector.md` was
    measured with, bit for bit and FLOP for FLOP.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels
    from whestfloor.mc import make_mlp

    sub = _load_submission()
    est = sub.Estimator()
    est.setup(_Ctx())
    est._beta2 = None                      # as if the file were unreadable
    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=3)]
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c1:
        a = np.asarray(est.predict(_MLP(W), int(1e12)))
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c2:
        b = np.asarray(kernels.corrected_sparse_kernel(
            W, tau=sub.TAU, n_samples=sub.N_SAMPLES, n_pilot=sub.N_PILOT,
            seed=7, beta=np.asarray(est._beta), damp=sub.DAMP,
            kmax=sub.CV_KMAX))
    assert np.array_equal(a, b)
    assert c1.flops_used == c2.flops_used


def test_scaled_head_matches_the_numpy_generator():
    """The training features and the deployed features must be the same map.

    `whestfloor.bigcorr.extract` built every row the 20 coefficients were
    fitted on; the flopscope path is what runs at grade time.  They are not
    bitwise equal -- the generator accumulates in float64, the shipped path
    stays in float32 outside the Gram solve -- so this pins the agreement at
    1e-6 absolute, three orders below the ~1e-3 residual being predicted.

    It also pins the COLUMN ORDER, which is the one thing a silent change
    would make catastrophic rather than merely wrong: `beta` is indexed by
    position and a permuted design would still run and still return finite
    numbers.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import bigcorr as BC
    from whestfloor import kernels
    from whestfloor.mc import make_mlp

    sub = _load_submission()
    channels = ("relu1", "mfv2", "mfm", "dpilot", "cv1")
    expect = ["one", "s", "Phi", "phi", "alpha"]
    for k in channels:
        expect += [k, f"{k}*Phi", f"{k}*alpha"]
    assert list(sub.FEATURES2) == expect

    rng = np.random.default_rng(0)
    beta2 = (rng.standard_normal(len(expect)) * 1e-3).astype(np.float32)
    Wn = make_mlp(128, 8, 4242)
    f = BC.extract(Wn, seed=4242, tau=2.5, n_samples=1200, n_pilot=60,
                   kmax=1, want_relu1=True, want_q2=False)
    cols = [np.ones_like(f["alpha"]), f["s"], f["Phi"], f["phi"], f["alpha"]]
    for k in channels:
        cols += [f[k], f[k] * f["Phi"], f[k] * f["alpha"]]
    pred_np = f["mu"] + np.stack(cols, -1) @ beta2.astype(np.float64)

    W = [fnp.asarray(w) for w in Wn]
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        out = kernels.corrected_sparse_kernel(
            W, tau=2.5, n_samples=1200, n_pilot=60, seed=4242, beta2=beta2,
            safe=False)
    assert np.abs(pred_np - np.asarray(out)[-1]).max() < 1e-6


def test_scaled_head_raise_still_falls_back_to_dense():
    """A raise anywhere in the new block must still cost accuracy, not the MLP.

    One zeroed prediction is O(1) against a score of O(1e-6), i.e. ~850x the
    whole score, and this session already lost one submission to a SymmetryError
    that eight local suites never produced.  The new block adds an eigh, an
    arcsin and a boolean-masked scatter, so the guard is re-asserted here
    against a forced failure inside it rather than at the outer boundary.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor.mc import make_mlp

    sub = _load_submission()
    est = sub.Estimator()
    est.setup(_Ctx())
    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=31)]

    good = est._transport_pair
    est._transport_pair = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("probe"))
    try:
        with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
            out = np.asarray(est.predict(_MLP(W, 33), int(1e12)))
    finally:
        est._transport_pair = good
    assert out.shape == (6, 64)
    assert np.isfinite(out).all()


def test_submission_contract():
    import warnings

    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor.contract import DEPTH, LAMBDA_FLOPS_PER_SECOND, WIDTH
    from whestfloor.mc import make_mlp

    sub = _load_submission()
    est = sub.Estimator()
    est.setup(_Ctx())
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
    # linearly, and the graded N sweep showed the optimum is at C/B ~ 0.2-0.3,
    # well past the 0.1 clamp. What still has to hold is a MARGIN under the
    # hard cap, because F is machine-independent while R is not: the grader's
    # residual is billed at 1e11 FLOP/s on one physical core and we cannot
    # measure its box. Half the budget leaves room for a residual three times
    # this box's worst case before anything is zeroed.
    flop_ratio = ctx.flops_used / 272_000_000_000
    assert flop_ratio <= 0.50, (
        f"FLOPs alone are {flop_ratio:.3f} of budget, leaving too little "
        f"margin under the hard cap for the grader's residual")


def test_submission_size_limits():
    """50 MiB / 50 files (whestbench limits.py)."""
    files = [p for p in (ROOT / "submission").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    total = sum(p.stat().st_size for p in files)
    assert len(files) <= 50, f"{len(files)} files"
    assert total <= 50 * 1024 * 1024, f"{total} bytes"
