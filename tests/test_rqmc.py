"""The randomised quasi-Monte Carlo draw must be a lattice, and unbiased.

Two properties carry the whole method and each is checked directly rather
than argued.

**Equidistribution.** Every one-dimensional projection of the rank-1 lattice
is the exact ``N``-point grid.  That is the mechanism: it makes the
shift-averaged squared error of any first-order ANOVA term ``1/(6N^2)``
instead of Monte Carlo's ``1/N``.  If a generating component ever shared a
factor with ``N`` the projection would collapse to a coarser grid and the
gain would quietly shrink.

**Unbiasedness.** ``docs/floor_theorem.md`` bounds every estimator from below
by the reference's own sampling variance, and the argument needs
``E[p] = mu_true``.  A Cranley-Patterson shift keeps that exactly: for each
FIXED lattice point, ``frac(p_i + Delta)`` is exactly uniform on the cube, so
each point is marginally a genuine standard Gaussian draw and only the
correlations between points are structured.  A biased sampler would beat the
"floor" on paper while being wrong, so this is tested against a closed form.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_lattice_projections_are_exact_grids():
    """Every 1-D projection is {0, 1/N, ..., (N-1)/N} -- for every dimension."""
    warnings.filterwarnings("ignore")
    import flopscope as flops

    from whestfloor import kernels as K

    N = K.RQMC_N
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        base = np.asarray(K.lattice_base(N, K.RQMC_Z), dtype=np.float64)
    assert base.shape == (N, len(K.RQMC_Z))
    grid = np.arange(N) / N
    for j in (0, 1, 37, 128, len(K.RQMC_Z) - 1):
        got = np.sort(base[:, j])
        assert np.allclose(got, grid, atol=2e-7), (
            f"dimension {j} is not an exact N-point grid; "
            f"max deviation {np.abs(got - grid).max():.2e}")
    # gcd(z_j, N) = 1 is what guarantees it, and N prime guarantees gcd
    assert all(math.gcd(int(z), N) == 1 for z in K.RQMC_Z)


def test_shifted_lattice_point_is_exactly_uniform():
    """Each point, over the shift, is uniform on the cube -- the unbiasedness
    mechanism.  Checked on the marginal of a fixed row."""
    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels as K

    N, d = 61, 8
    z = np.array([1, 13, 7, 22, 41, 5, 34, 19])
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        base = K.lattice_base(N, tuple(int(v) for v in z))
    rows = []
    with flops.BudgetContext(flop_budget=int(1e13), quiet=True):
        for s in range(4000):
            rng = fnp.random.default_rng(s)
            x = np.asarray(K.lattice_normals(base, rng), dtype=np.float64)
            rows.append(x[7])           # one FIXED lattice point
    a = np.array(rows)
    assert abs(a.mean()) < 0.02, a.mean()
    assert abs(a.std() - 1.0) < 0.02, a.std()
    # third and fourth standardised moments of a standard normal
    assert abs(float((a ** 3).mean())) < 0.06
    assert abs(float((a ** 4).mean()) - 3.0) < 0.12


def test_rqmc_is_unbiased_against_a_closed_form():
    """``E[relu(a.x)] = ||a|| / sqrt(2 pi)`` exactly.  The shift-averaged RQMC
    estimate must hit it, and must also beat iid at the same N."""
    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels as K

    N, d, R = 127, 16, 3000
    rng0 = np.random.default_rng(0)
    z = np.array([1] + [int(v) for v in rng0.choice(
        [k for k in range(2, N) if math.gcd(k, N) == 1], size=d - 1,
        replace=False)])
    a = rng0.standard_normal(d)
    truth = float(np.linalg.norm(a)) / math.sqrt(2 * math.pi)

    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        base = K.lattice_base(N, tuple(int(v) for v in z))
    q, m = [], []
    with flops.BudgetContext(flop_budget=int(1e13), quiet=True):
        for s in range(R):
            rng = fnp.random.default_rng(s)
            x = np.asarray(K.lattice_normals(base, rng), dtype=np.float64)
            q.append(float(np.maximum(x @ a, 0.0).mean()))
            y = np.asarray(fnp.random.default_rng(10_000 + s).standard_normal(
                (N, d), dtype=fnp.float32), dtype=np.float64)
            m.append(float(np.maximum(y @ a, 0.0).mean()))
    q, m = np.array(q), np.array(m)
    se = q.std(ddof=1) / math.sqrt(R)
    assert abs(q.mean() - truth) < 4.0 * se, (
        f"RQMC mean {q.mean():.6f} vs exact {truth:.6f}, {abs(q.mean()-truth)/se:.1f} sigma")
    # and it must actually be a variance reduction, or the point set is broken
    assert q.var() < 0.7 * m.var(), (q.var(), m.var())


def test_rqmc_ablation_is_exactly_the_shipped_sparse_kernel():
    """``rqmc=False`` must reproduce ``sparse_mc_kernel`` bit for bit.

    That is what makes the reported RQMC gain an ablation through one code
    path rather than a comparison between two programs.
    """
    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels as K
    from whestfloor.mc import make_mlp

    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=11)]
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c1:
        a = np.asarray(K.rqmc_sparse_kernel(W, tau=2.5, n_samples=512,
                                            n_pilot=64, seed=3, rqmc=False))
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c2:
        b = np.asarray(K.sparse_mc_kernel(W, tau=2.5, n_samples=512,
                                          n_pilot=64, seed=3))
    assert np.array_equal(a, b)
    assert c1.flops_used == c2.flops_used


# ---------------------------------------------------------------------------
# The x0_fn hook and the deployable lattice (whestfloor/rqmc.py)
# ---------------------------------------------------------------------------


def test_x0_fn_none_is_bitwise_the_shipped_corrected_kernel():
    """The hook must be invisible when unused.

    ``corrected_sparse_kernel`` gained an ``x0_fn`` argument so a lattice can
    replace the scored draw; ``x0_fn=None`` must be the previous program, to
    the bit and to the FLOP, or every A/B in ``docs/rqmc.md`` is comparing two
    different estimators rather than one estimator's two draws.
    """
    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels as K
    from whestfloor.mc import make_mlp

    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=11)]
    beta = np.zeros(15, dtype=np.float32)
    beta[1] = 0.25
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c1:
        a = np.asarray(K.corrected_sparse_kernel(
            W, tau=2.5, n_samples=512, n_pilot=64, seed=3, beta=beta))
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as c2:
        b = np.asarray(K.corrected_sparse_kernel(
            W, tau=2.5, n_samples=512, n_pilot=64, seed=3, beta=beta,
            x0_fn=None))
    assert np.array_equal(a, b)
    assert c1.flops_used == c2.flops_used


def test_billed_lattice_stays_float32_downstream():
    """A8: ``norm.ppf`` promotes to float64 and float64 bills at 2x.

    One promoted array reprices the entire 32-layer chain, which is a 2x loss
    on the score for a dtype slip.  The draw must come back float32 and the
    marginal FLOPs per sample must be the iid kernel's plus the draw, not
    twice the iid kernel's.
    """
    warnings.filterwarnings("ignore")
    import flopscope as flops
    import flopscope.numpy as fnp

    from whestfloor import kernels as K
    from whestfloor import rqmc as RQ
    from whestfloor.mc import make_mlp

    W = [fnp.asarray(x) for x in make_mlp(64, 6, seed=11)]
    z = RQ.get_z(521, 64, "cbc")
    with flops.BudgetContext(flop_budget=int(1e12), quiet=True):
        base = RQ.billed_lattice_base(521, z)
        x = RQ.billed_lattice_normals(base, fnp.random.default_rng(0))
    assert str(x.dtype) == "float32", x.dtype
    assert np.isfinite(np.asarray(x)).all()

    def bill(n, x0):
        with flops.BudgetContext(flop_budget=int(1e13), quiet=True) as c:
            K.corrected_sparse_kernel(W, tau=2.5, n_samples=n, n_pilot=64,
                                      seed=3, beta=None, x0_fn=x0)
        return int(c.flops_used)

    zs = {n: RQ.get_z(n, 64, "cbc") for n in (521, 1031)}
    bases = {}
    with flops.BudgetContext(flop_budget=int(1e13), quiet=True):
        for n, zz in zs.items():
            bases[n] = RQ.billed_lattice_base(n, zz)
    f_i = {n: bill(n, None) for n in (521, 1031)}
    f_l = {n: bill(n, RQ.lattice_x0_fn(bases[n])) for n in (521, 1031)}
    per_i = (f_i[1031] - f_i[521]) / 510
    per_l = (f_l[1031] - f_l[521]) / 510
    # The whole overhead must be the draw and nothing else: 157 FLOPs per
    # element, exactly, against ``standard_normal``'s 16.  A float64 chain
    # would show up as an overhead proportional to the *pass*, not to the
    # draw, so pinning the per-element figure is the sharp test.  (The ratio
    # itself is shape-dependent -- 1.20x on this 64x6 toy where the pass is
    # tiny, 1.015x at the scored 256x32.)
    assert abs((per_l - per_i) / 64 - 157.0) < 1.0, (per_l - per_i) / 64
    assert per_l / per_i < 1 + 2.0 * 157.0 * 64 / per_i, (
        f"lattice bills {per_l / per_i:.3f}x the iid marginal -- more than "
        "the draw can account for, so the float64 promotion escaped the cast")


def test_cranley_patterson_marginal_is_exactly_uniform():
    """``frac(k g + U) ~ U[0,1)^d`` — the unbiasedness mechanism, numerically.

    For a FIXED lattice index the shifted point is a measure-preserving
    rotation of ``U``, so its marginal is uniform for every ``N``.  Checked on
    the empirical CDF of one fixed row over many shifts.
    """
    from whestfloor import rqmc as RQ

    N, d = 1021, 8
    z = RQ.get_z(N, 256, "cbc")[:d]
    rows = []
    rng = np.random.default_rng(4)
    fixed = RQ.lattice_rows(613, 614, N, z)     # one arbitrary fixed point
    for _ in range(6000):
        u = fixed + rng.random(d)
        rows.append((u - np.floor(u))[0])
    a = np.sort(np.array(rows).ravel())
    ks = np.abs(a - (np.arange(1, a.size + 1) / a.size)).max()
    assert ks < 1.36 / math.sqrt(a.size) * 1.6, ks
