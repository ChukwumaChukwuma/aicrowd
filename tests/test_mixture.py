"""Assert the exactness boundary of the mixture machinery, don't assume it.

Four things have to be true for `docs/mixture.md`'s numbers to mean anything,
and each is checked here rather than argued:

1. ``MixtureState.split`` is a **quadrature identity**: splitting a single
   Gaussian along any direction leaves the mean and the covariance unchanged, to
   Gauss-Hermite accuracy.  If it did not, the propagator would be "improving"
   the answer by perturbing the state rather than by representing its shape.
2. Splitting therefore does not change ``E[relu]`` *at the layer where it
   happens* — it only changes what happens downstream.  This is the fact that
   makes the ceiling measurement, not the propagator, the decisive experiment.
3. ``CellAccumulator`` + ``predict_state`` reproduce a direct, unstreamed
   evaluation of ``sum_k w_k relu_mean(m_k, s_k)`` bitwise-closely, so the
   float32 fast path in the accumulator is not quietly moving the ceiling.
4. ``lloyd`` at ``r = 1`` returns a genuine Lloyd-Max fixed point (every
   centroid is the conditional mean of its own cell), which is what licenses
   calling the measured ceiling an optimum over ``K``-node quadratures.

No pytest in this environment: run it as ``$PY tests/test_mixture.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.mixture import (  # noqa: E402
    CellAccumulator,
    MixtureState,
    assign,
    frame_eig,
    gauss_hermite,
    lloyd,
    pooled_moments,
    predict_state,
)
from whestfloor.relu_moments import relu_mean  # noqa: E402


def _psd(n, seed):
    A = np.random.default_rng(seed).standard_normal((n, n))
    return (A @ A.T) / n + 0.05 * np.eye(n)


def test_split_preserves_mean_and_covariance():
    """A split is a quadrature of the parent, not a perturbation of it."""
    for nodes in (2, 3, 6, 12):
        _split_preserves(nodes)


def _split_preserves(nodes):
    n = 24
    rng = np.random.default_rng(7)
    m = rng.standard_normal(n)
    C = _psd(n, 11)
    st = MixtureState.gaussian(m, C)
    u = st.top_direction()
    sp = st.split(u, nodes)
    assert sp.K == nodes
    M, CC = sp.collapse()
    # Gauss-Hermite with `nodes` points is exact for polynomials of degree
    # 2*nodes-1, and mean/covariance are degrees 1 and 2.
    assert np.allclose(M, m, atol=1e-10)
    assert np.allclose(CC, C, atol=1e-9)
    assert abs(float(sp.w.sum()) - 1.0) < 1e-12


def test_split_does_not_move_the_answer_at_its_own_layer():
    """`sum_k w_k relu_mean` over the split equals the parent's `relu_mean`.

    Gauss-Hermite is a quadrature, so this converges with `nodes`; at 24 nodes
    it must agree to well below the 1e-8 scale the whole page works at.
    """
    n = 16
    m = np.random.default_rng(3).standard_normal(n) * 0.5
    C = _psd(n, 5)
    st = MixtureState.gaussian(m, C)
    base = relu_mean(m, np.sqrt(np.diag(C)))
    prev = None
    for nodes in (2, 4, 8, 16, 24, 40):
        got = st.split(st.top_direction(), nodes).relu_mean_mixture()
        err = float(np.sqrt(np.mean((got - base) ** 2)))
        if prev is not None:
            assert err < prev, (nodes, err, prev)
        prev = err
    # measured: 1.29e-2 / 2.26e-3 / 1.66e-4 / 1.82e-6 / 2.50e-8 / 3.53e-12
    assert prev < 1e-10


def test_gauss_hermite_weights_are_a_probability_measure():
    for K in (1, 2, 5, 6, 12):
        x, w = gauss_hermite(K)
        assert abs(float(w.sum()) - 1.0) < 1e-13
        assert abs(float(w @ x)) < 1e-12
        if K >= 2:
            assert abs(float(w @ (x * x)) - 1.0) < 1e-11


def test_accumulator_matches_a_direct_evaluation():
    """The streamed float32 group-by reproduces the unstreamed answer."""
    rng = np.random.default_rng(19)
    N, n, K, r = 40_000, 12, 8, 2
    Z = rng.standard_normal((N, n)).astype(np.float32) + 0.3
    T = np.ascontiguousarray(Z[:, :r]).astype(np.float32)
    C = lloyd(T.astype(np.float64), K, seed=1)
    acc = CellAccumulator(C, n)
    for lo in range(0, N, 4096):
        z = Z[lo:lo + 4096]
        ZZ = np.empty((z.shape[0], 2 * n), dtype=np.float32)
        ZZ[:, :n] = z
        np.multiply(z, z, out=ZZ[:, n:])
        acc.add(ZZ, T[lo:lo + 4096])
    got = predict_state(*acc.state(), n)

    lab = assign(T, C.astype(np.float32))
    Zd = Z.astype(np.float64)
    direct = np.zeros(n)
    for k in range(K):
        sel = Zd[lab == k]
        if not len(sel):
            continue
        s = np.sqrt(np.maximum(sel.var(axis=0), 1e-30))
        direct += len(sel) * relu_mean(sel.mean(axis=0), s)
    direct /= N
    assert np.allclose(got, direct, atol=2e-6), float(np.abs(got - direct).max())

    m, v = pooled_moments(*acc.state(), n)
    assert np.allclose(m, Zd.mean(axis=0), atol=2e-6)
    assert np.allclose(v, Zd.var(axis=0), atol=2e-5)


def test_lloyd_is_a_fixed_point_in_one_dimension():
    """Each centroid is the mean of its own cell: the Lloyd-Max condition."""
    for K in (2, 6, 32, 64):
        # no subsampling inside lloyd, so the fixed point is testable exactly
        T = np.random.default_rng(23).standard_normal((max(20_000, 40 * K), 1))
        C = lloyd(T, K, seed=4)
        lab = assign(T, C)
        for k in range(K):
            sel = T[lab == k, 0]
            assert len(sel) > 0
            # relative to the cell's own scale, so the statement is scale-free
            assert abs(float(sel.mean()) - float(C[k, 0])) < 1e-9 * sel.std()


def test_frame_eig_is_orthonormal_and_ordered():
    Z = (np.random.default_rng(29).standard_normal((5_000, 20))
         @ np.random.default_rng(31).standard_normal((20, 20)))
    m0, U, ev = frame_eig(Z)
    assert np.allclose(U.T @ U, np.eye(20), atol=1e-10)
    assert np.all(np.diff(ev) <= 1e-12)
    assert np.allclose(m0, Z.mean(axis=0))


def test_reduce_to_preserves_the_first_two_moments():
    """Runnalls' merge is moment-preserving; the state it hands on is honest."""
    n = 10
    rng = np.random.default_rng(37)
    st = MixtureState.gaussian(rng.standard_normal(n), _psd(n, 41))
    st = st.split(st.top_direction(), 6)
    M0, C0 = st.collapse()
    red = st.reduce_to(3)
    assert red.K == 3
    M1, C1 = red.collapse()
    assert np.allclose(M0, M1, atol=1e-10)
    assert np.allclose(C0, C1, atol=1e-9)


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
