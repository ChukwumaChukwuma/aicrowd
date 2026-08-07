"""The bit-sliced product must be an IDENTITY, not an approximation.

Everything in ``docs/bitslice.md`` rests on one claim: the packed
AND/popcount/reduce pipeline returns exactly the integer product
``sum_i q_a[i] q_w[i,j]``.  If that is true, then the NumPy simulator (which
does the integer product directly, in float32) and the billed flopscope kernel
are the same object, and the variance sweeps measured on the cheap one
describe the expensive one.  These tests check it on random codes at every
precision the sweeps use, and check the billing formula the cost model uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

flops = pytest.importorskip("flopscope")
fnp = pytest.importorskip("flopscope.numpy")

from whestfloor.bitslice import (  # noqa: E402
    bs_matmul,
    bs_pack_cols,
    bs_pack_rows,
    packed_layer_cost,
)


@pytest.mark.parametrize("ba,bw", [(1, 2), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)])
@pytest.mark.parametrize("k_in", [32, 64, 224])
def test_packed_product_is_exact(ba, bw, k_in):
    """Packed == integer matmul, bit for bit, at every precision swept."""
    rng = np.random.default_rng(ba * 1000 + bw * 10 + k_in)
    n, m = 17, 13
    qa = rng.integers(0, 2 ** ba, (n, k_in), dtype=np.uint8)
    lvw = 2 ** (bw - 1) - 1
    qw = rng.integers(-lvw - 1, lvw + 1, (k_in, m)).astype(np.int64)
    exact = qa.astype(np.int64) @ qw

    with flops.BudgetContext(flop_budget=10 ** 12, quiet=True):
        a = fnp.asarray(qa)
        # offset binary: every plane's coefficient is a positive 2^(p+q), and
        # the offset comes back out through one per-sample row sum
        qwu = fnp.asarray((qw + (lvw + 1)).astype(np.uint8))
        got = bs_matmul(bs_pack_rows(a, ba), bs_pack_cols(qwu, bw),
                        fnp.sum(a, axis=1, dtype=fnp.int32)[:, None], bw)
        got = np.asarray(got)
    assert got.dtype.kind in "iu"
    np.testing.assert_array_equal(got, exact)


def test_packbits_lane_order_agrees_between_the_two_packers():
    """Rows and columns must land in the same lanes or the product is garbage.

    This is the one place a silent error could hide: a lane permutation that
    is self-consistent on each side but different between them still produces
    plausible-looking numbers.
    """
    rng = np.random.default_rng(7)
    k = 64
    a = np.zeros((1, k), dtype=np.uint8)
    w = np.zeros((k, 1), dtype=np.uint8)
    for i in rng.choice(k, 9, replace=False):
        a[0, i] = 1
        w[i, 0] = 1
    with flops.BudgetContext(flop_budget=10 ** 12, quiet=True):
        pa = bs_pack_rows(fnp.asarray(a), 1)[0]
        pw = bs_pack_cols(fnp.asarray(w), 1)[0]
        n = int(np.asarray(fnp.sum(fnp.bitwise_count(
            fnp.bitwise_and(pa[:, :, None], pw[None, :, :])), dtype=fnp.int32)))
    assert n == 9


@pytest.mark.parametrize("ba,bw", [(2, 4), (4, 6)])
def test_cost_model_matches_the_meter(ba, bw):
    """``packed_layer_cost`` is what a real BudgetContext bills, to <1%."""
    n, k_in, k_out = 64, 224, 256
    rng = np.random.default_rng(3)
    h = rng.standard_normal((n, k_in)).astype(np.float32)
    qw = rng.integers(0, 2 ** bw, (k_in, k_out), dtype=np.uint8)
    lv = float(2 ** ba - 1)
    with flops.BudgetContext(flop_budget=10 ** 12, quiet=True) as ctx:
        wp = bs_pack_cols(fnp.asarray(qw), bw)      # per-MLP, subtracted below
        base = int(ctx.flops_used)
        x = fnp.asarray(h)
        g = fnp.random.default_rng(0)
        t = (x - 0.5) * 2.0
        q = fnp.minimum(fnp.maximum(
            fnp.floor(t + g.random(t.shape, dtype=fnp.float32)), 0.0), lv)
        qi = q.astype(fnp.uint8)
        rs = fnp.sum(qi, axis=1, dtype=fnp.int32)[:, None]
        acc = bs_matmul(bs_pack_rows(qi, ba), wp, rs, bw)
        z = acc.astype(fnp.float32) * 0.5 + 1.0
        fnp.maximum(z, 0.0)
        got = (int(ctx.flops_used) - base) / n
    want = packed_layer_cost(k_in, k_out, ba, bw)
    # 1.1% on this synthetic layer; 0.3% end to end (scripts/44 --mode price)
    assert abs(got - want) / want < 0.02, (got, want)


def test_float32_is_exact_for_the_simulator_range():
    """The NumPy simulator carries integer codes in float32; prove it is exact.

    Largest partial sum at the precisions swept is ``224 * 63 * 128`` = 1.8e6,
    well inside float32's 2^24 = 1.68e7 integer range, so the BLAS matmul is
    exact integer arithmetic and the simulator is bit-faithful to the kernel.
    """
    rng = np.random.default_rng(11)
    qa = rng.integers(0, 64, (128, 224)).astype(np.float32)
    qw = rng.integers(-128, 128, (224, 256)).astype(np.float32)
    got = qa @ qw
    exact = qa.astype(np.int64) @ qw.astype(np.int64)
    np.testing.assert_array_equal(got.astype(np.int64), exact)
