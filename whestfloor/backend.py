"""One kernel, two backends.

Every estimator in this repository is written **once**, against the small
operation set below.  It then runs either

* on plain NumPy — fast, for research iteration and for the thousands of
  offline calibration passes; or
* on flopscope — slow, but producing the exact analytic FLOP count the grader
  will charge.

Writing the math twice would let the researched kernel and the submitted
kernel drift apart, and a drift of that kind is exactly the sort of thing that
makes a reported number false.  So the two backends expose an identical
surface and ``tests/test_backend_parity.py`` pins them together.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SQRT_2 = float(np.sqrt(2.0))
_INV_SQRT_2PI = float(1.0 / np.sqrt(2.0 * np.pi))


class NumpyBackend:
    """Plain NumPy.  No FLOP accounting; used for research throughput."""

    name = "numpy"
    billed = False
    f32 = np.float32
    f64 = np.float64

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=np.float32):
        return np.ones(shape, dtype=dtype)

    def eye(self, n, dtype=np.float32):
        return np.eye(n, dtype=dtype)

    def matmul(self, a, b):
        return a @ b

    def einsum(self, sub, *ops):
        return np.einsum(sub, *ops, optimize=True)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def sqrt(self, a):
        return np.sqrt(a)

    def exp(self, a):
        return np.exp(a)

    def log(self, a):
        return np.log(a)

    def erf(self, a):
        # numpy has no erf; use the identity erf(x) = 2*Phi(x*sqrt2) - 1 via
        # the same rational approximation flopscope's norm.cdf uses, so the two
        # backends agree numerically as well as in cost.
        return _erf_np(a)

    def norm_pdf(self, a):
        return _INV_SQRT_2PI * np.exp(-0.5 * a * a)

    def norm_cdf(self, a):
        return 0.5 * (1.0 + _erf_np(a / _SQRT_2))

    def diag(self, a):
        return np.diag(a)

    def outer(self, a, b):
        return np.outer(a, b)

    def stack(self, xs, axis=0):
        return np.stack(xs, axis=axis)

    def sum(self, a, axis=None):
        return np.sum(a, axis=axis)

    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def as_symmetric(self, a, symmetry=(0, 1)):
        return a

    def fill_diagonal(self, a, v):
        out = a.copy()
        np.fill_diagonal(out, v)
        return out

    def to_numpy(self, a):
        return np.asarray(a)


def _erf_np(x):
    """Abramowitz & Stegun 7.1.26-class erf, vectorised, float64 accurate.

    NumPy ships no erf and SciPy is unavailable in the grader environment, so
    both backends go through an explicit implementation.  This one is the
    high-accuracy rational form (|err| < 1.2e-16 in float64), not the cheap
    7.1.26 approximation, because the estimator needs ~1e-11 relative accuracy
    from the Gaussian tail functions.
    """
    x = np.asarray(x, dtype=np.float64)
    return _erf_kernel(x)


def _erf_kernel(x: np.ndarray) -> np.ndarray:
    """erf via erfc with the Cody/W. J. rational Chebyshev approximations."""
    ax = np.abs(x)
    out = np.empty_like(ax)

    # Region 1: |x| <= 0.5 — series in x^2.
    m1 = ax <= 0.5
    if m1.any():
        z = x[m1] * x[m1]
        num = ((((-0.356098437018154e-1 * z + 0.699638348861914e1) * z
                 + 0.219792616182942e2) * z + 0.242667955230532e3))
        den = ((((z + 0.150827976304078e2) * z + 0.911649054045149e2) * z
                + 0.215058875869861e3))
        out[m1] = x[m1] * num / den

    # Region 2 & 3: |x| > 0.5 — erfc rational approximations.
    m2 = ~m1
    if m2.any():
        a = ax[m2]
        r = np.empty_like(a)
        lo = a <= 4.0
        if lo.any():
            z = a[lo]
            num = ((((((((-1.36864857382717e-7 * z + 5.64195517478974e-1) * z
                         + 7.21175825088309e0) * z + 4.31622272220567e1) * z
                       + 1.52989285046940e2) * z + 3.39320816734344e2) * z
                     + 4.51918953711873e2) * z + 3.00459261020162e2))
            den = ((((((((z + 1.27827273196294e1) * z + 7.70001529352295e1) * z
                        + 2.77585444743988e2) * z + 6.38980264465631e2) * z
                      + 9.31354094850610e2) * z + 7.90950925327898e2) * z
                    + 3.00459260956983e2))
            r[lo] = np.exp(-z * z) * num / den
        hi = ~lo
        if hi.any():
            z = a[hi]
            zi = 1.0 / (z * z)
            num = (((((1.63153871373020e-2 * zi + 3.05326634961232e-1) * zi
                      + 3.60344899949804e-1) * zi + 1.25781726111229e-1) * zi
                    + 1.60837851487423e-2))
            den = (((((zi + 2.56852019228982e0) * zi + 1.87295284992346e0) * zi
                     + 5.27905102951428e-1) * zi + 6.05183413124413e-2) * zi
                   + 2.33520497626869e-3)
            r[hi] = np.exp(-z * z) / z * (_INV_SQRT_PI - zi * num / den)
        sign = np.sign(x[m2])
        out[m2] = sign * (1.0 - r)
    return out


_INV_SQRT_PI = float(1.0 / np.sqrt(np.pi))


class FlopscopeBackend:
    """flopscope-billed backend.  Same surface, exact FLOP accounting."""

    name = "flopscope"
    billed = True

    def __init__(self) -> None:
        import flopscope as flops  # noqa: PLC0415
        import flopscope.numpy as fnp  # noqa: PLC0415

        self._flops = flops
        self._fnp = fnp
        self.f32 = fnp.float32
        self.f64 = fnp.float64

    def asarray(self, x, dtype=None):
        return self._fnp.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return self._fnp.zeros(shape, dtype=dtype or self.f32)

    def ones(self, shape, dtype=None):
        return self._fnp.ones(shape, dtype=dtype or self.f32)

    def eye(self, n, dtype=None):
        return self._fnp.eye(n, dtype=dtype or self.f32)

    def matmul(self, a, b):
        return self._fnp.matmul(a, b)

    def einsum(self, sub, *ops):
        return self._fnp.einsum(sub, *ops)

    def maximum(self, a, b):
        return self._fnp.maximum(a, b)

    def minimum(self, a, b):
        return self._fnp.minimum(a, b)

    def sqrt(self, a):
        return self._fnp.sqrt(a)

    def exp(self, a):
        return self._fnp.exp(a)

    def log(self, a):
        return self._fnp.log(a)

    def erf(self, a):
        return 2.0 * self.norm_cdf(a * _SQRT_2) - 1.0

    def norm_pdf(self, a):
        return self._flops.stats.norm.pdf(a)

    def norm_cdf(self, a):
        return self._flops.stats.norm.cdf(a)

    def diag(self, a):
        return self._fnp.diag(a)

    def outer(self, a, b):
        return self._fnp.outer(a, b)

    def stack(self, xs, axis=0):
        return self._fnp.stack(xs, axis=axis)

    def sum(self, a, axis=None):
        return self._fnp.sum(a, axis=axis)

    def mean(self, a, axis=None):
        return self._fnp.mean(a, axis=axis)

    def as_symmetric(self, a, symmetry=(0, 1)):
        return self._flops.as_symmetric(a, symmetry=symmetry)

    def fill_diagonal(self, a, v):
        self._fnp.fill_diagonal(a, v)
        return a

    def to_numpy(self, a) -> np.ndarray:
        return np.asarray(a)


def get_backend(kind: str = "numpy") -> Any:
    if kind == "numpy":
        return NumpyBackend()
    if kind == "flopscope":
        return FlopscopeBackend()
    raise ValueError(f"unknown backend {kind!r}")
