"""ARC White-Box Estimation Challenge — submission entry point.

This file is the **single source of truth for the graded algorithm**.  It
imports nothing but ``flopscope`` and ``whestbench``, because the grader
sandbox provides nothing else — no numpy, no scipy, no torch, and a reduced
standard library.  ``tests/test_submission_parity.py`` asserts that the
research kernel in ``whestfloor/kernels.py`` and the code below produce
bitwise-identical predictions and identical FLOP counts, so what is measured
is what is shipped.

Algorithm
---------
Full-covariance moment propagation with the **exact** post-ReLU covariance.

The pre-activation of every layer is modelled as jointly Gaussian; the linear
map is then exact,

    m_pre = Wᵀ m ,     Σ_pre = Wᵀ Σ W ,

and the rectifier's first two moments are exact per neuron.  The step that is
normally approximated is the *off-diagonal* post-ReLU covariance.  Writing
``a_k = E[relu(m + s t) He_k(t)]`` for the Hermite coefficients of the
rectifier about each neuron's own mean and scale, Mehler's formula gives the
covariance in closed form:

    Cov(relu(z_i), relu(z_j)) = Σ_{k≥1} a^i_k a^j_k ρ_ij^k / k!

with, derived by Stein's identity,

    a_1 = s Φ(α),      a_k = (-1)^k s He_{k-2}(α) φ(α)   for k ≥ 2,   α = m/s.

The ``k = 1`` term alone is ``Φ_i Φ_j Σ_ij`` — exactly the "gain" rule used by
the reference baseline.  Every further term is a correction that baseline
drops, and the whole series costs ``O(kmax · width²)`` against the layer's
``O(width³)`` contraction, i.e. a couple of percent.

Budget
------
The score multiplier is ``max(0.1, C/B)`` and clamps at the floor for any
``C ≤ 2.72e10``.  This estimator spends far less than that, so its ranked
score is exactly ``final_layer_mse / 10``.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator

#: Order at which Mehler's series is truncated.  Measured to saturate at 4:
#: k=2 gives 6.334e-5, k=4 gives 6.3156e-5, k=8/16 give 6.3153e-5 (scripts/12).
#: Higher k only adds residual wall time, and at k=24 that pushes C/B to 0.103,
#: crossing the 0.1 multiplier floor and making the score WORSE.
KMAX = 4

#: Floor applied to pre-activation variances before taking a square root.
VAR_FLOOR = 1e-12


class Estimator(BaseEstimator):
    """Covariance propagation with the exact post-ReLU covariance."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx) -> None:  # noqa: ANN001 - whestbench SetupContext
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp, budget: int):  # noqa: ANN001 - whestbench MLP
        _ = budget
        n = mlp.width
        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))
        rows = []

        for w in mlp.weights:
            # ---- exact linear map -------------------------------------
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            var_pre = fnp.maximum(fnp.diag(cov_pre), VAR_FLOOR)
            sig = fnp.sqrt(var_pre)
            alpha = mu_pre / sig
            ph = flops.stats.norm.pdf(alpha)
            Ph = flops.stats.norm.cdf(alpha)

            # ---- exact rectified-Gaussian marginals -------------------
            mu = mu_pre * Ph + sig * ph
            ez2 = (mu_pre * mu_pre + var_pre) * Ph + mu_pre * sig * ph
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            # ---- exact post-ReLU covariance via Mehler ----------------
            inv_sig = 1.0 / sig
            rho = cov_pre * fnp.outer(inv_sig, inv_sig)
            rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)

            a = _hermite_coeffs(alpha, sig, ph, Ph, KMAX)
            acc = fnp.outer(a[1], a[1]) * rho
            rho_k = rho
            fact = 1.0
            for k in range(2, KMAX + 1):
                rho_k = rho_k * rho
                fact *= k
                acc = acc + fnp.outer(a[k], a[k]) * (rho_k * (1.0 / fact))

            cov = acc
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))
            rows.append(mu)

        return fnp.stack(rows, axis=0)


def _hermite_coeffs(alpha, sig, ph, Ph, kmax):
    """``a_k = E[relu(m + s t) He_k(t)]`` for k = 1 … kmax.

    ``a_1 = s Φ(α)``;  ``a_k = (-1)^k s He_{k-2}(α) φ(α)`` for ``k ≥ 2``.
    He_j is built by the recurrence ``He_{j+1} = α He_j - j He_{j-1}``.
    """
    out = [None, sig * Ph]
    if kmax >= 2:
        s_phi = sig * ph
        h_prev = None   # He_{j-1}
        h = None        # He_j
        for k in range(2, kmax + 1):
            j = k - 2
            if j == 0:
                hj = None                       # He_0 == 1
            elif j == 1:
                h_prev, h = None, alpha         # He_1 == alpha
                hj = h
            else:
                base = alpha * h
                hj = base if h_prev is None else base - float(j - 1) * h_prev
                h_prev, h = h, hj
            term = s_phi if hj is None else s_phi * hj
            out.append(term if k % 2 == 0 else -term)
    return out
