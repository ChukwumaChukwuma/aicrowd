"""ARC White-Box Estimation Challenge — submission entry point.

This file is the **single source of truth for the graded algorithm**.  It
imports nothing but ``flopscope`` and ``whestbench``, because the grader
sandbox provides nothing else — no numpy, no scipy, no torch, and a reduced
standard library.  ``tests/test_submission_parity.py`` asserts that the
research kernel in ``whestfloor/kernels.py`` and the code below produce
bitwise-identical predictions and identical FLOP counts, so what is measured
is what is shipped.

The estimator in one paragraph
------------------------------
Sparse Monte Carlo (the always-off neurons pruned out of every matmul), plus
two things that cost 0.4% of the pass between them: **layer-1 Hermite control
variates**, whose expectations are known in closed form because ``z^1`` is
exactly Gaussian, and an **offline-trained linear head** over those and ~20
other predict-time features, loaded from ``corrector.npz`` at zero FLOPs.

Why sampling at all
-------------------
The grader reports a constant ``sampling_mse = 6.4695e-7`` on every
submission.  Because ``mse x C/B`` is flat in N for a sampler, that IS plain
Monte Carlo's adjusted plateau — better than the bundled covariance-
propagation baseline by ~10x, and better than every purely analytic estimator
in this repository.  Two full rounds of analytic work (Mehler covariance,
tree-diagram kappa_3 and kappa_4, coherent-bias shrink) landed at 2.28e-6
adjusted, i.e. 3.5x worse than plain sampling.  So the analytic arm is gone
and what is left is the cheapest correct sampler we can bill, made accurate.

1. Prune the neurons that never fire
------------------------------------
At depth the network is nearly decided.  With ``alpha = m/s`` the per-neuron
pre-activation ratio, ``rms|alpha|`` rises from 0 at layer 1 (where
``E[z] = 0`` exactly) to **3.44 by layer 32**.  A neuron with ``alpha < -tau``
emits ``relu(z) = 0`` on all but a vanishing fraction of samples, so its row
of the next weight matrix and its column of this one both drop out of every
per-sample matmul, and its small, nearly constant expected output folds into a
bias vector computed once.  Billed: **2,793,985 FLOPs/sample against
4,198,656 dense, i.e. 1.51x**, which buys 8,500 samples where the dense pass
affords 6,200.  The sign error traded away is 2-3e-7, measured paired against
the dense pass on the identical stream.

2. Layer-1 Hermite control variates
-----------------------------------
``z^1_i = x . W^1[:,i]`` is *exactly* Gaussian with known scale
``sigma_i = ||W^1[:,i]||``.  So for ``t_i = z^1_i / sigma_i``

    E[He_k(t_i)] = 0                                exactly, every k >= 1
    Cov(He_k(t_i), He_l(t_j)) = delta_kl k! rho_ij^k          (Mehler)

The first line makes the sample means of ``He_k(t)`` exactly-mean-zero control
variates that the forward pass already has in hand; the second makes their
Gram **analytic and block diagonal in k**, so the optimal coefficients need no
estimated covariance matrix.  Writing ``u_k = G_k^{-1} d_k`` for the sample
mean ``d_k`` collapses the whole correction to two length-N matvecs rather
than a ``width x width`` cross-moment matrix:

    correction_j = (1/N) sum_s w_s y_{sj},   w_s = sum_k u_k . (g_{k,s} - d_k)

The ``k=1`` block is evaluated in the input basis, where ``G = I`` exactly
(``u_1 = rho^{-1} d_1`` unwinds to ``W^{1,-1} xbar``): no solve, no
conditioning problem, and the block is thereby *proved* to be the optimal
input-linear control variate — the one the low-order barrier caps at 1.38x.
``k=2`` is what escapes that cap, because ``He_2(w_i . x)`` is an order-2
object along the network's own first-layer directions.  Measured population
shares of ``Var(relu(z^32_j))``: k=1 23-29%, k<=2 38-48%, k=3 adds ~2% and is
not worth its own estimation noise.

Coefficients are estimated by a **split of the sample** — ``dbar`` from one
half against the covariance from the other, both ways.  The one-pass form
leaves a self-term ``Cov(g' G^-1 g, y_j) / N`` that is a real bias rather than
noise; at k=3 it reverses the sign of the correction outright.

3. The offline-trained head
---------------------------
Each Hermite block costs ``p/N = 256/8500 = 3.0%`` of the residual in
estimation noise, and the sparse mask leaves a small closure bias.  Rather
than assume a coefficient of 1 on each correction, a linear head over the two
corrections and a handful of other predict-time features is fitted **offline
on 640 generated MLPs with fresh seeds**, disjoint from every evaluation
suite, and shipped as a 15-float vector in ``corrector.npz``.  ``fnp.load``
bills **0 FLOPs**, so the head is free at grade time.

The design was 28 columns and is now 15.  The thirteen that went — the
Rao-Blackwell gap, the two sample-Edgeworth terms, the multiplicative shrink,
the per-neuron MC noise scale, the weight-column moments and the per-MLP
scalars — each measured at **exactly 1.000x** in the leave-one-group-out table
(removing the group left validation at 1.504x, unchanged to three decimals),
and between them they cost **eight passes over the (8500, 256) sample array**:
``d^3`` and ``d^4`` for the Edgeworth skew/kurtosis, and ``mean(x*x)`` for
``Var(relu z)``.  That is participant wall time, billed at ``lambda = 1e11``
FLOP/s, and it was the single largest avoidable term in ``C/B``.

Safety
------
``DAMP = 0`` reproduces the uncorrected sparse estimator bit for bit through
the identical code path.  If ``corrector.npz`` is missing or unreadable the
head is simply absent and the estimator degrades to that same uncorrected
sparse pass rather than failing.  Every numerical path is wrapped so that a
raise falls back to a dense Monte-Carlo pass: a single zeroed MLP would cost
~850x the whole score.

Sizing
------
``C = F + 1e11 R`` and the multiplier is ``max(0.1, C/B)``.  ``F`` is
machine-independent; ``R`` is participant wall time on one physical core.
This estimator sits at ``F/B = 0.0919`` — the feature block is 0.35% of the
free budget — leaving room for the grader's residual before the floor is
crossed, and crossing it is a linear penalty, not a cliff.

``N`` was re-swept against the budget the lean feature block frees, and it does
NOT buy samples: ``N = 9000`` and ``N = 9500`` both lower the raw MSE (3.58e-6,
3.53e-6 against 3.72e-6) and both *raise* the adjusted score, to 0.996x and
0.964x of this configuration, because ``C/B`` is already above the 0.1 floor
and the sparse mask's closure bias does not shrink with ``N``.
"""

from __future__ import annotations

import os

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator

#: Threshold on ``alpha = m/s`` below which a neuron is treated as always-off.
#: CALIBRATED, not derived: swept on the official 100 MLPs.  The optimum is
#: broad (2.3 / 2.5 / 2.7 give adjusted 5.97 / 5.65 / 5.79 e-7 uncorrected).
#: The LOWER end is fixed by accuracy, not cost: tau = 2.0 is 1.12x cheaper
#: per sample again but its sign error is 20-30% of the final MSE.
TAU = 2.5

#: Scored Monte-Carlo samples.  Set by the ``F/B`` invariant, not by the
#: argmin of a sweep.
N_SAMPLES = 8500

#: Pilot samples.  A short DENSE pass doing three jobs at once: it supplies
#: ``alpha`` (which the threshold needs), the frozen constants for the pruned
#: neurons, and the depth-1 unscored filler rows.  150 is the ROBUST choice:
#: P = 80/100/150/250 land within 3%, P = 600 is clearly worse because the
#: pilot then eats the samples it was meant to improve.
N_PILOT = 150

#: Highest Hermite order used by the layer-1 control variate.  k=3 adds ~2%
#: of explained variance against 3.0% of estimation noise, so it loses.
CV_KMAX = 2

#: Scales the offline head's output.  ``0.0`` is the exact ablation: the
#: identical code path with the correction switched off, reproducing the
#: uncorrected sparse estimator bit for bit.
DAMP = 1.0

#: Relative jitter on the analytic Hermite Gram's diagonal.  Insurance only:
#: measured cond(2 rho .^ 2) = 2.41, because squaring O(1/16) correlations
#: makes them O(1/256).  ``rho`` itself -- the k=1 Gram -- has cond = 3.8e8 at
#: this shape, which is exactly why the k=1 block is evaluated in the input
#: basis where the Gram is the identity and no solve is needed at all.
CV_GRAM_JITTER = 1e-6

#: Samples for the defensive dense fallback.  A single raising MLP is
#: catastrophic — the grader zeroes that prediction, whose MSE is O(1) against
#: a score of O(1e-6), so one failure in 100 would dominate by ~850x.  Sized
#: so that even a raise on the very last operation, with the whole sparse pass
#: already billed, lands at C/B ~ 0.18.
FALLBACK_SAMPLES = 6000

#: Floor applied to pre-activation variances before taking a square root.
VAR_FLOOR = 1e-12

#: 1 / sqrt(2 pi).
INV_SQRT_2PI = 0.3989422804014327

#: File holding the offline-trained head.  Regenerable bit-identically by
#: ``scripts/28_learned_corrector.py --mode data`` then
#: ``--mode fit --install``.
COEF_FILE = "corrector.npz"

#: Column order of ``corrector.npz``'s ``beta``, for audit.  Must equal
#: ``whestfloor.corrector.FEATURES``; ``tests/test_submission_parity.py``
#: pins the two designs against each other bitwise.
FEATURES = (
    "one",
    "cv1", "cv1_Phi", "cv1_a",
    "cv2", "cv2_Phi", "cv2_a",
    "cv1mf", "cv1mf_Phi", "cv1mf_a",
    "s", "Phi", "phi", "a",
    "dpilot",
)


class Estimator(BaseEstimator):
    """Sparse Monte Carlo + layer-1 Hermite CVs + an offline-trained head."""

    def __init__(self) -> None:
        self._beta = None

    def setup(self, ctx) -> None:  # noqa: ANN001 - whestbench SetupContext
        # ``fnp.load`` is billed at 0 FLOPs (measured), and setup runs off
        # budget inside a ~5 s window that also covers imports, so this must
        # be a plain pickle-free npz read and nothing else.
        d = getattr(ctx, "submission_dir", None) or os.path.dirname(
            os.path.abspath(__file__))
        try:
            self._beta = fnp.load(os.path.join(d, COEF_FILE))["beta"]
        except Exception:  # noqa: BLE001 - no head is a valid degradation
            self._beta = None

    def predict(self, mlp, budget: int):  # noqa: ANN001 - whestbench MLP
        _ = budget
        # Seeded from mlp.seed per the whestbench contract: the grader supplies
        # the same seed to every submission, and self-seeded randomness risks
        # prize disqualification.  One generator feeds both the pilot and the
        # scored draw, so the scored samples are independent of the pilot
        # while every stream still descends from mlp.seed alone.
        try:
            return self._sparse(mlp, TAU, N_SAMPLES, N_PILOT, mlp.seed)
        except Exception:
            return self._dense(mlp, FALLBACK_SAMPLES, mlp.seed)

    # ------------------------------------------------------------------
    def _dense(self, mlp, n_samples, seed):
        """Plain MC over all layers.  Also the fallback for the sparse path."""
        rng = fnp.random.default_rng(seed)
        x = rng.standard_normal((n_samples, mlp.width), dtype=fnp.float32)
        rows = []
        for w in mlp.weights:
            x = fnp.maximum(x @ w, 0.0)
            rows.append(fnp.mean(x, axis=0))
        return fnp.stack(rows, axis=0)

    # ------------------------------------------------------------------
    def _hermite_cv(self, x, z1, y, w1, kmax):
        """Layer-1 Hermite control-variate corrections, one per output neuron.

        Two length-N matvecs per order plus one (width x width) solve; billed,
        the whole feature block is 0.40% of the scored pass.  The k=2 basis is
        kept as ``z1^2`` rather than ``He_2(z1/sigma)``: the per-column scale
        folds into ``d`` and ``u`` and the constant folds into the centring, so
        one pass over the (N, width) array is saved.  Normalising the columns
        of ``W^1`` before the Gram keeps ``rho`` exactly symmetric (no divide
        to lose the tag) with an exact unit diagonal.
        """
        n = x.shape[0]
        h = n // 2
        sig1 = fnp.sqrt(fnp.maximum(fnp.sum(w1 * w1, axis=0), VAR_FLOOR))
        inv1 = (1.0 / sig1).astype(z1.dtype)

        bases = [(x, None, None, 0.0)]
        if kmax >= 2:
            wn = w1 * inv1
            rho = wn.T @ wn
            G = (2.0 * rho) * rho
            # fill_diagonal keeps the matmul's symmetry tag; an explicit
            # flops.symmetrize would re-tag it at 2.3e7 FLOPs and save
            # nothing on the solve, so it is deliberately absent.
            fnp.fill_diagonal(G, 2.0 * (1.0 + CV_GRAM_JITTER))
            bases.append((z1 * z1, G, inv1 * inv1, -1.0))

        out = []
        for g, G, sc, off in bases:
            g1, g2 = g[:h], g[h:]
            us = []
            for gg in (g1, g2):
                d = fnp.mean(gg, axis=0)
                if sc is not None:
                    d = d * sc + off
                us.append(d if G is None else fnp.linalg.solve(G, d))
            u1, u2 = us
            wa = g1 @ (u2 if sc is None else u2 * sc)
            wa = wa - fnp.mean(wa)
            wb = g2 @ (u1 if sc is None else u1 * sc)
            wb = wb - fnp.mean(wb)
            out.append(0.5 * ((y[:h].T @ wa) / h + (y[h:].T @ wb) / (n - h)))
        while len(out) < 2:
            out.append(fnp.zeros(w1.shape[0], dtype=x.dtype))
        return out

    # ------------------------------------------------------------------
    def _meanfield_cv(self, h1m, weights, alpha, Ph):
        """Layer-1 mean gap pushed forward through the mean-field Jacobian.

        ``E[relu(z^1_i)] = ||W^1[:,i]|| / sqrt(2 pi)`` exactly, so
        ``d1 = mean_s relu(z^1) - E[relu(z^1)]`` is exactly mean zero and
        free.  Propagating it with the rectifier Jacobian replaced by its
        expectation ``P(z^l > 0) = Phi(alpha^l)`` costs 31 matvecs -- 4.1e6
        FLOPs -- and turns it into a per-output-neuron prediction of the
        sampling error.

        Same first-order channel as the k=1 Hermite block, reached the other
        way: analytic-but-approximate coefficients instead of exact-but-
        estimated ones.  It carries no ``p/N`` estimation noise, which is why
        it measures better with a unit coefficient (1.43x against 1.25x), and
        it is biased by the mean-field approximation, which is why both are
        offered to the head rather than one being chosen.
        """
        depth = len(weights)
        w1 = weights[0]
        sig1 = fnp.sqrt(fnp.maximum(fnp.sum(w1 * w1, axis=0), VAR_FLOOR))
        prop = h1m - sig1 * INV_SQRT_2PI
        for l in range(1, depth):
            prop = prop @ weights[l]
            if l < depth - 1:
                g = flops.stats.norm.cdf(alpha[l])
                prop = prop * g.astype(prop.dtype)
        return prop * Ph

    # ------------------------------------------------------------------
    def _sparse(self, mlp, tau, n_samples, n_pilot, seed):
        n = mlp.width
        depth = len(mlp.weights)
        rng = fnp.random.default_rng(seed)

        # ---- pilot: a short dense pass -------------------------------
        x = rng.standard_normal((n_pilot, n), dtype=fnp.float32)
        alpha, mean_h = [], []
        for w in mlp.weights:
            z = x @ w
            m = fnp.mean(z, axis=0)
            v = fnp.maximum(fnp.mean(z * z, axis=0) - m * m, VAR_FLOOR)
            alpha.append(m / fnp.sqrt(v))
            x = fnp.maximum(z, 0.0)
            mean_h.append(fnp.mean(x, axis=0))

        # ---- masks and pre-sliced weights: billed once, not per sample
        # The last layer keeps all n output columns.  Pruning them would save
        # ~1% of the pass and would force a scatter back into n slots, whose
        # only failure mode (an all-dead layer) is the one thing that must
        # never raise.
        subs, biases = [], []
        keep_prev = None
        for l, w in enumerate(mlp.weights):
            keep = (None if (tau is None or l == depth - 1)
                    else alpha[l] > -tau)
            wr = w if keep_prev is None else w[keep_prev, :]
            subs.append(wr if keep is None else wr[:, keep])
            if keep_prev is None:
                biases.append(None)
            else:
                dead = mean_h[l - 1] * (1.0 - keep_prev.astype(fnp.float32))
                wd = w if keep is None else w[:, keep]
                biases.append(dead @ wd)
            keep_prev = keep

        # ---- scored pass ---------------------------------------------
        x0 = rng.standard_normal((n_samples, n), dtype=fnp.float32)
        x = x0
        z1 = h1 = None
        for l in range(depth):
            z = x @ subs[l]
            if biases[l] is not None:
                z = z + biases[l]
            if l == 0:
                z1 = z
            x = fnp.maximum(z, 0.0)
            if l == 0:
                h1 = x          # kept, not reduced: a reduction here
                                # would be billed before the damp=0
                                # early return and break the ablation
        mu = fnp.mean(x, axis=0)
        # Only the final row is scored.  The others come free from the pilot;
        # they are not blended with the scored pass, which would correlate the
        # estimate with the mask that was derived from the same samples.
        if self._beta is None or DAMP == 0.0:
            return fnp.stack(mean_h[:-1] + [mu], axis=0)

        # ---- features and the offline head ---------------------------
        # FIFTEEN columns.  Thirteen more were fitted, measured at exactly
        # 1.000x on validation, and deleted -- see FEATURES below.  ``v`` comes
        # from the raw second moment rather than a centred copy of the
        # (N, width) array, which is one pass fewer; at rms|alpha| = 3.44 the
        # cancellation costs 1e-6 relative on ``v``, six orders under the
        # ~1.7e-3 residual the head predicts, and the pilot above already uses
        # exactly this form.
        cvs = self._hermite_cv(x0, z1, x, mlp.weights[0], CV_KMAX)
        m = fnp.mean(z, axis=0)
        v = fnp.maximum(fnp.mean(z * z, axis=0) - m * m, VAR_FLOOR)
        s = fnp.sqrt(v)
        a = m / s
        # norm.cdf/pdf promote float32 -> float64 to match scipy and float64
        # bills at 2x; cast straight back so nothing downstream inherits it.
        Ph = flops.stats.norm.cdf(a).astype(a.dtype)
        ph = flops.stats.norm.pdf(a).astype(a.dtype)
        one = fnp.ones_like(a)
        cv1, cv2 = cvs[0], cvs[1]
        mf = self._meanfield_cv(fnp.mean(h1, axis=0), mlp.weights,
                                alpha, Ph)
        cols = [
            one,
            cv1, cv1 * Ph, cv1 * a,
            cv2, cv2 * Ph, cv2 * a,
            mf, mf * Ph, mf * a,
            s, Ph, ph, a,
            mu - mean_h[-1],
        ]
        corr = fnp.stack(cols, axis=1) @ self._beta
        if DAMP != 1.0:
            corr = corr * DAMP
        return fnp.stack(mean_h[:-1] + [mu + corr], axis=0)
