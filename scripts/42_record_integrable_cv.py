#!/usr/bin/env python
"""Round 12: expressive AND analytically integrable control variates.

Every bar was fixed before its run.  The headline bar is the parent brief's:
held-out explained variance > 60% net of ``p/N``, scored on the real objective
``adjusted = 0.1 b^2 + (1 - R^2) V0`` rather than on ``R^2`` alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment  # noqa: E402

LOC = dict(gt_samples=2_000_000, seed=900_000, n_mlps=3)
V0 = 4.06e-07  # (1 - R^2) V0 is the sampling term of adjusted at the ship point
SHIP_ADJ = 2.47e-07

# ---------------------------------------------------------------------------
Experiment(
    name="depth_ladder_integrability_trade",
    script="scripts/41_integrable_cv.py",
    hypothesis=(
        "docs/hermite_rank_ceiling.md sec 5.3 says functions explaining 90% of "
        "sum_j Var(y_j) exist.  They do, they are cheap, and the network "
        "already computes them: a LINEAR control variate in the layer-L "
        "activations reaches R^2 = 86% at L=16 and 99% at L=32.  The only "
        "thing missing is E[g].  Propagating it by the Gaussian closure "
        "(exact at L=1, exact second moments at L=2) makes the estimator "
        "biased, and the bias enters the score as 0.1 b^2.  If R^2 grows "
        "faster than b^2 the ladder opens; the argmin of "
        "adjusted(L) = 0.1 b(L)^2 + (1 - R^2(L)) V0 decides it."),
    acceptance_bar=SHIP_ADJ, bar_metric="adjusted_final_layer_score",
    bar_direction="lower_is_better",
).record(
    **LOC,
    estimator="linear CV in relu(z^L), analytic Gaussian-closure mean, L swept",
    adjusted_final_layer_score=2.536e-07,     # the argmin, at L = 1
    raw_final_layer_mse=2.536e-06,            # adjusted / 0.1 (clamped)
    compute_ratio=0.1,
    b=8.842e-05, v_eff_r2=0.3773, c=2.79e6,
    n_span_samples=120_000, n_ref_samples=2_000_000, n_ship=27_000,
    argmin_L=1,
    R2_eff_by_L={"1": 0.3773, "2": 0.4615, "3": 0.5260, "4": 0.5821,
                 "6": 0.6590, "8": 0.7164, "12": 0.8001, "16": 0.8596,
                 "20": 0.9043, "24": 0.9383, "28": 0.9669, "32": 0.9886},
    rms_bias_by_L={"1": 8.842e-05, "2": 1.558e-03, "3": 2.552e-03,
                   "4": 3.360e-03, "6": 4.398e-03, "8": 5.262e-03,
                   "12": 5.777e-03, "16": 6.022e-03, "20": 6.071e-03,
                   "24": 6.683e-03, "28": 6.754e-03, "32": 6.996e-03},
    adjusted_by_L={"1": 2.536e-07, "2": 4.612e-07, "3": 8.436e-07,
                   "4": 1.298e-06, "6": 2.073e-06, "8": 2.884e-06,
                   "12": 3.418e-06, "16": 3.683e-06, "20": 3.725e-06,
                   "24": 4.492e-06, "28": 4.575e-06, "32": 4.899e-06},
    x_ship_by_L={"1": 0.974, "2": 0.536, "4": 0.190, "8": 0.086,
                 "16": 0.067, "24": 0.055, "32": 0.050},
    closure_accuracy_factor_needed=4.4,
    best_if_bias_were_zero={"8": 2.14, "16": 4.33, "24": 9.86, "32": 53.4},
    L1_bias_is_reference_noise=True,
    L1_reference_noise_floor=1.35e-04,
    notes=(
        "FAILS: the argmin is L = 1, which is where the shipped basis already "
        "is (0.974x is the run-to-run agreement between this measurement and "
        "the ship, not a loss).  Every deeper layer is worse, monotonically, "
        "down to 0.050x at L = 32.  MECHANISM: R^2 and b^2 both saturate, but "
        "b^2 saturates FIRST -- b is already 88% of its L=32 value by L=8, "
        "while R^2 is only 72% of the way from 38% to 99%.  So the trade is "
        "decided in the first few layers and it is decided against.  The "
        "measurement that matters for the future is the LAST column: 'r "
        "needed' is 4.2-4.6 at EVERY depth from 6 to 32, i.e. one uniform "
        "4.4x improvement in analytic layer-mean accuracy would open the "
        "entire ladder at once, and with a perfect mean L=16 is 4.3x, L=24 "
        "9.9x and L=32 53x.  The best analytic machinery in this repository "
        "(kappa_3 star diagrams, docs/state_of_play.md) is 1.5x better than "
        "the plain Gaussian closure measured here, and the oracle-cumulant "
        "ceiling from scripts/11 is 8.2x on the reconstruction step alone.  "
        "So the control-variate question REDUCES to the closure-accuracy "
        "question: the basis is not the lever, E[g] is.  L=1 bias 8.8e-05 is "
        "the 2e6-sample reference's own noise floor (1.35e-04 predicted), so "
        "layer 1 is confirmed exactly integrable to the precision measured."),
).__str__()
print("recorded: depth_ladder_integrability_trade")
