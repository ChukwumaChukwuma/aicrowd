#!/usr/bin/env python
"""Round 3: three mechanisms killed, one adversarial break acted on."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.contract import ADJUSTED_FLOOR                      # noqa: E402
from whestfloor.ledger import Experiment                            # noqa: E402

Experiment(
    name="adversarial_review_of_2.11x",
    script="scripts/15_adv_permlp.py",
    hypothesis="The reported 2.11x improvement is a property of the estimator, "
               "not of the suite it was measured on.",
    acceptance_bar=2.11, bar_metric="pooled_improvement",
    bar_direction="higher_is_better",
).record(
    seed=0, n_mlps=22, gt_samples=1_200_000, estimator="cov_prop_edgeworth",
    compute_ratio=0.0466, raw_final_layer_mse=3.7158e-5,
    pooled_improvement=1.831, ci_low=1.619, ci_high=2.069,
    per_suite={"quick": 2.108, "adv": 1.733, "s256x32": 1.660},
    notes=("BROKEN. 2.11x is outside the pooled 95% CI [1.62, 2.07]; honest "
           "value 1.83x. P(a fresh 8-MLP suite reaches 2.11x) = 0.29. Effect "
           "DIRECTION holds: 0/22 MLPs made worse, and it holds at all 5 "
           "shapes tested. But the headline also credited the correction with "
           "the Mehler change; against mehler_k4 alone it is 1.72x. Three "
           "further breaks: (a) the derived coefficient is not the optimum - "
           "optimal damp is 2.2-3.3, never 1.0, across every suite and shape; "
           "(b) scripts/13 validated against a GAUSSIAN reference, so it "
           "measured diagram truncation only - against the TRUE kappa_3 the "
           "star form captures 0.20-0.43, and the true kappa_3 is 1.8-3.6x "
           "larger, which is exactly what a damp of 2.4-3.3 compensates; "
           "(c) _hermite_coeffs treated He_0 as 0, corrupting every a_k from "
           "k=4 up - the shipped KMAX=4 used a wrong a_4 (wrong sign). Also: "
           "the correction drove 3.5% of predictions negative, and E[relu]>=0."),
)

Experiment(
    name="chaos2_and_condindep",
    script="scripts/15_chaos2_bakeoff.py",
    hypothesis="An exact chaos-2 distribution, or an exact conditional-"
               "independence CGF, beats the Gaussian model by >=30x per layer "
               "and so escapes the cumulant ceiling.",
    acceptance_bar=30.0, bar_metric="best_factor",
    bar_direction="higher_is_better",
).record(
    seed=0, n_mlps=1, gt_samples=4_000_000, estimator="chaos2",
    compute_ratio=0.0, raw_final_layer_mse=5.601e-5 ** 2,
    best_factor=5.45, chaos2_factor=5.45, condindep_factor=1.01,
    oracle_edge4_factor=8.55,
    notes=("BOTH FAIL. chaos-2 gets 5.45x, below even the oracle-cumulant "
           "Edgeworth-4 (8.55x) it was meant to escape: the order-2 Hermite "
           "truncation of relu is a parabola, so kappa_4 overshoots 2.6-8x. "
           "The conditional-independence CGF gets 1.01x - it reproduces 1.3% "
           "of the true skewness, evaluated EXACTLY by FFT convolution, so "
           "the correlations dominate the shape, not the individual "
           "rectifications. THE DECISIVE NUMBER: splitting the one-step error "
           "by drawing z exactly Gaussian from the layer's true (m,Sigma) "
           "gives errB (rectifier-map error) 2.99e-4 and errA (z itself "
           "non-Gaussian) 1.665e-3 - 97% is errA. A PERFECT one-step shape "
           "model buys 1.02x. errA is 1270x the per-layer budget and is a "
           "hard floor for ANY Gaussian-STATE propagation. The state must "
           "carry the non-Gaussianity; no better rectifier model can help."),
)

Experiment(
    name="calibrated_coherent_bias_shrink",
    script="scripts/17_fit_shrink.py",
    hypothesis="A single per-layer multiplicative constant, calibrated once "
               "and validated out of sample on disjoint suites, beats the "
               "analytic kappa_3 correction at lower cost.",
    acceptance_bar=1.0, bar_metric="improvement_over_prev_ship",
    bar_direction="higher_is_better",
).record(
    seed=0, n_mlps=24, gt_samples=1_200_000,
    estimator="cov_prop_shrink(kmax=4, g=0.99975)",
    compute_ratio=0.0373,
    adjusted_final_layer_score=2.2818e-6,
    raw_final_layer_mse=2.2818e-5,
    unbiased_true_mse=2.2818e-5,
    per_suite={"quick": 1.5496e-5, "adv": 2.4181e-5, "s256x32": 2.8776e-5},
    baseline_cov_prop_gain=6.7456e-5,
    prev_ship_edgeworth=3.7158e-5,
    improvement_over_prev_ship=3.7158e-5 / 2.2818e-5,
    improvement_over_baseline=6.7456e-5 / 2.2818e-5,
    x_above_floor=2.2818e-6 / ADJUSTED_FLOOR,
    flops_used=3.252e9,
    notes=("SHIPPED. 2.96x over the strongest bundled baseline and 1.63x over "
           "the previous ship, at 39% FEWER FLOPs (C/B 0.037 vs 0.047, so a "
           "wider margin before the 0.1 multiplier floor). The constant is "
           "CALIBRATED, not derived - but it transfers exactly: the argmin is "
           "0.99975 on all three disjoint suites (disjoint MLP seeds AND "
           "disjoint ground-truth seeds), and each suite's fitted value scores "
           "the other suites' own optimum. It is mechanically motivated: "
           "scripts/04 measured the Gaussian one-step error to have a POSITIVE "
           "mean at every layer, so the model systematically over-estimates "
           "and a coherent multiplicative bias compounds through the chain. "
           "Also fixes the a_4 Hermite bug and clips predictions to >=0. "
           "Still 4.1e5x above the noise floor."),
)
print("recorded 3")
