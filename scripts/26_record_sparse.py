#!/usr/bin/env python
"""Round 6: the sign-stable sparse sampler.

Two rows, because two different things happened and conflating them would
launder a failed bar into a success.

1. The pre-committed bar was **4x per-sample cost reduction** at a sign error
   under 10% of the final MSE.  It **FAILED**: the best usable point is 1.50x,
   and 4x is the theoretical ceiling of the mechanism, reachable only at a
   threshold that destroys the estimator.

2. Separately, and despite that, the **score improved 1.44x** and the result
   shipped, because a 1.50x cheaper sample is still a cheaper sample and the
   estimator now sits below the grader's own plain-MC constant for the first
   time.

Numbers from ``scripts/25_sparse_sign_stable.py`` on the official 100-MLP
suite (N=1e9 reference, so raw_final_layer_mse is leaderboard-comparable).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment  # noqa: E402

OFF = dict(seed=0, n_mlps=100, gt_samples=1_000_000_000)
SCRIPT = "scripts/25_sparse_sign_stable.py"

Experiment(
    name="sparse_sign_stable_cost_bar",
    script=SCRIPT,
    hypothesis="A sign-stable sparse pass is at least 4x cheaper per sample "
               "than a dense one, at a sign-error rate contributing less than "
               "10% of the final MSE.",
    acceptance_bar=4.0,
    bar_metric="cost_reduction_per_sample",
    bar_direction="higher_is_better",
).record(
    **OFF,
    estimator="sparse_mc_kernel(tau=2.5) vs dense, per-sample FLOPs",
    cost_reduction_per_sample=1.50,
    compute_ratio=0.0999,
    raw_final_layer_mse=5.6530e-6,
    dense_flops_per_sample=4_198_656,
    sparse_flops_per_sample=2_793_985,
    modal_fusion_flops_per_sample=20_361_710,
    flips_per_sample=987.3,
    flips_per_sample_pct=12.05,
    kink_fraction_tau3=0.714,
    kink_per_flip_tau3=5.9,
    sign_error_mse_tau2p5_bracket=[1e-8, 1.5e-6],
    sign_error_mse_tau3=3e-8,
    sign_error_mse_tau2=1.5e-6,
    sign_error_mse_tau1=1.9e-4,
    var_modal_arm=1.968,
    var_correction_arm=1.921,
    var_z32=0.128,
    var_h32_plain_mc=0.062,
    rao_blackwell_gain_tau2=1.001,
    notes=(
        "FAILS at 1.50x against a 4x bar, and the ceiling is structural: "
        "pruning cost scales as (|ON|/n)^2 and ON must contain every neuron "
        "with positive mean (~50% by construction), so 4x is reachable only "
        "at tau=0 -- measured 3.83x there, with a sign error of 4.5e-2, "
        "7700x the whole score. Three separate premises were refuted by "
        "measurement. (a) Flips are 987/8192 = 12.05% per sample, not the "
        "410/5% assumed; worse, a flip-correcting scheme must EVALUATE the "
        "kink set {|alpha|<tau}, which is 71.4% of neurons at tau=3 -- 5.9x "
        "more than actually flip -- because alpha is ~Gaussian across "
        "neurons so its density near zero falls only linearly. rms|alpha| at "
        "layer 32 is 3.44, not 4.4, and is exactly 0 at layer 1. (b) The "
        "modal-fusion decomposition z32 = xA + sum eps R is exact (verified "
        "to 3e-5) but useless: mu_input = 0 exactly so A predicts NOTHING, "
        "and the arms cancel 15.3x (Var 1.968 and 1.921 summing to 0.128), "
        "leaving the correction arm with 31x MORE variance than plain MC. "
        "Billed, that scheme costs 4.95x MORE per sample than the dense pass "
        "it replaces, plus 51% of the free budget in setup, because the "
        "kink-to-kink coupling is O(depth^2) over sets half the width. (c) "
        "There is no Rao-Blackwellisation: the decided neurons carry 0.102% "
        "of the estimator variance at tau=2 and 0.001% at tau=3, so "
        "conditioning on them is worth 1.001x. Their ReLU deviation has "
        "Var ~1e-7 against z's 0.1. The variance lives entirely in the kink "
        "neurons, which must still be sampled."
    ),
)

Experiment(
    name="ship_sparse_sign_stable_mc",
    script=SCRIPT,
    hypothesis="Pruning the always-off neurons out of every matmul beats "
               "plain Monte Carlo through the identical code path, and beats "
               "the shipped blend.",
    acceptance_bar=1.0,
    bar_metric="improvement_over_prev_ship",
    bar_direction="higher_is_better",
).record(
    **OFF,
    estimator="sparse_mc_kernel(tau=2.5, n=8500, P=150)",
    adjusted_final_layer_score=5.8700e-7,
    raw_final_layer_mse=5.8050e-6,
    unbiased_true_mse=5.8050e-6,
    compute_ratio=0.1011,
    flop_ratio=0.0915,
    flops_used=2.489e10,
    adjusted_at_2x_residual=6.4258e-7,
    adjusted_at_3x_residual=6.9816e-7,
    ablation_dense_same_path_raw=8.6466e-6,
    ablation_dense_same_path_adjusted=8.8488e-7,
    improvement_over_ablation=1.53,
    improvement_over_ablation_4seed_mean=1.44,
    improvement_over_prev_ship=7.7791e-7 / 5.8700e-7,
    prev_ship_adjusted=7.7791e-7,
    blend_with_sparse_arm_adjusted=7.7532e-7,
    grader_sampling_mse=6.4695e-7,
    competitor_sign_stable_adjusted=1.60e-7,
    leader_adjusted=3.63e-10,
    x_above_floor=5.8700e-7 / 4.949e-12,
    n_failed_mlps=0.0,
    worst_single_mlp_mse=4.2164e-5,
    max_compute_ratio_over_suite=0.1335,
    notes=(
        "SHIPPED, and it drops the analytic arm entirely. 1.33x over the "
        "previous ship and 1.53x over plain MC through the IDENTICAL code "
        "path (tau=None reproduces dense bit for bit -- pinned by "
        "tests/test_submission_parity.py). Across four independent seeds the "
        "ablation gain is 1.53/1.37/1.33/1.56, mean 1.44x, so it is not a "
        "single-draw artefact. First estimator in this repo below the "
        "grader's own plain-MC constant 6.4695e-7, by 1.10x. The analytic "
        "arm was dropped on measurement, not taste: blending it back in "
        "gives 7.75e-7 because its 2.7e9 FLOPs push C/B from 0.100 to 0.124 "
        "and the 1.24x multiplier penalty exceeds what it buys -- a second "
        "estimator only pays below the multiplier floor, and there is no "
        "room below the floor. Sized by F/B = 0.0915 (machine-independent) "
        "rather than C/B, leaving 23ms of residual before the floor; "
        "shrinking N to hedge against the grader's single core was tested "
        "and is WRONG (N=7073 sized to the floor at 3x residual scores "
        "7.12e-7 there, against 6.82e-7 for N=8500), because above the floor "
        "the adjusted score is flat in N. CAVEAT: tau was swept on the same "
        "100 official MLPs it is scored on, so it is CALIBRATED, not "
        "derived; the optimum is broad (2.3/2.5/2.7 -> 5.97/5.65/5.79e-7) "
        "but the private re-evaluation uses a fresh suite. The mask is also "
        "derived from the sample stream, so the estimator is unbiased only "
        "conditionally on the mask. 0 raises over all 100 official MLPs, "
        "worst single MLP 4.2164e-5, max C/B 0.1335, and the defensive dense "
        "fallback was exercised and verified. See docs/sparse_sign_stable.md."
    ),
)
print("recorded 2")
