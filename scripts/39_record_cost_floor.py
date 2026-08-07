#!/usr/bin/env python
"""Round 11: the 6-9x cut in billed FLOPs per sample, measured and killed.

Three bars, all fixed before the runs and none re-rolled.  The headline bar was
the graded score, ``adjusted < 2.6082e-07``; the two enabling bars were the
per-route conditions that would have had to hold for anything to reach it.

The kill that matters is ``cheap_model_floor``, because it is a bound on the
whole family of sub-forward-pass estimators rather than a verdict on one
construction: it uses ORACLE moments, so no practical scheme beats the curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment  # noqa: E402

LOC = dict(gt_samples=1_000_000_000)

# ---------------------------------------------------------------------------
Experiment(
    name="linearise_and_fold",
    script="scripts/37_cost_floor.py",
    hypothesis=(
        "A neuron is affine to first order everywhere, not only at large "
        "|alpha|: relu(z) = a + Phi(alpha) z + delta with E[delta] = 0 and "
        "Var(delta) = Var(relu z) - Phi^2 Var(z), which is 8.5x smaller than "
        "the Var(relu z) that freezing at a constant injects.  Linearised "
        "neurons leave the per-sample matmuls entirely because consecutive "
        "affine blocks compose into one precomputed matrix.  If the "
        "linearisation damage is CONCENTRATED, a few hundred exact neurons "
        "suffice and c collapses by 6-9x at tolerable bias."),
    acceptance_bar=6.0, bar_metric="cost_gain_at_affordable_bias",
    bar_direction="higher_is_better",
).record(
    **LOC, seed=700_000, n_mlps=2,
    estimator="linearise-and-fold over sparse MC tau=2.5, exact set |E| swept",
    cost_gain_at_affordable_bias=1.00,
    raw_final_layer_mse=1.57e-9,          # b^2 at the |E| that is affordable
    compute_ratio=0.2324,
    n_pilot_samples=120_000,
    n_linearisable_neurons=7936,
    resid_over_var_relu=0.1178,
    stein_check_max_abs_err=9.38e-2,
    damage_share_top8=0.0190, damage_share_top256=0.2728,
    damage_share_top2048=0.8441, damage_share_top4096=0.9923,
    b2_at_E256=1.410e-5, cost_gain_at_E256=9.19,
    b2_at_E2048=6.479e-7, cost_gain_at_E2048=1.46,
    b2_at_E4096=1.567e-9, cost_gain_at_E4096=1.00,
    fold_pays_iff="m < d/2 (exact set under half the state)",
    undecided_over_alwayson_at_tau2p5=3.34,
    notes=(
        "FAILS at 1.00x against a 6x bar.  The trade curve crosses from "
        "unusable to free without passing through useful: |E| = 256 folds to "
        "9.19x with b^2 = 1.4e-05 (1,400x the budget), |E| = 4096 has "
        "b^2 = 1.6e-09 (affordable) and folds to exactly 1.00x.  MECHANISM: "
        "the fold cost of k layers with state d and m exact neurons is "
        "2 d^2 + 4(k-1) d m + (k-1)(k-2) m^2, minimised at k=1 -- no fold at "
        "all -- as soon as m >= d/2 = 128; and the damage is spread almost "
        "uniformly over the 7,936 linearisable neurons (worst 256 carry 27%), "
        "so the exact set that keeps the bias affordable is 4096 = half the "
        "network, exactly where the fold stops paying.  Same rank obstruction "
        "as docs/hermite_rank_ceiling.md sec 5.2 in a different coordinate.  "
        "COROLLARY, and it kills the weaker 'prune the always-ON neurons too' "
        "version outright: the fold needs UND < ON and at tau = 2.5 the "
        "undecided set is 62.4% of a layer against 18.7% always-on, a ratio of "
        "3.34, so k=1 is optimal at every layer and there is no fold to make."),
)

# ---------------------------------------------------------------------------
Experiment(
    name="cheap_model_floor",
    script="scripts/37_cost_floor.py",
    hypothesis=(
        "An estimator cheaper than a forward pass must replace a PREFIX of the "
        "network by a distribution.  Give that model every advantage -- an "
        "exact Gaussian with the ORACLE mean and full covariance of z^j from a "
        "400k-sample pass, then layers j+1..32 exactly -- and measure the "
        "irreducible squared bias against billed cost.  The curve is a FLOOR "
        "on the entire family.  For the family to be alive, some point at "
        "c <= 4 layer-equivalents must beat the graded 2.6082e-07, which needs "
        "0.1 * b^2 < 2.6082e-07, i.e. b^2 < 2.61e-06."),
    acceptance_bar=2.61e-6, bar_metric="b2_at_4_layer_equivalents",
    bar_direction="lower_is_better",
).record(
    **LOC, seed=700_000, n_mlps=3,
    estimator="oracle-Gaussian truncation at z^j + exact layers j+1..32",
    b2_at_4_layer_equivalents=3.767e-6,
    raw_final_layer_mse=3.767e-6,
    compute_ratio=0.1,
    n_ref_samples=400_000, n_surrogate_samples=400_000,
    b2_closure_from_exact_m_s=9.036e-7,
    b2_j30=3.251e-6, b2_j29=3.767e-6, b2_j28=4.082e-6,
    b2_j26=5.660e-6, b2_j24=6.872e-6, b2_j20=8.867e-6,
    c_j29_flops_per_sample=524_288,
    best_possible_adjusted_at_4_layer_eq=3.767e-7,
    shipped_graded_adjusted=2.6082e-7,
    notes=(
        "FAILS at 3.77e-06 against a 2.61e-06 bar, and the failure is "
        "monotone: the floor gets WORSE the earlier the truncation, i.e. "
        "exactly where the savings are (3.25e-06 at 3 layer-equivalents "
        "rising to 8.87e-06 at 13), because the model error is injected once "
        "and then amplified by the remaining nonlinear layers.  Since "
        "adjusted >= 0.1 b^2, EVERY point on the curve is worse than the "
        "graded 2.6082e-07 at any N, with oracle moments.  The 1-layer-"
        "equivalent endpoint -- no truncation, just the Gaussian closure of "
        "the final rectifier with the exact (m,s) of z^32 -- floors at "
        "b^2 = 9.04e-07, independently reproducing the ledger's "
        "'Gaussian closure / Rao-Blackwell the last layer = 0.809x'.  "
        "CONSEQUENCE FOR THE LEADERBOARD READ: graded submission 323861 (raw "
        "4.24e-8, C/B 1.12) back-solves to ~478,000 billed FLOPs/sample = 3.6 "
        "layer-equivalents only if its v_eff equals ours; this curve says a "
        "3.6-layer-equivalent sampler has b^2 >= 3.8e-06, which is 89,000x its "
        "measured raw.  So 323861 is not a cheap-per-sample sampler -- it is "
        "either a normal-cost sampler with ~5x better v_eff, or deterministic "
        "with 2.1e-4 rms model error.  The lever is not c."),
)

# ---------------------------------------------------------------------------
Experiment(
    name="damage_ranked_mask",
    script="scripts/37_cost_floor.py",
    hypothesis=(
        "alpha ranks neurons by Var(relu z) alone.  The damage a frozen neuron "
        "does is that TIMES its downstream sensitivity sum_j (phi_j/s_j) "
        "R_l[i,j]^2, so a damage-ranked mask should freeze materially more "
        "neurons at the same bias.  Bar: 1.3x on c, which is the smallest cut "
        "worth the 31 extra n^3 matmuls the Jacobian costs."),
    acceptance_bar=1.3, bar_metric="cost_gain_over_alpha_mask",
    bar_direction="higher_is_better",
).record(
    **LOC, seed=700_000, n_mlps=2,
    estimator="sparse MC with the mask ranked by Var(relu z) x downstream sens",
    cost_gain_over_alpha_mask=1.037,
    raw_final_layer_mse=1.1144e-6,
    compute_ratio=0.2287,
    neurons_frozen_alpha_tau2p5=1477,
    neurons_frozen_damage_ranked=1569,
    sensitivity_p90_over_p10_layer1=5.4,
    sensitivity_p90_over_p10_layer31=1.5,
    jacobian_cost_flops=1.0e9,
    notes=(
        "FAILS at 1.037x against a 1.3x bar.  MECHANISM: the downstream "
        "sensitivity barely varies WITHIN a layer -- p90/p10 is 5.4 at layer 1 "
        "and only 1.5 at layer 31 -- while Var(relu z) spans five orders of "
        "magnitude, so alpha is already nearly the right order and there is "
        "almost nothing for the second factor to re-rank.  The 3.7% is real "
        "but sits inside the grader's own ~2% run-to-run noise and needs a "
        "0.4%-of-budget Jacobian to compute.  Not shipped.  A first-order "
        "VARIANCE criterion was also built (freezing coordinate i changes "
        "Var(z^32_j) by -2 R[i,j](C_l R_l)[i,j] + R[i,j]^2 C_l[i,i]) and is "
        "NOT usable: it says 68% of neurons would REDUCE the variance if "
        "frozen, which the measured tau sweep (v_raw +356% at tau=1.0) "
        "refutes -- the marginal formula is not additive over a set because "
        "the omitted quadratic term R_S' C_SS R_S grows as |S|^2."),
)

print("recorded 3 cost-lever experiments")
