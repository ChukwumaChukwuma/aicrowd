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

ONLY = set(sys.argv[1:])


def want(name: str) -> bool:
    """Records are immutable and append-only; re-running a name duplicates it."""
    return not ONLY or name in ONLY


LOC = dict(gt_samples=2_000_000, seed=900_000, n_mlps=3)
V0 = 4.06e-07  # (1 - R^2) V0 is the sampling term of adjusted at the ship point
SHIP_ADJ = 2.47e-07

# ---------------------------------------------------------------------------
if want("depth_ladder_integrability_trade"):
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

# ---------------------------------------------------------------------------
if want("eigenfunction_degree_content"):
    Experiment(
        name="eigenfunction_degree_content",
        script="scripts/41_integrable_cv.py",
        hypothesis=(
            "If the top-8 eigenfunctions of Cov(y) -- the ones that carry 90.1% of "
            "sum_j Var(y_j) -- are well approximated by low-degree Hermite "
            "polynomials in x, their means are exact by Wick and the 10x lever is "
            "open.  Measure the approximation quality: this single measurement "
            "decides the direction."),
        acceptance_bar=0.75, bar_metric="top8_degree_le_2_share",
        bar_direction="higher_is_better",
    ).record(
        gt_samples=100_000, seed=900_000, n_mlps=3,
        estimator="degree decomposition of the top-8 eigenfunctions of Cov(y)",
        top8_degree_le_2_share=0.514,
        raw_final_layer_mse=3.7157e-06, compute_ratio=0.1,
        b=0.0, v_eff_r2=0.392, c=2.79e6,
        top1_share=[0.5730, 0.7744, 0.6584], top8_share=[0.8781, 0.9366, 0.9031],
        top8_share_mean=0.9059,
        top_eigfn_deg1=[0.3129, 0.3291, 0.2841],
        top_eigfn_deg2=[0.2807, 0.3116, 0.2918],
        top8_vw_deg_le_2=[0.4839, 0.5754, 0.4832],
        ou_bound_top_eigfn_deg_le_2=[0.8529, 0.8826, 0.8235],
        ou_bound_top_eigfn_deg_le_3=[0.9497, 0.9773, 0.9453],
        full_degree_le_2_ceiling=0.456, full_degree_le_2_features=32896,
        p_over_N_of_full_degree_2=1.219,
        notes=(
            "FAILS at 51.4% against a 75% bar, and the failure is structural.  "
            "FIRST: the '90.1% at p=8' bound is a TAUTOLOGY.  The operator "
            "T = sum_j ybar_j (x) ybar_j sends every function into "
            "span{ybar_1..ybar_256}, so its eigenfunctions ARE linear combinations "
            "of the centred outputs and its non-zero spectrum IS that of the "
            "256x256 matrix Cov(y).  'The top-8 eigenfunctions explain 90.1%' "
            "means 'Cov(y) has effective rank 8'; their means are zero only "
            "because E[y_j] was subtracted, which is the answer.  SECOND: they are "
            "not low-degree.  Degrees 1 and 2 measured in closed form (||E[Y x]||^2 "
            "and (1/2)||E[Y(xx'-I)]||_F^2, cross-half unbiased) give 51.4% of the "
            "top-8 energy at degree <= 2, and an eigendecomposition cannot change "
            "the degree content, so the ceiling for ALL degree-<=2 polynomials is "
            "f_1 + f_2 = 45.6% of Var(y) -- of which the shipped 512-feature "
            "rank-one basis already realises 39.3%.  The remaining 6.3 points cost "
            "32,384 more fitted coefficients than N contains (p/N = 1.22)."),
    ).__str__()
    print("recorded: eigenfunction_degree_content")

# ---------------------------------------------------------------------------
if want("symmetry_derived_variates"):
    Experiment(
        name="symmetry_derived_variates",
        script="scripts/41_integrable_cv.py",
        hypothesis=(
            "Two exact symmetries give mean-known variates with no model at all.  "
            "x -> -x preserves N(0,I), so the antithetic pair mean has the same "
            "mean as y and NO odd Hermite degree; the networks have no biases so y "
            "is exactly positively homogeneous, y(x) = ||x|| Y(x/||x||) with "
            "E||x|| known in closed form, which Rao-Blackwellises the radius "
            "exactly.  Free, exact, and worth measuring against the shipped basis."),
        acceptance_bar=1.05, bar_metric="x_ship_resid_per_cost",
        bar_direction="higher_is_better",
    ).record(
        gt_samples=150_000, seed=900_000, n_mlps=3,
        estimator="antithetic pairing and radial Rao-Blackwell over sparse MC + t,He2",
        x_ship_resid_per_cost=0.958,
        raw_final_layer_mse=3.7157e-06, compute_ratio=0.1,
        b=0.0, v_eff_r2=0.392, c=2.79e6,
        E_norm_x_exact=15.9843826666,
        anti_raw_var_gain=1.084, homog_raw_var_gain=1.031,
        anti_x_ship=[0.9624, 0.9522, 0.9606],
        homog_x_ship=[0.9811, 0.9651, 0.9779],
        both_x_ship=[0.9281, 0.8909, 0.9193],
        R2_plain=[0.3293, 0.4307, 0.3379], R2_anti=[0.2443, 0.3498, 0.2689],
        R2_homog=[0.2956, 0.3826, 0.2977],
        notes=(
            "FAILS at 0.958x (antithetic), 0.975x (homogeneity), 0.929x (both) "
            "against a 1.05x bar.  Both symmetries are exactly what they claim: "
            "the antithetic pair mean really does carry only even degrees (46.1% "
            "of a single draw's variance, i.e. 1.084x per unit cost on RAW "
            "variance) and the homogeneity identity is exact to float (1.031x).  "
            "They lose because the shipped dictionary already spans what they "
            "remove.  26.7 of the 45 points of odd-degree variance are degree 1, "
            "which the layer-1 Hermite k=1 block removes exactly at HALF the cost, "
            "so pairing spends a factor two in samples to buy the odd degrees >= 3 "
            "while cutting the CV's R^2 from 32.9% to 24.4%.  Likewise "
            "||x||^2 - n IS sum_i He_2 on any orthonormal frame, so the k=2 block "
            "already contains the radial variate.  MECHANISM: an exactly-mean-zero "
            "variate is worth nothing if the dictionary you already ship spans it, "
            "and both of these are inside the shipped span at strictly higher cost "
            "per unit of it."),
    ).__str__()
    print("recorded: symmetry_derived_variates")

# ---------------------------------------------------------------------------
if want("exactly_integrable_quadratic_in_h1"):
    Experiment(
        name="exactly_integrable_quadratic_in_h1",
        script="scripts/41_integrable_cv.py",
        hypothesis=(
            "Cov(relu(z^1)) is the arc-cosine kernel -- closed form, exact -- so "
            "Cov(z^2) = W^2' Cov(h^1) W^2 is exact too, and EVERY quadratic form "
            "in the layer-1 activations has an exactly known mean.  z^2 is "
            "already computed by the forward pass, so this dictionary is nearly "
            "free.  It is not a polynomial in x (relu(z_i)relu(z_j) carries every "
            "even degree) so it is not bounded by f_1 + f_2 = 45.6%, and it is "
            "not rank-one at degree 2 so it is not bounded by "
            "docs/hermite_rank_ceiling.md sec 5 either.  It is the only "
            "unexplored corner of the exactly-integrable region."),
        acceptance_bar=0.60, bar_metric="R2_eff_best",
        bar_direction="higher_is_better",
    ).record(
        gt_samples=150_000, seed=900_000, n_mlps=3,
        estimator="SHIP(t,He2) + h1 + q2(k) on the kink frame, exact means",
        R2_eff_best=0.4166,
        raw_final_layer_mse=3.7157e-06, compute_ratio=0.1,
        b=0.0, v_eff_r2=0.4166, c=2.79e6,
        v_eff_gain_over_ship=1.0871,
        n_ship=27_000, argmax_k=32, p_at_argmax=1296,
        R2_eff_ship=0.3658, R2_pop_ship=0.3848,
        R2_eff_h1_alone=0.3768, h1_gain=1.0177, h1_features=256,
        q2_pop_by_k_kink={"8": 0.0564, "16": 0.0936, "24": 0.1258,
                          "32": 0.1567, "48": 0.2051},
        q2_pop_by_k_unadapted={"8": 0.0191, "16": 0.0368, "24": 0.0595,
                               "32": 0.0782, "48": 0.1210},
        combo_gain_by_k_kink={"8": 1.0397, "16": 1.0625, "24": 1.0774,
                              "32": 1.0871, "48": 1.0804},
        combo_gain_by_k_unadapted={"8": 1.0037, "16": 1.0031, "24": 1.0050,
                                   "32": 0.9992, "48": 0.9802},
        q1_pop_by_k_kink={"8": 0.0283, "16": 0.0484, "32": 0.0823,
                          "48": 0.1088},
        rank_bound_5p2_points={"8": 3.56, "16": 6.20, "32": 10.23, "48": 12.8},
        kink_frame_fraction_of_rank_bound={"8": 0.79, "16": 0.78, "32": 0.80,
                                           "48": 0.85},
        kink_weight_share_layers_21_to_32=0.87,
        kink_weight_share_layers_28_to_32=0.53,
        per_sample_flops=1.9e4, per_sample_share_of_c=0.0065,
        frame_setup_dispatches=60,
        notes=(
            "FAILS the 60% bar at 41.66%, and PASSES as an increment: 1.0871x on "
            "v_eff over the shipped basis with an EXACTLY known mean, at 0.65% "
            "of c per sample.  q2 is worth roughly twice q1 at every k (15.7% "
            "against 8.2% at k=32) precisely because relu(z_i)relu(z_j) is not a "
            "polynomial -- q1 is capped by f_2 = 19% and q2 is not.  h1 = "
            "relu(z^1) alone matches the shipped 512-feature t+He_2 basis with "
            "256 features and an exact mean (1.0177x), because relu mixes "
            "degrees 1 and 2 in 73.4/23.4 which is close to what this target "
            "wants; adding it ON TOP of the pair is worth 0.9997x, so it is a "
            "replacement not an addition.  THE CAVEAT IS THE FRAME: with the "
            "unadapted frame (first k coordinates of z^2, zero setup) the same "
            "block gives 1.004x instead of 1.087x, and the kink weight is "
            "concentrated at the deep end (87% in layers 21-32) so Q needs the "
            "full forward normal recursion AND the backward Jacobian sweep -- "
            "~60 flopscope dispatches, the same object sec 9.4 of "
            "hermite_rank_ceiling priced at 5.5% of the free budget and which "
            "turned cva's 1.032x raw into 0.945x adjusted.  Net is positive but "
            "Strassen-sized.  SIDE RESULT: the kink frame reaches 78-85% of sec "
            "5.2's every-subspace-at-once rank bound at every m, against the "
            "mean-field frame's 69% at m=16 -- the only figure this round "
            "improves on a published one."),
    ).__str__()
    print("recorded: exactly_integrable_quadratic_in_h1")
