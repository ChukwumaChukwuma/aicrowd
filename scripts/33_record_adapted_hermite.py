#!/usr/bin/env python
"""Round 9: high-degree Hermite CVs on network-adapted directions.

Four pre-registered bars.  The headline mechanism failed and the failure is
the result again, but for a *different* reason than round 8's: the blocker is
neither the ANOVA barrier nor ``p/N``, it is **rank**.  ``h_d(<a,x>)`` is
exactly a unit rank-one tensor of the degree-``d`` chaos, and past degree 2
the network's chaos content -- 43% of the variance, mean Hermite degree 10.5 --
has no rank-one component for any direction to see.

Every figure here is printed by ``scripts/32_adapted_hermite.py``
(``--mode span | chaos | ceiling``) or ``scripts/28_learned_corrector.py``
(``--mode score | --mode ship``).  Numbers are filled in from
``ledger/round9.json`` so the ledger rows and ``docs/hermite_rank_ceiling.md``
quote the same bytes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "ledger" / "round9.json"
R = json.loads(RESULTS.read_text())

SCRIPT32 = "scripts/32_adapted_hermite.py"
SCRIPT28 = "scripts/28_learned_corrector.py"


# ---------------------------------------------------------------------------
S = R["span"]
Experiment(
    name="adapted_hermite_span_r2",
    script=SCRIPT32,
    hypothesis="The k<=2 instance of the low-order barrier does not bound "
               "degree-3+ Hermite control variates, because He_d of a "
               "layer-1 pre-activation is a degree-d object.  What killed "
               "k=3 in the ship was p/N, not approximation power, so a "
               "SMALLER, better-chosen basis -- high degree along 10-50 "
               "directions taken from the network itself rather than degree "
               "2 along an arbitrary 256 -- spans more than R^2 = 0.75 "
               "(> 4x) at p/N < 0.15.",
    acceptance_bar=0.75,
    bar_metric="best_span_r2_under_p_budget",
    bar_direction="higher_is_better",
).record(
    seed=12345, n_mlps=S["n_mlps"], gt_samples=S["n_samples"],
    estimator="held-out span over 2,494 Hermite features with an ANALYTIC "
              "(Mehler) Gram: layer-1 coordinate pure powers to degree 8, a "
              "24-direction mean-field orthonormal frame with all products "
              "to total degree 4 and pure powers to degree 16, all cross "
              "terms",
    compute_ratio=0.0,
    raw_final_layer_mse=S["baseline_mse_ref"],
    best_span_r2_under_p_budget=S["coord_d2_shipped"],
    p_budget=1275,
    span_r2_all_2494_features=S["everything"],
    span_r2_shipped_k2=S["coord_d2_shipped"],
    span_r2_coord_d1=S["coord_d1"],
    span_r2_coord_d3=S["coord_d3"],
    span_r2_coord_d8=S["coord_d8"],
    span_r2_per_degree_alone=S["coord_deg_alone"],
    span_r2_adapt_m1_deg16=S["adapt_pure_m1_d16"],
    span_r2_adapt_m24_deg1=S["adapt_pure_m24_d1"],
    span_r2_adapt_m24_deg16=S["adapt_pure_m24_d16"],
    span_r2_adapt_tensor_m16_deg2=S["adapt_tensor_m16_deg2"],
    span_r2_adapt_all=S["adapt_all"],
    span_r2_per_neuron_meanfield=S["per_neuron_mf_d1"],
    span_r2_per_neuron_jacobian=S["per_neuron_jac_d1"],
    best_effective_dictionary=S["best_effective_dictionary"],
    best_effective_r2=S["best_effective_r2"],
    adapted_frame=S["frame"],
    n_features_total=S["p_total"],
    official_confirmation=R["official_confirmation"],
    notes=S["notes"],
)

# ---------------------------------------------------------------------------
C = R["chaos"]
Experiment(
    name="hermite_degree_spectrum_of_relu_z32",
    script=SCRIPT32,
    hypothesis="The Hermite DEGREE spectrum of relu(z^32) -- distinct from "
               "the coordinate-subset ANOVA spectrum of "
               "docs/floor_theorem.md -- is measurable exactly by the "
               "Ornstein-Uhlenbeck semigroup, at one forward pass per "
               "coupling and with no curse of dimensionality, and it "
               "reproduces the independently measured f_1 = 0.276 +- 0.016.",
    acceptance_bar=0.05,
    bar_metric="f1_absolute_error_against_prior_measurement",
    bar_direction="lower_is_better",
).record(
    seed=4242, n_mlps=C["n_mlps"], gt_samples=C["n_samples"],
    estimator="OU/Mehler coupling C(t) = sum_j Cov(y_j(x), y_j(x_t)) / "
              "sum_j Var(y_j) = sum_d f_d t^d, inverted by NNLS (diagnostic)",
    compute_ratio=0.0,
    raw_final_layer_mse=R["span"]["baseline_mse_ref"],
    f1_absolute_error_against_prior_measurement=abs(C["f_d"][0] - 0.276),
    f1_prior_measurement=0.276,
    coupling_grid=C["t_grid"],
    C_of_t=C["C_of_t"],
    f_d=C["f_d"],
    f1_per_mlp=C["f1_per_mlp"],
    cumulative_degree_le_2=C["cum_d2"],
    cumulative_degree_le_4=C["cum_d4"],
    cumulative_degree_le_8=C["cum_d8"],
    mean_hermite_degree=C["mean_hermite_degree"],
    assumption_free_upper_degree_le_1=C["hard_upper_d1"],
    assumption_free_upper_degree_le_2=C["hard_upper_d2"],
    assumption_free_upper_degree_le_3=C["hard_upper_d3"],
    mass_strictly_above_degree_2=1.0 - C["cum_d2"],
    notes=C["notes"],
)

# ---------------------------------------------------------------------------
K = R["ceiling"]
Experiment(
    name="hermite_family_rank_ceiling",
    script=SCRIPT32,
    hypothesis="Because h_d(<a,x>) is exactly the unit RANK-ONE tensor "
               "a^(x)d of the degree-d chaos, the closed span of any Hermite "
               "dictionary on m directions is L^2(sigma(A^T x)), which "
               "bounds R^2 at every degree at once; and the degree-2 "
               "reachability spectrum of P = sum_j S_j^2 bounds what any m "
               "directions can reach.  The two together must reproduce the "
               "measured span surface to within the bound's one-sidedness.",
    acceptance_bar=0.02,
    bar_metric="degree_free_ceiling_minus_hermite_truncation",
    bar_direction="lower_is_better",
).record(
    seed=12345, n_mlps=K["n_mlps"], gt_samples=K["n_samples"],
    estimator="degree-free conditional-expectation ceiling (equiprobable "
              "binning, per-bin means from independent halves) + exact "
              "degree-2 rank spectrum of sum_j S_j^2 (cross-half pair "
              "kernel) (diagnostic)",
    compute_ratio=0.0,
    raw_final_layer_mse=R["span"]["baseline_mse_ref"],
    degree_free_ceiling_minus_hermite_truncation=abs(
        K["cond_mf_m1"] - K["hermite_m1_d16"]),
    degree_free_ceiling_meanfield_m1=K["cond_mf_m1"],
    degree_free_ceiling_meanfield_m3=K["cond_mf_m3"],
    degree_free_ceiling_jacobian_m3=K["cond_jac_m3"],
    degree_free_ceiling_random_m3=K["cond_rand_m3"],
    hermite_m1_degree_le_16=K["hermite_m1_d16"],
    degree2_reachable_share_of_f2=K["deg2_reach_of_f2"],
    degree2_coord_basis_share_of_f2=K["deg2_coord_share_of_f2"],
    degree2_f2_pair_estimate=K["deg2_f2_pair_estimate"],
    any_p_features_bound_from_cov_y=K["covy_topp"],
    family_ceiling_r2=K["family_ceiling_r2"],
    family_ceiling_variance_reduction=K["family_ceiling_x"],
    shipped_realises_r2=K["shipped_realises_r2"],
    shipped_realises_variance_reduction=K["shipped_realises_x"],
    notes=K["notes"],
)

# ---------------------------------------------------------------------------
P = R["ship"]
Experiment(
    name="shipped_estimator_unchanged_round9",
    script=SCRIPT28,
    hypothesis="Round 9 changes no shipped byte -- the argmax of the "
               "R^2(p, d) surface over 71 dictionaries IS the shipped "
               "k <= 2 basis -- so the official-suite figures must reproduce "
               "docs/stein_cv.md sec 7 with 0 raises and a bitwise damp=0 "
               "ablation.",
    acceptance_bar=4.10e-07,
    bar_metric="adjusted_final_layer_score",
    bar_direction="lower_is_better",
).record(
    seed=0, n_mlps=100, gt_samples=1_000_000_000,
    estimator=P["estimator"],
    adjusted_final_layer_score=P["adj1"],
    adjusted_at_2x_residual=P["adj2"],
    adjusted_at_3x_residual=P["adj3"],
    raw_final_layer_mse=P["raw"],
    compute_ratio=P["cb"],
    flop_ratio=P["fb"],
    ablation_damp0_raw=P["abl_raw"],
    ablation_damp0_adjusted=P["abl_adj"],
    ship_mode_adjusted=P["ship_mode_adj1"],
    ship_mode_compute_ratio=P["ship_mode_cb"],
    ship_mode_max_compute_ratio=P["ship_mode_maxc"],
    dense_fallback_raw=P["fallback_raw"],
    dense_fallback_compute_ratio=P["fallback_cb"],
    n_raises=P["raises"],
    worst_single_mlp_mse=P["worst"],
    notes=P["notes"],
)

print("recorded 4 rows")

# ---------------------------------------------------------------------------
# Round 9b: cashing the follow-up sec 8 quantified.
# ---------------------------------------------------------------------------
V = R["cva"]
Experiment(
    name="adapted_degree1_cv_mechanism",
    script=SCRIPT28,
    hypothesis="Degrees are exactly orthogonal, so span R^2 is additive "
               "across degree blocks: replacing the 256-feature input-basis "
               "k=1 control variate with m ~ 24 directions spanning the "
               "mean-field Jacobian, and keeping the coordinate k=2 block, "
               "raises effective R^2 from 27.2% to 31.0% and must therefore "
               "cut the unbiased true MSE of the raw (unit-coefficient) "
               "control-variate arm by about 3%.",
    acceptance_bar=1.02,
    bar_metric="mechanism_gain_over_shipped_arm",
    bar_direction="higher_is_better",
).record(
    seed=0, n_mlps=V["n_train_mlps"], gt_samples=200_000,
    estimator="sparse MC - cva24 - cv2, coefficient fixed at 1, no head",
    compute_ratio=0.0,
    raw_final_layer_mse=V["unit_coef_cva24_cv2"],
    mechanism_gain_over_shipped_arm=V["mechanism_gain_over_shipped_arm"],
    unit_coef_sparse_mc=V["unit_coef_sparse_mc"],
    unit_coef_cv1=V["unit_coef_cv1"],
    unit_coef_cv1_cv2=V["unit_coef_cv1_cv2"],
    unit_coef_cv1mf=V["unit_coef_cv1mf"],
    unit_coef_cva24=V["unit_coef_cva24"],
    unit_coef_cva24_cv2=V["unit_coef_cva24_cv2"],
    m_grid=V["m_grid"], m_validation_gain=V["m_validation_gain"],
    m_selected=V["m_selected"],
    notes=V["notes"],
)

Experiment(
    name="adapted_degree1_cv_end_to_end",
    script=SCRIPT28,
    hypothesis="That 3% survives the offline head and the cost of extracting "
               "the directions, so the adapted degree-1 block beats the "
               "shipped adjusted score of 4.0142e-07 on the official suite.",
    acceptance_bar=4.0142e-07,
    bar_metric="adjusted_final_layer_score",
    bar_direction="lower_is_better",
).record(
    seed=0, n_mlps=100, gt_samples=1_000_000_000,
    estimator="sparse MC + layer-1 Hermite k<=2 + adapted degree-1 block "
              "(m=48) + 18-column offline ridge head",
    adjusted_final_layer_score=V["official_adj1"],
    adjusted_at_2x_residual=V["official_adj2"],
    adjusted_at_3x_residual=V["official_adj3"],
    raw_final_layer_mse=V["official_raw"],
    compute_ratio=V["official_cb"],
    flop_ratio=V["official_fb"],
    same_run_shipped_raw=V["ship_raw"],
    same_run_shipped_adjusted=V["ship_adj1"],
    same_run_shipped_compute_ratio=V["ship_cb"],
    raw_ratio_against_ship=V["raw_ratio"],
    adjusted_ratio_against_ship=V["adj_ratio"],
    ablation_damp0_raw=V["abl_raw"],
    n_raises=V["official_raises"],
    worst_single_mlp_mse=V["official_worst"],
    validation_gain=V["val_gain_18col"],
    validation_gain_previous_design=V["val_gain_15col"],
    test_split_gain=V["test_gain_18col"],
    test_split_gain_previous_design=V["test_gain_15col"],
    leave_out_cva_validation=V["loo_without_cva"],
    only_cva_validation=V["only_cva"],
    only_cv1_validation=V["only_cv1"],
    shipped_corrector_sha256=V["corrector_npz_sha256"],
    notes=V["notes"],
)

print("recorded 2 more rows (round 9b)")
