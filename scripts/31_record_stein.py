#!/usr/bin/env python
"""Round 8: Stein control variates from the network's own gradient.

Three pre-registered bars.  The headline mechanism failed, and the failure is
the result: the construction is exactly unbiased (bar 1) and genuinely escapes
the low-order barrier, and it is still worth nothing (bar 2), for a reason that
is a property of the Hermite raising operator rather than of this suite.  What
shipped out of the round is the cost saving that was sitting in the feature
block (bar 3).

Every figure here is printed by ``scripts/30_stein_cv.py`` (bars 1 and 2) or
``scripts/28_learned_corrector.py --mode fit / --mode score / --mode ship``
(bar 3).  Numbers are filled in from ``ledger/round8.json`` so the ledger row
and ``docs/stein_cv.md`` quote the same bytes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "ledger" / "round8.json"
R = json.loads(RESULTS.read_text())

SCRIPT30 = "scripts/30_stein_cv.py"
SCRIPT28 = "scripts/28_learned_corrector.py"


# ---------------------------------------------------------------------------
Experiment(
    name="stein_identity_from_the_network_gradient",
    script=SCRIPT30,
    hypothesis="For x ~ N(0, I) and psi taken from the network's own "
               "internals, h(x) = c.grad psi(x) - (c.x) psi(x) has E[h] = 0 "
               "EXACTLY -- no Gaussianity is needed anywhere inside the "
               "network, only at the input -- so a control variate built "
               "from a forward-mode tangent pass is unbiased by construction "
               "and does not violate the floor argument's premise.",
    acceptance_bar=2.0,
    bar_metric="worst_rms_z_score",
    bar_direction="lower_is_better",
).record(
    seed=4242, n_mlps=R["verify"]["n_mlps"],
    gt_samples=R["verify"]["n_samples"],
    estimator="Stein CV h = c.grad psi - (c.x) psi, forward-mode tangent "
              "pass (diagnostic)",
    compute_ratio=0.0,
    raw_final_layer_mse=R["verify"]["rms_mean_h_relu"],
    worst_rms_z_score=R["verify"]["worst_rms_z"],
    rms_mean_h_relu=R["verify"]["rms_mean_h_relu"],
    rms_mc_stderr_relu=R["verify"]["rms_se_relu"],
    ratio_relu=R["verify"]["ratio_relu"],
    max_abs_z=R["verify"]["max_abs_z"],
    n_tests_per_family=R["verify"]["n_tests"],
    psi_families=R["verify"]["families"],
    notes=R["verify"]["notes"],
)

# ---------------------------------------------------------------------------
Experiment(
    name="stein_cv_span_r2",
    script=SCRIPT30,
    hypothesis="A dictionary of Stein control variates driven by the "
               "network's own gradient -- which is as high-order as the "
               "network itself and therefore not bounded by the low-order "
               "barrier -- spans more than R^2 = 0.75 of Var(relu(z^32_j)), "
               "i.e. more than 4x, materially above the k<=2 ceiling of "
               "1.75x, which is what justifies the 2x per-sample cost of the "
               "extra gradient pass.",
    acceptance_bar=0.75,
    bar_metric="stein_span_r2",
    bar_direction="higher_is_better",
).record(
    seed=12345, n_mlps=R["span"]["n_mlps"],
    gt_samples=R["span"]["n_samples"],
    estimator="Stein CV span, held-out projection over 2048 features "
              "(4 psi families x 2 directions x 256 source neurons)",
    compute_ratio=0.0,
    raw_final_layer_mse=R["span"]["baseline_mse_ref"],
    stein_span_r2=R["span"]["stein_alone"],
    hermite_k2_span_r2=R["span"]["hermite_k2"],
    hermite_k1_span_r2=R["span"]["hermite_k1"],
    joint_span_r2=R["span"]["joint"],
    stein_increment_points=R["span"]["increment_points"],
    exact_ceiling_all_256_directions=R["span"]["ceiling"],
    exact_ceiling_per_mlp=R["span"]["ceiling_per_mlp"],
    per_block_r2=R["span"]["per_block"],
    notes=R["span"]["notes"],
)

# ---------------------------------------------------------------------------
Experiment(
    name="lean_corrector_head",
    script=SCRIPT28,
    hypothesis="Dropping the thirteen feature columns that measure at "
               "exactly 1.000x on validation -- the Rao-Blackwell/Edgeworth "
               "group, the multiplicative shrink, the weight/suite scalars, "
               "cv3 and sd_mc -- removes eight passes over the (N, width) "
               "sample array and therefore beats the shipped estimator's "
               "adjusted score, at no cost in accuracy on the untouched test "
               "split.",
    acceptance_bar=R["lean"]["bar_adjusted"],
    bar_metric="adjusted_final_layer_score",
    bar_direction="lower_is_better",
).record(
    seed=0, n_mlps=100, gt_samples=1_000_000_000,
    estimator=R["lean"]["estimator"],
    adjusted_final_layer_score=R["lean"]["adj1"],
    adjusted_at_2x_residual=R["lean"]["adj2"],
    adjusted_at_3x_residual=R["lean"]["adj3"],
    raw_final_layer_mse=R["lean"]["raw"],
    compute_ratio=R["lean"]["cb"],
    flop_ratio=R["lean"]["fb"],
    same_run_prev_ship_adjusted=R["lean"]["prev_adj1"],
    same_run_prev_ship_raw=R["lean"]["prev_raw"],
    same_run_prev_ship_cb=R["lean"]["prev_cb"],
    improvement_over_prev_ship=R["lean"]["gain"],
    improvement_at_2x_residual=R["lean"]["gain2"],
    improvement_at_3x_residual=R["lean"]["gain3"],
    ablation_damp0_raw=R["lean"]["abl_raw"],
    ablation_damp0_adjusted=R["lean"]["abl_adj"],
    n_raises=R["lean"]["raises"],
    worst_single_mlp_mse=R["lean"]["worst"],
    n_features=15, n_features_before=28,
    validation_gain=R["lean"]["val_gain"],
    test_split_gain=R["lean"]["test_gain"],
    test_split_gain_full_design=R["lean"]["test_gain_full"],
    notes=R["lean"]["notes"],
)

print("recorded 3 rows")
