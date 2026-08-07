#!/usr/bin/env python
"""Round 7: the offline-trained residual corrector.

Three pre-registered bars, and they did not all pass.

1. **Ridge corrector vs the shipped estimator, bar 1.25x** on the official
   100-MLP suite, fitted only on generated data.
2. **MLP head vs ridge, bar 1.5x** -- the heavier artifact only ships if it
   earns its FLOPs.
3. The **premise** the whole thing rests on, recorded separately because it is
   the number that decided the design: the population share of
   ``Var(relu(z^32_j))`` explained by the layer-1 Hermite control-variate
   family at each order, which says ``k <= 2`` and not more.

Every figure here is printed by ``scripts/28_learned_corrector.py``.  Numbers
are filled in from the recorded runs in ``docs/learned_corrector.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "ledger" / "round7.json"
R = json.loads(RESULTS.read_text())

OFF = dict(seed=0, n_mlps=100, gt_samples=1_000_000_000)
SCRIPT = "scripts/28_learned_corrector.py"


# ---------------------------------------------------------------------------
Experiment(
    name="hermite_cv_anova_premise",
    script=SCRIPT,
    hypothesis="The layer-1 Hermite control-variate family -- exactly "
               "mean-zero by Gaussianity of z^1 and with an analytic Mehler "
               "Gram -- explains more than the 27.6% first-order ANOVA share "
               "that bounds every low-order surrogate at 1.38x, and the "
               "k<=2 truncation is where its gain per unit of estimation "
               "noise peaks.",
    acceptance_bar=0.276,
    bar_metric="population_share_explained_kmax2",
    bar_direction="higher_is_better",
).record(
    seed=11, n_mlps=4, gt_samples=400_000,
    estimator="layer-1 Hermite CV, population projection (diagnostic)",
    compute_ratio=0.0,
    raw_final_layer_mse=R["anova"]["raw_final_layer_mse_ref"],
    population_share_explained_k1=R["anova"]["k1"],
    population_share_explained_kmax2=R["anova"]["k2"],
    population_share_explained_kmax3=R["anova"]["k3"],
    population_share_explained_kmax6=R["anova"]["k6"],
    per_mlp_k1=R["anova"]["per_mlp_k1"],
    per_mlp_k2=R["anova"]["per_mlp_k2"],
    notes=R["anova"]["notes"],
)

# ---------------------------------------------------------------------------
Experiment(
    name="offline_ridge_corrector",
    script=SCRIPT,
    hypothesis="A ridge head on predict-time features, fitted only on 640 "
               "generated MLPs with fresh seeds, beats the shipped sparse "
               "Monte-Carlo estimator by at least 1.25x on the official "
               "100-MLP suite with 0 raises.",
    acceptance_bar=R["ridge"]["bar_adjusted"],
    bar_metric="adjusted_final_layer_score",
    bar_direction="lower_is_better",
).record(
    **OFF,
    estimator=R["ridge"]["estimator"],
    adjusted_final_layer_score=R["ridge"]["adj1"],
    adjusted_at_2x_residual=R["ridge"]["adj2"],
    adjusted_at_3x_residual=R["ridge"]["adj3"],
    raw_final_layer_mse=R["ridge"]["raw"],
    compute_ratio=R["ridge"]["cb"],
    flop_ratio=R["ridge"]["fb"],
    ablation_damp0_raw=R["ridge"]["abl_raw"],
    ablation_damp0_adjusted=R["ridge"]["abl_adj"],
    improvement_over_prev_ship=R["ridge"]["gain"],
    n_raises=R["ridge"]["raises"],
    worst_single_mlp_mse=R["ridge"]["worst"],
    n_train_mlps=R["ridge"]["n_train"],
    n_val_mlps=R["ridge"]["n_val"],
    n_test_mlps=R["ridge"]["n_test"],
    gt_samples_per_training_mlp=200_000,
    validation_gain=R["ridge"]["val_gain"],
    test_split_gain=R["ridge"]["test_gain"],
    feature_block_flops=R["ridge"]["feature_flops"],
    feature_block_residual_ms=R["ridge"]["feature_residual_ms"],
    notes=R["ridge"]["notes"],
)

# ---------------------------------------------------------------------------
Experiment(
    name="offline_mlp_head",
    script=SCRIPT,
    hypothesis="A small tanh MLP head over the same features beats the ridge "
               "head by at least 1.5x, which is what a heavier artifact and "
               "a nonlinear predict-time head would have to earn.",
    acceptance_bar=1.5,
    bar_metric="mlp_gain_over_ridge",
    bar_direction="higher_is_better",
).record(
    seed=0, n_mlps=R["mlp"]["n_test"], gt_samples=200_000,
    estimator=R["mlp"]["estimator"],
    compute_ratio=0.0,
    raw_final_layer_mse=R["mlp"]["test_mse"],
    mlp_gain_over_ridge=R["mlp"]["gain_over_ridge"],
    ridge_test_gain=R["mlp"]["ridge_test_gain"],
    mlp_test_gain=R["mlp"]["mlp_test_gain"],
    hidden_units=R["mlp"]["hidden"],
    notes=R["mlp"]["notes"],
)

print("recorded 3 rows")
