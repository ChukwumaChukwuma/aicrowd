#!/usr/bin/env python
"""Round 15: scaling the offline-trained corrector, and where it stops.

Every bar was fixed before its run and is recorded verbatim.  The headline bar
is the parent brief's: beat the current ship's GRADED 2.4646e-07.

Numbers are from ``scripts/45_big_corrector.py`` on 475 locally generated MLPs
(seeds 400000+) x 4 estimator seeds at the DEPLOYED operating point
``(tau, N, P) = (2.5, 25000, 225)``, split by MLP seed 60/20/20.  Every quoted
figure is the held-out TEST split; the adjusted scores are PROJECTIONS through
the score model in ``scripts/45`` (validated: it returns 2.470e-07 at
N* = 24,160 against the graded 2.4646e-07 at N = 25,000).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment  # noqa: E402

ONLY = set(sys.argv[1:])


def want(name: str) -> bool:
    return not ONLY or name in ONLY


LOC = dict(gt_samples=2 * 32_768, seed=400_000, n_mlps=475)
SHIP_ADJ = 2.4646e-07          # the GRADED score of the current submission
SCRIPT = "scripts/45_big_corrector.py"

# ---------------------------------------------------------------------------
if want("bigcorr_scaled_head"):
    Experiment(
        name="bigcorr_scaled_head",
        script=SCRIPT,
        hypothesis=(
            "The shipped corrector is a 15-float ridge head fitted on 640 MLPs "
            "at (tau, N, P) = (2.5, 8500, 150) and DEPLOYED at (2.5, 25000, "
            "225).  Regenerating at the deployed point, adding every channel "
            "whose population mean is exactly known (the layer-1 Hermite "
            "family, the exactly-integrable relu1 block, and a NEW two-channel "
            "first-order transport of the layer-1 mean and variance gaps "
            "anchored on the exact Cov(z^2)), and fitting a linear head over "
            "them should beat the shipped head on freshly generated held-out "
            "networks by enough to clear the graded 2.4646e-07."),
        acceptance_bar=SHIP_ADJ, bar_metric="adjusted_final_layer_score",
        bar_direction="lower_is_better",
    ).record(
        **LOC,
        estimator=("sparse MC + relu1/mfv2/mfm/dpilot/mfv2g/cv1 channels x "
                   "{1, Phi, alpha} + 14 shape columns, 32-float ridge head"),
        adjusted_final_layer_score=1.9707e-07,   # projected, score model of s.5
        raw_final_layer_mse=1.0857e-06,          # held-out TEST unbiased MSE
        compute_ratio=0.2287,                    # C/B at N* = 21,754
        n_test_mlps=95, n_val_mlps=95, n_train_mlps=285,
        n_seeds_per_mlp=4, n_train_neurons=291_840,
        n_params=32, artifact_bytes=5920,
        test_umse_uncorrected=2.0476e-06,
        test_umse_shipped_head=1.3962e-06,
        test_umse_refit15=1.3306e-06,
        test_umse_rich83=1.0949e-06,
        x_over_shipped_head=1.286,
        x_over_uncorrected=1.886,
        new_channel_flops=2.35e8,
        new_channel_share_of_B=0.00086,
        is_projection=True,
        note=("adjusted is a PROJECTION: the MEASURED quantity is the 1.286x "
              "ratio against the shipped head on 95 held-out networks in the "
              "same run.  The score model reproduces the graded 2.4646e-07 to "
              "0.2% and the graded argmin N to 3%."),
    )

if want("bigcorr_capacity_is_worthless"):
    Experiment(
        name="bigcorr_capacity_is_worthless",
        script=SCRIPT,
        hypothesis=(
            "The parent brief's premise: 15 floats is a toy and the lever is "
            "the SIZE of the offline model -- 13M parameters fit in the 50 MiB "
            "cap and cost 0.01% of the budget to evaluate.  Test it properly: "
            "random tanh features NESTED in the rich linear design (same "
            "columns as the linear block, penalty in the scaled basis, "
            "unweighted loss matching the metric), swept over 256 to 4096 "
            "features, plus a per-neuron numpy MLP boosted on the ridge "
            "residual.  Bar: the best nonlinear head beats the best linear one "
            "on held-out TEST by >= 1.05x."),
        acceptance_bar=1.05, bar_metric="best_nonlinear_over_linear_test",
        bar_direction="higher_is_better",
    ).record(
        **LOC,
        estimator="RFRidge(256|1024|4096) and SGDHead(64 | 256-128), nested",
        best_nonlinear_over_linear_test=0.974,
        raw_final_layer_mse=1.1246e-06,          # the best of the six
        adjusted_final_layer_score=2.0995e-07,   # projected
        compute_ratio=0.2287, seed_note="rf_seed 0",
        test_x_over_raw={"linear_83": 1.870, "rf_256": 1.821, "rf_1024": 1.792,
                         "rf_4096": 1.787, "mlp_64": 1.240,
                         "mlp_256_128": 1.229},
        n_params={"linear_83": 83, "rf_256": 8305, "rf_1024": 32881,
                  "rf_4096": 131185, "mlp_64": 7361, "mlp_256_128": 62209},
        selected_penalty={"rf_256": 3.0, "rf_1024": 10.0, "rf_4096": 30.0},
        note=("monotonically WORSE with capacity: validation raises the "
              "penalty until every added feature is shrunk to zero and only "
              "its degrees of freedom remain.  Three implementation details "
              "had to be right before this was a fair test -- nesting, "
              "column-scaled penalty, unweighted loss -- and getting any of "
              "them wrong moved the number by up to 0.5x."),
    )

if want("bigcorr_per_mlp_coefficients"):
    Experiment(
        name="bigcorr_per_mlp_coefficients",
        script=SCRIPT,
        hypothesis=(
            "A head with MLP-specific coefficients bounds every pooled head on "
            "the same columns, so measuring it prices ALL remaining capacity "
            "at once.  Fit per test MLP on reference half a, score against "
            "half b, subtract Var(b)(1 + p/n) -- the reference noise plus the "
            "exact OLS estimation-variance term -- so there is no in-sample "
            "credit.  Bar: per-MLP coefficients beat the pooled vector on the "
            "identical columns by >= 1.05x."),
        acceptance_bar=1.05, bar_metric="per_mlp_over_pooled_wellconditioned",
        bar_direction="higher_is_better",
    ).record(
        **LOC,
        estimator="per-MLP OLS on the same design, honest split scoring",
        per_mlp_over_pooled_wellconditioned=0.937,
        raw_final_layer_mse=1.1629e-06,
        adjusted_final_layer_score=2.1128e-07,   # projected
        compute_ratio=0.2287,
        pooled_x={"ch_14": 1.876, "ch_x_mod_40": 1.878, "plus_shape_57": 1.863,
                  "plus_pooled_83": 1.870},
        per_mlp_x={"ch_14": 1.566, "ch_x_mod_40": 1.761,
                   "plus_shape_57": 1.912, "plus_pooled_83": 2.189},
        p_over_n={"ch_14": 0.055, "ch_x_mod_40": 0.156, "plus_shape_57": 0.223,
                  "plus_pooled_83": 0.324},
        note=("on both well-conditioned designs (p/n <= 0.16) knowing each "
              "network's own optimal coefficients is WORSE than one shared "
              "vector: 0.835x and 0.937x.  The two larger designs subtract a "
              "correction 1.3x the residual being measured and are reported "
              "but not relied on."),
    )

if want("bigcorr_learning_curve"):
    Experiment(
        name="bigcorr_learning_curve",
        script=SCRIPT,
        hypothesis=(
            "The deliverable: held-out gain as a function of training-set "
            "size.  If it is still improving at the data budget, more data is "
            "the answer rather than more parameters.  Bar: the 83-column "
            "design gains >= 1.05x going from 120 to 240 training MLPs, i.e. "
            "the curve is still live at the budget."),
        acceptance_bar=1.05, bar_metric="gain_120_to_240_mlps",
        bar_direction="higher_is_better",
    ).record(
        **LOC,
        estimator="rich 83-column linear head, held-out TEST",
        gain_120_to_240_mlps=1.008,
        raw_final_layer_mse=1.0693e-06,          # 2.0476e-06 / 1.915
        adjusted_final_layer_score=1.9640e-07,   # projected
        compute_ratio=0.2287,
        curve_ridge15={"8": 1.543, "15": 1.535, "30": 1.538, "60": 1.536,
                       "120": 1.569, "240": 1.558},
        curve_rich83={"8": 1.736, "15": 1.712, "30": 1.750, "60": 1.847,
                      "120": 1.900, "240": 1.915},
        curve_rf256={"8": 1.652, "15": 1.680, "30": 1.698, "60": 1.825,
                     "120": 1.862, "240": 1.855},
        curve_rf4096={"8": 1.616, "15": 1.634, "30": 1.650, "60": 1.788,
                      "120": 1.834, "240": 1.838},
        note=("FAIL is the result: the shipped 15-column design is saturated "
              "at EIGHT training MLPs and the 83-column one by 120.  The "
              "binding constraint is the feature set, not the data."),
    )

if want("bigcorr_label_precision"):
    Experiment(
        name="bigcorr_label_precision",
        script=SCRIPT,
        hypothesis=(
            "Reference compute is the binding constraint, so the MLPs-versus-"
            "precision trade has to be priced.  It is priced EXACTLY rather "
            "than by generating a second dataset: reference error is zero-mean "
            "noise of known per-neuron variance vh_j/(n_gt r), so a reference "
            "of n_gt/f is simulated by adding independent noise of variance "
            "(f-1) vh_j/(n_gt r) to each TRAINING half.  Bar: cutting the "
            "reference 16-fold costs >= 5% of held-out gain, i.e. precision "
            "is worth paying for."),
        acceptance_bar=1.05, bar_metric="gain_ratio_full_over_sixteenth",
        bar_direction="higher_is_better",
    ).record(
        **LOC,
        estimator="rich 83-column linear head, training labels perturbed",
        gain_ratio_full_over_sixteenth=1.008,
        raw_final_layer_mse=1.1062e-06,          # 2.0476e-06 / 1.851
        adjusted_final_layer_score=2.0800e-07,   # projected
        compute_ratio=0.2287,
        ref_cv_variance_gain={"16384": 1.139, "32768": 1.333, "65536": 1.585},
        gain_by_inflation_240_mlps={"1": 1.851, "2": 1.851, "4": 1.861,
                                    "8": 1.856, "16": 1.837},
        note=("FAIL, decisively: at n_gt/16 the label noise is 3x the signal "
              "it labels and the held-out gain is unchanged to 1%.  Reference "
              "compute should buy MLPs, not precision -- and since the MLP "
              "axis saturates at 120 too, neither is the constraint."),
    )

print("recorded")
