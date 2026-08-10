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

# ---------------------------------------------------------------------------
# ROUND 15b: the head refitted for a randomised lattice scored draw.
# ---------------------------------------------------------------------------
LAT = dict(gt_samples=2 * 32_768, seed=400_000, n_mlps=375)

if want("bigcorr_lattice_refit"):
    Experiment(
        name="bigcorr_lattice_refit",
        script=SCRIPT,
        hypothesis=(
            "A sibling measured a randomised rank-1 lattice at 1.427x "
            "projected graded, and the head at 0.922x UNDER that lattice -- net "
            "negative.  The mechanism is not a tuning artefact: the k=1 Hermite "
            "block is provably the optimal input-linear control variate and a "
            "rank-1 lattice annihilates exactly those first-order ANOVA terms, "
            "so the shipped coefficients are fitted against a residual whose "
            "first-order part the lattice has already removed.  That makes the "
            "head MIS-FITTED rather than intrinsically redundant.  Refit it on "
            "lattice draws at (tau, N, P) = (2.5, 24989, 225) and score at "
            "deployable weights on unbiased true MSE."),
        acceptance_bar=1.10, bar_metric="test_x_over_damp0_under_lattice",
        bar_direction="higher_is_better",
    ).record(
        **LAT,
        estimator=("sparse MC on a shifted rank-1 lattice + mfv2g/cv1mfg/"
                   "dpilot/cv1/mfv2 x {1, Phi, alpha} + 5 shape, 20-float head"),
        test_x_over_damp0_under_lattice=1.1324,
        raw_final_layer_mse=1.0909e-06,     # held-out TEST unbiased MSE
        compute_ratio=0.2799,               # F/B of the lattice pass, sibling
        adjusted_final_layer_score=None,
        n_test_mlps=75, n_val_mlps=75, n_seeds_per_mlp=4, n_params=20,
        artifact_bytes=342, n_samples=24989, n_pilot=225, tau=2.5,
        ci95_low=1.0688, ci95_high=1.1978,
        test_umse_damp0=1.2354e-06,
        test_umse_iid_fitted_head=1.5858e-06,
        x_over_iid_fitted_head=1.454,
        iid_head_under_lattice=0.921,       # reproduces the sibling's 0.922
        channel_unit_gain_iid={"cv1": 1.321, "relu1": 1.556, "cv2": 1.037,
                               "mfm": 1.529, "mfv": 1.100, "mfv2": 1.146},
        channel_unit_gain_lattice={"cv1": 0.374, "relu1": 0.541, "cv2": 0.821,
                                   "mfm": 0.989, "mfv": 1.091, "mfv2": 1.055},
        selected_channels=["mfv2g", "cv1mfg", "dpilot", "cv1", "mfv2"],
        note=("PASS on the point estimate; the 95% bootstrap interval over MLPs "
              "is [1.0688, 1.1978] and so does NOT exclude a miss.  45 -> 75 "
              "test MLPs did not tighten it ([1.0703, 1.1835] -> "
              "[1.0688, 1.1978]), because the variance is MLP heterogeneity "
              "rather than count.  --interleave, which should recover cv1/cv2/"
              "relu1 from actively harmful, is implemented and UNMEASURED."),
    )

# ---------------------------------------------------------------------------
# ROUND 15c: the same refit on the COMPLETE lattice set.  Records are immutable
# and append-only, so this is a new row rather than an edit of the 375-MLP one
# above -- and the delta between them is itself the result (my claim that
# another 75 networks "would buy little" was wrong).
# ---------------------------------------------------------------------------
if want("bigcorr_lattice_refit_full"):
    Experiment(
        name="bigcorr_lattice_refit_full",
        script=SCRIPT,
        hypothesis=(
            "bigcorr_lattice_refit read 1.1324x with a 95% bootstrap CI of "
            "[1.0688, 1.1978] on 375 of the 475 generated MLPs, and I claimed "
            "the interval had stopped tightening because 45 -> 75 test MLPs had "
            "not moved it.  Re-run on the complete set: if that claim was right "
            "the point estimate and the interval both stay put, and if it was "
            "wrong they move.  Same pre-registered bar, not re-rolled: beta "
            "refitted on lattice draws must beat damp=0 under a lattice by "
            ">= 1.10x on held-out TEST, paired on the same MLPs."),
        acceptance_bar=1.10, bar_metric="test_x_over_damp0_under_lattice",
        bar_direction="higher_is_better",
    ).record(
        gt_samples=2 * 32_768, seed=400_000, n_mlps=475,
        estimator=("sparse MC on a shifted rank-1 lattice + mfv2/cv1mf/cv1/cv2/"
                   "mfv/dpilot/mfm/mfvg x {1, Phi, alpha} + 5 shape, 29 floats"),
        test_x_over_damp0_under_lattice=1.1762,
        raw_final_layer_mse=9.4400e-07,      # held-out TEST unbiased MSE
        compute_ratio=0.2799,                # F/B of the lattice pass, sibling
        adjusted_final_layer_score=None,
        n_test_mlps=95, n_val_mlps=95, n_seeds_per_mlp=4, n_params=29,
        artifact_bytes=378, n_samples=24989, n_pilot=225, tau=2.5,
        ci95_low=1.1126, ci95_high=1.2421,
        test_umse_damp0=1.1103e-06,
        test_umse_iid_fitted_head=1.4316e-06,
        x_over_iid_fitted_head=1.517,
        selected_channels=["mfv2", "cv1mf", "cv1", "cv2", "mfv", "dpilot",
                           "mfm", "mfvg"],
        n_channels_by_train_size={"75": 5, "225": 6, "375": 5, "475": 8},
        superseded_by_grader=True,
        note=("PASS and the interval now EXCLUDES a miss (lower bound 1.1126). "
              "375 -> 475 MLPs moved the point estimate 1.1324 -> 1.1762 and "
              "tightened the CI from [1.0688, 1.1978] to [1.1126, 1.2421], so "
              "the earlier 'another 75 networks would buy little' was wrong -- "
              "the 45 -> 75 plateau was noise in the interval estimate, not a "
              "variance floor.  The GAIN is stable; the selected channel SET is "
              "not (5/6/5/8 channels at 75/225/375/475 MLPs, never the same "
              "five) because the channels are collinear -- read it as a span, "
              "not a ranking.  mfv2 leads at every size; relu1 is never "
              "selected.  NOTE: 01b2a80 measured the lattice at 0.932x on the "
              "grader, so this head is fitted for a sampler that is not "
              "currently shipping; heads/iid_head.npz is the live candidate."),
    )
