#!/usr/bin/env python
"""Record round-3: two *valid distributions* for S_j, and why neither helps.

The cumulant route died because a truncated cumulant vector does not determine
``E[relu]``.  Both models recorded here avoid moments entirely -- each is a
genuine distribution, so neither can diverge the way an Edgeworth series does.
Both are killed, for different and precisely identified reasons, and the third
record is the measurement that explains why the whole family was never going to
be worth much.

Numbers come from ``scripts/15_chaos2_bakeoff.py``; nothing is entered by hand
that the script did not print.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.ledger import Experiment  # noqa: E402

# Fixed before any run: 30x over the Gaussian model in one-step RMS.
BAR = 30.0


def main() -> int:
    Experiment(
        name="chaos2_generalized_chisquare",
        script="scripts/15_chaos2_bakeoff.py",
        hypothesis=(
            "Truncating the Hermite expansion of each rectifier at chaos order "
            "2 and treating chaos 3+ as an independent Gaussian gives a valid "
            "distribution for S_j -- chaos 1 + chaos 2 is a generalized "
            "chi-square, not a series -- and beats the Gaussian model by at "
            "least 30x in one-step RMS. The whole oracle-cumulant family "
            "manages only ~10x, so anything below 30x is a repackaging of the "
            "old mechanism, not a new one."
        ),
        acceptance_bar=BAR,
        bar_metric="rms_reduction_vs_gaussian",
        bar_direction="higher_is_better",
    ).record(
        seed=0, n_mlps=2, gt_samples=12_000_000, estimator="chaos2-exact",
        compute_ratio=0.0,
        raw_final_layer_mse=5.6012e-5 ** 2,
        rms_reduction_vs_gaussian=5.45,
        w256d32_layer32={"gauss": 3.0553e-4, "gauss_oracle_v": 3.0531e-4,
                         "chaos2": 5.6012e-5, "chaos2_xi0.8": 9.6664e-5,
                         "edge3_oracle": 3.7967e-5, "edge4_oracle": 3.5734e-5,
                         "cond_indep": 3.0174e-4, "factor_chaos2": 5.45},
        w256d32_layer17={"gauss": 7.2297e-4, "chaos2": 1.8260e-4,
                         "edge3_oracle": 8.1215e-5, "edge4_oracle": 6.8652e-5,
                         "cond_indep": 7.1337e-4, "factor_chaos2": 3.96},
        w48d8_factor_by_layer={"2": 3.59, "4": 4.19, "6": 4.64},
        w48d8_rms_gauss_by_layer={"2": 6.3640e-3, "4": 5.6474e-3, "6": 4.2716e-3},
        w48d8_rms_chaos2_by_layer={"2": 1.7729e-3, "4": 1.3489e-3, "6": 9.2061e-4},
        kappa3_captured=0.777, kappa4_captured=-2.122,
        notes=(
            "FAILS: 3.6-5.5x against a 30x bar, and it never beats the "
            "cumulant oracle it was supposed to escape -- Edgeworth-3 fed the "
            "EXACT kappa_3 does 8.1x and Edgeworth-4 8.6x on the same layer "
            "(consistent with the ledger's 10.2x, an independent replication "
            "of that row on this harness). Mechanism of failure identified and "
            "structural: the order-2 Hermite truncation of relu is a PARABOLA, "
            "so the tails of every summand are wrong and the model's kappa_4 "
            "overshoots the truth by 2.6-8x (captured fraction NEGATIVE, "
            "-0.28 to -0.92) while it captures 73-82% of kappa_3. Damping the "
            "quadratic form (A -> xi A, variance held) to trade kappa_4 for "
            "kappa_3 makes it worse at every xi tried. Residual diagnostics: "
            "Cov(P, eta) ~ 0 as assumed, but eta has excess kurtosis 1.69 and "
            "E[P^2 eta] = 0.11 in normalised units, so both the Gaussianity "
            "and the independence of chaos 3+ are violated. Cost was NOT the "
            "blocker: the per-j eigendecomposition is 3.9e10/layer (45x the "
            "free budget over 32 layers) but eig(diag(d_j) R) = eig(Q' D_j Q) "
            "for R ~ QQ', which is ~2e9 over 32 layers at rank 32, i.e. 8% of "
            "free. Method: chaos-2 control variate, truth = chaos2_exact + "
            "E[Y-X] with E[X] = chaos2_exact exactly, two independent halves, "
            "so the MSE is unbiased; CF inversion validated against MC and "
            "quadrature-converged to 2e-13."
        ),
    )

    Experiment(
        name="cond_independence_cgf_shape",
        script="scripts/15_chaos2_bakeoff.py",
        hypothesis=(
            "A sum of INDEPENDENT rectified Gaussians has an exact, "
            "everywhere-valid CGF. Taking the mean and variance exactly from "
            "the full covariance and letting that CGF supply only the SHAPE "
            "(affine renormalisation, under which only its standardised "
            "cumulants survive) beats the Gaussian model by at least 30x, "
            "because the correlations then only have to be right in their "
            "effect on the shape."
        ),
        acceptance_bar=BAR,
        bar_metric="rms_reduction_vs_gaussian",
        bar_direction="higher_is_better",
    ).record(
        seed=0, n_mlps=2, gt_samples=12_000_000, estimator="cond-indep-exact",
        compute_ratio=0.0,
        raw_final_layer_mse=3.0174e-4 ** 2,
        rms_reduction_vs_gaussian=1.74,
        w256d32_factor={"17": 1.01, "32": 1.01},
        w48d8_factor_by_layer={"2": 1.74, "4": 1.43, "6": 1.17},
        gamma1_true={"w256_l32": 0.0430, "w256_l17": 0.0596, "w48_l4": 0.2036},
        gamma1_indep_model={"w256_l32": 0.0015, "w256_l17": 0.0068, "w48_l4": 0.0893},
        gamma1_captured_kfactor_w48={"K0": 0.332, "K1": 0.471, "K2": 0.608},
        gamma1_captured_kfactor_w256={"K0": 0.013, "K1": 0.064, "K2": 0.250},
        notes=(
            "FAILS: 1.01x at the competition shape (1.4-1.7x at width 48). It "
            "answers cleanly the question it was built to ask -- the shape of "
            "S_j is dominated by the CORRELATIONS, not by the individual "
            "rectifications. At width 256 the independence model reproduces "
            "1.3% of the true skewness (gamma1 0.0015 vs 0.0430); at width 48 "
            "it reaches 17-41%. Evaluated EXACTLY by FFT convolution "
            "(grid-converged to 3.4e-8, validated against MC), so this kills "
            "the MODEL, not a saddlepoint approximation of it -- no "
            "saddlepoint was involved, and the ledger's earlier saddle4 row is "
            "not implicated. Adding factors does not rescue it: a K-factor "
            "conditional-independence model captures 1.3 / 6.4 / 25 % of "
            "gamma1 at K = 0 / 1 / 2 at width 256, at 40^K quadrature nodes."
        ),
    )

    Experiment(
        name="one_step_error_split",
        script="scripts/15_chaos2_bakeoff.py",
        hypothesis=(
            "The one-step error of Gaussian propagation is dominated by 'S_j "
            "is not Gaussian even when z is' -- the part every rectifier-shape "
            "model attacks -- rather than by 'z^l itself is not Gaussian'. "
            "Bar: that part must be at least half of the total one-step RMS, "
            "or the whole one-step-shape programme is capped no matter how "
            "good the shape model is."
        ),
        acceptance_bar=0.5,
        bar_metric="errB_fraction_of_total_rms",
        bar_direction="higher_is_better",
    ).record(
        seed=0, n_mlps=1, gt_samples=12_000_000, estimator="diagnostic",
        compute_ratio=0.0,
        raw_final_layer_mse=1.687e-3 ** 2,
        errB_fraction_of_total_rms=0.177,
        width=256, depth=32,
        layer32={"total": 1.687e-3, "errB": 2.988e-4, "errA": 1.665e-3},
        layer17={"total": 3.159e-3, "errB": 6.597e-4, "errA": 2.924e-3},
        gamma1_true_by_layer={"2": 0.0634, "5": 0.1405, "9": 0.2500,
                              "17": 0.3543, "25": 0.4086},
        gamma1_generated_by_layer={"2": 0.0364, "5": 0.0452, "9": 0.0475,
                                   "17": 0.0575, "25": 0.0434},
        gamma1_ratio_by_layer={"2": 1.74, "5": 3.11, "9": 5.27,
                               "17": 6.16, "25": 9.42},
        gamma1_regression_corr_by_layer={"2": 0.835, "5": 0.605, "9": 0.534,
                                         "17": 0.562, "25": 0.307},
        notes=(
            "FAILS the bar at 0.18, and this is the most consequential number "
            "in the round. Drawing z EXACTLY Gaussian from the layer's true "
            "(m, Sigma) leaves a one-step error of only 2.99e-4 at layer 32; "
            "the true one-step error at the same layer is 1.69e-3. So 97% of "
            "the one-step MSE comes from z^l itself not being Gaussian, which "
            "no model of the rectifier map can remove -- it needs a "
            "non-Gaussian STATE. A perfect one-step shape model buys 1.02x at "
            "layer 32 and 1.08x at layer 17, which is why chaos-2's 5.5x on "
            "the part it does address is worth almost nothing end to end, and "
            "why 1.67e-3 (1270x the 1.31e-6 per-layer budget) is a floor for "
            "any Gaussian-state propagation however perfect its rectifier "
            "model. Corroborated independently: gamma1 of z^{l+1} GENERATED by "
            "one layer acting on a Gaussian z^l saturates at 0.043-0.058 while "
            "the network's true gamma1 grows 0.06 -> 0.14 -> 0.25 -> 0.35 -> "
            "0.41 -> 0.44 (ledger, layer 32) with depth; the skewness is "
            "inherited, not generated. The shipped kappa_3 star-diagram term "
            "computes only the GENERATED kappa_3 and is therefore under-scaled "
            "by 1.7x at layer 2 rising to ~10x at layer 32 -- which explains "
            "why umax=2,3 (more diagrams of the same generated term) did not "
            "help. Note the per-neuron pattern also decorrelates with depth "
            "(regression corr 0.84 -> 0.31), so a scalar rescale of the "
            "existing term recovers at most ~10-30% of the variance at depth: "
            "the lever is to PROPAGATE kappa_3 in the state, not to rescale "
            "it."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
