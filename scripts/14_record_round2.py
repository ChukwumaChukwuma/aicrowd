#!/usr/bin/env python
"""Record round-2 results: the corrected floor, the killed cumulant route, and
the one mechanism that survived and shipped."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import ADJUSTED_FLOOR, PER_LAYER_BAR  # noqa: E402
from whestfloor.ledger import Experiment  # noqa: E402

SUITE = dict(seed=0, n_mlps=8, gt_samples=1_200_000)


def main() -> int:
    Experiment(
        name="reconstruction_from_cumulants",
        script="scripts/11_reconstruction_exact.py",
        hypothesis=(
            "Some reconstruction of E[relu] from exact cumulants through order "
            "6 reaches the per-layer budget. Tested with NO Monte Carlo: "
            "targets built by FFT convolution, exact to ~1e-12."
        ),
        acceptance_bar=PER_LAYER_BAR,
        bar_metric="best_rms_error",
    ).record(
        **SUITE, estimator="reconstruction-oracle", compute_ratio=0.0,
        raw_final_layer_mse=1.0966e-3 ** 2,
        best_rms_error=1.0966e-3,
        rms_gauss=8.9998e-3, rms_edge3=1.8495e-3, rms_edge4=1.0966e-3,
        rms_edge6=1.8177e-3, rms_gamma_gauss=1.3424e-3,
        notes=(
            "KILLS THE CUMULANT ROUTE. Best is 8.2x against a required ~3800x, "
            "and Edgeworth-6 is WORSE than Edgeworth-4 (1.82e-3 vs 1.10e-3), so "
            "the series diverges. A moment-matched Gamma-convolved-Gaussian - a "
            "genuine density, unlike a truncated series - does not beat it "
            "either (1.34e-3), so this is not merely a bad reconstruction: the "
            "information is not in the low-order cumulants. Population matched "
            "to the network's measured regime (|gamma1| 0.37, gamma2 0.23)."
        ),
    )

    Experiment(
        name="kappa3_star_diagrams",
        script="scripts/13_validate_kappa3.py",
        hypothesis=(
            "Diagrams with a zero edge multiplicity factorise through the "
            "rank-one R^(0), so kappa_3 is computable for all outputs at O(n^3) "
            "instead of O(n^4), and they carry >=80% of the true kappa_3."
        ),
        acceptance_bar=0.80,
        bar_metric="fraction_captured",
        bar_direction="higher_is_better",
    ).record(
        seed=11, n_mlps=1, gt_samples=3_000_000,
        estimator="kappa3_star", compute_ratio=0.0,
        raw_final_layer_mse=0.0,
        fraction_captured=0.884,
        captured_by_layer={"2": 0.864, "4": 0.850, "6": 0.872, "8": 0.884},
        notes=(
            "Validated against brute-force MC on a real layer so R carries the "
            "correlation structure the network actually produces. umax=1 wins "
            "shallow, umax=2 deep; higher umax degrades, so the omitted "
            "triangle diagrams are not negligible and the star series should "
            "not be pushed further."
        ),
    )

    Experiment(
        name="edgeworth_kappa3_end_to_end",
        script="scripts/12_evaluate.py",
        hypothesis=(
            "Adding the analytically-computed kappa_3 Edgeworth correction to "
            "Mehler covariance propagation improves the end-to-end adjusted "
            "score by at least 1.5x over the strongest bundled baseline, "
            "without crossing the 0.1 multiplier floor."
        ),
        acceptance_bar=1.5,
        bar_metric="improvement_over_baseline",
        bar_direction="higher_is_better",
    ).record(
        **SUITE, estimator="cov_prop_edgeworth(kmax=4,umax=1)",
        compute_ratio=0.0553,
        adjusted_final_layer_score=3.2338e-6,
        raw_final_layer_mse=3.2370e-5,
        unbiased_true_mse=3.2338e-5,
        unbiased_true_mse_stderr=5.8e-7,
        flops_used=5.349e9,
        improvement_over_baseline=6.8218e-5 / 3.2338e-5,
        baseline_cov_prop_gain=6.8218e-5,
        mehler_only=6.3156e-5,
        ablated_damp0=6.3156e-5,
        x_above_floor=3.2338e-6 / ADJUSTED_FLOOR,
        notes=(
            "PASSES at 2.11x. Ablation is exact: setting the correction "
            "coefficient to zero reproduces the uncorrected 6.3156e-5 to the "
            "last digit through the identical code path, so the gain is "
            "attributable to this term and nothing else. umax=2 (3.61e-5) and "
            "umax=3 (3.78e-5) are worse, and umax=3 also pushes C/B to 0.114, "
            "crossing the multiplier floor. Still 5.9e5x above the noise floor: "
            "this is a real improvement over the baselines, not a solution."
        ),
    )
    print("recorded 3 experiments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
