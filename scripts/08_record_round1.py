#!/usr/bin/env python
"""Record round-1 diagnostic results in the ledger, failures included.

Each bar below was fixed before the corresponding run: every one of scripts
04-07 states the 1.34e-5 per-neuron RMS target in its module docstring, and
that target — together with the per-layer budget derived from it — is the bar.
Nothing here is re-rolled after the fact.

Per-layer budget.  End-to-end RMS must reach 1.34e-5.  Measured accumulation is
incoherent across layers: covariance propagation injects ~2.0e-3 per layer and
ends at ~9.2e-3 end-to-end, and sqrt(32) x 2.0e-3 = 1.13e-2, so the gain per
layer is ~1.  A per-layer injection of eps therefore lands at sqrt(32)*eps, and
the per-layer bar is 1.34e-5 / sqrt(32) = 2.4e-6.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.ledger import Experiment  # noqa: E402

PER_LAYER_BAR = 1.34e-5 / (32 ** 0.5)  # 2.37e-6

COMMON = dict(
    seed=0,
    n_mlps=1,
    compute_ratio=0.0,          # diagnostics; no estimator was billed
    estimator="diagnostic",
)


def main() -> int:
    e1 = Experiment(
        name="one_step_gaussian_error",
        script="scripts/04_layer_error_anatomy.py",
        hypothesis=(
            "Given the exact previous-layer mean and covariance, the Gaussian "
            "assumption alone injects less than the per-layer budget of "
            "2.37e-6 RMS."
        ),
        acceptance_bar=PER_LAYER_BAR,
        bar_metric="one_step_rms_layer32",
    )
    e1.record(
        **COMMON,
        gt_samples=200_000,
        raw_final_layer_mse=1.314e-3 ** 2,
        one_step_rms_layer32=1.314e-3,
        one_step_rms_layer2=1.693e-3,
        one_step_mean_layer32=4.956e-4,
        skew_rms_layer32=0.4389,
        excess_kurtosis_layer32=0.4044,
        r_mean_fraction_layer32=0.9514,
        measured_v_layer32=0.0898,
        notes=(
            "Layer 1 reads 3.33e-4, which is the paired-measurement noise "
            "floor, not an error: z^1 is exactly Gaussian, so the diagnostic "
            "validates itself. Deeper layers are 1.0-2.4e-3, ~550x over the "
            "per-layer bar. Closed-form Gaussian propagation cannot reach the "
            "noise floor by any arrangement of the chain. Also: measured "
            "per-neuron final-layer variance is 0.090 for this MLP, not the "
            "0.18 quoted in the challenge docs - the floor may be 9e-12 "
            "adjusted rather than 1.8e-11. Being re-derived independently."
        ),
    )

    e2 = Experiment(
        name="conditional_gaussianity_low_rank",
        script="scripts/06_conditional_gaussianity.py",
        hypothesis=(
            "The non-Gaussianity of z^l is carried by its dominant covariance "
            "directions, so conditioning on the top k<=3 coordinates and "
            "assuming Gaussianity only within cells cuts the one-step error by "
            "at least 100x, making a shared-covariance Gaussian mixture the "
            "right state to propagate."
        ),
        acceptance_bar=100.0,
        bar_metric="reduction_factor",
        bar_direction="higher_is_better",
    )
    e2.record(
        **COMMON,
        gt_samples=300_000,
        raw_final_layer_mse=4.530e-4 ** 2,
        reduction_factor=2.9,
        rms_k0=1.3191e-3,
        rms_k1=9.7223e-4,
        rms_k2=5.1587e-4,
        rms_k3=4.5304e-4,
        mean_err_k0=4.973e-4,
        mean_err_k3=1.185e-5,
        notes=(
            "REFUTED, and cheaply. Converged in bin count (k=2: 5.24e-4 at 64 "
            "cells vs 5.16e-4 at 1024), so this is not binning error. The "
            "fluctuation is low-rank (participation ratio 2.2, top eigenvalue "
            "67% of trace, scripts/05) but the NON-GAUSSIANITY is not: it is "
            "spread across many directions. Low-rank exact + Gaussian "
            "remainder is dead as a mechanism. One useful residue: the "
            "coherent bias falls 42x (4.97e-4 -> 1.19e-5) under conditioning, "
            "so the bias and the scatter have different origins."
        ),
    )

    e3 = Experiment(
        name="edgeworth_oracle_cumulants",
        script="scripts/07_edgeworth_test.py",
        hypothesis=(
            "An Edgeworth correction through fourth order, supplied with the "
            "EXACT measured cumulants, brings the one-step error under the "
            "2.37e-6 per-layer bar."
        ),
        acceptance_bar=PER_LAYER_BAR,
        bar_metric="one_step_rms_layer32_k3k4",
    )
    e3.record(
        **COMMON,
        gt_samples=250_000,
        raw_final_layer_mse=1.474e-4 ** 2,
        one_step_rms_layer32_gauss=1.337e-3,
        one_step_rms_layer32_k3=4.871e-4,
        one_step_rms_layer32_k3k4=1.474e-4,
        one_step_rms_layer32_k3k4k3sq=1.313e-4,
        noise_control_layer32=3.320e-5,
        noise_control_layer8=1.128e-4,
        reduction_factor=10.2,
        notes=(
            "FAILS the bar by 62x, but is the strongest mechanism measured so "
            "far: 10-22x per layer with oracle cumulants. Adding the gamma_1^2 "
            "term buys nothing (1.31e-4 vs 1.47e-4, within noise), which says "
            "the series is already near its asymptotic limit at gamma_1=0.44 - "
            "the expansion parameter is 0.44, not n^-1/2=0.06, so this is a "
            "slowly-converging asymptotic series, not a convergent one. "
            "Extrapolating 15x per-layer to end-to-end gives RMS ~6e-4, MSE "
            "~3.7e-7, adjusted ~3.7e-8 - which is about where the public "
            "leaderboard sits, and 2000x above the floor. Consistent with the "
            "brief's statement about the board. An upper bound on this route, "
            "since a real estimator must also COMPUTE the cumulants."
        ),
    )
    print("recorded 3 experiments (1 informative, 2 FAIL)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
