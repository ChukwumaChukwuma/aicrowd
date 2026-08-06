#!/usr/bin/env python
"""Round 6: multilevel Monte Carlo over network surrogates, measured and killed.

Bars were fixed before the runs, in the order the brief specified: the level
variances had to clear a 20x cost-to-accuracy bar BEFORE anything was built,
and only then would the kernel ship.  They did not, so this records the kill
and the two structural numbers that make it general.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment  # noqa: E402

OFF = dict(seed=0, n_mlps=100, gt_samples=1_000_000_000)

Experiment(
    name="mlmc_rank_levels",
    script="scripts/22_mlmc_variance_decay.py",
    hypothesis=(
        "Multilevel Monte Carlo over rank-truncated copies of the network "
        "beats plain Monte Carlo's cost-to-accuracy V*C_full by at least 20x. "
        "Levels are coupled (one input draw through both networks), allocation "
        "is the MLMC rule N_k ~ sqrt(V_k/C_k), and the figure of merit is "
        "(sum_k sqrt(V_k C_k))^2."),
    acceptance_bar=20.0, bar_metric="mlmc_gain_over_plain_mc",
    bar_direction="higher_is_better",
).record(
    **OFF,
    estimator="mlmc_kernel(levels=[(r,N_r),(256,N)]) vs mc_kernel",
    mlmc_gain_over_plain_mc=0.9413,
    compute_ratio=0.0868,
    raw_final_layer_mse=9.1010e-6,
    adjusted_final_layer_score=9.1010e-7,
    mc_plain_raw_final_layer_mse=1.0994e-5,
    mc_plain_compute_ratio=0.0829,
    endtoend_gain_measured=1.208,
    endtoend_gain_predicted=0.938,
    flops_used=2.208e10,
    n_probe_mlps=3, n_probe_samples=8192,
    surrogate_families="svd, svd_rescale, covproj(oracle E[hh^T]), sketch",
    best_rho_any_surrogate=0.875,
    best_rho_cost_ratio=1.75,
    rho_required_for_bar_necessary=0.975,
    rho_required_for_bar_realistic=0.9989,
    best_rho_below_half_rank=0.229,
    perturbation_amplification_K=393.0,
    delta_required_for_bar=2.4e-3,
    rank_for_delta_required=251,
    notes=(
        "FAILS at 0.941x against a 20x bar -- every genuine multilevel ladder "
        "is WORSE than plain sampling, and the MLMC allocator's own optimum is "
        "to put ONE sample on level 0 and degenerate back to plain MC. Two "
        "independent obstructions, either fatal. (A) COST: flopscope charges a "
        "factored layer 4nr against 2n^2 dense, so c(r) = 2r/n and the level-0 "
        "discount alone caps the gain at n/(2 r_0) = 128/r_0; a 20x bar needs "
        "r_0 <= 6, and r = 128 already costs exactly what the dense layer does. "
        "Even a PERFECT "
        "rank-8 surrogate caps at 15.8x. (B) COUPLING: perturbing every weight "
        "matrix by a norm-preserving relative amount d gives a clean quadratic "
        "law Var(f-f~)/V = K d^2 with K = 393 over two decades (d = 1e-4..3e-3), "
        "so the top level needs d <= 2.4e-3 -- r = 251 of 256 by the W spectrum, at "
        "1.96x the dense cost. "
        "Best coupling measured anywhere: rho = 0.875 at r = 224, which costs "
        "1.75x the dense network, and at every rank that actually saves FLOPs "
        "(r < 128) the best rho anywhere is 0.229, against a NECESSARY rho >= "
        "0.975 (which already grants a free level 0; a realistic rank-4 level 0 "
        "needs 0.9989). An input-adapted "
        "ORACLE projector onto the top-r eigenspace of the measured E[h h^T] is "
        "much better per rank (deep activations are near rank-one: 98.3% of "
        "E||h||^2 in ONE direction at layer 32) but still loses, because layer "
        "1's activation second moment is the IDENTITY -- the early layers are "
        "incompressible and they dominate (perturbing only the last 4 of 32 "
        "layers costs 0.34x the uniform prediction). Solving for the per-layer "
        "ranks that hit even BREAK-EVEN gives a surrogate costing 0.97x dense; "
        "for the 20x target, 1.00x. Plain SVD truncation additionally outputs "
        "literal ZEROS: keeping fraction kappa of ||W||_F gives per-layer gain "
        "sqrt(kappa) and 0.376^16 = 6e-8, so the naive construction in the brief "
        "fails for a second, trivial reason; rescaling to fix it measures WORSE. "
        "END TO END on the official 100-MLP N=1e9 suite at matched 2.2e10 FLOPs, "
        "all at the 0.1 multiplier floor: plain MC 1.0994e-5 raw, mlmc[128,256] "
        "2.4393e-5 (0.45x), mlmc[224,256] 4.9742e-5 (0.22x), mlmc[192,224,256] "
        "1.3808e-4 (0.08x). Cost model audited against a real BudgetContext to "
        "0.2-0.4%, and mlmc_kernel(levels=((256,N),)) is BITWISE identical to "
        "mc_kernel(N) with identical FLOP counts, so the plain-MC arm is a true "
        "ablation through the same code path. NOTHING IS SHIPPED. This is the "
        "same wall docs/floor_theorem.md records for input-anchored control "
        "variates, reached from a third direction: depth-32 ReLU mixing "
        "decorrelates anything that is not the network itself -- an affine "
        "function of the input gets rho = 0.25, a half-rank copy of the network "
        "rho = 0.39, and a copy costing 1.75x the original rho = 0.875."),
)

Experiment(
    name="activation_spectrum_is_near_rank_one_at_depth",
    script="scripts/22c_mlmc_ceiling.py",
    hypothesis=(
        "The activation second moment E[h_l h_l^T] concentrates enough at depth "
        "that a top-r projector captures 99.9% of E||h||^2 at r <= 128 for every "
        "layer, which is what a cheap input-adapted surrogate would need."),
    acceptance_bar=0.999, bar_metric="min_capture_r128_over_layers",
    bar_direction="higher_is_better",
).record(
    **OFF,
    estimator="oracle projector onto top-r eigenspace of measured E[h h^T]",
    compute_ratio=0.0,
    raw_final_layer_mse=0.0,
    min_capture_r128_over_layers=0.6058,
    capture_r128_layer1=0.6058,
    capture_r128_layer1_population=0.5,
    capture_r128_layer16=0.99925,
    capture_r128_layer32=0.999974,
    capture_r1_layer32=0.983076,
    notes=(
        "FAILS on the EARLY layers and passes handsomely on the late ones, and "
        "that split is the finding. At layer 32 a SINGLE direction carries "
        "98.3% of E||h||^2 and rank 128 carries 99.997%; at layer 16 rank 128 "
        "carries 99.92%. But h_0 = x is isotropic, so the POPULATION capture at "
        "layer 1 is cap_1(r) = r/n exactly -- 0.5 at r = 128, not the 0.6058 "
        "recorded here, which is a 4096-sample estimate inflated by "
        "Marchenko-Pastur spread and therefore optimistic -- and layer 1 needs "
        "the full 256. Since a factored layer only saves FLOPs "
        "below half rank, and since perturbations injected EARLY dominate the "
        "output error (measured: perturbing the last 4 of 32 layers costs 0.34x "
        "the uniform-quadrature prediction, the last 8 0.47x, the last 16 0.78x), "
        "the compressibility that exists is in exactly the layers where it is "
        "worth the least. Recorded because the low-rank structure is REAL and "
        "may be usable by a mechanism that does not pay 4nr per layer -- it just "
        "cannot be turned into a cheaper forward pass."),
)
print("recorded 2")
