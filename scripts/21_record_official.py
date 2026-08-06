#!/usr/bin/env python
"""Round 5: real leaderboard-comparable numbers, and a crash that would have
cost ~850x the entire score."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment                            # noqa: E402

OFF = dict(seed=0, n_mlps=100, gt_samples=1_000_000_000)

Experiment(
    name="symmetry_crash_on_official_mlp19",
    script="scripts/19_fetch_official_suite.py",
    hypothesis="The shipped estimator runs without raising on all 100 official "
               "MLPs.",
    acceptance_bar=0.0, bar_metric="n_failed_mlps",
).record(
    **OFF, estimator="cov_prop_edgeworth(f32=True) [pre-fix]",
    compute_ratio=0.0, raw_final_layer_mse=6.6202,
    n_failed_mlps=1.0,
    notes=("CAUGHT ONLY BY THE OFFICIAL SUITE. The shipped config raised "
           "flopscope SymmetryError at layer 27 of official MLP 19 (seed "
           "3254440928728510558) - 1 of 100, and invisible on all 8 locally "
           "generated suites. Not an overflow: everything finite, max|cov_pre| "
           "= 2.95 vs a 100-MLP median of 2.52; the f32 cast makes the einsum "
           "result asymmetric just beyond as_symmetric's atol=1e-6. The grader "
           "zeroes a raising MLP, whose MSE is 6.6202, adding 6.62e-2 to the "
           "mean - ~850x the entire score. Fixed by replacing as_symmetric "
           "(validates, raises) with symmetrize (projects, cannot raise), plus "
           "a try/except that falls back to the MC arm alone. Verified: 0 "
           "failures over 100 MLPs, worst single-MLP MSE 4.888e-5."),
)

Experiment(
    name="official_suite_leaderboard_comparable",
    script="scripts/12_evaluate.py",
    hypothesis="Against the official N=1e9 reference, the shipped estimator "
               "beats plain Monte Carlo's grader-reported adjusted plateau of "
               "6.4695e-7.",
    acceptance_bar=6.4695e-7, bar_metric="adjusted_final_layer_score",
).record(
    **OFF, estimator="blend_kernel(n=4600, wmc=0.70)",
    compute_ratio=0.111,
    adjusted_final_layer_score=7.7791e-7,
    raw_final_layer_mse=6.9885e-6,
    unbiased_true_mse=6.9885e-6,
    flops_used=2.47e10,
    official_avg_variance=0.049491,
    official_raw_floor=4.949e-11,
    baseline_cov_prop_gain=8.3663e-5,
    mc_n6000=9.2101e-7,
    leader_adjusted=3.63e-10,
    x_above_leader=7.7791e-7 / 3.63e-10,
    notes=("FAILS the bar by 1.20x, and that is the honest headline: after all "
           "of this the estimator is still marginally WORSE than plain "
           "sampling as the grader measures it. It is 10.8x better than the "
           "strongest bundled baseline (8.3663e-5 raw) and 1.18x better than "
           "our own best pure MC (9.2101e-7), but the bundled baseline was "
           "never the right reference. C/B = 0.111 is deliberately just over "
           "the 0.1 floor: above the floor the adjusted score is flat in N "
           "while blending still lowers raw MSE, so the small penalty is "
           "repaid. Official avg_variance over 100 MLPs = 0.049491 +- 0.003336 "
           "(CV 0.674), confirming contract.py's 0.0551 to 1.4sigma and "
           "refuting the docs' 0.18 at 39sigma; the true raw floor is 4.949e-11. "
           "Leader is at 3.63e-10 adjusted, so we remain ~2140x behind."),
)
print("recorded 2")
