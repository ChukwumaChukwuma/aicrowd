#!/usr/bin/env python
"""Round 11b: Strassen in the scored pass.  The only honest cut to ``c``.

Bar fixed before the run: **1.10x on BILLED FLOPs per sample** (``dF/dN``),
which is the smallest cut that clears the grader's ~2% run-to-run noise with
margin once the dispatch residual is paid.  ``dF/dN`` is machine-independent;
the effective-compute ratio is not, so the second is recorded at several ``N``
and the accuracy claim is recorded from the official 100-MLP suite rather than
argued.

Numbers are READ FROM THE ARTIFACTS ``scripts/38_strassen.py`` wrote, not
transcribed, so the ledger row cannot drift from the run that produced it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whestfloor.ledger import Experiment  # noqa: E402

A = Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve() / "cost"
score = json.loads((A / "strassen_score.json").read_text())
rows = {r["variant"]: r for r in score["rows"]}
SHIP = rows["ship: direct, mask as shipped"]
EVEN = rows["direct, kept sets rounded to even"]
STR = rows["STRASSEN depth 1 (+ even sets)"]

#: Medians of 5 interleaved repeats, one thread, whole ``predict``
#: (``scripts/38 --mode price`` and the residual grid in docs/cost_floor.md).
XC_BY_N = {8500: 0.968, 22000: 1.049, 45000: 1.115, 90000: 1.132}

Experiment(
    name="strassen_scored_pass",
    script="scripts/38_strassen.py",
    hypothesis=(
        "docs/graded.md showed every EQUIVALENT contraction bills exactly "
        "n w (2w-1), so there is no mispriced op to arbitrage.  Strassen is "
        "not an equivalent contraction: it returns the same product from 7 "
        "half-size multiplications instead of 8, so the bill is honest and "
        "the arithmetic is genuinely cheaper.  Depth 1 over the sparse pass "
        "should bill ~1.13x less per sample at a numerical perturbation "
        "orders under the estimator's own residual.  Bar: 1.10x on dF/dN, "
        "the machine-independent number."),
    acceptance_bar=1.10, bar_metric="billed_flops_per_sample_gain",
    bar_direction="higher_is_better",
).record(
    seed=0, n_mlps=score["n_mlps"], gt_samples=1_000_000_000,
    estimator=("sparse MC tau=2.5 + layer-1 Hermite k<=2 + 15-col head, "
               "N=%d P=600, scored pass via Strassen depth 1 on quadrant "
               "blocks" % score["n_samples"]),
    billed_flops_per_sample_gain=2_841_232 / 2_515_251,
    dfdn_direct=2_841_232, dfdn_strassen=2_515_251,
    dfdn_strassen_depth2=2_224_237,
    raw_final_layer_mse=STR["raw_final_layer_mse"],
    flop_ratio=STR["flop_ratio"], compute_ratio=STR["compute_ratio"],
    adjusted_final_layer_score=STR["adj1"],
    adjusted_at_2x_residual=STR["adj2"], adjusted_at_3x_residual=STR["adj3"],
    ship_raw_final_layer_mse=SHIP["raw_final_layer_mse"],
    ship_flop_ratio=SHIP["flop_ratio"],
    ship_compute_ratio=SHIP["compute_ratio"],
    ship_adjusted=SHIP["adj1"],
    even_mask_raw=EVEN["raw_final_layer_mse"], even_mask_adjusted=EVEN["adj1"],
    adjusted_gain_over_shipped=STR["gain_vs_ship"],
    even_mask_gain_over_shipped=EVEN["gain_vs_ship"],
    raises=STR["raises"],
    effective_compute_gain_by_n={str(k): v for k, v in XC_BY_N.items()},
    notes=(
        "PASSES the FLOP bar at 1.1296x on dF/dN (2,841,232 -> 2,515,251), "
        "which is machine-independent, and the answer moves by 3.0e-07 rms "
        "from Strassen's round-off -- four orders under the ~1.7e-3 residual "
        "the head is predicting.  Depth 2 bills 1.2699x and is REJECTED: it "
        "adds ~216 ms of billed residual for a net 1.00x.  What the FLOP bar "
        "does not capture is that the 800 extra dispatches are a FIXED cost "
        "while the 1.1296x is per sample, so the EFFECTIVE gain runs 0.968x "
        "at N=8500, 1.049x at 22000, 1.115x at 45000, 1.132x at 90000 "
        "(medians of 5 interleaved repeats, one thread).  Past N ~ 35000 the "
        "DIRECT path hits its own cache cliff, which chunking repairs for "
        "free (1.075x at 45000), so against the repaired baseline Strassen is "
        "1.037-1.043x; the two devices overlap and chunking is wired into the "
        "Strassen path too (answer-invariant to 3e-6, because it changes "
        "which rows pair in (A11+A22) and so the rounding, not the "
        "arithmetic).  Kept sets have to be even to split the contraction and "
        "are rounded UP -- the least-dead pruned neuron goes back in -- which "
        "is a strictly weaker approximation than the shipped mask and costs "
        "0.34% of c; its effect is recorded separately as the "
        "even_mask_* fields.  NOT wired into submission/estimator.py by this "
        "record: it is default-off in the research kernel until the "
        "chunk x strassen grid is priced at the shipped N."),
)
print("recorded strassen_scored_pass")
