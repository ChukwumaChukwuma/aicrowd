#!/usr/bin/env python
"""Append the Gaussian-mixture results to the experiment ledger.

Every bar below was fixed in commit 432a7cd, before any of these runs, and is
not re-rolled.  They come from three places and nowhere else:

``1.0e-6``  our shipped **graded** raw MSE is 1.0754e-6 at 2.4x the compute of
            a six-component mixture.  A ceiling above 1.0e-6 means the family
            cannot beat what is already submitted even with a perfect
            propagator, so the bar is the ship, not an aspiration.
``4.0e-9``  dpskv5's raw, read off the public per-MLP telemetry.  This is the
            bar the moonshot is actually aimed at: it is what "the top of the
            leaderboard is a mixture" would require.
``9.0e-9``  huang_chung_yi's raw, at 6.30e9 billed FLOPs = 1.9 measured
            covariance propagations.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.ledger import Experiment  # noqa: E402

ART = Path(os.environ.get("WHEST_ARTIFACTS", "artifacts")) / "mixture"
SCRIPT = "scripts/47_mixture.py"


def jload(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def record_ceiling():
    d = jload("ceiling.json")
    if d is None:
        print("  (no ceiling.json)")
        return
    t = d["table_extrap"]
    gt = 2 * d["n_cells"]
    common = dict(compute_ratio=0.0, seed=int(d["mlp_seeds"][0]),
                  n_mlps=int(d["n_mlps"]), gt_samples=gt)

    r1 = min(v for k, v in t.items()
             if k.startswith("eig|1|") and int(k.split("|")[2]) >= 16)
    e = Experiment(
        name="mixture_ceiling_r1",
        script=SCRIPT,
        hypothesis=(
            "The answer of every Gaussian-mixture method is "
            "sum_k w_k relu_mean(m_kj, s_kj), so the family's ceiling is the "
            "mixture whose components are the TRUE conditional laws of z^32 "
            "given a partition of the top-r eigendirections of Cov(z^32).  At "
            "r = 1 -- the direction docs/state_of_play.md records as carrying "
            "59.7% of the activation covariance and the direction the moonshot "
            "proposes conditioning on -- and with the cell count taken to "
            "saturation (K >= 16 Lloyd cells), that ceiling must reach raw MSE "
            "below 1.0e-6, our shipped graded raw at 2.4x the compute.  Above "
            "that bar an r = 1 mixture cannot beat the ship even with a "
            "perfect propagator and a perfect final-layer readout."),
        acceptance_bar=1.0e-6,
        bar_metric="raw_final_layer_mse",
        bar_direction="lower_is_better",
    )
    e.record(
        estimator="oracle conditional-Gaussian mixture, r=1, K>=16",
        raw_final_layer_mse=r1,
        ceiling_by_r_K=t,
        ceiling_by_r_K_finiteN=d["table"],
        baselines=d["baselines_extrap"],
        notes=(
            "Exact conditional moments from Monte Carlo; no propagation error "
            "and no closure error are charged, so this bounds every mixture of "
            "this shape from above.  Error estimated by pairing each stream's "
            "predictor with its OWN sample truth and multiplying the two "
            "independent streams' residuals, which cancels the leading "
            "Monte-Carlo fluctuation; N/4, N/2 and N agree to 1-2%."),
        **common)
    print(f"  mixture_ceiling_r1              {r1:.4e}  {e.verdict(r1)}")

    k6 = min(v for k, v in t.items()
             if k.startswith("eig|") and k.endswith("|6"))
    e = Experiment(
        name="mixture_ceiling_K6",
        script=SCRIPT,
        hypothesis=(
            "The telemetry-implied budget is about six covariance "
            "propagations, i.e. a six-component mixture.  Over every frame and "
            "every r measured, the best K = 6 ceiling must reach raw < 1.0e-6."),
        acceptance_bar=1.0e-6,
        bar_metric="raw_final_layer_mse",
        bar_direction="lower_is_better",
    )
    e.record(estimator="oracle conditional-Gaussian mixture, K=6, best r",
             raw_final_layer_mse=k6, ceiling_by_r_K=t, **common)
    print(f"  mixture_ceiling_K6              {k6:.4e}  {e.verdict(k6)}")

    best8 = min(v for k, v in t.items() if int(k.split("|")[2]) <= 8)
    e = Experiment(
        name="mixture_ceiling_leaderboard",
        script=SCRIPT,
        hypothesis=(
            "If the top of the leaderboard is a six-component Gaussian "
            "mixture, then some mixture with K <= 8 components must be capable "
            "of dpskv5's raw 4.0e-9.  This is the moonshot's own claim, tested "
            "against its own ceiling with every advantage granted: exact "
            "conditional moments, an optimal Lloyd partition, and a free "
            "choice among variance, kurtosis and input-side frames."),
        acceptance_bar=4.0e-9,
        bar_metric="raw_final_layer_mse",
        bar_direction="lower_is_better",
    )
    e.record(estimator="oracle conditional-Gaussian mixture, K<=8, best (frame, r)",
             raw_final_layer_mse=best8, ceiling_by_r_K=t, **common)
    print(f"  mixture_ceiling_leaderboard     {best8:.4e}  {e.verdict(best8)}")

    best = min(t.values())
    e = Experiment(
        name="mixture_ceiling_any",
        script=SCRIPT,
        hypothesis=(
            "Drop the budget entirely: over EVERY configuration measured -- up "
            "to r = 32 directions and K = 1024 components, which is 170x the "
            "affordable component count -- some conditional-Gaussian mixture "
            "reaches dpskv5's raw 4.0e-9.  Failing this bar says the "
            "leaderboard is not a mixture at any price, not merely that it is "
            "not one at our price."),
        acceptance_bar=4.0e-9,
        bar_metric="raw_final_layer_mse",
        bar_direction="lower_is_better",
    )
    e.record(estimator="oracle conditional-Gaussian mixture, best of all (frame, r, K)",
             raw_final_layer_mse=best, ceiling_by_r_K=t, **common)
    print(f"  mixture_ceiling_any             {best:.4e}  {e.verdict(best)}")


def record_propagate():
    d = jload("propagate.json")
    if d is None:
        print("  (no propagate.json)")
        return
    rows = d["rows"]
    best_key = min(rows, key=lambda k: rows[k])
    best = rows[best_key]
    nodes, r, split_at, resplit = (int(v) for v in best_key.split("|"))
    K = nodes ** r
    F = d["bill"][f"{K}|{split_at}"]
    fb = F / 272_000_000_000
    e = Experiment(
        name="mixture_propagation_deployable",
        script=SCRIPT,
        hypothesis=(
            "The self-contained mixture propagator -- K Gaussians, each "
            "rectified by the exact rectified-Gaussian machinery, no sampling "
            "anywhere -- reaches raw MSE below 1.0e-6 at a cost under the 0.1 "
            "multiplier clamp.  Same bar as the ceiling: below it the method "
            "beats the ship, above it there is no reason to integrate it."),
        acceptance_bar=1.0e-6,
        bar_metric="raw_final_layer_mse",
        bar_direction="lower_is_better",
    )
    e.record(
        estimator=f"mixture propagation nodes={nodes} r={r} split@{split_at} "
                  f"resplit={resplit} (K={K})",
        raw_final_layer_mse=best,
        adjusted_final_layer_score=best * max(0.1, fb),
        compute_ratio=fb,
        flops_used=F,
        seed=970_000,
        n_mlps=6,
        gt_samples=2 * 500_000,
        chain_baseline=d["chain"],
        by_config=rows,
        billed_flops_by_K=d["bill"],
        notes=("F measured in a real flopscope.BudgetContext on the same "
               "kernel that produced the answer."),
    )
    print(f"  mixture_propagation_deployable  {best:.4e}  {e.verdict(best)}  "
          f"({best_key}, F/B={fb:.4f})")


def record_anatomy():
    d = jload("anatomy.json")
    if d is None:
        print("  (no anatomy.json)")
        return
    pr_v = float(np.mean([m["pr_variance"] for m in d["mlps"]]))
    pr_k = float(np.mean([m["pr_abs_excess_kurtosis"] for m in d["mlps"]]))
    share1 = float(np.mean([m["err_cum_share_oracle"][0] for m in d["mlps"]]))
    rms = float(np.mean([m["rms_oracle"] for m in d["mlps"]]))
    e = Experiment(
        name="mixture_premises",
        script=SCRIPT,
        hypothesis=(
            "The mixture hypothesis rests on the claim that the "
            "non-Gaussianity of z^32 is concentrated in very few directions, "
            "inferred from the collapse of the activation-covariance "
            "participation ratio to ~2.7 with the top eigendirection carrying "
            "59.7%.  Variance concentration is not non-Gaussianity "
            "concentration.  Bar: the participation ratio of |excess kurtosis| "
            "over the eigenframe must itself be below 10 -- i.e. the "
            "non-Gaussianity really is low-rank -- for the premise to hold."),
        acceptance_bar=10.0,
        bar_metric="pr_abs_excess_kurtosis",
        bar_direction="lower_is_better",
    )
    e.record(estimator="diagnostic",
             raw_final_layer_mse=rms ** 2,
             compute_ratio=0.0, seed=int(d["mlps"][0]["seed"]),
             n_mlps=len(d["mlps"]), gt_samples=int(d["n_samples"]),
             pr_variance=pr_v, pr_abs_excess_kurtosis=pr_k,
             err_share_top1_eigendirection=share1,
             per_mlp=d["mlps"])
    print(f"  mixture_premises   PR(var)={pr_v:.1f}  PR(|kurt|)={pr_k:.1f}  "
          f"{e.verdict(pr_k)}")


#: Marginal billed cost of one extra mixture component, measured in a real
#: BudgetContext (scripts/47 --mode propagate): 2.364e9 FLOPs.
COMPONENT_FLOPS = 2_364_000_000
FLOP_BUDGET = 272_000_000_000


def frontier():
    """The cost-accuracy frontier of the whole family, and its extrapolation.

    ``K`` is what a mixture *pays* for: each component is one covariance
    propagation.  So the object that decides the moonshot is
    ``min over (frame, r) of ceiling(frame, r, K)`` as a function of ``K``, and
    the question is what ``K`` a log-log extrapolation puts at 4.0e-9.
    """
    d = jload("ceiling.json")
    if d is None:
        return
    t = d["table_extrap"]
    by_k: dict[int, float] = {}
    arg: dict[int, str] = {}
    for key, v in t.items():
        K = int(key.split("|")[2])
        if v > 0 and (K not in by_k or v < by_k[K]):
            by_k[K], arg[K] = v, key
    by_k[1] = d["baselines_extrap"]["gauss"]
    arg[1] = "single Gaussian, exact mean and variance"
    Ks = sorted(by_k)
    print()
    print("# the frontier: best ceiling at each component count")
    print(f"  {'K':>5} {'best raw MSE':>13} {'rms':>10} {'F = K*2.364e9':>15} "
          f"{'F/B':>7}  argmin")
    for K in Ks:
        F = K * COMPONENT_FLOPS
        print(f"  {K:>5} {by_k[K]:13.4e} {np.sqrt(by_k[K]):10.3e} "
              f"{F:15,d} {F / FLOP_BUDGET:7.3f}  {arg[K]}")
    fit = [K for K in Ks if K >= 4]
    if len(fit) >= 3:
        b, a = np.polyfit(np.log([float(K) for K in fit]),
                          np.log([by_k[K] for K in fit]), 1)
        print()
        print(f"  log-log fit over K >= 4:  raw ~ {np.exp(a):.3e} * K^{b:+.3f}")
        for target, who in ((9.0e-9, "huang_chung_yi"), (4.0e-9, "dpskv5")):
            Kneed = float(np.exp((np.log(target) - a) / b))
            F = Kneed * COMPONENT_FLOPS
            print(f"  to reach raw {target:.1e} ({who}): K = {Kneed:.3g} "
                  f"components = {F:.3g} FLOPs = {F / FLOP_BUDGET:.4g} x the "
                  f"whole budget")
        return {"by_K": {str(k): v for k, v in by_k.items()},
                "argmin": {str(k): v for k, v in arg.items()},
                "loglog_a": float(a), "loglog_b": float(b)}
    return None


def main() -> int:
    #: The ledger is append-only by design -- a re-run appends, it never edits
    #: -- so name the sections to record and do not re-append the others.
    only = set(sys.argv[1:]) or {"anatomy", "ceiling", "propagate"}
    print(f"# recording mixture results: {sorted(only)}")
    if "anatomy" in only:
        record_anatomy()
    if "ceiling" in only:
        record_ceiling()
    if "propagate" in only:
        record_propagate()
    fr = frontier()
    if fr is not None:
        (ART / "frontier.json").write_text(json.dumps(fr, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
