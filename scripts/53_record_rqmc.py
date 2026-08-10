#!/usr/bin/env python
"""Append the RQMC results to ``ledger/experiments.jsonl``.

Every bar below is written here **before** its run and is never re-rolled; the
ledger computes the verdict from the bar rather than taking one from the
caller.  Sources are the JSON artifacts written by ``scripts/50_rqmc_rate.py``
and ``scripts/51_rqmc_deploy.py`` -- this script computes nothing, so it cannot
launder a number.

Bars, as handed down with the task:

* **RATE** -- fitted ``p >= 1.40`` for the lattice in ``v = v_0/N^p`` on the
  official suite.  This is the bar that would justify building the deployable
  version on rate grounds; iid is ``p = 1`` and the classical lattice rate is
  ``p ~ 2``.
* **REDUNDANCY** -- the lattice must be worth ``>= 1.20x`` *on top of* the
  shipped layer-1 Hermite control variates and offline head.  1.20x on raw
  against 1.51% extra billed cost is ~1.18x on adjusted, which would take the
  graded 2.4646e-07 to 2.09e-07 and clear the headline bar with margin.
  radiant-allomancer (18085 §2.1) measured 1.40x for the same construction over
  identical-budget iid and diagnosed the overlap; 1.20x is that, discounted.
* **HEADLINE** -- graded adjusted below **2.4646e-07**, the current ship.  Only
  the grader ranks; local raw runs ~1.45x above graded on identical seeds, so
  the local number recorded here is reported as a RATIO against the iid arm
  measured in the same run, and the bar is applied to the projection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.ledger import Experiment  # noqa: E402

SHIP_GRADED_ADJUSTED = 2.4646e-07
BAR_RATE_P = 1.40
BAR_REDUNDANCY = 1.20


def art() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve() / "rqmc"


def load(name):
    p = art() / name
    return json.loads(p.read_text()) if p.is_file() else None


def record_rate(n_mlps: int):
    d = load("rate_fit.json")
    if not d or "cbc" not in d.get("fits", {}):
        print("no rate_fit.json with a cbc fit; skipping")
        return
    p_lat, se_lat, v0_lat, r2_lat = d["fits"]["cbc"]
    p_iid, se_iid, v0_iid, _ = d["fits"]["iid"]
    Ns = d["N"]
    v_iid, v_cbc = d["V"]["iid"], d["V"]["cbc"]
    ratios = [a / b for a, b in zip(v_iid, v_cbc)]
    e = Experiment(
        name="rqmc_convergence_rate",
        script="scripts/50_rqmc_rate.py",
        hypothesis=(
            "A randomly-shifted rank-1 lattice (CBC order-2 generating vector, "
            "Cranley-Patterson shift, float64 inverse CDF) converges at "
            "p >= 1.40 in v = v_0/N^p, against iid Monte Carlo's p = 1, on the "
            "integrand relu(z^32(x)) of the official MLPs.  The iid arm is the "
            "control and must return p = 1."),
        acceptance_bar=BAR_RATE_P,
        bar_metric="fitted_p",
        bar_direction="higher_is_better",
    )
    e.record(
        estimator="dense forward pass, lattice vs iid draw, variance across "
                  "independent randomisations",
        fitted_p=p_lat, fitted_p_stderr=se_lat, fitted_p_r2=r2_lat,
        fitted_p_iid_control=p_iid, fitted_p_iid_stderr=se_iid,
        v0_lattice=v0_lat, v0_iid=v0_iid,
        n_grid=Ns, variance_iid=v_iid, variance_lattice=v_cbc,
        variance_ratio=ratios,
        # the ledger's required reporting fields: this experiment measures a
        # sampler's variance, not a scored prediction, so raw IS the variance
        # at the shipped operating point and there is no compute multiplier.
        raw_final_layer_mse=v_cbc[-1],
        compute_ratio=None if False else float(Ns[-1] * 4.198656e6 / 2.72e11),
        seed=0, n_mlps=n_mlps, gt_samples=0,
        notes=("Variance measured across randomisations, which for an unbiased "
               "estimator IS its MSE, so no ground truth enters.  Credit: "
               "evaaaz 18053 for the construction."),
    )
    print(f"recorded rate: p_lattice = {p_lat:.3f} +- {se_lat:.3f} "
          f"(bar {BAR_RATE_P}), iid control {p_iid:.3f} +- {se_iid:.3f}")


def record_redundancy(n: int, n_mlps: int, reps: int):
    d = load(f"redundancy_n{n}.json")
    if not d:
        print(f"no redundancy_n{n}.json; skipping")
        return
    s = d["_summary"]
    off = d["lattice,  head off"]
    on = d["lattice,  head ON "]
    iid_on = d["iid,      head ON "]
    gain_on_top = iid_on["mse"] / on["mse"]
    e = Experiment(
        name="rqmc_x_control_variates",
        script="scripts/51_rqmc_deploy.py",
        hypothesis=(
            "The lattice is worth >= 1.20x ON TOP OF the shipped layer-1 "
            "Hermite control variates and offline head, not only against a "
            "bare sampler.  radiant-allomancer 18085 section 2.1 reports the "
            "two levers are partly redundant -- covariance shrinkage already "
            "removes most of the variance RQMC targets -- and that is what "
            "turned 5-7x into 1.40x for them.  Our k=1 Hermite block is the "
            "optimal input-linear control variate and a rank-1 lattice "
            "annihilates exactly the first-order ANOVA terms, so overlap is "
            "expected by construction; the question is what survives at "
            "order >= 2."),
        acceptance_bar=BAR_REDUNDANCY,
        bar_metric="gain_on_top_of_head",
        bar_direction="higher_is_better",
    )
    e.record(
        estimator=f"corrected_sparse_kernel, tau=2.5, N={n}, P=225, "
                  "x0_fn=lattice",
        gain_on_top_of_head=gain_on_top,
        lattice_alone=s["lattice"], head_alone=s["head"], both=s["both"],
        redundancy_factor=s["redundancy"],
        mse_iid_head_off=d["iid,      head off"]["mse"],
        mse_iid_head_on=iid_on["mse"],
        mse_lattice_head_off=off["mse"],
        mse_lattice_head_on=on["mse"],
        bias2_lattice_head_on=on["b2"], bias2_iid_head_on=iid_on["b2"],
        raw_final_layer_mse=on["mse"],
        # matched N in both arms, so the compute ratio is the draw overhead
        # alone: 157 FLOPs/element = 1.51% of the pass (section 3).
        compute_ratio=1.0151, seed=0, n_mlps=n_mlps, gt_samples=1_000_000_000,
        n_randomisations=reps,
        notes="MSE against the official 1e9 reference; variance and bias^2 "
              "separated across randomisations.",
    )
    print(f"recorded redundancy: gain on top of the head = {gain_on_top:.3f}x "
          f"(bar {BAR_REDUNDANCY}), redundancy factor {s['redundancy']:.3f}")


def record_score(n_mlps: int):
    d = load("score.json")
    if not d:
        print("no score.json; skipping")
        return
    lat = {k: v for k, v in d.items() if k.startswith("lat")}
    iid = {k: v for k, v in d.items() if k.startswith("iid")}
    if not lat or not iid:
        print("score.json lacks both arms; skipping")
        return
    best_lat = min(lat.items(), key=lambda kv: kv[1]["adj"])
    best_iid = min(iid.items(), key=lambda kv: kv[1]["adj"])
    ratio = best_lat[1]["adj"] / best_iid[1]["adj"]
    projected = SHIP_GRADED_ADJUSTED * ratio
    e = Experiment(
        name="rqmc_official_suite_score",
        script="scripts/51_rqmc_deploy.py",
        hypothesis=(
            "End to end on the official 100-MLP suite through the shipped "
            "kernel, the lattice arm's adjusted score, projected onto the "
            "grader by the local iid arm measured in the SAME run, is below "
            "the shipped graded 2.4646e-07."),
        acceptance_bar=SHIP_GRADED_ADJUSTED,
        bar_metric="projected_graded_adjusted",
        bar_direction="lower_is_better",
    )
    e.record(
        estimator=f"corrected_sparse_kernel + lattice: {best_lat[0].strip()}",
        projected_graded_adjusted=projected,
        local_adjusted_lattice=best_lat[1]["adj"],
        local_adjusted_iid=best_iid[1]["adj"],
        local_adjusted_ratio=ratio,
        local_iid_variant=best_iid[0].strip(),
        raw_final_layer_mse=best_lat[1]["raw"],
        raw_final_layer_mse_iid=best_iid[1]["raw"],
        compute_ratio=best_lat[1]["CB"],
        compute_ratio_iid=best_iid[1]["CB"],
        worst_mlp=best_lat[1]["worst"], raises=best_lat[1]["nfail"],
        seed=0, n_mlps=n_mlps, gt_samples=1_000_000_000,
        notes="Only the grader ranks.  The local harness reads ~1.45x above "
              "graded on identical seeds, so the bar is applied to the "
              "projection through the paired local iid arm, not to the local "
              "number.",
    )
    print(f"recorded score: local ratio {ratio:.4f}, projected graded "
          f"{projected:.4e} (bar {SHIP_GRADED_ADJUSTED:.4e})")


def record_ship(n_mlps: int):
    """The configuration recommended for the ship, paired against the iid arm
    measured in the SAME run -- not the best-adjusted variant, which sits at
    N = 49,999 with C/B = 0.615 and is 1.06x better at 1x residual and WORSE at
    2x (``docs/rqmc.md`` section 7)."""
    d = load("score.json")
    if not d:
        print("no score.json; skipping")
        return
    key_l = next((k for k in d if k.startswith("lat") and "n=24989" in k
                  and "damp=0.0" in k), None)
    key_i = next((k for k in d if k.startswith("iid") and "n=25000" in k
                  and "damp=1.0" in k), None)
    if not (key_l and key_i):
        print("score.json lacks the shipping pair; skipping")
        return
    lat, iid = d[key_l], d[key_i]
    ratio = lat["adj"] / iid["adj"]
    e = Experiment(
        name="rqmc_shipping_config",
        script="scripts/51_rqmc_deploy.py",
        hypothesis=(
            "The RECOMMENDED shipping configuration -- lattice at N = 24,989 "
            "(prime, CBC order-2 vector), tau = 2.5, P = 225, offline head OFF "
            "-- projected onto the grader through the local iid arm measured in "
            "the same run, is below the shipped graded 2.4646e-07.  N is kept at "
            "the shipped operating point rather than the local argmin at 49,999, "
            "because that point sits at C/B = 0.615, reverses under a 2x "
            "residual, and is past the measured cache cliff."),
        acceptance_bar=SHIP_GRADED_ADJUSTED,
        bar_metric="projected_graded_adjusted",
        bar_direction="lower_is_better",
    )
    e.record(
        estimator="corrected_sparse_kernel(tau=2.5, n_samples=24989, "
                  "n_pilot=225, beta=None, damp=0.0, x0_fn=lattice)",
        projected_graded_adjusted=SHIP_GRADED_ADJUSTED * ratio,
        local_adjusted_lattice=lat["adj"], local_adjusted_iid=iid["adj"],
        local_adjusted_ratio=ratio,
        local_adjusted_lattice_2x=lat["adj2"], local_adjusted_iid_2x=iid["adj2"],
        raw_final_layer_mse=lat["raw"], raw_final_layer_mse_iid=iid["raw"],
        raw_ratio=iid["raw"] / lat["raw"],
        compute_ratio=lat["CB"], compute_ratio_iid=iid["CB"],
        worst_mlp=lat["worst"], worst_mlp_iid=iid["worst"],
        raises=lat["nfail"], seed=0, n_mlps=n_mlps,
        gt_samples=1_000_000_000,
        notes="Only the grader ranks.  Worst-MLP also improves 3.2x "
              "(1.315e-05 -> 4.163e-06), which matters because the score is a "
              "mean over MLPs and ours is worst-MLP dominated.",
    )
    print(f"recorded ship: local ratio {ratio:.4f} (raw "
          f"{iid['raw'] / lat['raw']:.4f}x), projected graded "
          f"{SHIP_GRADED_ADJUSTED * ratio:.4e} (bar {SHIP_GRADED_ADJUSTED:.4e})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=("all", "rate", "redundancy", "score", "ship"))
    ap.add_argument("--n-mlps", type=int, default=6)
    ap.add_argument("--N", type=int, default=24989)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    if a.what in ("all", "rate"):
        record_rate(a.n_mlps)
    if a.what in ("all", "redundancy"):
        record_redundancy(a.N, a.n_mlps, a.reps)
    if a.what in ("all", "score"):
        record_score(100)
    if a.what in ("all", "ship"):
        record_ship(100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
