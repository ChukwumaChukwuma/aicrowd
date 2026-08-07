"""Append the trajectory-calibrated-closure results to the experiment ledger.

Every bar below was fixed before its run and is not re-rolled.  The single bar
that decides the direction is ``r > 4.4``: ``docs/integrable_cv.md`` sec 3.2
measures the break-even of the deep ``relu(z^L)`` control variate at
``r = 4.2-4.6`` at every depth from 6 to 32, where ``r`` is the factor by which
a closure's ``rms(E[relu(z^L)] - truth)`` beats the Gaussian closure's.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.ledger import Experiment  # noqa: E402

ART = Path(os.environ.get("WHEST_ARTIFACTS", "artifacts")) / "traj"
SCRIPT = "scripts/43_traj_closure.py"
#: The reference is 2,000,000 samples in two independent halves per MLP.
GT = 2_000_000


def jload(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def record_baseline():
    d = jload("baseline_loc.json")
    if d is None:
        return
    by = {r["arm"]: r for r in d["rows"]}
    e = Experiment(
        name="closure_accuracy_r_baseline",
        script=SCRIPT,
        hypothesis=(
            "Before building anything, put the two analytic arms this "
            "repository already has on the yardstick docs/integrable_cv.md "
            "sec 3.2 defines: r = rms(mu_gauss^L - mu_true^L) / "
            "rms(mu_arm^L - mu_true^L) for mu^L = E[relu(z^L)], against a "
            "2e6-sample reference with the two-half unbiased estimator, at "
            "L = 8, 16, 24, 32.  The Gaussian closure must come out at r = 1 "
            "by construction and the kappa_3 star-diagram arm at the "
            "published r = 1.45; if either does not reproduce, the yardstick "
            "is not measuring what the payoff curve was priced against."),
        acceptance_bar=1.30,
        bar_metric="r_kappa3_L8",
        bar_direction="higher_is_better",
    )
    e.record(
        estimator="cov_prop_edgeworth (kappa_3 star) and the tree catalogue",
        raw_final_layer_mse=by["gauss"]["rms"]["32"] ** 2,
        compute_ratio=0.0,
        seed=int(d["seeds"][0]),
        n_mlps=len(d["seeds"]),
        gt_samples=GT,
        b=by["gauss"]["rms"]["32"],
        v_eff=None,
        c=None,
        r_kappa3_L8=by["k3star"]["r"]["8"],
        r_by_arm={a: by[a]["r"] for a in by},
        rms_by_arm={a: by[a]["rms"] for a in by},
        seeds=[int(s) for s in d["seeds"]],
        notes=(
            "REPRODUCED.  Gaussian closure r = 1.000 by construction, rms "
            f"{by['gauss']['rms']['32']:.3e} at L=32 against the sibling's "
            "6.38e-03 on the official pair (different MLPs, same order).  "
            "kappa_3 star arm r = "
            f"{by['k3star']['r']['8']:.3f}/{by['k3star']['r']['16']:.3f}/"
            f"{by['k3star']['r']['24']:.3f}/{by['k3star']['r']['32']:.3f} at "
            "L = 8/16/24/32, against the published 1.45.  The COMPLETE tree "
            "catalogue (injectivity corrections, both Ursell terms, kappa_4) "
            "is NOT better than the incomplete star form -- 1.42/1.34/1.31/"
            "1.23 -- which is the first sign that the source diagrams are not "
            "the binding constraint."),
    )


def record_ceiling():
    d = jload("ceiling_loc.json")
    if d is None:
        return
    by = {r["oracle"]: r for r in d["rows"]}
    e = Experiment(
        name="closure_accuracy_r_oracle_ceiling",
        script=SCRIPT,
        hypothesis=(
            "A per-layer correction, however it is parameterised and however "
            "it is fitted, cannot do better than overwriting the state with "
            "truth at every layer boundary.  Measure that oracle directly: "
            "re-anchor E[relu(z^l)] to a 2e6-sample reference at every layer, "
            "leave the covariance to the closure, and read r at the same four "
            "depths.  This is jamesrahenry's sec 1(b) reset experiment "
            "(discourse 18097) on our networks, and it bounds the entire "
            "trajectory-calibration programme."),
        acceptance_bar=4.4,
        bar_metric="r_mean_oracle_L8",
        bar_direction="higher_is_better",
    )
    e.record(
        estimator="Gaussian closure with per-layer oracle re-anchoring",
        raw_final_layer_mse=by["mean"]["mse"]["32"],
        compute_ratio=0.0,
        seed=int(d["seeds"][0]),
        n_mlps=len(d["seeds"]),
        gt_samples=GT,
        b=by["mean"]["mse"]["32"] ** 0.5,
        v_eff=None,
        c=None,
        r_mean_oracle_L8=by["mean"]["r"]["8"],
        r_by_oracle={a: by[a]["r"] for a in by},
        seeds=[int(s) for s in d["seeds"]],
        notes=(
            "THE CEILING, and it is the result that decides the direction.  A "
            "PERFECT per-layer mean correction reaches r = "
            f"{by['mean']['r']['8']:.2f}/{by['mean']['r']['16']:.2f}/"
            f"{by['mean']['r']['24']:.2f}/{by['mean']['r']['32']:.2f} at "
            "L = 8/16/24/32.  At L = 8 that is BELOW the 4.4 break-even and "
            "at L = 16 it is level with it; only L = 24 and L = 32 have "
            "oracle headroom, and there it is 5.5-5.9x, i.e. the payoff at "
            "the ceiling is ~2.5x, not the 5.4x that r = 10 would buy.  Two "
            "control rows reproduce jamesrahenry's sec 1(b) mechanism for "
            "mechanism on our networks: re-anchoring the post-ReLU VARIANCE "
            f"to truth is r = {by['postvar']['r']['32']:.2f} at L=32, i.e. it "
            "makes the chain THREE TIMES WORSE, and re-anchoring mean and "
            f"variance together is only {by['mean+postvar']['r']['32']:.2f} "
            "against the mean alone's "
            f"{by['mean']['r']['32']:.2f}.  The covariance error is what "
            "compensates the mean error downstream; fixing it destroys the "
            "compensation.  His numbers on the official nets: plain chain "
            "6.0e-5, mean-only reset 1.64e-6 (r = 6.05) -- we measure 5.85."),
    )


def record_fit(tag, name, hypothesis, bar, notes_extra=""):
    d = jload(f"fit_{tag}.json")
    if d is None:
        return
    rows = {int(r["L"]): r for r in d["rows"]}
    rmin = min(r["r"] for r in rows.values())
    e = Experiment(
        name=name,
        script=SCRIPT,
        hypothesis=hypothesis,
        acceptance_bar=bar,
        bar_metric="r_min",
        bar_direction="higher_is_better",
    )
    e.record(
        estimator=f"trajectory-calibrated closure, pack={d['pack']} "
                  f"src={d['src']} cum={d['cum']}",
        raw_final_layer_mse=rows[32]["mse_traj"],
        compute_ratio=0.0,
        seed=int(d["eval"][0]),
        n_mlps=len(d["train"]) + len(d["eval"]),
        gt_samples=GT,
        b=rows[32]["mse_traj"] ** 0.5,
        v_eff=None,
        c=None,
        r_min=rmin,
        r_by_L={L: rows[L]["r"] for L in rows},
        rms_by_L={L: rows[L]["mse_traj"] ** 0.5 for L in rows},
        rms_gauss_by_L={L: rows[L]["mse_gauss"] ** 0.5 for L in rows},
        n_train=len(d["train"]), n_eval=len(d["eval"]),
        train_seeds=d["train"], eval_seeds=d["eval"],
        pack=d["pack"], src=d["src"], cum=d["cum"], lam=d["lam"],
        notes=notes_extra,
    )


if __name__ == "__main__":
    record_baseline()
    record_ceiling()
    for tag, nm, hyp, bar, note in json.loads(
            Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else []:
        record_fit(tag, nm, hyp, bar, note)
    print("recorded")
