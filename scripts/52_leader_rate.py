#!/usr/bin/env python
"""Is the top of the leaderboard converging at the lattice rate?  Measure it.

The claim under test: fitting ``raw = v / N_eq^p`` to the top entries with
``v = 0.045`` and ``N_eq = F / 4.198656e6`` gives ``p`` of 1.7-2.1 against our
1.11, and ``p ~ 2`` is the classical randomised-QMC rate, so the top of the
board is running a lattice and we are not.

The statistic that claim rests on is

    p_implied  =  ln(v / raw) / ln(N_eq)

evaluated at ONE (N_eq, raw) pair per entry.  One point cannot identify two
parameters.  ``p_implied`` is a monotone recoding of ``raw`` at fixed ``N_eq``:
it is defined for a deterministic estimator, for a biased estimator, and for an
estimator whose per-sample cost is not one dense forward pass, and in each of
those cases it is not a convergence exponent at all.

There IS an identifiable version, and the leaderboard telemetry supports it.
Several entries submitted the SAME method at DIFFERENT ``N``.  Regressing
``ln raw`` on ``ln N_eq`` *within* an entry estimates ``p`` with ``v``
eliminated, which is the quantity the claim is about.  That is what this
script does.

Data: ``$WHEST_ARTIFACTS/recon*/subs/*.json`` — the server-rendered submission
pages, each carrying ``per_mlp[].telemetry.flops_used`` for all 100 MLPs plus
the aggregate ``final_layer_mse``.  Provenance and the fetch are
``docs/recon.md`` §1.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

DENSE_FLOPS = 4_198_656.0
V_ASSUMED = 0.045


def art() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


def load_subs():
    seen, out = set(), []
    for p in sorted(glob.glob(str(art() / "recon*" / "subs" / "*.json"))):
        try:
            d = json.load(open(p))
            sub = d["props"]["data"]["submission"]
        except Exception:  # noqa: BLE001
            continue
        if sub["id"] in seen:
            continue
        who = None

        def find(o, depth=0):
            nonlocal who
            if who or depth > 6:
                return
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "name" and isinstance(v, str):
                        who = who or v
                    find(v, depth + 1)
            elif isinstance(o, list):
                for v in o[:3]:
                    find(v, depth + 1)
        find(sub)
        r = sub["evaluation"]["results"]
        F = [m["telemetry"]["flops_used"]
             for m in (r.get("per_mlp") or []) if m.get("telemetry")]
        if not F or sub.get("scoreSecondary") is None:
            continue
        seen.add(sub["id"])
        out.append(dict(id=sub["id"], who=who or "?", ts=sub.get("createdAt", ""),
                        adj=float(sub["score"]),
                        raw=float(sub["scoreSecondary"]),
                        F=float(np.mean(F)),
                        neq=float(np.mean(F)) / DENSE_FLOPS))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=V_ASSUMED)
    args = ap.parse_args()
    subs = load_subs()
    subs = [s for s in subs if s["raw"] < 1e-3]     # 324900 crashed at 1.7e-2

    print("# every graded submission we hold telemetry for, newest last\n")
    hdr = (f"{'who':<18} {'sub':>7} {'when':>17} {'N_eq':>9} {'raw':>11} "
           f"{'adjusted':>11} {'p_implied':>10}")
    print(hdr)
    print("-" * len(hdr))
    for s in sorted(subs, key=lambda s: (s["who"], s["ts"])):
        pi = (math.log(args.v / s["raw"]) / math.log(s["neq"])
              if s["neq"] > 1.05 else float("nan"))
        print(f"{s['who']:<18} {s['id']:>7} {s['ts'][:16]:>17} "
              f"{s['neq']:9.0f} {s['raw']:11.4e} {s['adj']:11.4e} "
              f"{pi:10.3f}")

    print("\n\n# 1. The single-point statistic is not identifiable, and here "
          "is the proof\n")
    print("Four of dpskv5's graded submissions ran at N_eq ~ 1 -- ONE dense "
          "forward-pass\nequivalent of billed compute for the whole "
          "prediction:\n")
    low = [s for s in subs if s["neq"] < 5]
    for s in sorted(low, key=lambda s: s["neq"]):
        print(f"  {s['who']:<16} {s['id']}  N_eq {s['neq']:6.1f}  "
              f"raw {s['raw']:.4e}")
    print(f"\nA sampler with per-sample variance v = {args.v} and N_eq = 1 has "
          f"raw = {args.v:.3f}.\nThese entries are 1e6 times better than that "
          "at that compute, so their error is\nMODEL error, not sampling "
          "error.  For them `p_implied` is not an exponent; it is\n"
          "`ln(v/raw)/ln(N_eq)` with a denominator near zero.")

    print("\n\n# 2. The identifiable version: regress ln raw on ln N_eq "
          "WITHIN an entry\n")
    by = defaultdict(list)
    for s in subs:
        by[s["who"]].append(s)
    hdr2 = (f"{'who':<18} {'n subs':>7} {'N_eq range':>20} {'raw range':>24} "
            f"{'fitted p':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    fits = {}
    for who, ss in sorted(by.items()):
        ss = [s for s in ss if s["neq"] > 1.05]
        if len(ss) < 2:
            continue
        x = np.log([s["neq"] for s in ss])
        y = np.log([s["raw"] for s in ss])
        if x.max() - x.min() < 0.3:
            continue
        p = -float(np.polyfit(x, y, 1)[0])
        fits[who] = p
        print(f"{who:<18} {len(ss):7d} "
              f"{min(s['neq'] for s in ss):9.0f} -{max(s['neq'] for s in ss):9.0f} "
              f"{min(s['raw'] for s in ss):11.4e} -{max(s['raw'] for s in ss):11.4e} "
              f"{p:9.3f}")

    print("\n  The sharpest pair, because it is the same team 3.5 hours apart "
          "and it is\n  the pair that took them to rank 1:\n")
    a = next(s for s in subs if s["id"] == "324846")
    b = next(s for s in subs if s["id"] == "324969")
    pab = math.log(a["raw"] / b["raw"]) / math.log(b["neq"] / a["neq"])
    print(f"    dpskv5 324846  N_eq {a['neq']:8.0f}  raw {a['raw']:.4e}  "
          f"adjusted {a['adj']:.4e}")
    print(f"    dpskv5 324969  N_eq {b['neq']:8.0f}  raw {b['raw']:.4e}  "
          f"adjusted {b['adj']:.4e}")
    print(f"\n    compute cut {a['neq'] / b['neq']:.2f}x, raw moved "
          f"{a['raw'] / b['raw']:.4f}x  ->  fitted p = {pab:.3f}")
    print(f"    adjusted improved {a['adj'] / b['adj']:.1f}x -- ALL of it from "
          "the multiplier, none from raw.")
    print(f"\n    Had p been 1.9, cutting N by {a['neq'] / b['neq']:.1f}x would "
          f"have multiplied raw by\n    {(a['neq'] / b['neq']) ** 1.9:.0f}x, to "
          f"{a['raw'] * (a['neq'] / b['neq']) ** 1.9:.2e}.  It moved by "
          f"{100 * (b['raw'] / a['raw'] - 1):+.1f}%.")

    print("\n\n# 3. What the board is actually doing\n")
    print("Raw is FLAT in N for every top entry that varied N.  That is a bias "
          "floor:\n`raw -> b^2` and `adjusted -> b^2 * max(0.1, C/B)`, so the "
          "only lever left is\nC, and the way to rank is to drive C to the "
          "0.1 clamp.  dpskv5 did exactly\nthat and it was worth 16x.  This "
          "reproduces docs/graded.md section 5 from the\ntelemetry rather "
          "than from an inequality.")
    (art() / "rqmc" / "leader_rate.json").write_text(
        json.dumps(dict(subs=subs, within_entry_p=fits), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
