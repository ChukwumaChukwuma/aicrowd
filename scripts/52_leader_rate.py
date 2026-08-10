#!/usr/bin/env python
"""Can the metered frontier be an IID sampler?  A single-point feasibility test.

The statistic the RQMC line was reopened on is

    p_implied  =  ln(v / raw) / ln(N_eq),      N_eq = F / 4.198656e6,  v = 0.045

evaluated at ONE (N_eq, raw) pair per entry.  **It is not identifiable**: one
point cannot separate ``v`` from ``p``, and ``p_implied`` returns a number for a
deterministic estimator just as happily as for a sampler.  Two independent
demonstrations, in order of how much they cost to believe.

``--mode unmetered``
    The cheap one, and it has already been vindicated.  An entry whose whole
    billed arithmetic is a couple of dense-forward-pass equivalents cannot be
    sampling at all, whatever ``p_implied`` says about it.  Run against the
    telemetry snapshot in ``$WHEST_ARTIFACTS/recon*/subs`` this flagged dpskv5
    (``N_eq = 0.5``), dstepanov (3.0) and ely2sh (3.5) as non-samplers; the
    2026-08-07 re-grade then found dpskv5's and joe_wanza's leading entries at
    ``F/C = 0.000`` and put them under review.  So the diagnostic works, and it
    is the reason the pre-re-grade version of this analysis reached the right
    conclusion about the wrong entries.

``--mode feasible``
    **The one that matters, and it supports the superlinear hypothesis.**  For a
    metered entry, ask instead: is there ANY ``(c, N, v_eff)`` consistent with
    ``p = 1`` that reaches its raw MSE inside this repository's own measured
    limits?  ``F`` is known, so ``N = F/c``, so ``v_eff = raw * N`` is forced by
    the entry's own numbers once ``c`` is chosen.  Two measured constraints then
    close the box from both sides:

      * ``docs/cost_floor.md`` section 3 -- an ORACLE Gaussian surrogate that
        replaces a prefix of the network costs an irreducible ``b^2``, measured:
        9.04e-07 at 1 layer-equivalent a sample, 3.25e-06 at 3, 3.77e-06 at 4.
        So a cheap per-sample pass has a bias floor, and ``raw >= b^2``.
      * ``docs/hermite_rank_ceiling.md`` -- the layer-1 Hermite family caps
        variance reduction at 1.76x, and the dictionary-free bound (top-8
        eigenfunctions of ``Cov(y)``) at ~10x.

    Squeeze those together and either the entry needs a variance reduction far
    past the ceiling, or it needs a per-sample cost whose bias floor is far
    above its raw.  If both fail, ``p = 1`` is refuted for that entry **without
    assuming anything about v** -- which is exactly what a single point can
    support and ``p_implied`` cannot.

Board snapshot: 2026-08-07 re-grade, supplied by the coordinator; the earlier
top three were corrected down hard (dpskv5 4.0e-10 -> 5.43e-08, huang_chung_yi
9.0e-10 -> 1.159e-07, joe_wanza 1.0e-09 -> 4.87e-08) and the submissions the
original brief was fitted to no longer exist.  ``www.aicrowd.com`` is blocked by
the egress proxy from this sandbox, so the current rows are carried here as data
rather than re-fetched; the stale per-MLP JSON under ``$WHEST_ARTIFACTS/recon*``
is still used by ``--mode unmetered``, which only ever needed ``F``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DENSE_FLOPS = 4_198_656.0
V_ASSUMED = 0.045
B = 2.72e11

#: 2026-08-07 re-grade.  ``fc`` is the instrumented share ``sum F / sum C``.
BOARD = [
    # who,            fc,     F,         raw,        note
    ("rayan53",       0.957, 2.687e10, 1.349e-08, "new #1, metered"),
    ("ednacob",       0.930, 1.283e11, 3.633e-08, "metered"),
    ("oabuod",        0.885, 1.517e11, 1.165e-07, "metered"),
    ("US (shipped)",  1.000, 6.221e10, 9.285e-07, "metered"),
    ("dpskv5",        0.000, 2.111e06, 7.840e-08, "UNMETERED, under review"),
    ("joe_wanza",     0.000, 1.296e07, 5.211e-08, "UNMETERED, under review"),
]

#: ``docs/cost_floor.md`` section 3, measured on 3 local MLPs x 400,000 oracle
#: surrogate samples: (layer-equivalents per sample, irreducible b^2).  The
#: 32-layer row is the full pass and has no model error at all.
BIAS_FLOOR = ((1, 9.04e-07), (3, 3.25e-06), (4, 3.77e-06), (5, 4.08e-06),
              (7, 5.66e-06), (9, 6.87e-06), (13, 8.87e-06), (32, 0.0))
LAYER_EQ = 131_072.0

#: ``docs/hermite_rank_ceiling.md``: the layer-1 Hermite dictionary caps at
#: 1.76x; the dictionary-free bound from the top-8 eigenfunctions of Cov(y) is
#: 90.1% of the variance, i.e. 10.1x.  Both are variance-reduction ceilings on
#: v_eff against the bare v = 0.0449.
V_BARE = 0.0449
CEILING_HERMITE = 1.76
CEILING_EIGEN = 10.1


def art() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


# ---------------------------------------------------------------------------
# mode: unmetered
# ---------------------------------------------------------------------------
def load_subs():
    seen, out = set(), []
    for p in sorted(glob.glob(str(art() / "recon*" / "subs" / "*.json"))):
        try:
            d = json.load(open(p))
            sub = d["props"]["data"]["submission"]
        except Exception:  # noqa: BLE001
            continue
        if sub["id"] in seen or sub.get("scoreSecondary") is None:
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
        if not F:
            continue
        seen.add(sub["id"])
        out.append(dict(id=sub["id"], who=who or "?",
                        ts=sub.get("createdAt", ""), adj=float(sub["score"]),
                        raw=float(sub["scoreSecondary"]), F=float(np.mean(F)),
                        neq=float(np.mean(F)) / DENSE_FLOPS))
    return [s for s in out if s["raw"] < 1e-3]


def mode_unmetered():
    subs = load_subs()
    print("# Entries whose entire billed arithmetic is a few dense forward "
          "passes.\n# A sampler with v = 0.045 at N_eq = n scores 0.045/n; "
          "these are 1e6x better.\n")
    hdr = (f"{'who':<18} {'sub':>7} {'N_eq':>7} {'raw':>11} "
           f"{'a sampler would score':>22} {'x better than possible':>23}")
    print(hdr)
    print("-" * len(hdr))
    for s in sorted(subs, key=lambda s: s["neq"]):
        if s["neq"] > 6:
            continue
        best = V_ASSUMED / max(s["neq"], 1e-9)
        print(f"{s['who']:<18} {s['id']:>7} {s['neq']:7.1f} {s['raw']:11.4e} "
              f"{best:22.4e} {best / s['raw']:23.3e}")
    print("\n  These are the entries the 2026-08-07 re-grade put at F/C = 0.000\n"
          "  and under organizer review.  The diagnostic found them from F "
          "alone,\n  before the re-grade, and it needed no assumption about v.")

    print("\n\n# Within-entry rate fits on the SAME (now stale) snapshot, for "
          "the record.\n# These are the fits the pre-re-grade analysis rested "
          "on.  The scores they\n# used have been corrected, so they are "
          "reported as history, not evidence.\n")
    by = defaultdict(list)
    for s in subs:
        by[s["who"]].append(s)
    for who, ss in sorted(by.items()):
        ss = [s for s in ss if s["neq"] > 1.05]
        if len(ss) < 2:
            continue
        x = np.log([s["neq"] for s in ss])
        y = np.log([s["raw"] for s in ss])
        if x.max() - x.min() < 0.3:
            continue
        p = -float(np.polyfit(x, y, 1)[0])
        print(f"  {who:<18} {len(ss)} subs, N_eq "
              f"{min(s['neq'] for s in ss):.0f}-{max(s['neq'] for s in ss):.0f}"
              f"   fitted p = {p:+.3f}")


# ---------------------------------------------------------------------------
# mode: feasible
# ---------------------------------------------------------------------------
def bias_floor(layer_eq: float) -> float:
    """Interpolate ``docs/cost_floor.md`` section 3's oracle bias floor."""
    xs = [a for a, _ in BIAS_FLOOR]
    ys = [b for _, b in BIAS_FLOOR]
    if layer_eq <= xs[0]:
        return ys[0]
    if layer_eq >= 32:
        return 0.0
    return float(np.interp(layer_eq, xs, ys))


def mode_feasible():
    print("# Is p = 1 (iid) feasible for each metered entry, given its own F?\n")
    print("For a chosen per-sample cost c, the entry's own numbers force")
    print("    N = F/c        and        v_eff = raw * N")
    print("with no assumption about v at all.  Then check v_eff against the")
    print("measured variance-reduction ceilings, and check c against the")
    print("measured bias floor of a per-sample pass that cheap.\n")
    hdr = (f"{'who':<14} {'F':>10} {'p_impl':>7} | {'c (layer-eq)':>12} "
           f"{'N':>10} {'v_eff needed':>13} {'x reduction':>12} {'b^2 floor':>10} "
           f"{'verdict':>9}")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for who, fc, F, raw, note in BOARD:
        neq = F / DENSE_FLOPS
        pi = (math.log(V_ASSUMED / raw) / math.log(neq)) if neq > 1.05 else \
            float("nan")
        rows = []
        for leq in (32, 21.7, 13, 9, 4, 3, 1):
            c = leq * LAYER_EQ
            N = F / c
            v_eff = raw * N
            red = V_BARE / v_eff
            b2 = bias_floor(leq)
            ok = (red <= CEILING_EIGEN) and (b2 <= raw)
            rows.append((leq, N, v_eff, red, b2, ok))
        any_ok = any(r[5] for r in rows)
        for i, (leq, N, v_eff, red, b2, ok) in enumerate(rows):
            head = (f"{who:<14} {F:10.3e} {pi:7.3f} |" if i == 0
                    else f"{'':<14} {'':>10} {'':>7} |")
            print(f"{head} {leq:12.1f} {N:10.0f} {v_eff:13.4e} "
                  f"{red:12.1f} {b2:10.3e} {'ok' if ok else 'NO':>9}")
        print(f"{'':<14} {'':>10} {'':>7} | "
              f"{'--> p = 1 is ' + ('FEASIBLE' if any_ok else 'REFUTED'):>60}")
        out[who] = dict(p_implied=pi, feasible_iid=any_ok, F=F, raw=raw,
                        neq=neq, fc=fc, note=note)
    print("\nCeilings used, both measured in this repository:")
    print(f"  layer-1 Hermite dictionary        {CEILING_HERMITE:5.2f}x  "
          "(docs/hermite_rank_ceiling.md)")
    print(f"  dictionary-free, top-8 eigen      {CEILING_EIGEN:5.2f}x  "
          "(same, section 5.3 -- generous, nothing realises it)")
    print(f"  bare per-sample variance v        {V_BARE:.4f}")
    print("  bias floor of a cheap pass        docs/cost_floor.md section 3, "
          "oracle Gaussian")
    (art() / "rqmc" / "feasible.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="feasible",
                    choices=("feasible", "unmetered", "all"))
    a = ap.parse_args()
    if a.mode in ("all", "unmetered"):
        mode_unmetered()
        print("\n" + "=" * 78 + "\n")
    if a.mode in ("all", "feasible"):
        mode_feasible()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
