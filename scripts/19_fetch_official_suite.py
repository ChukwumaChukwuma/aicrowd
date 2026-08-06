#!/usr/bin/env python
"""Build a :class:`whestfloor.suite.Suite` from the OFFICIAL WhestBench ground truth.

Why this matters
----------------
Every suite in this repository is baked locally, and the best of them reaches
~1.2e6 reference samples per MLP.  Its reference therefore carries noise
``v/1.2e6 ~ 4e-8``, three hundred times the true error we are chasing, which is
why ``whestfloor.harness`` has to run the two-half unbiased estimator at all.

The official dataset is baked at ``N = 1e9``.  Its reference noise is
``v/1e9 ~ 5e-11`` — the leaderboard's own floor.  Against this suite the plain
``final_layer_mse`` **is** the leaderboard number; no unbiasing, no correction.

How the data is obtained
------------------------
``https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026``
revision ``v1-phase1``, config ``default``, split ``mini``: 100 MLPs, width
256, depth 32, in three ~300 MB Parquet shards.

We download ~1.9 MB of each shard, not 300 MB.  Two facts make that possible:

1. ``weights`` is 99.4% of the bytes and we do not need it.  Weights are a
   deterministic function of ``mlp_seed`` under seed protocol 3.0 — see
   :mod:`whestfloor.official_seeds`, and see ``--verify-rows`` below, which
   proves it bit-for-bit against the file itself.
2. Parquet is columnar and HF serves ``Range:`` on ``/resolve/`` URLs, so
   :mod:`whestfloor.parquet_lite` reads the 5.5 KB footer, learns the byte
   extent of the four columns we want, and fetches only those.

(HF's ``datasets-server`` cannot serve this dataset at all: the shards use a
single ~309 MB row group, over the server's 300 MB scan limit, so ``/rows`` and
``/first-rows`` both fail with ``Scan size limit exceeded`` regardless of
``columns=``.  The ``refs/convert/parquet`` branch is byte-identical and fails
the same way.)

The single-reference caveat
---------------------------
``Suite`` is built around two *independent* reference halves so that
``mean((p-a)(p-b))`` estimates the true MSE with the reference's own variance
removed.  The official bake gives ONE reference.  This script therefore sets
``gt_a = gt_b = official``, at which point the cross-product degenerates:

    mean((p-a)(p-b)) == mean((p-g)^2) == the ordinary, biased MSE

so ``unbiased_true_mse`` on this suite is **not** an unbiased estimate of
anything, it is just the raw MSE printed twice, and ``leaderboard_equivalent``
(which adds ``v/1e9`` to it) would double-count the reference noise.  The suite
name says so, ``Suite.seed_protocol`` says so, and this script prints it in
red-flag terms at the end.  Read ``raw_mse``; ignore the other two columns.

Usage
-----
    python scripts/19_fetch_official_suite.py                  # full run
    python scripts/19_fetch_official_suite.py --mc-samples 0   # skip the slow MC
    python scripts/19_fetch_official_suite.py --refresh        # re-download

Idempotent: decoded columns are cached under ``$WHEST_ARTIFACTS/official_cache``
so reruns are offline and instant, and the suite is rebuilt deterministically
from the cache every time.  Nothing is written inside the repository.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import (  # noqa: E402
    DEPTH,
    DOC_QUOTED_AVG_VARIANCE,
    GT_SAMPLES_OFFICIAL,
    MEASURED_AVG_VARIANCE,
    MEASURED_AVG_VARIANCE_SE,
    WIDTH,
)
from whestfloor.mc import layer_means, make_mlp  # noqa: E402
from whestfloor.official_seeds import SEED_PROTOCOL, make_official_mlp  # noqa: E402
from whestfloor.parquet_lite import (  # noqa: E402
    ColumnChunk,
    RangeFetcher,
    _TCompact,
    parse_footer,
    read_column,
    read_columns_over_http,
    read_struct,
)
from whestfloor.suite import Suite  # noqa: E402

REPO = "aicrowd/arc-whestbench-public-2026"
SHARDS = [f"data/mini-{i:05d}-of-00003.parquet" for i in range(3)]

C_SEED = "mlp_seed"
C_ID = "mlp_id"
C_FINAL = "final_means.list.element"
C_ALL = "all_layer_means.list.element.list.element"
C_VAR = "avg_variance"
C_WEIGHTS = "weights.list.element.list.element.list.element"
WANTED = [C_ID, C_SEED, C_ALL, C_FINAL, C_VAR]

#: The suite name is the warning label.  It shows up in every report the
#: harness and the ledger emit, which is exactly where the caveat belongs.
SUITE_NAME = "official-mini-v1phase1-N1e9-SINGLEREF-UNBIASED_MSE_INVALID"


def artifacts_dir() -> Path:
    return Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve()


def shard_url(shard: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{shard}"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_columns(revision: str, cache: Path, *, refresh: bool) -> dict[str, np.ndarray]:
    """Return the four ground-truth columns for all 100 MLPs, in file order."""
    if cache.exists() and not refresh:
        z = np.load(cache)
        print(f"[fetch] cache hit: {cache}  ({cache.stat().st_size / 1e6:.2f} MB)")
        return {k: z[k] for k in z.files}

    ids, seeds, alm, avv, nrows = [], [], [], [], []
    t0 = time.time()
    for shard in SHARDS:
        url = shard_url(shard, revision)
        print(f"[fetch] {shard}")
        meta, cols = read_columns_over_http(url, WANTED)
        n = meta.num_rows
        a = cols[C_ALL].reshape(n, DEPTH, WIDTH)
        f = cols[C_FINAL].reshape(n, WIDTH)
        # Cheap structural check: whestbench computes final_mean as a copy of
        # all_layer_means[-1] (simulation.py), so these must be bit-equal.  If
        # our nested-list decode or reshape were wrong, this would not hold.
        if not np.array_equal(f, a[:, -1, :]):
            raise SystemExit(f"{shard}: final_means != all_layer_means[:, -1]; "
                             "the parquet decode is wrong, refusing to continue")
        print(f"    rows={n}  writer={meta.created_by}  "
              f"final_means == all_layer_means[:,-1]  OK")
        ids.append(cols[C_ID])
        seeds.append(cols[C_SEED])
        alm.append(a)
        avv.append(cols[C_VAR])
        nrows.append(n)

    out = {
        "mlp_id": np.concatenate(ids).astype(np.int64),
        "mlp_seed": np.concatenate(seeds).astype(np.int64),
        "all_layer_means": np.concatenate(alm).astype(np.float32),
        "avg_variance": np.concatenate(avv).astype(np.float64),
        "shard_rows": np.asarray(nrows, dtype=np.int64),
        "revision": np.asarray(revision),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"[fetch] {time.time() - t0:.1f}s -> cached {cache} "
          f"({cache.stat().st_size / 1e6:.2f} MB)")
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_bit_exact(revision: str, seeds: np.ndarray, shard_rows: np.ndarray,
                     rows: list[tuple[int, int]]) -> bool:
    """Fetch the real ``weights`` for a few rows and compare bit-for-bit.

    ``rows`` is a list of ``(shard_index, row_within_shard)``.  Each check costs
    one ~8.4 MB ranged GET: the shard stores exactly one row per data page, so
    a single page is one MLP's weights.  Nothing is written to disk.
    """
    print("\n[verify] bit-exact weight derivation (gold standard)")
    per_row = DEPTH * WIDTH * WIDTH
    offsets = np.concatenate([[0], np.cumsum(shard_rows)])
    ok = True
    for shard_i, row in rows:
        url = shard_url(SHARDS[shard_i], revision)
        f = RangeFetcher(url, verbose=False)
        meta = parse_footer(f.get(-1, 1 << 16))
        col = meta.columns[C_WEIGHTS]
        # Walk page headers (256 B each) until we reach the page holding `row`.
        off, seen = col.data_offset, 0
        while True:
            r = _TCompact(f.get(off, 256), 0)
            ph = read_struct(r)
            nv = ph[5][1]
            if seen <= row * per_row < seen + nv:
                break
            off += r.p + ph[3]
            seen += nv
        if seen != row * per_row or nv != per_row:
            raise SystemExit(f"weights page boundaries are not one-per-row "
                             f"(page starts at value {seen}, holds {nv})")
        page_len = r.p + ph[3]
        one = ColumnChunk(path=col.path, physical=col.physical, codec=col.codec,
                          num_values=per_row, start=off, length=page_len,
                          dict_offset=None, data_offset=off,
                          max_def=col.max_def, max_rep=col.max_rep)
        official = read_column(f.get(off, page_len), one).reshape(DEPTH, WIDTH, WIDTH)

        gi = int(offsets[shard_i]) + row
        s = int(seeds[gi])
        mine = np.stack(make_official_mlp(WIDTH, DEPTH, s))
        same = bool(np.array_equal(official, mine))
        # control: the local convention, which must NOT match
        wrong = np.stack(make_mlp(WIDTH, DEPTH, s))
        wrong_same = bool(np.array_equal(official, wrong))
        ok &= same and not wrong_same
        print(f"    shard {shard_i} row {row:>2} (global {gi:>2}) seed={s}")
        print(f"        protocol 3.0 : identical={same}  "
              f"max|diff|={np.abs(official.astype(np.float64) - mine).max():.3g}  "
              f"({per_row:,} float32 values, {page_len / 1e6:.2f} MB fetched)")
        print(f"        local  seed  : identical={wrong_same}  "
              f"max|diff|={np.abs(official.astype(np.float64) - wrong).max():.4g}"
              f"   <- control, must differ")
        del official, mine, wrong
    print(f"    => {'PASS' if ok else 'FAIL'}")
    return ok


def verify_monte_carlo(seeds: np.ndarray, alm: np.ndarray, avv: np.ndarray,
                       *, n_mlps: int, n_samples: int, seed_base: int) -> bool:
    """Re-derive weights from the seed, run our own MC, compare to the official means.

    The expected disagreement is our own MC noise: per neuron
    ``sqrt(v / n_samples)``, since the official reference's ``sqrt(v/1e9)`` is
    ~250x smaller and contributes nothing.  We also report a z-score, because
    the naive ``RMS^2 ~ v/n`` has a large spread: final-layer noise is strongly
    correlated across neurons (the 256 of them share one input vector), so
    ``sd(RMS^2) = sqrt(2 tr(C^2)) / (width * n)`` is order 50% of the mean, not
    the ``sqrt(2/width)`` = 9% an independence assumption would suggest.

    Pass/fail is deliberately the O(1) test the task of this check demands: a
    wrong derivation puts us on a *different network* and the RMS lands near
    ``sqrt(2 v)`` ~ 1, four orders of magnitude out, not 2x out.  The z-score
    is reported as a diagnostic, not a gate: ``eps'eps`` is a low-effective-rank
    weighted chi-square, so it is strongly right-skewed and a +3 is ordinary.
    Bit-exactness (``--verify-rows``) is the check that actually settles the
    derivation; this one confirms the *ground truth column* is what we think.
    """
    print(f"\n[verify] Monte-Carlo agreement, {n_samples:,} samples/MLP "
          f"vs the official N={GT_SAMPLES_OFFICIAL:.0e} reference")
    ok = True
    for i in range(n_mlps):
        s = int(seeds[i])
        t0 = time.time()
        w = make_official_mlp(WIDTH, DEPTH, s)
        mine, var, cov = layer_means(w, n_samples, seed=seed_base + i,
                                     want_var=True, want_cov=True)
        del w
        d = mine[-1] - alm[i, -1].astype(np.float64)
        mse = float(np.mean(d ** 2))
        exp = float(avv[i]) / n_samples                       # E[RMS^2]
        C = cov / n_samples
        sd = float(np.sqrt(2.0 * np.sum(C * C))) / WIDTH      # sd(RMS^2)
        z = (mse - exp) / sd
        all_rms = float(np.sqrt(np.mean((mine - alm[i].astype(np.float64)) ** 2)))
        good = mse < 100.0 * exp          # O(1) disagreement is the failure mode
        ok &= good
        flag = "ok" if good else "FAIL (O(1) disagreement -- derivation is WRONG)"
        if good and abs(z) > 3.0:
            flag = "ok (high z; see docstring -- chi-square tail, not a defect)"
        print(f"    MLP {i} seed={s}")
        print(f"        final-layer RMS diff {np.sqrt(mse):.3e}   "
              f"expected {np.sqrt(exp):.3e}   z({mse:.3e} vs {exp:.3e} "
              f"+-{sd:.1e}) = {z:+.2f}  {flag}")
        print(f"        all-layer   RMS diff {all_rms:.3e}   "
              f"avg_variance ours {var.mean():.5f} vs official {avv[i]:.5f}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
    print(f"    => {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="v1-phase1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the column cache and re-download")
    ap.add_argument("--verify-rows", type=int, default=3,
                    help="bit-exact weight checks (~8.4 MB each); 0 to skip")
    ap.add_argument("--mc-samples", type=int, default=2_000_000,
                    help="Monte-Carlo samples per verified MLP; 0 to skip")
    ap.add_argument("--mc-mlps", type=int, default=3)
    ap.add_argument("--mc-seed-base", type=int, default=90_000)
    args = ap.parse_args()

    art = artifacts_dir()
    out = Path(args.out) if args.out else art / "suites" / "official_mini.npz"
    cache = art / "official_cache" / f"mini_{args.revision}_columns.npz"

    print(f"# official WhestBench ground truth: {REPO} @ {args.revision}, "
          f"config=default split=mini")
    print(f"# target: {out}\n")

    cols = fetch_columns(args.revision, cache, refresh=args.refresh)
    seeds = cols["mlp_seed"]
    ids = cols["mlp_id"]
    alm = cols["all_layer_means"]
    avv = cols["avg_variance"]

    n = len(seeds)
    print(f"\n[check] rows={n}  shape={alm.shape}  "
          f"unique seeds={len(set(seeds.tolist()))}  "
          f"mlp_id contiguous 0..{n - 1}={np.array_equal(ids, np.arange(n))}")
    if n != 100 or len(set(seeds.tolist())) != 100:
        raise SystemExit(f"expected 100 distinct MLPs, got {n}")
    if alm.shape != (100, DEPTH, WIDTH):
        raise SystemExit(f"unexpected ground-truth shape {alm.shape}")
    if not np.all(np.isfinite(alm)):
        raise SystemExit("ground truth contains non-finite values")
    if not np.array_equal(ids, np.arange(n)):
        raise SystemExit("rows are not in mlp_id order; the shard concatenation "
                         "would mismatch seeds against ground truth")

    ok = True
    if args.verify_rows > 0:
        # spread the checks over the shards so shard handling is covered too
        plan = [(0, 1), (1, 5), (2, 11)][:args.verify_rows]
        ok &= verify_bit_exact(args.revision, seeds, cols["shard_rows"], plan)
    if args.mc_samples > 0 and args.mc_mlps > 0:
        ok &= verify_monte_carlo(seeds, alm, avv, n_mlps=args.mc_mlps,
                                 n_samples=args.mc_samples,
                                 seed_base=args.mc_seed_base)
    if not ok:
        raise SystemExit("\nVERIFICATION FAILED — refusing to write a suite. "
                         "The seed derivation or the parquet decode is wrong.")

    # ---------------- avg_variance, the number the challenge docs get wrong --
    m, sd = float(avv.mean()), float(avv.std(ddof=1))
    se = sd / np.sqrt(n)
    print(f"\n[avg_variance] official, N=1e9, {n} MLPs")
    print(f"    mean = {m:.6f} +- {se:.6f} (se)      sd = {sd:.6f}  CV = {sd / m:.3f}")
    print(f"    range [{avv.min():.5f}, {avv.max():.5f}]   "
          f"median {np.median(avv):.5f}")
    print(f"    our local generation : {MEASURED_AVG_VARIANCE} +- "
          f"{MEASURED_AVG_VARIANCE_SE}   -> "
          f"{abs(m - MEASURED_AVG_VARIANCE) / np.hypot(se, MEASURED_AVG_VARIANCE_SE):.1f} sigma")
    print(f"    challenge docs quote : {DOC_QUOTED_AVG_VARIANCE}          -> "
          f"{abs(m - DOC_QUOTED_AVG_VARIANCE) / se:.0f} sigma  (refuted)")
    print(f"    official raw MSE floor v/1e9 = {m / GT_SAMPLES_OFFICIAL:.4e}")

    # ---------------- build the suite ---------------------------------------
    gt = alm.astype(np.float64)
    # SINGLE REFERENCE.  Both halves are the same array on purpose; see the
    # module docstring.  Consequences, in one place:
    #   * harness.unbiased_mse computes mean((p-a)*(p-b)) which with a == b is
    #     identically mean((p-g)^2) — the ordinary BIASED MSE.  The
    #     `unbiased_true_mse` column of scripts/12_evaluate.py is therefore a
    #     duplicate of `raw_mse` on this suite and means nothing on its own.
    #   * harness.leaderboard_equivalent adds v/1e9 to that column, which
    #     DOUBLE-COUNTS the reference noise here.  Ignore `lb_equiv_adj`.
    #   * `raw_mse` is the number to read: the reference already has 1e9
    #     samples, so it is the leaderboard quantity with no correction at all.
    gt_a = gt
    gt_b = gt
    # n_per_half is chosen so that Suite.gt_samples == 2 * n_per_half reports
    # the TRUE 1e9, rather than pretending there are two 1e9-sample halves.
    n_per_half = GT_SAMPLES_OFFICIAL // 2
    # The official parquet stores only the scalar mean-over-neurons variance,
    # not a per-neuron vector, so final_var is that scalar broadcast: its mean
    # is exact, its across-neuron structure is not real.  Its only consumer is
    # the standard error of the (degenerate) unbiased estimator.
    final_var = np.repeat(avv[:, None], WIDTH, axis=1)

    suite = Suite(
        name=SUITE_NAME,
        width=WIDTH,
        depth=DEPTH,
        mlp_seeds=[int(s) for s in seeds],
        gt_a=gt_a,
        gt_b=gt_b,
        n_per_half=n_per_half,
        gt_seed_a=[0] * n,          # not ours; baked by the organisers
        gt_seed_b=[0] * n,
        final_var=final_var,
        final_cov=None,             # not published
        seed_protocol="official",   # <- weights come from official_seeds.py
    )
    suite.save(out)
    print(f"\n[suite] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"    name           {suite.name}")
    print(f"    n_mlps         {suite.n_mlps}   width {suite.width} depth {suite.depth}")
    print(f"    gt_samples     {suite.gt_samples:,}  (protocol {SEED_PROTOCOL})")
    print(f"    reference noise per neuron  v/N = {m / GT_SAMPLES_OFFICIAL:.4e}")
    print(f"    vs best local suite (1.2e6) v/N = {m / 1_200_000:.4e}  "
          f"({GT_SAMPLES_OFFICIAL / 1_200_000:.0f}x lower here)")

    rt = Suite.load(out)
    assert rt.seed_protocol == "official" and rt.n_mlps == 100
    assert np.array_equal(np.stack(rt.weights(1)),
                          np.stack(make_official_mlp(WIDTH, DEPTH, int(seeds[1]))))
    print("    round-trip load + weights() reproduce the official MLPs: OK")

    print("\n" + "!" * 78)
    print("!! SINGLE-REFERENCE SUITE.  gt_a == gt_b, so the unbiased-MSE")
    print("!! estimator is INVALID here: `unbiased_true_mse` degenerates to the")
    print("!! plain biased MSE and `lb_equiv_adj` double-counts v/1e9.")
    print("!! Read `raw_mse`.  With a 1e9-sample reference it IS the")
    print("!! leaderboard-comparable number, uncorrected.")
    print("!" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
