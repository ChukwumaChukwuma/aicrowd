#!/usr/bin/env python
"""ADVERSARIAL: audit the ablation and the kappa_3 validation.

Three checks the claim rests on that nobody has tested:

A. **Is damp=0 a true ablation?**  The claim is that ``damp=0`` "reproduces the
   uncorrected number exactly through the identical code path".  Verify that
   (i) the k3 matmuls really are executed (FLOP counts must differ from
   mehler_k4), and (ii) the damp=0 output is bitwise identical to plain
   ``cov_prop_mehler``.  If (ii) holds, the "ablation" is a tautology: it can
   only ever reproduce mehler_k4, so it is evidence about the plumbing, not
   about the mechanism.  Also note the ledger ablates at ``umax=2`` while the
   shipped estimator runs ``umax=1``.

B. **Does the shipped submission match the researched kernel?**  Run
   ``submission/estimator.py`` against ``cov_prop_edgeworth(kmax=4, umax=1)``.

C. **What does scripts/13 actually validate?**  It draws z from a *Gaussian*
   with the measured (m, cov) in BOTH the star formula and the brute-force
   reference, so it validates the diagram truncation only -- it says nothing
   about whether the kappa_3 the estimator actually feeds in is right, because
   the estimator supplies its own propagated, non-exact (m, cov, rho).  Here
   the star kappa_3 is compared against a brute-force kappa_3 of the *true*
   network pre-activation, which is the quantity the Edgeworth term needs.
   The gap between the two is the part scripts/13 does not measure.

   scripts/13 also has a wrap-around: at the last requested layer it silently
   substitutes ``W[0]`` for the (nonexistent) next weight matrix.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor import kernels  # noqa: E402
from whestfloor.cumulants import kappa3_brute, kappa3_star  # noqa: E402
from whestfloor.harness import run_billed  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402


# --------------------------------------------------------------------------
def check_A():
    print("=" * 76)
    print("A. is damp=0 a true ablation?")
    print("=" * 76)
    W = make_mlp(256, 32, 4321)
    meh, f_meh, _ = run_billed(functools.partial(kernels.cov_prop_mehler,
                                                 kmax=4), W)
    for u in (1, 2):
        ab, f_ab, _ = run_billed(functools.partial(
            kernels.cov_prop_edgeworth, kmax=4, umax=u, damp=0.0), W)
        same = np.array_equal(meh, ab)
        print(f"  damp=0 umax={u}: flops {f_ab:,} vs mehler_k4 {f_meh:,} "
              f"(+{f_ab - f_meh:,});  output bitwise identical to "
              f"mehler_k4: {same};  max|diff| {np.max(np.abs(meh - ab)):.3e}")
    print("  => the extra matmuls ARE billed, but the ablation's OUTPUT is "
          "mehler_k4 by construction:")
    print("     mu - (0/6)*k3*(...) == mu whenever k3 is finite, so the "
          "'exact reproduction' is arithmetic,")
    print("     not evidence that the correction is the mechanism it is "
          "claimed to be.")
    print("  NOTE: the ledger ablation uses umax=2; the shipped estimator "
          "uses umax=1.")
    print()

    # the one case where damp=0 is NOT a no-op: non-finite k3 -> 0*inf = nan
    print("  is `damp*0` safe? 0 * non-finite = nan.  Probing whether k3 can "
          "go non-finite:")
    n, d = 32, 12
    rng = np.random.default_rng(3)
    Wbad = [(rng.standard_normal((n, n)) * 3e2).astype(np.float32)
            for _ in range(d)]
    try:
        out0, _f, _ = run_billed(functools.partial(
            kernels.cov_prop_edgeworth, kmax=4, umax=1, damp=0.0), Wbad)
        outm, _f, _ = run_billed(functools.partial(kernels.cov_prop_mehler,
                                                   kmax=4), Wbad)
        nb = int(np.sum(~np.isfinite(out0)))
        nm = int(np.sum(~np.isfinite(outm)))
        print(f"     large-scale weights: non-finite in damp=0 output {nb}, "
              f"in mehler_k4 output {nm} -> "
              f"{'ABLATION DIVERGES' if nb != nm else 'no divergence'}")
    except Exception as e:  # noqa: BLE001
        print(f"     raised {type(e).__name__}: {e}")
    print()


# --------------------------------------------------------------------------
def check_B():
    print("=" * 76)
    print("B. submission/estimator.py vs the researched kernel")
    print("=" * 76)
    import importlib.util  # noqa: PLC0415
    import types  # noqa: PLC0415
    if "whestbench" not in sys.modules:
        wb = types.ModuleType("whestbench")
        wb.BaseEstimator = type("BaseEstimator", (), {})
        sys.modules["whestbench"] = wb
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "shipped_estimator", root / "submission" / "estimator.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    print(f"  shipped constants: KMAX={sub.KMAX} UMAX={sub.UMAX} "
          f"VAR_FLOOR={sub.VAR_FLOOR}")
    W = make_mlp(256, 32, 4321)

    class _M:
        width = 256
        depth = 32

    import flopscope as flops  # noqa: PLC0415
    import flopscope.numpy as fnp  # noqa: PLC0415
    m = _M()
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as ctx:
        m.weights = [fnp.asarray(w) for w in W]
        e = sub.Estimator()
        out = np.asarray(e.predict(m, 272_000_000_000), dtype=np.float64)
        f_sub = int(ctx.flops_used)
    ref, f_ref, _ = run_billed(functools.partial(kernels.cov_prop_edgeworth,
                                                 kmax=4, umax=1, damp=1.0), W)
    print(f"  max|diff| {np.max(np.abs(out - ref)):.3e}   flops "
          f"{f_sub:,} vs {f_ref:,}")
    print()


# --------------------------------------------------------------------------
def check_C(width, depth, layers, n_state, n_brute, seed):
    print("=" * 76)
    print("C. what scripts/13 validates vs what the estimator needs")
    print("=" * 76)
    W = make_mlp(width, depth, seed)
    rng = np.random.default_rng(97)

    print(f"  width={width} depth={depth} n_brute={n_brute:,}")
    print("  layer  cap_vs_GAUSSIAN_ref  cap_vs_TRUE_ref   "
          "||k3_gauss||  ||k3_true||   ratio")
    print("  " + "-" * 76)
    for li in layers:
        # --- true joint state of x^{li} = relu(z^{li}) by direct simulation
        n = width
        s1 = np.zeros(n)
        g = np.zeros((n, n))
        S1 = np.zeros(n)
        S2 = np.zeros(n)
        S3 = np.zeros(n)
        Wn = W[li + 1] if li + 1 < depth else W[0]
        Wn64 = Wn.astype(np.float64)
        done, total, chunk = 0, n_brute, 4096
        while done < total:
            nb = min(chunk, total - done)
            x = rng.standard_normal((nb, n), dtype=np.float32)
            for lj, w in enumerate(W):
                z = x @ w
                if lj == li:
                    break
                x = np.maximum(z, np.float32(0.0))
            zf = z.astype(np.float64)
            s1 += zf.sum(axis=0)
            g += zf.T @ zf
            Sv = np.maximum(zf, 0.0) @ Wn64
            S1 += Sv.sum(axis=0)
            S2 += (Sv * Sv).sum(axis=0)
            S3 += (Sv ** 3).sum(axis=0)
            done += nb
        m = s1 / done
        cov = g / done - np.outer(m, m)
        m1, m2, m3 = S1 / done, S2 / done, S3 / done
        k3_true = m3 - 3 * m1 * m2 + 2 * m1 ** 3   # the REAL kappa_3

        sd = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        R = cov / np.outer(sd, sd)
        np.fill_diagonal(R, 1.0)

        # scripts/13's reference: brute force under a GAUSSIAN with that (m,cov)
        k3_gauss = kappa3_brute(Wn64, m, cov, n_state, 909 + li)
        k3_s = kappa3_star(Wn64, m, sd, R, umax=1)

        cap_g = 1.0 - np.linalg.norm(k3_s - k3_gauss) / np.linalg.norm(k3_gauss)
        cap_t = 1.0 - np.linalg.norm(k3_s - k3_true) / np.linalg.norm(k3_true)
        print(f"  {li + 1:5d}  {cap_g:18.3f}  {cap_t:15.3f}   "
              f"{np.linalg.norm(k3_gauss):11.4e}  "
              f"{np.linalg.norm(k3_true):10.4e}   "
              f"{np.linalg.norm(k3_true) / np.linalg.norm(k3_gauss):.3f}")
    print()
    print("  scripts/13 reports column 1.  The Edgeworth term needs column 2.")
    print("  scripts/13 also substitutes W[0] for the next weight matrix at "
          "the last layer it")
    print("  is asked for (kernels are indexed li+1, guarded by "
          "`if li+1 < depth else W[0]`), so the")
    print("  deepest row of its table is not a real layer transition.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--layers", type=str, default="1,3,5")
    ap.add_argument("--n-state", type=int, default=1_000_000)
    ap.add_argument("--n-brute", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip", type=str, default="")
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    if "A" not in skip:
        check_A()
    if "B" not in skip:
        check_B()
    if "C" not in skip:
        check_C(args.width, args.depth,
                [int(x) for x in args.layers.split(",")],
                args.n_state, args.n_brute, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
