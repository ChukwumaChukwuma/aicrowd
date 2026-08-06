#!/usr/bin/env python
"""ADVERSARIAL: numerical failure hunting in the shipped estimator.

Instruments a byte-identical re-implementation of ``submission/estimator.py``
(the kernel path, ``cov_prop_edgeworth``) and counts, per layer and per MLP:

  * how often ``var_pre`` is clamped at the 1e-12 floor;
  * how often ``rho`` is clipped at +-1 off the diagonal (a clip means the
    propagated covariance is not PSD-consistent with its own diagonal);
  * how often ``sig`` underflows or ``alpha`` is non-finite;
  * how often the correction produces a NEGATIVE predicted mean.  The target is
    E[relu(.)], which is >= 0 by construction; a negative prediction is a hard
    violation of the estimand's range, not a matter of taste;
  * whether the correction is ever larger in magnitude than the uncorrected
    mean (i.e. the "correction" dominates the thing it corrects);
  * any non-finite entry in the returned prediction.

Also builds deliberately hostile weight matrices -- rank-deficient, scaled to
underflow, scaled to overflow -- to see whether anything worse than a clamp
happens.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.mc import make_mlp  # noqa: E402
from whestfloor.relu_moments import relu_hermite_coeffs  # noqa: E402

VAR_FLOOR = 1e-12


def _relu_gauss_np(m, v, s):
    from scipy_free import ncdf, npdf  # placeholder, never imported
    raise RuntimeError


def _phi(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _Phi(x):
    # float64 erf via numpy-only rational approx is not needed: use math.erf
    import math
    f = np.vectorize(math.erf)
    return 0.5 * (1.0 + f(x / np.sqrt(2.0)))


def kappa3_star(w, a, rho, umax):
    G = [None] + [a[m][:, None] * w for m in range(1, 2 * umax + 1)]
    Rp = [None, rho]
    for u in range(2, umax + 1):
        Rp.append(Rp[u - 1] * rho)
    RG = [None] + [Rp[u] @ G[u] for u in range(1, umax + 1)]
    fact = [1.0]
    for i in range(1, 2 * umax + 2):
        fact.append(fact[-1] * i)
    acc = None
    for u in range(1, umax + 1):
        for v in range(1, umax + 1):
            t = np.sum(G[u + v] * RG[u] * RG[v], axis=0) / (fact[u] * fact[v])
            acc = t if acc is None else acc + t
    return 3.0 * acc


def hermite_coeffs(alpha, sig, ph, Ph, kmax):
    out = [None, sig * Ph]
    if kmax >= 2:
        s_phi = sig * ph
        h_prev = h = None
        for k in range(2, kmax + 1):
            j = k - 2
            if j == 0:
                hj = None
            elif j == 1:
                h_prev, h = None, alpha
                hj = h
            else:
                base = alpha * h
                hj = base if h_prev is None else base - float(j - 1) * h_prev
                h_prev, h = h, hj
            term = s_phi if hj is None else s_phi * hj
            out.append(term if k % 2 == 0 else -term)
    return out


def instrumented(weights, kmax=4, umax=1, damp=1.0):
    """Mirror of cov_prop_edgeworth in float32 numpy, with counters."""
    n = weights[0].shape[0]
    mu = np.zeros(n, dtype=np.float32)
    cov = np.eye(n, dtype=np.float32)
    prev = None
    st = dict(var_floor_hits=0, rho_clip_hi=0, rho_clip_lo=0, sig_zero=0,
              neg_mean=0, neg_mean_before=0, corr_dominates=0,
              nonfinite=0, n_cells=0, n_offdiag=0,
              max_corr_ratio=0.0, worst_neg=0.0, k3_nonfinite=0)
    rows = []
    for w in weights:
        w = np.asarray(w, dtype=np.float32)
        mu_pre = w.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w, w,
                            optimize=True).astype(np.float32)
        raw_var = np.diag(cov_pre)
        st["var_floor_hits"] += int(np.sum(raw_var <= VAR_FLOOR))
        st["n_cells"] += n
        var_pre = np.maximum(raw_var, VAR_FLOOR)
        sig = np.sqrt(var_pre)
        st["sig_zero"] += int(np.sum(sig == 0.0))
        alpha = mu_pre / sig
        ph = _phi(alpha.astype(np.float64))
        Ph = _Phi(alpha.astype(np.float64))
        mu = mu_pre * Ph + sig * ph
        ez2 = (mu_pre * mu_pre + var_pre) * Ph + mu_pre * sig * ph
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        st["neg_mean_before"] += int(np.sum(mu < 0.0))

        if prev is not None:
            k3 = kappa3_star(w.astype(np.float64), prev[0], prev[1], umax)
            st["k3_nonfinite"] += int(np.sum(~np.isfinite(k3)))
            delta = (damp / 6.0) * k3 * (mu_pre / (var_pre * sig)) * ph
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.abs(delta) / np.maximum(np.abs(mu), 1e-300)
            st["corr_dominates"] += int(np.sum(r > 1.0))
            st["max_corr_ratio"] = max(st["max_corr_ratio"],
                                       float(np.nanmax(r)))
            mu = mu - delta
        nneg = int(np.sum(mu < 0.0))
        st["neg_mean"] += nneg
        if nneg:
            st["worst_neg"] = min(st["worst_neg"], float(mu.min()))

        inv_sig = 1.0 / sig
        rho = cov_pre * np.outer(inv_sig, inv_sig)
        off = ~np.eye(n, dtype=bool)
        st["rho_clip_hi"] += int(np.sum(rho[off] > 1.0))
        st["rho_clip_lo"] += int(np.sum(rho[off] < -1.0))
        st["n_offdiag"] += int(off.sum())
        rho = np.clip(rho, -1.0, 1.0)

        a = hermite_coeffs(alpha, sig, ph, Ph, max(kmax, 2 * umax))
        acc = np.outer(a[1], a[1]) * rho
        rho_k = rho
        fact = 1.0
        for k in range(2, kmax + 1):
            rho_k = rho_k * rho
            fact *= k
            acc = acc + np.outer(a[k], a[k]) * (rho_k / fact)
        cov = acc
        np.fill_diagonal(cov, var_post)
        cov = np.asarray(0.5 * (cov + cov.T), dtype=np.float32)
        prev = (a, rho)
        rows.append(mu.astype(np.float32))
        mu = mu.astype(np.float32)
    out = np.stack(rows, axis=0)
    st["nonfinite"] = int(np.sum(~np.isfinite(out)))
    return out, st


def run_batch(seeds, width, depth, damp, umax, label):
    tot = {}
    per_neg = []
    for sd in seeds:
        W = make_mlp(width, depth, sd)
        out, st = instrumented(W, umax=umax, damp=damp)
        for k, v in st.items():
            if k.startswith("max_"):
                tot[k] = max(tot.get(k, 0.0), v)
            elif k.startswith("worst_"):
                tot[k] = min(tot.get(k, 0.0), v)
            else:
                tot[k] = tot.get(k, 0) + v
        per_neg.append(st["neg_mean"])
    n_cells = tot["n_cells"]
    print(f"[{label}] width={width} depth={depth} damp={damp} umax={umax} "
          f"n_mlps={len(seeds)}  ({n_cells:,} neuron-layer cells)")
    print(f"   var_pre at 1e-12 floor : {tot['var_floor_hits']:,} "
          f"({tot['var_floor_hits'] / n_cells:.3%})")
    print(f"   sig == 0               : {tot['sig_zero']:,}")
    print(f"   |rho| clipped off-diag : {tot['rho_clip_hi'] + tot['rho_clip_lo']:,}"
          f" of {tot['n_offdiag']:,} "
          f"({(tot['rho_clip_hi'] + tot['rho_clip_lo']) / tot['n_offdiag']:.3%})"
          f"  [hi {tot['rho_clip_hi']:,} / lo {tot['rho_clip_lo']:,}]")
    print(f"   NEGATIVE predicted mean: {tot['neg_mean']:,} "
          f"({tot['neg_mean'] / n_cells:.3%})   worst value "
          f"{tot['worst_neg']:.4e}   (before correction: "
          f"{tot['neg_mean_before']:,})")
    print(f"   |correction| > |mean|  : {tot['corr_dominates']:,} "
          f"({tot['corr_dominates'] / n_cells:.3%})   max ratio "
          f"{tot['max_corr_ratio']:.3e}")
    print(f"   non-finite in output   : {tot['nonfinite']:,}   "
          f"k3 non-finite: {tot['k3_nonfinite']:,}")
    print(f"   per-MLP negative counts: {per_neg}")
    print()
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mlps", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=31000)
    args = ap.parse_args()

    seeds = [args.seed_base + i for i in range(args.n_mlps)]
    print("=" * 78)
    print("A. competition shape, shipped configuration")
    print("=" * 78)
    run_batch(seeds, 256, 32, 1.0, 1, "shipped")
    run_batch(seeds[:8], 256, 32, 0.0, 1, "damp=0 control")
    run_batch(seeds[:8], 256, 32, 2.25, 1, "damp=2.25")

    print("=" * 78)
    print("B. deeper / narrower, where the correction compounds further")
    print("=" * 78)
    run_batch(seeds[:8], 256, 64, 1.0, 1, "depth 64")
    run_batch(seeds[:8], 64, 64, 1.0, 1, "width 64 depth 64")
    run_batch(seeds[:8], 32, 96, 1.0, 1, "width 32 depth 96")

    print("=" * 78)
    print("C. hostile weight matrices (not He-initialised)")
    print("=" * 78)
    n, d = 64, 16
    rng = np.random.default_rng(7)
    cases = {
        "rank-1 every layer": [np.outer(rng.standard_normal(n),
                                        rng.standard_normal(n)).astype(np.float32)
                               for _ in range(d)],
        "tiny scale 1e-8": [(rng.standard_normal((n, n)) * 1e-8).astype(np.float32)
                            for _ in range(d)],
        "huge scale 1e3": [(rng.standard_normal((n, n)) * 1e3).astype(np.float32)
                           for _ in range(d)],
        "all zeros": [np.zeros((n, n), dtype=np.float32) for _ in range(d)],
        "identity": [np.eye(n, dtype=np.float32) for _ in range(d)],
        "one dead column": None,
    }
    base = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32)
            for _ in range(d)]
    dead = [w.copy() for w in base]
    dead[3][:, 0] = 0.0
    cases["one dead column"] = dead

    for nm, W in cases.items():
        try:
            out, st = instrumented(W, umax=1, damp=1.0)
            print(f"  {nm:22s} floor {st['var_floor_hits']:6d}  "
                  f"rho-clip {st['rho_clip_hi'] + st['rho_clip_lo']:8d}  "
                  f"neg-mean {st['neg_mean']:5d}  "
                  f"nonfinite {st['nonfinite']:5d}  "
                  f"max|corr|/|mu| {st['max_corr_ratio']:.3e}  "
                  f"out range [{np.nanmin(out):.3e}, {np.nanmax(out):.3e}]")
        except Exception as e:  # noqa: BLE001
            print(f"  {nm:22s} RAISED {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
