#!/usr/bin/env python
"""Is there 6-9x of BILLED FLOPs per sample to cut?  Three routes, measured.

The premise this script tests
-----------------------------
For any sampler ``adjusted = v_eff * c / B`` in the zero-bias limit, so the
score is the product of residual per-sample variance and billed FLOPs per
sample.  ``docs/hermite_rank_ceiling.md`` caps ``v_eff`` at 1.76x and 1.66x is
realised, so a 5.9x score gain has to come from ``c``: 2.82e6 -> ~4.8e5.

``c`` can only fall three ways.  (1) Evaluate fewer neurons per layer -- that
is the shipped ``alpha < -tau`` mask, and ``scripts/35`` shows its own optimum
is ``tau = 2.5``.  (2) Fold neurons OUT of the per-sample matmuls by making
them affine, so consecutive affine blocks compose into one precomputed matrix
(``--mode linfold``).  (3) Stop doing a full forward pass at all: replace a
PREFIX of the network by a distribution and sample from that (``--mode
gauss``).  A fourth, weaker, route is to keep the mask but choose it better
(``--mode mask``).

All three are measured here and all three fail, each for a different and
quantified reason.  ``--mode gauss`` is the important one, because it is a
FLOOR on the entire family rather than a statement about one construction: it
uses ORACLE moments, so no practical scheme beats it.

Selection data
--------------
LOCAL MLPs only (``--seed-base``, default 700000), disjoint from the official
suite and from the corrector's training seeds.  ``official_mini.npz`` is never
read.  Nothing here is fitted; every number is a measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whestfloor.contract import DEPTH, WIDTH  # noqa: E402
from whestfloor.corrector import norm_cdf, norm_pdf  # noqa: E402
from whestfloor.mc import make_mlp  # noqa: E402

CHUNK = 1024
VAR_FLOOR = 1e-12
#: One layer of the dense per-sample pass: ``width * width * 2``.  Every cost
#: in this file is quoted in these units so the comparison is scale free.
LAYER_EQ = WIDTH * WIDTH * 2


def artifacts() -> Path:
    p = Path(os.environ.get("WHEST_ARTIFACTS", "_artifacts")).resolve() / "cost"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Shared: a long dense pass that returns whatever moments the caller asks for.
# ---------------------------------------------------------------------------
def dense_pass(W, seed, N, *, want_cov=(), want_relu_moments=False):
    """``E[relu(z^32)]`` plus per-layer moments.  One streamed pass.

    ``want_cov`` is a set of 1-based layer indices for which the FULL
    ``(width, width)`` pre-activation covariance is accumulated.
    """
    n, L = W[0].shape[0], len(W)
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    s1 = np.zeros((L, n))
    s2 = np.zeros((L, n))
    h1 = np.zeros((L, n))
    h2 = np.zeros((L, n))
    hz = np.zeros((L, n))
    cov = {j: np.zeros((n, n)) for j in want_cov}
    done = 0
    while done < N:
        b = min(CHUNK, N - done)
        x = rng.standard_normal((b, n), dtype=np.float32)
        for l, w in enumerate(W):
            z = x @ w
            s1[l] += z.sum(0, dtype=np.float64)
            s2[l] += (z.astype(np.float64) ** 2).sum(0)
            if (l + 1) in cov:
                cov[l + 1] += (z.T @ z).astype(np.float64)
            x = np.maximum(z, 0.0)
            if want_relu_moments:
                h1[l] += x.sum(0, dtype=np.float64)
                h2[l] += (x.astype(np.float64) ** 2).sum(0)
                hz[l] += (x.astype(np.float64) * z).sum(0)
        out += x.sum(0, dtype=np.float64)
        done += b
    m = s1 / N
    v = np.maximum(s2 / N - m * m, VAR_FLOOR)
    res = {"mu": out / N, "m": m, "s": np.sqrt(v), "alpha": m / np.sqrt(v),
           "cov": {j: cov[j] / N - np.outer(m[j - 1], m[j - 1]) for j in cov}}
    if want_relu_moments:
        eh = h1 / N
        res["eh"] = eh
        res["vh"] = np.maximum(h2 / N - eh * eh, 0.0)
        # Cov(relu z, z) = Var(z) Phi(alpha) by Stein; measured, not assumed.
        res["czh"] = hz / N - eh * m
    return res


def jacobians(W, Phi):
    """``R[l]`` maps ``x^l`` to ``z^32`` under the mean-field linearisation.

    ``R[L-2] = W[L-1]``; ``R[l] = W[l+1] diag(Phi^{l+1}) R[l+1]``.  31 matmuls.
    """
    L = len(W)
    R = [None] * L
    R[L - 2] = W[L - 1].astype(np.float64)
    for l in range(L - 3, -1, -1):
        R[l] = W[l + 1] @ (Phi[l + 1][:, None] * R[l + 1])
    return R


# ---------------------------------------------------------------------------
# mode linfold -- can affine neurons be folded out of the matmuls?
# ---------------------------------------------------------------------------
def fold_cost(m, d, k):
    """Billed FLOPs/sample for ONE layer inside a fold of ``k`` layers.

    Inside a fold nothing but the EXACT neurons is materialised: the affine
    part of every layer composes into one precomputed matrix.  With state dim
    ``d`` re-materialised every ``k`` layers and ``m`` exact neurons a layer,
    the fold's total is

        2 d^2 + 4 (k-1) d m + (k-1)(k-2) m^2

    (``d`` sources to the ``m`` exact columns of each layer, the accumulated
    exact outputs feeding every later layer in the window, and one ``d x d``
    re-materialisation at the end).  ``k = 1`` returns ``2 d^2``, the
    unfolded cost, so the formula is its own ablation.
    """
    return (2.0 * d * d + 4.0 * (k - 1) * d * m
            + (k - 1) * (k - 2) * m * m) / k


def mode_linfold(n_mlps, seed_base, n_samples, out):
    """A neuron is affine to first order, not just when ``|alpha|`` is large.

    ``relu(z) = a + b z + delta`` with ``b = Phi(alpha)`` the L2-optimal gain
    (Stein) and ``E[delta] = 0`` by choice of ``a``.  Every neuron treated that
    way leaves the per-sample matmuls entirely -- the affine parts compose.
    What it costs is ``Var(delta) = Var(relu z) - Phi^2 Var(z)``, pushed to the
    output by the mean-field Jacobian.  So the question is whether that
    "damage" is CONCENTRATED: if a few hundred neurons carry it, the exact set
    is small, the fold is deep and ``c`` collapses.
    """
    print(f"# linearise-and-fold, {n_mlps} LOCAL MLPs (seeds {seed_base}..), "
          f"{n_samples:,} samples")
    print("# damage(l,i) = 1/2 * Var(relu - Phi z) * sum_j (phi_j/s_j) "
          "R_l[i,j]^2\n#   = the bias that linearising neuron (l,i) puts on "
          "the scored mean\n")
    tops = (8, 32, 128, 256, 512, 1024, 2048, 4096)
    share = {k: [] for k in tops}
    rows = []
    for i in range(n_mlps):
        W = make_mlp(WIDTH, DEPTH, seed_base + i)
        st = dense_pass(W, seed_base + i + 991, n_samples,
                        want_relu_moments=True)
        s, alpha = st["s"], st["alpha"]
        Phi = norm_cdf(alpha)
        b_opt = st["czh"] / (s * s)
        resid = np.maximum(st["vh"] - st["czh"] ** 2 / (s * s), 0.0)
        R = jacobians(W, Phi)
        w32 = norm_pdf(alpha[-1]) / s[-1]
        dmg = np.stack([0.5 * resid[l] * ((R[l] ** 2) @ w32)
                        for l in range(DEPTH - 1)])
        flat = np.sort(dmg.ravel())[::-1]
        tot = float(flat.sum())
        for k in tops:
            share[k].append(float(flat[:k].sum()) / tot)
        rows.append({"seed": seed_base + i, "stein_max_err":
                     float(np.abs(b_opt - Phi).max()), "total_damage": tot,
                     "resid_over_vrelu": float(resid.sum() / st["vh"].sum())})
        print(f"  mlp {seed_base+i}: |b_opt - Phi|max = "
              f"{rows[-1]['stein_max_err']:.2e} (Stein check), "
              f"sum resid / sum Var(relu) = "
              f"{rows[-1]['resid_over_vrelu']:.4f}", flush=True)

    print(f"\n# concentration of the damage over all {DEPTH-1}*{WIDTH} = "
          f"{(DEPTH-1)*WIDTH} linearisable neurons")
    print(f"{'exact set |E|':>14}{'share of damage':>18}"
          f"{'damage left':>14}{'implied b^2':>13}")
    mean_tot = float(np.mean([r["total_damage"] for r in rows]))
    tab = []
    for k in tops:
        sh = float(np.mean(share[k]))
        left = mean_tot * (1.0 - sh)
        b2 = (left / WIDTH) ** 2          # mean bias per output neuron
        tab.append({"E": k, "share": sh, "b2": b2})
        print(f"{k:>14}{sh:>18.4f}{left:>14.4e}{b2:>13.3e}")

    print(f"\n# what that exact set costs, from fold_cost(m, d={WIDTH}, k) -- "
          f"m = |E| / {DEPTH-1} per layer")
    print(f"{'|E|':>7}{'m/layer':>9}{'best k':>8}{'c/sample':>12}"
          f"{'x dense':>9}{'b^2':>11}")
    for t in tab:
        m = t["E"] / (DEPTH - 1)
        best = min(((fold_cost(m, WIDTH, k), k) for k in range(1, 33)))
        c = best[0] * DEPTH
        print(f"{t['E']:>7}{m:>9.1f}{best[1]:>8d}{c:>12,.0f}"
              f"{DEPTH*LAYER_EQ/c:>9.2f}{t['b2']:>11.3e}")
    print("\nfold_cost is minimised at k=1 (no fold at all) as soon as "
          f"m >= d/2 = {WIDTH//2}: the fold pays only while the exact set is\n"
          "under half the state, and the damage table says the exact set has "
          "to be about half the network.")
    (artifacts() / out).write_text(json.dumps(
        {"n_mlps": n_mlps, "seed_base": seed_base, "n_samples": n_samples,
         "per_mlp": rows, "table": tab}, indent=1))
    print(f"\nwrote {artifacts() / out}")


# ---------------------------------------------------------------------------
# mode gauss -- the FLOOR on every cheap-model sampler
# ---------------------------------------------------------------------------
def run_from(W, j, m, C, N, seed):
    """Draw ``z^j ~ N(m, C)`` and run layers ``j+1 .. 32`` exactly."""
    n = W[0].shape[0]
    ev, U = np.linalg.eigh(C)
    Lh = ((U * np.sqrt(np.maximum(ev, 0.0))) @ U.T).astype(np.float32)
    m32 = m.astype(np.float32)
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    done = 0
    while done < N:
        b = min(CHUNK, N - done)
        x = np.maximum(
            rng.standard_normal((b, n), dtype=np.float32) @ Lh + m32, 0.0)
        for w in W[j:]:
            x = np.maximum(x @ w, 0.0)
        acc += x.sum(0, dtype=np.float64)
        done += b
    return acc / N


def mode_gauss(n_mlps, seed_base, n_ref, n_sim, js, out):
    """How cheap can a sampler be before its MODEL error dominates?

    To cost less than a full forward pass a sampler must replace a prefix of
    the network by a distribution.  Give it the most favourable one there is:
    an exact Gaussian with the ORACLE mean and covariance of ``z^j``, measured
    from a long Monte-Carlo pass.  Then run layers ``j+1..32`` exactly.  Cost
    is ``(33-j)`` layer-equivalents a sample (one for the correlated draw).

    ``b^2`` is measured the way ``scripts/35`` measures pruning bias: two
    independent (surrogate, reference) pairs, ``b^2 = mean_j e_1j e_2j``, which
    is unbiased because the two errors are independent.  No practical scheme
    beats this curve, because no practical scheme has the oracle covariance.
    """
    print(f"# cheap-model FLOOR, {n_mlps} LOCAL MLPs, oracle Sigma from "
          f"{n_ref:,} samples, {n_sim:,} surrogate samples")
    print("# b^2 = mean_j e_1j e_2j over two independent streams "
          "(unbiased).  ORACLE moments: this is a floor.\n")
    acc = {j: [] for j in js}
    acc["closure"] = []
    t0 = time.time()
    for i in range(n_mlps):
        W = make_mlp(WIDTH, DEPTH, seed_base + i)
        e = {}
        for rep in (0, 1):
            st = dense_pass(W, (seed_base + i) * 13 + rep, n_ref,
                            want_cov=tuple(js) + (DEPTH,))
            A = st["mu"]
            C32 = st["cov"][DEPTH]
            s32 = np.sqrt(np.maximum(np.diag(C32), VAR_FLOOR))
            a32 = st["m"][-1] / s32
            e.setdefault("closure", []).append(
                st["m"][-1] * norm_cdf(a32) + s32 * norm_pdf(a32) - A)
            for j in js:
                p = run_from(W, j, st["m"][j - 1], st["cov"][j], n_sim,
                             (seed_base + i) * 13 + 97 * rep + j)
                e.setdefault(j, []).append(p - A)
        for k in acc:
            acc[k].append(float(np.mean(e[k][0] * e[k][1])))
        print(f"  mlp {seed_base+i} ({time.time()-t0:.0f}s): closure "
              f"{acc['closure'][-1]:.3e}  " +
              "  ".join(f"j{j}:{acc[j][-1]:.3e}" for j in js), flush=True)

    print(f"\n{'scheme':<46}{'c (layer-eq)':>13}{'c/sample':>11}"
          f"{'x dense':>9}{'b^2':>12}{'rms b':>10}")
    rows = []
    def emit(label, k, ce):
        b2 = float(np.mean(acc[k]))
        rows.append({"scheme": label, "layer_eq": ce, "b2": b2})
        print(f"{label:<46}{ce:>13d}{ce*LAYER_EQ:>11,d}"
              f"{DEPTH/ce:>9.2f}{b2:>12.3e}{np.sqrt(abs(b2)):>10.3e}")
    emit("closed form m Phi + s phi from exact (m,s)", "closure", 1)
    for j in js:
        emit(f"gaussian at z^{j}, {DEPTH-j} exact layers after", j,
             DEPTH - j + 1)
    print(f"\n{'':<46}{'':>13}{'':>11}{DEPTH:>9d}"
          f"{'0 (exact)':>12}{'':>10}   <- the full pass")
    (artifacts() / out).write_text(json.dumps(
        {"n_mlps": n_mlps, "seed_base": seed_base, "n_ref": n_ref,
         "n_sim": n_sim, "rows": rows}, indent=1))
    print(f"\nwrote {artifacts() / out}")


# ---------------------------------------------------------------------------
# mode mask -- is ``alpha < -tau`` the wrong mask?
# ---------------------------------------------------------------------------
def mode_mask(n_mlps, seed_base, n_samples, taus):
    """``alpha`` ranks neurons by ``Var(relu z)`` alone.  The damage a frozen
    neuron does is ``Var(relu z)`` TIMES its downstream sensitivity, and the
    mask ought to be ranked by the product.  How much does that buy?
    """
    print(f"# damage-ranked mask vs the shipped alpha rule, {n_mlps} MLPs\n")
    for i in range(n_mlps):
        W = make_mlp(WIDTH, DEPTH, seed_base + i)
        st = dense_pass(W, seed_base + i + 991, n_samples,
                        want_relu_moments=True)
        alpha, s = st["alpha"], st["s"]
        R = jacobians(W, norm_cdf(alpha))
        w32 = norm_pdf(alpha[-1]) / s[-1]
        sens = np.stack([(R[l] ** 2) @ w32 for l in range(DEPTH - 1)])
        D = (st["vh"][:DEPTH - 1] * sens).ravel()
        a = alpha[:DEPTH - 1].ravel()
        o = np.argsort(D)
        cs = np.cumsum(D[o])
        print(f"  mlp {seed_base+i}: downstream sensitivity spread within a "
              f"layer (p90/p10) = " +
              " ".join(f"L{l+1}:{np.quantile(sens[l],.9)/np.quantile(sens[l],.1):.1f}"
                       for l in (0, 7, 15, 30)))
        for tau in taus:
            sel = a < -tau
            k_a = int(sel.sum())
            k_d = int(np.searchsorted(cs, D[sel].sum()))
            print(f"    tau={tau:4.2f}: alpha freezes {k_a:5d}, damage-ranked "
                  f"freezes {k_d:5d} at the same damage -> {k_d/max(k_a,1):.3f}x"
                  f"  (cost x{((1-k_a/D.size)/(1-k_d/D.size))**2:.3f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("linfold", "gauss", "mask"))
    ap.add_argument("--n-mlps", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=700_000)
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--n-ref", type=int, default=400_000)
    ap.add_argument("--n-sim", type=int, default=400_000)
    ap.add_argument("--js", default="30,29,28,26,24,20")
    ap.add_argument("--taus", default="3.0,2.5,2.0,1.5")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.mode == "linfold":
        mode_linfold(a.n_mlps, a.seed_base, a.n_samples,
                     a.out or "linfold.json")
    elif a.mode == "gauss":
        mode_gauss(a.n_mlps, a.seed_base, a.n_ref, a.n_sim,
                   [int(v) for v in a.js.split(",")], a.out or "gauss.json")
    else:
        mode_mask(a.n_mlps, a.seed_base, a.n_samples,
                  [float(v) for v in a.taus.split(",")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
